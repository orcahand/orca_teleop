"""MANUS glove bridge: MANUS SDK (Linux, integrated mode) -> ``GloveState`` messages on ZMQ.

Runs on the Linux computer the gloves' dongle is plugged into; the MANUS SDK drives the
dongle itself (no MANUS Core). Teleop on any computer listens to it with
``orca-teleop-hand --source manus --glove-address <this computer>:2044``.

Needs only this folder (``bridge.py``, ``conversion.py``, ``wire.py``, ``source/``), cffi,
numpy, pyzmq, protobuf, and the MANUS SDK's ``Python`` folder on ``PYTHONPATH``.

Each SDK skeleton callback goes out as one message, unfiltered and at the SDK's rate,
stamped with this computer's clock (Unix ns), with every glove the callback holds. Each
hand is 25 nodes in ``conversion.MANUS_NODE_NAMES`` order, sorted by the SDK's node labels
(``NodeInfo`` chain and joint type), never by the SDK's undocumented array order. Each
glove loads its wearer's calibration, ``Calibration_left.mcal`` or
``Calibration_right.mcal`` in ``--calibration-dir``, as the C++ manus-client does.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from orca_teleop.ingress.manus import conversion, wire
else:  # run as a file on the glove computer, without orca_teleop installed
    import conversion
    import wire

logger = logging.getLogger("manus_bridge")

DEFAULT_BIND = f"tcp://127.0.0.1:{wire.DEFAULT_PORT}"  # this computer only
CALIBRATION_FILE = "Calibration_{side}.mcal"  # manus-client's and its CalibrationGUI's names
_SIDE_NAMES = {1: "left", 2: "right"}  # ManusSDKTypes.h Side
_HAND_CHAIN = 13  # ChainType_Hand
# A MANUS SDK bug (3.2.x integrated): CoreSdk_InitializeIntegrated aborts with
# "basic_string::_M_construct null not valid" (PlatformSpecific::GetLocale, from its
# telemetry's GetLanguage) unless all of these are set, so over SSH, from a service or in
# a bare environment the bridge sets the missing ones.
_LOCALE_VARS = ("LANG", "LC_ALL", "LC_MESSAGES", "LANGUAGE")
_STATS_EVERY_S = 5.0


def sort_nodes(infos: list[tuple], node_ids: list[int]) -> tuple[str, list[int]]:
    """One glove's side, and the rows of its node data in wire order.

    ``infos`` are the SDK's node labels ``(nodeId, parentId, chainType, side,
    fingerJointType)``, ``node_ids`` the node data's ids, each in the SDK's (any) order;
    rows are matched to labels by node id. ``ValueError`` if the labels aren't one hand.
    """
    sides = {_SIDE_NAMES.get(side) for _, _, chain, side, _ in infos if chain} - {None}
    if len(sides) != 1:
        raise ValueError(f"a glove skeleton should have one side, got {sides or 'none'}")
    ids = {node_id for node_id, *_ in infos}
    names = []
    for node_id, parent, chain, _, joint in infos:
        chain_name = conversion.CHAIN_TYPES.get(chain, "")
        name = conversion.node_name(chain_name, conversion.FINGER_JOINT_TYPES.get(joint, ""))
        # The wrist is the hand chain's root: its parent is itself or outside the skeleton.
        wrist = chain != _HAND_CHAIN or parent == node_id or parent not in ids
        names.append(name if wrist else None)
    row = {node_id: i for i, node_id in enumerate(node_ids)}
    try:
        return sides.pop(), [row[infos[k][0]] for k in conversion.wire_order(names)]
    except KeyError as err:
        raise ValueError(f"node {err} has a label but no data") from None


def fill_locale(env: Any = os.environ) -> list[str]:
    """Set the locale variables the MANUS SDK needs that are missing, from LANG (else
    C.UTF-8; LANGUAGE: en_US:en). Returns the names it set."""
    missing = [name for name in _LOCALE_VARS if not env.get(name)]
    lang = env.get("LANG") or "C.UTF-8"
    for name in missing:
        env[name] = "en_US:en" if name == "LANGUAGE" else lang
    return missing


def _name(enum: Any, value: Any) -> Any:
    """An SDK return code's name, for the log."""
    return next((code.name for code in enum if code == value), value)


def _node_row(node: Any) -> list[float]:
    """A ``SkeletonNode``'s position and rotation quaternion (w, x, y, z)."""
    p, r = node.transform.position, node.transform.rotation
    return [p.x, p.y, p.z, r.w, r.x, r.y, r.z]


class ManusBridge:
    """The MANUS SDK (integrated mode) on one side, a ZMQ PUB socket on the other."""

    def __init__(self, bind: str = DEFAULT_BIND, calibration_dir: str | Path = "."):
        self.bind = bind
        self.calibration_dir = Path(calibration_dir)
        # (t_ns, {glove id: (node ids, (N, 7) nodes)}) per SDK callback, in order.
        self._frames: queue.SimpleQueue = queue.SimpleQueue()
        self._layouts: dict[int, tuple[list[int], str, list[int]]] = {}  # ids, side, rows
        self._calibrated_since: dict[int, int] = {}  # glove id -> t_ns; 0: none loaded
        self._warned: set[str] = set()
        self._stop = threading.Event()
        self.lib: Any = None  # the MANUS SDK, once loaded

    def stop(self) -> None:
        self._stop.set()

    def _ok(self, result: Any) -> bool:
        return result == self.enums.SDKReturnCode.Success

    def _check(self, result: Any, what: str) -> None:
        if not self._ok(result):
            raise RuntimeError(
                f"MANUS SDK: {what} failed ({_name(self.enums.SDKReturnCode, result)})"
            )

    def start_sdk(self) -> None:
        """Initialize (integrated), register the skeleton callback, set the frame, connect."""
        if filled := fill_locale():
            logger.info("Set %s for the MANUS SDK, which aborts without them", ", ".join(filled))
        try:
            import manus_sdk
            from manus_sdk import ffi, lib
            from manus_sdk.generated import _enums as enums
        except ImportError as err:
            raise SystemExit(
                "The MANUS Python SDK isn't found: put the Python folder of the MANUS SDK "
                f"package on PYTHONPATH, and `pip install cffi` ({err})"
            ) from None
        if lib is None:  # manus_sdk only warns when its binary doesn't load
            raise SystemExit(
                f"The MANUS SDK has no binary for Python {platform.python_version()} on "
                f"{platform.machine()} (see the warning above)"
            )
        self.ffi, self.lib, self.enums = ffi, lib, enums
        self._check(lib.CoreSdk_InitializeIntegrated(), "InitializeIntegrated")
        sdk = manus_sdk.ManusSDK()
        self._check(sdk.register_raw_skeleton_callback(self._on_skeletons), "callback")
        vuh = ffi.new("CoordinateSystemVUH*")
        vuh.view = enums.AxisView.XFromViewer  # the frame conversion.py documents
        vuh.up = enums.AxisPolarity.PositiveZ
        vuh.handedness = enums.Side.Right
        vuh.unitScale = 1.0  # metres
        self._check(lib.CoreSdk_InitializeCoordinateSystemWithVUH(vuh[0], True), "coordinates")
        # Integrated mode still goes through the SDK's host search, as MANUS's examples do.
        while not self._stop.is_set() and not self._connect():
            logger.info("The MANUS SDK is not connected yet (dongle, licence?); retrying")
            time.sleep(1.0)
        if self._stop.is_set():
            return
        # The hand stays at the origin: the retargeter wants the fingers, not the wrist path.
        if not self._ok(lib.CoreSdk_SetRawSkeletonHandMotion(enums.HandMotion.None_)):
            logger.warning("Could not turn off the skeleton's hand motion; continuing")

    def _connect(self) -> bool:
        lib, ffi = self.lib, self.ffi
        count = ffi.new("uint32_t*")
        if not self._ok(lib.CoreSdk_LookForHosts(1, False)):
            return False
        if not self._ok(lib.CoreSdk_GetNumberOfAvailableHostsFound(count)) or count[0] == 0:
            return False
        hosts = ffi.new("ManusHost[]", count[0])
        if not self._ok(lib.CoreSdk_GetAvailableHostsFound(hosts, count[0])):
            return False
        if not self._ok(lib.CoreSdk_ConnectToHost(hosts[0])):
            return False
        logger.info("MANUS SDK connected (%s)", ffi.string(hosts[0].hostName).decode())
        return True

    def _on_skeletons(self, stream_info: Any) -> None:
        """SDK thread: copy each glove's nodes out of C memory and queue them, nothing more."""
        t_ns = time.time_ns()
        lib, ffi = self.lib, self.ffi
        try:
            gloves = {}
            for i in range(stream_info.skeletonsCount):
                info = ffi.new("RawSkeletonInfo*")
                if not self._ok(lib.CoreSdk_GetRawSkeletonInfo(i, info)):
                    continue
                nodes = ffi.new("SkeletonNode[]", info.nodesCount)
                if not self._ok(lib.CoreSdk_GetRawSkeletonData(i, nodes, info.nodesCount)):
                    continue
                gloves[int(info.gloveId)] = (
                    [int(nd.id) for nd in nodes],
                    np.array([_node_row(nd) for nd in nodes], dtype=np.float64),
                )
            if gloves:
                self._frames.put((t_ns, gloves))
        except Exception:  # never raise into the SDK's thread
            logger.exception("Error reading a MANUS skeleton")

    def _load_calibration(self, glove_id: int, side: str) -> None:
        """Once per glove, when found: its side's .mcal, as manus-client does. Frames from
        before it are not sent. A missing or rejected file: a warning, uncalibrated data."""
        self._calibrated_since[glove_id] = 0
        path = self.calibration_dir / CALIBRATION_FILE.format(side=side)
        fix = "calibrate it with MANUS's SDK client (README, MANUS gloves)"
        try:
            data = path.read_bytes()
        except OSError as err:
            logger.warning("Streaming UNCALIBRATED %s glove: %s (%s); %s", side, path, err, fix)
            return
        buf = self.ffi.new(f"unsigned char[{len(data)}]")  # as MANUS's sdk_client.py does
        self.ffi.memmove(buf, data, len(data))
        result = self.ffi.new("SetGloveCalibrationReturnCode*")
        call = self.lib.CoreSdk_SetGloveCalibration(glove_id, buf, len(data), result)
        code = _name(self.enums.SetGloveCalibrationReturnCode, result[0])
        if self._ok(call) and result[0] == self.enums.SetGloveCalibrationReturnCode.Success:
            self._calibrated_since[glove_id] = time.time_ns()
            logger.info("Loaded the %s glove's calibration %s (%s)", side, path, code)
        else:
            call = _name(self.enums.SDKReturnCode, call)
            logger.warning(
                "Streaming UNCALIBRATED %s glove: the MANUS SDK rejected %s (%s, %s); %s",
                *(side, path, call, code, fix),
            )

    def wire_hand(self, glove_id: int, ids: list[int], nodes: np.ndarray) -> tuple[str, Any]:
        """A glove's side and its nodes in wire order (labels read once per glove)."""
        cached = self._layouts.get(glove_id)
        if cached is None or cached[0] != ids:
            labels = self.ffi.new("NodeInfo[]", len(ids))
            got = self.lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, labels, len(ids))
            self._check(got, "node labels")
            infos = [(x.nodeId, x.parentId, x.chainType, x.side, x.fingerJointType) for x in labels]
            cached = (ids, *sort_nodes(infos, ids))
            self._layouts[glove_id] = cached
            logger.info("Glove %d is the %s hand, %d nodes", glove_id, cached[1], len(ids))
            twins = [g for g, (_, side, _) in self._layouts.items() if side == cached[1]]
            if len(twins) > 1:
                logger.warning("Gloves %s all say %s hand", twins, cached[1])
            if glove_id not in self._calibrated_since:
                self._load_calibration(glove_id, cached[1])
        return cached[1], nodes[cached[2]]

    def publish_loop(self, socket: Any) -> None:
        sent, t_stats, counts = 0, time.monotonic(), {}
        while not self._stop.is_set():
            try:
                t_ns, gloves = self._frames.get(timeout=1.0)
            except queue.Empty:
                continue
            hands = {}
            for glove_id, (ids, nodes) in gloves.items():
                try:
                    side, hand = self.wire_hand(glove_id, ids, nodes)
                except (RuntimeError, ValueError) as err:
                    if f"{glove_id}: {err}" not in self._warned:  # once, not per frame
                        self._warned.add(f"{glove_id}: {err}")
                        logger.warning("Skipping glove %d: %s", glove_id, err)
                    continue
                if t_ns >= self._calibrated_since[glove_id]:
                    hands[side] = hand
            if not hands:
                continue
            socket.send(wire.encode(hands, t_ns, sent))
            sent += 1
            for side in hands:
                counts[side] = counts.get(side, 0) + 1
            now = time.monotonic()
            if now - t_stats >= _STATS_EVERY_S:
                rates = (f"{k} {n / (now - t_stats):.0f}" for k, n in sorted(counts.items()))
                logger.info("Frames sent per second: %s", ", ".join(rates))
                t_stats, counts = now, {}

    def run(self) -> None:
        import zmq

        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        # At most 100 messages (~0.2 s) wait for a slow subscriber, then ZMQ drops: a slow
        # link must not put seconds of old hands in front of the new ones.
        socket.setsockopt(zmq.SNDHWM, 100)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(self.bind)
            logger.info("Publishing glove frames on %s", self.bind)
            self.start_sdk()
            self.publish_loop(socket)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            if self.lib is not None:
                self.lib.CoreSdk_ShutDown()
            socket.close()
            ctx.term()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help=f"ZMQ address to publish on (default: {DEFAULT_BIND}, this computer only; "
        f"tcp://0.0.0.0:{wire.DEFAULT_PORT} for the network)",
    )
    parser.add_argument(
        "--calibration-dir",
        default=".",
        metavar="DIR",
        help="the folder with Calibration_left.mcal and Calibration_right.mcal "
        "(default: the current folder)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
    bridge = ManusBridge(wire.zmq_address(args.bind), args.calibration_dir)
    for stop_signal in (signal.SIGTERM, signal.SIGHUP):  # kill, or the SSH session closing
        signal.signal(stop_signal, lambda *_: bridge.stop())  # end cleanly: SDK shutdown
    bridge.run()


if __name__ == "__main__":
    main()
