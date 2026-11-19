"""MANUS glove bridge: MANUS SDK (Linux, integrated mode) -> ``GloveState`` messages on ZMQ.

Runs on the Linux computer the gloves' dongle is plugged into; the MANUS SDK drives the
dongle itself (no MANUS Core). Teleop on any computer listens to it with
``orca-teleop-hand --source manus --glove-address <this computer>:2044``.

Needs only this folder (``bridge.py``, ``conversion.py``, ``wire.py``, ``source/``), cffi,
numpy, pyzmq, protobuf, and the MANUS SDK's ``Python`` folder on ``PYTHONPATH``.

Every SDK callback goes out as one message, unfiltered and at the SDK's rate, stamped with
this computer's clock (Unix ns). Raw skeleton messages hold the hands: 25 nodes each, in
``conversion.MANUS_NODE_NAMES`` order, sorted by the SDK's node labels (``NodeInfo`` chain
and joint type), never by the SDK's undocumented array order. Every stream in ``STREAMS``
also carries the SDK's own structs as JSON (``wire.sdk_data``), unless the licence lacks
the feature it needs. Each glove loads its wearer's calibration, ``Calibration_left.mcal``
or ``Calibration_right.mcal`` in ``--calibration-dir``, as the C++ manus-client does.
Every message also carries the bridge's active warnings (``wire.bridge_warnings``: the SDK
not connected, licence, uncalibrated or lost glove), and a status message carries them at
least once a second, so a teleop that connects later, or gets no hands, hears them too.
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
from collections.abc import Callable
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
# The SDK streams sent, with the licence features (LicenseInfo, ManusSDKTypes.h) each needs.
# MANUS's own ROS 2 node sends nothing in integrated mode without "integrated"; that "raw"
# gates the raw device (sensor) data is our reading of its name: the SDK doesn't say. The
# SDK enforces the licence; the bridge says what is missing and leaves those streams out,
# except the hands: whether they come is the SDK's call.
STREAMS = {
    "raw_skeleton": ("integrated",),
    "ergonomics": ("integrated",),
    "raw_device_data": ("integrated", "raw"),
    "landscape": (),
    "system": (),
}
_FEATURE_NAMES = {"integrated": "SDK Integrated", "raw": "Raw"}
_SIDE_NAMES = {1: "left", 2: "right"}  # ManusSDKTypes.h Side
_HAND_CHAIN = 13  # ChainType_Hand
# A MANUS SDK bug (3.2.x integrated): CoreSdk_InitializeIntegrated aborts with
# "basic_string::_M_construct null not valid" (PlatformSpecific::GetLocale, from its
# telemetry's GetLanguage) unless all of these are set, so over SSH, from a service or in
# a bare environment the bridge sets the missing ones.
_LOCALE_VARS = ("LANG", "LC_ALL", "LC_MESSAGES", "LANGUAGE")
_STATS_EVERY_S = 5.0
LOST_AFTER_S = 1.0  # a glove that sends nothing this long is reported lost
STATUS_EVERY_S = 1.0  # with nothing else to send, a status message this often


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


def sdk_plain(ffi: Any, value: Any, ctype: Any = None) -> Any:
    """A MANUS SDK struct (a cffi pointer, or a value of ``ctype``) as JSON-ready Python.

    Structs become dicts, arrays lists cut to their count field (``gloves`` to
    ``gloveCount``), char arrays text, enums their names without the type's prefix, and
    floats are rounded to 1e-6. No per-field code: new SDK fields come along by themselves.
    """
    if ctype is None:
        ctype, value = ffi.typeof(value).item, value[0]
    if ctype.kind == "struct":
        fields, out = dict(ctype.fields), {}
        for name, field in ctype.fields:
            item, kind = getattr(value, name), field.type
            count = _count_field(fields, name) if kind.kind == "array" else None
            if count:
                n = min(getattr(value, count), kind.length)
                out[name] = [sdk_plain(ffi, item[i], kind.item) for i in range(n)]
            else:
                out[name] = sdk_plain(ffi, item, kind)
        return out
    if ctype.kind == "array":
        if ctype.item.cname == "char":
            return ffi.string(value).decode(errors="replace")
        return [sdk_plain(ffi, item, ctype.item) for item in value]
    if ctype.kind == "enum":
        name = ctype.elements.get(value, value)
        return name.removeprefix(ctype.cname + "_") if isinstance(name, str) else name
    if ctype.kind in ("pointer", "function"):
        return None
    return round(value, 6) if isinstance(value, float) else value


def _count_field(fields: dict, name: str) -> str | None:
    """The field that counts array ``name``: gloves -> gloveCount, data -> dataCount,
    sensorData -> sensorCount, netDevices -> numberOfNetDevices."""
    for count in (
        name[:-1] + "Count",
        name + "Count",
        name.removesuffix("Data") + "Count",
        "numberOf" + name[:1].upper() + name[1:],
    ):
        if count != name and count in fields:
            return count
    return None


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
        # (t_ns, stream, the SDK's data, copied) per SDK callback, in order.
        self._frames: queue.SimpleQueue = queue.SimpleQueue()
        self._layouts: dict[int, tuple[list[int], str, list[int]]] = {}  # ids, side, rows
        self._calibrated_since: dict[int, int] = {}  # glove id -> t_ns; 0: none loaded
        self._blocked: set[str] = set()  # streams left out for the licence
        self._licence_logged = False
        self._last_seen: dict[int, tuple[float, str]] = {}  # glove id -> (monotonic, side)
        self._sent, self._t_sent = 0, time.monotonic()  # messages sent, and when the last
        self.warnings: dict[str, str] = {}  # code -> text, in every message while they last
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

    def start_sdk(self, socket: Any = None) -> None:
        """Initialize (integrated), register the stream callbacks, set the frame, connect."""
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
        for stream, register, callback in (
            ("raw_skeleton", sdk.register_raw_skeleton_callback, self._on_skeletons),
            ("ergonomics", sdk.register_ergonomics_callback, self._copier("ergonomics")),
            ("raw_device_data", sdk.register_raw_device_data_callback, self._on_devices),
            ("landscape", sdk.register_landscape_callback, self._copier("landscape")),
            ("system", sdk.register_system_callback, self._copier("system")),
        ):
            self._check(register(callback), f"the {stream} callback")
        vuh = ffi.new("CoordinateSystemVUH*")
        vuh.view = enums.AxisView.XFromViewer  # the frame conversion.py documents
        vuh.up = enums.AxisPolarity.PositiveZ
        vuh.handedness = enums.Side.Right
        vuh.unitScale = 1.0  # metres
        self._check(lib.CoreSdk_InitializeCoordinateSystemWithVUH(vuh[0], True), "coordinates")
        # Integrated mode still goes through the SDK's host search, as MANUS's examples do.
        while not self._stop.is_set() and not self._connect():
            self._warn("not_connected", "The MANUS SDK is not connected (dongle? licence?)")
            if socket is not None:  # so teleop hears why no hands come
                self._send_status(socket)
            time.sleep(1.0)
        if self._stop.is_set():
            return
        self._clear("not_connected", "The MANUS SDK is connected now")
        # The hand stays at the origin: the retargeter wants the fingers, not the wrist path.
        if not self._ok(lib.CoreSdk_SetRawSkeletonHandMotion(enums.HandMotion.None_)):
            self._warn("hand_motion", "Could not turn off the skeleton's hand motion")

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

    # ----- SDK threads: copy the data out of C memory and queue it, nothing more ----------

    def _on_skeletons(self, stream_info: Any) -> None:
        t_ns = time.time_ns()
        lib, ffi = self.lib, self.ffi
        try:
            gloves = {}
            for i in range(stream_info.skeletonsCount):
                info = ffi.new("RawSkeletonInfo*")
                if not self._ok(lib.CoreSdk_GetRawSkeletonInfo(i, info)):
                    continue
                info.publishTime = stream_info.publishTime  # as MANUS's examples do
                nodes = ffi.new("SkeletonNode[]", info.nodesCount)
                if not self._ok(lib.CoreSdk_GetRawSkeletonData(i, nodes, info.nodesCount)):
                    continue
                gloves[int(info.gloveId)] = (
                    [int(nd.id) for nd in nodes],
                    np.array([_node_row(nd) for nd in nodes], dtype=np.float64),
                    info,
                )
            if gloves:
                self._frames.put((t_ns, "raw_skeleton", gloves))
        except Exception:  # never raise into the SDK's thread
            logger.exception("Error reading a MANUS skeleton")

    def _on_devices(self, info: Any) -> None:
        t_ns = time.time_ns()
        try:
            devices = []
            for i in range(info.rawDeviceDataCount):
                device = self.ffi.new("RawDeviceData*")
                if self._ok(self.lib.CoreSdk_GetRawDeviceData(i, device)):
                    devices.append(device)
            self._frames.put((t_ns, "raw_device_data", (int(info.publishTime.time), devices)))
        except Exception:
            logger.exception("Error reading MANUS raw device data")

    def _copier(self, stream: str) -> Callable[[Any], None]:
        """A callback for a stream whose callback gets all its data as one struct."""

        def on_data(pointer: Any) -> None:
            t_ns = time.time_ns()
            try:
                copy = self.ffi.new(self.ffi.typeof(pointer))
                self.ffi.memmove(copy, pointer, self.ffi.sizeof(copy[0]))
                self._frames.put((t_ns, stream, copy))
            except Exception:
                logger.exception("Error reading MANUS %s", stream)

        return on_data

    # ----- the publish thread --------------------------------------------------------------

    def _load_calibration(self, glove_id: int, side: str) -> None:
        """Once per glove, when found: its side's .mcal, as manus-client does. Frames from
        before it are not sent. A missing or rejected file: a warning, uncalibrated data."""
        self._calibrated_since[glove_id] = 0
        path = self.calibration_dir / CALIBRATION_FILE.format(side=side)
        fix = "calibrate it with MANUS's SDK client (README, MANUS gloves)"
        try:
            data = path.read_bytes()
        except OSError as err:
            self._warn(f"uncalibrated_{side}", f"Streaming UNCALIBRATED {side} glove: {err}; {fix}")
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
            self._warn(
                f"uncalibrated_{side}",
                f"Streaming UNCALIBRATED {side} glove: the MANUS SDK rejected {path} "
                f"({call}, {code}); {fix}",
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
                self._warn(f"same_side_{cached[1]}", f"Gloves {twins} all say {cached[1]} hand")
            if glove_id not in self._calibrated_since:
                self._load_calibration(glove_id, cached[1])
        return cached[1], nodes[cached[2]]

    def _hands(self, t_ns: int, gloves: dict) -> tuple[dict, dict]:
        """A raw skeleton callback's hands in wire order, and each glove's RawSkeletonInfo."""
        hands, infos = {}, {}
        for glove_id, (ids, nodes, info) in gloves.items():
            try:
                side, hand = self.wire_hand(glove_id, ids, nodes)
            except (RuntimeError, ValueError) as err:
                self._warn(f"glove_{glove_id}_unusable", f"Skipping glove {glove_id}: {err}")
                continue
            self._clear(f"glove_{glove_id}_unusable", f"Glove {glove_id} works now")
            self._last_seen[glove_id] = (time.monotonic(), side)
            self._clear(f"glove_{glove_id}_lost", f"The {side} glove ({glove_id}) is back")
            if t_ns >= self._calibrated_since[glove_id]:
                hands[side], infos[side] = hand, sdk_plain(self.ffi, info)
        return hands, infos

    def _sdk_data(self, stream: str, data: Any) -> Any:
        """Any other stream's data as JSON-ready Python."""
        if stream == "raw_device_data":
            publish_time, devices = data
            devices = [sdk_plain(self.ffi, device) for device in devices]
            return {"publishTime": {"time": publish_time}, "devices": devices}
        data = sdk_plain(self.ffi, data)
        if stream == "ergonomics":  # name each value by its ErgonomicsDataType
            kind = self.ffi.typeof("ErgonomicsDataType")
            for glove in data["data"]:
                names = (sdk_plain(self.ffi, i, kind) for i in range(len(glove["data"])))
                glove["data"] = dict(zip(names, glove["data"], strict=True))
        elif stream == "landscape" and data["gloveDevices"]["dongleCount"]:
            self._check_licence(data["settings"]["license"])  # read once there's a dongle
        elif stream == "system":
            logger.info("MANUS system message: %s", data)
        return data

    def _check_licence(self, licence: dict) -> None:
        """On each landscape with a dongle: warn about each stream the licence lacks a
        feature for, and leave it out, all but the hands; clear the warning once it's there."""
        if not self._licence_logged:
            self._licence_logged = True
            logger.info("MANUS licence: %s", ", ".join(k for k, v in licence.items() if v))
        self._blocked = set()
        for stream, features in STREAMS.items():
            name, lacking = stream.replace("_", " "), [f for f in features if not licence.get(f)]
            if not lacking:
                self._clear(f"licence_{stream}", f"{name}: the MANUS licence covers it now")
                continue
            if stream != "raw_skeleton":
                self._blocked.add(stream)
            what = "hands may not come" if stream == "raw_skeleton" else "is not streamed"
            lacking = " and ".join(_FEATURE_NAMES[f] for f in lacking)
            text = f"{name} {what}: your MANUS licence does not include {lacking}"
            self._warn(f"licence_{stream}", text)

    def _check_gloves(self) -> None:
        """Warn about each glove that sent nothing for LOST_AFTER_S."""
        now = time.monotonic()
        for glove_id, (seen, side) in self._last_seen.items():
            if now - seen > LOST_AFTER_S and f"glove_{glove_id}_lost" not in self.warnings:
                text = f"No data from the {side} glove ({glove_id}) for {LOST_AFTER_S:g} s"
                self._warn(f"glove_{glove_id}_lost", text + " (off, out of range, battery?)")

    def _warn(self, code: str, text: str, exc_info: bool = False) -> None:
        """Log a warning once, and send it in every message until it's cleared."""
        if self.warnings.get(code) != text:
            self.warnings[code] = text
            logger.warning(text, exc_info=exc_info)

    def _clear(self, code: str, text: str) -> None:
        if self.warnings.pop(code, None) is not None:
            logger.info(text)

    def _send(self, socket: Any, raw: bytes) -> None:
        socket.send(raw)
        self._sent, self._t_sent = self._sent + 1, time.monotonic()

    def _send_status(self, socket: Any) -> None:
        """A message with only the warnings: at least one goes out every STATUS_EVERY_S."""
        self._send(
            socket, wire.encode({}, time.time_ns(), self._sent, "status", None, self.warnings)
        )

    def publish_loop(self, socket: Any) -> None:
        t_stats, counts = time.monotonic(), {}
        while not self._stop.is_set():
            self._check_gloves()
            if time.monotonic() - self._t_sent >= STATUS_EVERY_S:
                self._send_status(socket)
            try:
                t_ns, stream, data = self._frames.get(timeout=STATUS_EVERY_S / 4)
            except queue.Empty:
                continue
            if stream in self._blocked:
                continue
            try:
                if stream == "raw_skeleton":
                    hands, data = self._hands(t_ns, data)
                    if not hands:
                        continue
                else:
                    hands, data = {}, self._sdk_data(stream, data)
                raw = wire.encode(hands, t_ns, self._sent, stream, data, self.warnings)
            except Exception as err:  # one odd message must not end the bridge
                if f"bad_{stream}" not in self.warnings:
                    text = f"A {stream} message failed (first error: {err!r})"
                    self._warn(f"bad_{stream}", text, True)
                continue
            self._send(socket, raw)
            for key in hands or (stream,):
                counts[key] = counts.get(key, 0) + 1
            if self._t_sent - t_stats >= _STATS_EVERY_S:
                elapsed = self._t_sent - t_stats
                rates = (f"{k} {n / elapsed:.0f}" for k, n in sorted(counts.items()))
                logger.info("Sent per second: %s", ", ".join(rates))
                t_stats, counts = self._t_sent, {}

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
            self.start_sdk(socket)
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
