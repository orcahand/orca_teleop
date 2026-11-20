"""MANUS gloves: node naming and order, the bridge's sorting, the wire, the publisher,
and that a glove skeleton retargets exactly like the same hand's landmarks.

No MANUS SDK and no gloves: the bridge's SDK calls are replaced by node lists shaped
like the SDK's (``NodeInfo`` tuples in an SDK-like order) and by a stand-in ``manus_sdk``
module. The retargeting tests need the v2 hand model (ORCAHAND_DESCRIPTION_DIR), like
test_fidelity.
"""

from __future__ import annotations

import os
import pathlib
import queue
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from orca_teleop.constants import DEFAULT_MANUS_ADDRESS
from orca_teleop.ingress.manus import bridge, conversion, fake_bridge, wire
from orca_teleop.ingress.manus.publisher import ManusPublisher
from orca_teleop.ingress.server import HandLandmarks, IngressServer, landmark_problem

zmq = pytest.importorskip("zmq")

CLIP = pathlib.Path(__file__).parent / "data" / "pinch_clip.npz"

# ManusSDKTypes.h values: ChainType Hand 13, thumb..pinky 5..9; FingerJointType 1..5.
_CHAIN = {"thumb": 5, "index": 6, "middle": 7, "ring": 8, "pinky": 9}
_JOINT = {"cmc": 1, "mcp": 2, "pip": 3, "dip": 4, "tip": 5}
_THUMB_JOINT = {"cmc": 1, "mcp": 2, "ip": 3, "tip": 5}  # the thumb has no Distal


def _sdk_skeleton(order: list[str], side: int = 2) -> list[tuple[int, int, int, int, int]]:
    """``NodeInfo`` tuples ``(nodeId, parentId, chainType, side, fingerJointType)`` for a
    skeleton whose nodes are ``order`` (our names)."""
    infos, root = [], order.index("wrist")
    for i, name in enumerate(order):
        if name == "wrist":  # the first is the root, its own parent; another hangs off it
            infos.append((i, root, 13, side, 0))
            continue
        finger, joint = name.split("_")
        table = _THUMB_JOINT if finger == "thumb" else _JOINT
        infos.append((i, 0, _CHAIN[finger], side, table[joint]))
    return infos


# Orders seen in the wild for MANUS's 25 nodes: thumb first (IsaacTeleop's plugin),
# pinky before ring (GR00T's client), and ours (orcahand/manus-client).
_FINGERS_THUMB_FIRST = ["wrist"] + [
    f"{f}_{j}"
    for f, joints in (
        ("thumb", ("cmc", "mcp", "ip", "tip")),
        ("index", ("cmc", "mcp", "pip", "dip", "tip")),
        ("middle", ("cmc", "mcp", "pip", "dip", "tip")),
        ("ring", ("cmc", "mcp", "pip", "dip", "tip")),
        ("pinky", ("cmc", "mcp", "pip", "dip", "tip")),
    )
    for j in joints
]
_PINKY_BEFORE_RING = (
    list(conversion.MANUS_NODE_NAMES[:11])
    + list(conversion.MANUS_NODE_NAMES[16:21])
    + list(conversion.MANUS_NODE_NAMES[11:16])
    + list(conversion.MANUS_NODE_NAMES[21:])
)
SDK_ORDERS = [list(conversion.MANUS_NODE_NAMES), _FINGERS_THUMB_FIRST, _PINKY_BEFORE_RING]


def _sort(order: list[str], side: int = 2) -> tuple[str, list[int]]:
    return bridge.sort_nodes(_sdk_skeleton(order, side), list(range(len(order))))


def _clip(n: int) -> np.ndarray:
    """The first ``n`` tracked frames of the pinch clip (it has gaps, as NaN)."""
    if not CLIP.exists():
        pytest.skip("tests/data/pinch_clip.npz is missing")
    world = np.load(CLIP)["world"].astype(np.float64)
    return world[np.all(np.isfinite(world), axis=(1, 2))][:n]


def _sdk_nodes(landmarks: np.ndarray, order: list[str]) -> np.ndarray:
    """What the SDK would hand the bridge for this hand: ``(25, 7)`` in ``order``, in a
    Z-up frame (rotated, not mirrored, from the landmarks' frame)."""
    wire_nodes = fake_bridge.landmarks_to_wire(landmarks @ fake_bridge._INTO_MANUS.T)
    by_name = dict(zip(conversion.MANUS_NODE_NAMES, wire_nodes, strict=True))
    positions = np.array([by_name[name] for name in order])
    return np.hstack([positions, np.tile([1.0, 0.0, 0.0, 0.0], (len(order), 1))])


# ----- names and order ----------------------------------------------------------------


def test_the_landmark_pick_is_the_one_the_legacy_client_was_aligned_to():
    assert conversion.NUM_NODES == 25
    assert conversion.MANUS_TO_MANO == [
        0, 21, 22, 23, 24, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20,
    ]  # fmt: skip


def test_nodes_are_named_by_chain_and_joint_type():
    assert conversion.node_name("Hand", "Invalid") == "wrist"
    assert conversion.node_name("FingerIndex", "Metacarpal") == "index_cmc"
    assert conversion.node_name("FingerIndex", "Proximal") == "index_mcp"
    assert conversion.node_name("FingerPinky", "Distal") == "pinky_dip"
    # The thumb has no Distal node: its Intermediate node is the IP joint.
    assert conversion.node_name("FingerThumb", "Intermediate") == "thumb_ip"
    assert conversion.node_name("FingerThumb", "Metacarpal") == "thumb_cmc"
    assert conversion.node_name("Arm", "Invalid") is None


@pytest.mark.parametrize("order", SDK_ORDERS, ids=["ours", "thumb-first", "pinky-before-ring"])
def test_the_bridge_sorts_any_sdk_order_into_the_wire_order(order):
    side, rows = _sort(order)
    assert side == "right"
    assert [order[i] for i in rows] == list(conversion.MANUS_NODE_NAMES)


def test_node_data_is_matched_to_its_info_by_node_id():
    """The data array may list the nodes in another order than the info array."""
    infos = _sdk_skeleton(_FINGERS_THUMB_FIRST)
    data_ids = list(range(25))[::-1]  # data row k holds node 24 - k
    _, rows = bridge.sort_nodes(infos, data_ids)
    names_by_row = [_FINGERS_THUMB_FIRST[node_id] for node_id in data_ids]
    assert [names_by_row[row] for row in rows] == list(conversion.MANUS_NODE_NAMES)
    with pytest.raises(ValueError, match="no data"):
        bridge.sort_nodes(infos, list(range(1, 26)))


def test_a_skeleton_that_is_not_a_hand_is_refused():
    order = list(conversion.MANUS_NODE_NAMES)
    with pytest.raises(ValueError, match="missing \\['thumb_tip'\\]"):
        _sort(order[:-1])
    infos = _sdk_skeleton(order)
    infos[3] = (3, 0, 6, 1, 3)  # one node on the left hand
    with pytest.raises(ValueError, match="one side"):
        bridge.sort_nodes(infos, list(range(25)))


def test_a_second_hand_chain_node_is_not_the_wrist():
    order = list(conversion.MANUS_NODE_NAMES) + ["wrist"]  # e.g. a palm node
    _, rows = _sort(order)
    assert rows[0] == 0 and 25 not in rows


# ----- the wire and the publisher -----------------------------------------------------


def test_a_message_round_trips_and_the_publisher_takes_its_hand():
    landmarks = _clip(1)[0]
    nodes = _sdk_nodes(landmarks, _FINGERS_THUMB_FIRST)
    _, rows = _sort(_FINGERS_THUMB_FIRST, side=1)
    raw = wire.encode({"left": nodes[rows]}, 1_700_000_000_123_456_789)
    t_ns, hands = wire.decode(raw)
    assert t_ns == 1_700_000_000_123_456_789
    assert list(hands) == ["left"] and hands["left"].shape == (25, 3)

    keypoints, sides = ManusPublisher(handedness="left").keypoints_from_message(raw)
    assert sides == ["left"]
    assert landmark_problem(keypoints) is None
    expected = (landmarks - landmarks[0]) @ fake_bridge._INTO_MANUS.T
    np.testing.assert_allclose(keypoints, expected, atol=1e-6)
    assert ManusPublisher(handedness="right").keypoints_from_message(raw) == (None, ["left"])


def _free_tcp(host: str = "127.0.0.1") -> tuple[object, object, str]:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.IPV6, int("[" in host))
    sock.bind(f"tcp://{host}:*")
    return ctx, sock, sock.getsockopt_string(zmq.LAST_ENDPOINT)


@pytest.mark.parametrize("host", ["localhost", "[::1]"])
def test_glove_frames_reach_the_ingress_server_as_landmarks(host):
    """Bridge socket -> ManusPublisher (ZMQ, then gRPC) -> IngressServer's queue, with
    another SDK stream's message right behind each hand (it must not displace the hand).
    The server and bridge addresses may be a name or an IPv6 address in brackets."""
    if host == "[::1]" and not socket.has_ipv6:
        pytest.skip("no IPv6 on this computer")
    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    server = IngressServer(landmarks_q, stop, port=0)
    port = server.start()
    ctx, sock, address = _free_tcp("[::1]" if host == "[::1]" else "127.0.0.1")
    publisher = ManusPublisher(f"{host}:{port}", "right", address)
    thread = threading.Thread(target=publisher.run, name="manus-publisher")
    thread.start()
    hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(1.0))
    try:
        deadline = time.monotonic() + 10.0
        got = None
        while got is None and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand, "left": hand}, time.time_ns()))
            sock.send(wire.encode({}, time.time_ns(), 0, "landscape", {"gloveDevices": {}}))
            try:
                got = landmarks_q.get(timeout=0.05)
            except queue.Empty:
                pass
        assert isinstance(got, HandLandmarks), "no landmarks reached the server"
        assert got.handedness == "right"
        np.testing.assert_allclose(got.keypoints, conversion.wire_to_keypoints(hand), atol=1e-6)
    finally:
        publisher.stop()
        thread.join(timeout=5.0)
        stop.set()
        server.stop(grace=0.5)
        sock.close()
        ctx.term()
    assert not thread.is_alive()


def test_the_newest_hand_wins_with_other_streams_behind_it():
    """Each hand followed by another SDK stream's message, in a burst: the publisher must
    keep the last hand, which a conflating socket (newest message only) would lose."""
    ctx, sock, address = _free_tcp()
    publisher = ManusPublisher("localhost:1", "right", address)  # its ZMQ side only
    thread = threading.Thread(target=publisher._zmq_receiver, name="zmq-receiver")
    thread.start()
    hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(1.0))
    try:
        deadline = time.monotonic() + 10.0
        while not publisher._fresh.is_set() and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand}, time.time_ns()))
            time.sleep(0.02)
        assert publisher._fresh.is_set(), "the publisher never got a hand"
        for i in range(1, 21):
            sock.send(wire.encode({"right": hand * (1 + i / 100)}, time.time_ns()))
            sock.send(wire.encode({}, time.time_ns(), 0, "ergonomics", {"data": []}))
        time.sleep(0.5)
        with publisher._lock:
            latest = publisher._latest_keypoints
        np.testing.assert_allclose(latest, conversion.wire_to_keypoints(hand * 1.2), atol=1e-6)
    finally:
        publisher.stop()
        thread.join(timeout=5.0)
        sock.close()
        ctx.term()


def test_a_glove_stream_that_fails_ends_the_publisher():
    """A bad bridge address must end the session, not leave it waiting forever."""
    stop = threading.Event()
    server = IngressServer(queue.Queue(maxsize=4), stop, port=0)
    port = server.start()
    publisher = ManusPublisher(f"localhost:{port}", "right", "tcp://no such host:port")
    thread = threading.Thread(target=publisher.run, name="manus-publisher")
    thread.start()
    thread.join(timeout=5.0)
    stop.set()
    server.stop(grace=0.5)
    assert not thread.is_alive()


def test_a_bridge_address_may_leave_out_tcp():
    assert wire.zmq_address("192.168.1.10:2044") == "tcp://192.168.1.10:2044"
    assert wire.zmq_address("tcp://10.0.0.2:2044") == "tcp://10.0.0.2:2044"
    from orca_teleop.constants import MANUS_BRIDGE_PORT

    assert MANUS_BRIDGE_PORT == wire.DEFAULT_PORT  # bridge.py can't import constants


# ----- retargeting --------------------------------------------------------------------


def _retargeter():
    """A fresh retargeter on orca_core's v2 right hand, or skip without the model."""
    try:
        from orca_teleop.utils import default_urdf_path

        default_urdf_path("right")
    except (FileNotFoundError, ValueError) as err:
        pytest.skip(f"no v2 hand model: {err}")
    from orca_teleop.retargeting.retargeter import Retargeter

    return Retargeter.from_paths(None, None)


def _retarget_all(retargeter, clip: np.ndarray) -> np.ndarray:
    from orca_teleop.retargeting.retargeter import TargetPose

    out = []
    for keypoints in clip:
        action = retargeter.retarget(TargetPose(joint_positions=keypoints, source="mediapipe"))
        if action is not None:
            out.append(action.as_array(retargeter._finger_joint_ids))
    return np.array(out, dtype=float)


def test_a_glove_skeleton_retargets_like_the_same_hands_landmarks():
    """The whole glove path (SDK order, bridge sort, wire, publisher) changes nothing the
    retargeter sees: the same joint targets as the landmarks themselves, pinches included."""
    clip = _clip(200)
    _, rows = _sort(_FINGERS_THUMB_FIRST)
    publisher = ManusPublisher(handedness="right")
    via_gloves = []
    for landmarks in clip:
        raw = wire.encode({"right": _sdk_nodes(landmarks, _FINGERS_THUMB_FIRST)[rows]}, 0)
        via_gloves.append(publisher.keypoints_from_message(raw)[0])
    direct = _retarget_all(_retargeter(), clip.astype(np.float32))
    gloves = _retarget_all(_retargeter(), np.array(via_gloves))
    assert direct.shape == gloves.shape and len(direct) > 100
    error = np.abs(direct - gloves)  # degrees
    # The solver amplifies float32 rounding on a few frames: moving the clip to the wrist
    # alone, no glove path at all, already shifts a frame by up to 1.6 degrees.
    assert np.median(error) < 0.01 and np.percentile(error, 99) < 1.0 and error.max() < 3.0


def test_a_mirrored_skeleton_would_not_retarget_alike():
    """The check above can fail: a mirror (a left-handed frame) changes the targets."""
    clip = _clip(200)
    mirrored = clip * np.array([1.0, 1.0, -1.0])
    direct = _retarget_all(_retargeter(), clip.astype(np.float32))
    wrong = _retarget_all(_retargeter(), mirrored.astype(np.float32))
    assert np.max(np.abs(direct - wrong)) > 10.0


def test_the_made_up_hand_closes_the_robot_fingers():
    retargeter = _retargeter()
    from orca_teleop.retargeting.retargeter import TargetPose

    def pose(t_s):
        kp = fake_bridge.landmarks_to_wire(
            fake_bridge.made_up_hand(t_s) @ fake_bridge._INTO_MANUS.T
        )
        return conversion.wire_to_keypoints(kp)

    action = None
    for t_s in np.linspace(0.0, 0.1, 60):  # through the auto-scale window, hand open
        action = retargeter.retarget(TargetPose(joint_positions=pose(t_s), source="mediapipe"))
    opened = action.as_dict()
    for t_s in np.linspace(0.1, 2.0, 60):  # close it
        action = retargeter.retarget(TargetPose(joint_positions=pose(t_s), source="mediapipe"))
    closed = action.as_dict()
    for finger in ("index", "middle", "ring", "pinky"):
        flex_open = opened[f"{finger}_mcp"] + opened[f"{finger}_pip"]
        flex_closed = closed[f"{finger}_mcp"] + closed[f"{finger}_pip"]
        assert flex_closed > flex_open + 60.0, finger


# ----- the bridge against a stand-in MANUS SDK ----------------------------------------

# The MANUS SDK types the bridge touches, as in the SDK's _cdef_generated.h but with fewer
# fields and enum values, so the stand-in SDK hands the bridge real cffi data.
_SDK_CDEF = """
typedef enum Side { Side_Invalid, Side_Left, Side_Right, Side_Center } Side;
typedef enum ErgonomicsDataType {
    ErgonomicsDataType_LeftFingerIndexMCPStretch,
    ErgonomicsDataType_RightFingerIndexMCPStretch,
    ErgonomicsDataType_MAX_SIZE
} ErgonomicsDataType;
typedef int SetGloveCalibrationReturnCode;
typedef struct ManusVec3 { float x; float y; float z; } ManusVec3;
typedef struct ManusQuaternion { float w; float x; float y; float z; } ManusQuaternion;
typedef struct ManusTransform {
    ManusVec3 position; ManusQuaternion rotation; ManusVec3 scale;
} ManusTransform;
typedef struct ManusTimestamp { uint64_t time; } ManusTimestamp;
typedef struct ManusHost { char hostName[256]; char ipAddress[40]; } ManusHost;
typedef struct CoordinateSystemVUH {
    int view; int up; int handedness; float unitScale;
} CoordinateSystemVUH;
typedef struct SkeletonStreamInfo {
    ManusTimestamp publishTime; uint32_t skeletonsCount;
} SkeletonStreamInfo;
typedef struct RawSkeletonInfo {
    uint32_t gloveId; uint32_t nodesCount; ManusTimestamp publishTime;
} RawSkeletonInfo;
typedef struct SkeletonNode { uint32_t id; ManusTransform transform; } SkeletonNode;
typedef struct NodeInfo {
    uint32_t nodeId; uint32_t parentId; int chainType; Side side; int fingerJointType;
} NodeInfo;
typedef struct ErgonomicsData { uint32_t id; bool isUserID; float data[2]; } ErgonomicsData;
typedef struct ErgonomicsStream {
    ManusTimestamp publishTime; ErgonomicsData data[32]; uint32_t dataCount;
} ErgonomicsStream;
typedef struct RawDeviceData {
    uint32_t id; uint32_t sensorCount; ManusTransform sensorData[5]; ManusQuaternion rotation;
} RawDeviceData;
typedef struct RawDeviceDataInfo {
    ManusTimestamp publishTime; uint32_t rawDeviceDataCount;
} RawDeviceDataInfo;
typedef struct GloveLandscapeData {
    uint32_t id; Side side; uint32_t batteryPercentage;
} GloveLandscapeData;
typedef struct DeviceLandscape {
    uint32_t dongleCount; GloveLandscapeData gloves[32]; uint32_t gloveCount;
} DeviceLandscape;
typedef struct LicenseInfo { bool sdk; bool raw; bool integrated; bool linuxTarget; } LicenseInfo;
typedef struct SettingsLandscape { LicenseInfo license; } SettingsLandscape;
typedef struct Landscape { DeviceLandscape gloveDevices; SettingsLandscape settings; } Landscape;
typedef struct SystemMessage { int type; char infoString[256]; uint32_t infoUInt; } SystemMessage;
"""
# The struct each stream's callback gets.
_CALLBACK_STRUCTS = {
    "raw_skeleton": "SkeletonStreamInfo",
    "ergonomics": "ErgonomicsStream",
    "raw_device_data": "RawDeviceDataInfo",
    "landscape": "Landscape",
    "system": "SystemMessage",
}


def _fake_manus_sdk(monkeypatch, skeletons: dict[int, np.ndarray], node_orders: dict[int, list]):
    """A ``manus_sdk`` module shaped like MANUS's cffi package (real cffi types), streaming
    ``skeletons`` (glove id -> (N, 7) nodes) whose node labels are ``node_orders``."""
    import enum
    import types

    cffi = pytest.importorskip("cffi")
    for name in bridge._LOCALE_VARS:  # start_sdk sets missing ones: the test's end undoes it
        value = os.environ.get(name)
        monkeypatch.setenv(name, "x" if value is None else value)
        if value is None:
            monkeypatch.delenv(name)
    ffi = cffi.FFI()
    ffi.cdef(_SDK_CDEF)
    enums = types.SimpleNamespace(
        SDKReturnCode=enum.IntEnum("SDKReturnCode", {"Success": 0, "NotConnected": 1}),
        SetGloveCalibrationReturnCode=enum.IntEnum(
            "SetGloveCalibrationReturnCode", {"Error": 0, "Success": 1, "VersionError": 2}
        ),
        HandMotion=enum.IntEnum("HandMotion", {"None_": 0, "Auto": 4}),
        AxisView=enum.IntEnum("AxisView", {"XFromViewer": 3}),
        AxisPolarity=enum.IntEnum("AxisPolarity", {"PositiveZ": 6}),
        Side=enum.IntEnum("Side", {"Left": 1, "Right": 2}),
    )
    calls = []
    ids = list(skeletons)

    class Lib:
        calibration_result = 1  # SetGloveCalibrationReturnCode.Success
        hosts = 1  # 0: the SDK finds no host (no dongle, no licence)
        calibrations: dict[int, bytes] = {}

        def __getattr__(self, name):
            def call(*args):
                calls.append(name)
                return 0

            return call

        def CoreSdk_InitializeIntegrated(self):
            calls.append("CoreSdk_InitializeIntegrated")
            self.locale = {name: os.environ.get(name) for name in bridge._LOCALE_VARS}
            return 0

        def CoreSdk_GetNumberOfAvailableHostsFound(self, count):
            count[0] = self.hosts
            return 0

        def CoreSdk_GetAvailableHostsFound(self, hosts, n):
            hosts[0].hostName = b"glove-pc"
            return 0

        def CoreSdk_GetRawSkeletonInfo(self, i, info):
            info.gloveId, info.nodesCount = ids[i], len(skeletons[ids[i]])
            return 0  # no publishTime: MANUS's examples copy the stream's in

        def CoreSdk_GetRawSkeletonData(self, i, nodes, n):
            rows = skeletons[ids[i]]
            for j in range(n):  # in reverse: the bridge must go by node id
                node = nodes[j]
                node.id = len(rows) - 1 - j
                p, r = node.transform.position, node.transform.rotation
                p.x, p.y, p.z, r.w, r.x, r.y, r.z = rows[node.id]
            return 0

        def CoreSdk_GetRawSkeletonNodeInfoArray(self, glove_id, infos, n):
            for i, row in enumerate(node_orders[glove_id]):
                info = infos[i]
                info.nodeId, info.parentId, info.chainType, info.side = row[:4]
                info.fingerJointType = row[4]
            return 0

        def CoreSdk_SetGloveCalibration(self, glove_id, data, n, result):
            calls.append("CoreSdk_SetGloveCalibration")
            self.calibrations[glove_id] = bytes(ffi.buffer(data, n))
            result[0] = self.calibration_result
            return 0

        def CoreSdk_GetRawDeviceData(self, i, device):
            device.id, device.sensorCount = 100 + i, 2
            device.sensorData[1].position.x = 0.5
            return 0

    class ManusSDK:
        callbacks: dict[str, object] = {}

        def __getattr__(self, name):  # register_<stream>_callback
            def register(callback):
                ManusSDK.callbacks[name.removeprefix("register_").removesuffix("_callback")] = (
                    callback
                )
                return 0

            return register

    module = types.ModuleType("manus_sdk")
    module.ffi, module.lib, module.ManusSDK = ffi, Lib(), ManusSDK
    generated = types.ModuleType("manus_sdk.generated")
    generated._enums = enums
    module.generated = generated
    monkeypatch.setitem(sys.modules, "manus_sdk", module)
    monkeypatch.setitem(sys.modules, "manus_sdk.generated", generated)
    monkeypatch.setitem(sys.modules, "manus_sdk.generated._enums", enums)
    return module, calls


def _fire(module, stream: str, **fields) -> None:
    """Call the bridge's callback for ``stream`` as the SDK would: a pointer to its struct."""
    pointer = module.ffi.new(_CALLBACK_STRUCTS[stream] + "*", fields)
    module.ManusSDK.callbacks[stream](pointer)


class _Running:
    """A bridge running in a thread, and a SUB socket on it once it delivers."""

    def __init__(self, module, calibration_dir) -> None:
        self.module = module
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        # A free port for the bridge: a plain socket frees it on close (a ZMQ socket only
        # later, in its IO thread, so under load the bridge's bind could still find it taken).
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            address = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
        self.bridge = bridge.ManusBridge(address, calibration_dir)
        self.thread = threading.Thread(target=self.bridge.run, name="bridge")
        self.thread.start()
        self.sub.connect(address)

    def first_with_both_hands(self) -> bytes:
        """Fire skeletons until a message with both hands comes, then drain the rest."""
        deadline, raw = time.monotonic() + 10.0, None
        while raw is None and time.monotonic() < deadline:
            if "raw_skeleton" in self.module.ManusSDK.callbacks:
                _fire(self.module, "raw_skeleton", skeletonsCount=2, publishTime={"time": 7_000})
            if self.sub.poll(50):
                raw = self.sub.recv()
                raw = raw if len(wire.decode(raw)[1]) == 2 else None
        assert raw is not None, "the bridge sent no message with both hands"
        while self.sub.poll(200):
            self.sub.recv()
        return raw

    def receive(self, n: int, timeout_s: float = 5.0) -> list[bytes]:
        """The next ``n`` messages, leaving out status messages (a slow machine's)."""
        got, deadline = [], time.monotonic() + timeout_s
        while len(got) < n and time.monotonic() < deadline:
            if self.sub.poll(100) and wire.sdk_data(raw := self.sub.recv())[0] != "status":
                got.append(raw)
        return got

    def stop(self) -> None:
        self.bridge.stop()
        self.thread.join(timeout=5.0)
        self.sub.close(linger=0)
        self.ctx.term()
        assert not self.thread.is_alive()


def _two_gloves(monkeypatch, landmarks: np.ndarray):
    right = _sdk_nodes(landmarks, _FINGERS_THUMB_FIRST)
    left = _sdk_nodes(landmarks * np.array([-1.0, 1.0, 1.0]), _PINKY_BEFORE_RING)
    return _fake_manus_sdk(
        monkeypatch,
        {11: right, 22: left},
        {11: _sdk_skeleton(_FINGERS_THUMB_FIRST, 2), 22: _sdk_skeleton(_PINKY_BEFORE_RING, 1)},
    )


def test_the_bridge_streams_sorted_hands_from_the_sdk(monkeypatch, tmp_path, caplog):
    """Integrated mode, the frame set up, both gloves in one message sorted by their
    labels, each calibrated from its file if there is one; then every SDK callback is one
    message, in order, stamped when it came."""
    landmarks = _clip(1)[0]
    module, calls = _two_gloves(monkeypatch, landmarks)
    (tmp_path / "Calibration_right.mcal").write_bytes(b"right glove .mcal")  # no left one
    caplog.set_level("INFO", logger="manus_bridge")
    running = _Running(module, tmp_path)
    try:
        raw = running.first_with_both_hands()
        before = time.time_ns()
        for _ in range(5):
            _fire(module, "raw_skeleton", skeletonsCount=2)
        after = time.time_ns()
        messages = running.receive(5)
    finally:
        running.stop()
    assert calls[0] == "CoreSdk_InitializeIntegrated" and "CoreSdk_ConnectToHost" in calls
    assert "CoreSdk_InitializeCoordinateSystemWithVUH" in calls
    assert calls[-1] == "CoreSdk_ShutDown"
    assert module.lib.calibrations == {11: b"right glove .mcal"}
    log = caplog.text
    assert f"Loaded the right glove's calibration {tmp_path / 'Calibration_right.mcal'}" in log
    assert "Streaming UNCALIBRATED left glove" in log and "Calibration_left.mcal" in log
    assert list(wire.bridge_warnings(raw)) == ["uncalibrated_left"]  # teleop hears it too

    expected = (landmarks - landmarks[0]) @ fake_bridge._INTO_MANUS.T
    keypoints, sides = ManusPublisher(handedness="right").keypoints_from_message(raw)
    assert sides == ["left", "right"]
    np.testing.assert_allclose(keypoints, expected, atol=1e-6)
    left_kp, _ = ManusPublisher(handedness="left").keypoints_from_message(raw)
    mirrored = ((landmarks * [-1.0, 1.0, 1.0]) - landmarks[0] * [-1.0, 1.0, 1.0]) @ (
        fake_bridge._INTO_MANUS.T
    )
    np.testing.assert_allclose(left_kp, mirrored, atol=1e-6)
    stream, infos = wire.sdk_data(raw)
    assert stream == "raw_skeleton"
    assert infos["right"] == {"gloveId": 11, "nodesCount": 25, "publishTime": {"time": 7_000}}
    assert infos["left"]["gloveId"] == 22

    assert len(messages) == 5, "a callback was dropped or merged"
    parsed = []
    for message in messages:
        state = wire.manus_stream_pb2.GloveState()
        state.ParseFromString(message)
        parsed.append(state)
    ids = [int(m.header.frame_id) for m in parsed]
    assert ids == list(range(ids[0], ids[0] + 5))
    stamps = [m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec for m in parsed]
    assert before <= stamps[0] and stamps == sorted(stamps) and stamps[-1] <= after
    assert all(len(m.left_mocap_pose) == len(m.right_mocap_pose) == 25 for m in parsed)


def test_teleop_hears_why_no_hands_come_while_the_sdk_is_not_connected(monkeypatch, tmp_path):
    module, calls = _two_gloves(monkeypatch, _clip(1)[0])
    module.lib.hosts = 0
    running = _Running(module, tmp_path)
    try:
        status, deadline = None, time.monotonic() + 10.0
        while status is None and time.monotonic() < deadline:
            if running.sub.poll(500):
                status = running.sub.recv()
        module.lib.hosts = 1  # the dongle is in
        hands = running.first_with_both_hands()
    finally:
        running.stop()
    assert status is not None and wire.sdk_data(status)[0] == "status"
    assert "not connected" in wire.bridge_warnings(status)["not_connected"]
    assert "not_connected" not in wire.bridge_warnings(hands)
    assert calls.count("CoreSdk_SetRawSkeletonHandMotion") == 1


def test_every_sdk_stream_goes_out_and_a_missing_licence_feature_is_reported(
    monkeypatch, tmp_path, caplog
):
    """Ergonomics, raw device data, landscape and system messages go out as the SDK's
    structs in JSON. A landscape without a dongle has no licence yet: nothing stops. Then
    the licence lacks Raw: a warning, but raw device data still goes out if the SDK gives it.
    Teleop's publisher skips those messages."""
    module, _ = _two_gloves(monkeypatch, _clip(1)[0])
    caplog.set_level("INFO", logger="manus_bridge")
    running = _Running(module, tmp_path)
    try:
        running.first_with_both_hands()
        _fire(module, "landscape")  # no dongle yet, so no licence: all False
        _fire(module, "raw_device_data", rawDeviceDataCount=1)  # still goes out
        licence = {"sdk": True, "raw": False, "integrated": True, "linuxTarget": True}
        _fire(
            module,
            "landscape",
            gloveDevices={
                "dongleCount": 1,
                "gloves": [{"id": 11, "side": 2, "batteryPercentage": 80}],
                "gloveCount": 1,
            },
            settings={"license": licence},
        )
        _fire(
            module,
            "ergonomics",
            publishTime={"time": 5},
            data=[{"id": 11, "data": [10.0, 20.123456789]}],
            dataCount=1,
        )
        _fire(module, "raw_device_data", rawDeviceDataCount=2)  # not licensed: still sent
        _fire(module, "system", type=3, infoString=b"hello", infoUInt=7)
        _fire(module, "raw_skeleton", skeletonsCount=2)
        messages = running.receive(7)
    finally:
        running.stop()
    streams = [wire.sdk_data(raw) for raw in messages]
    assert [name for name, _ in streams] == [
        *("landscape", "raw_device_data", "landscape", "ergonomics", "raw_device_data"),
        *("system", "raw_skeleton"),
    ]
    landscape, ergonomics, system = (streams[i][1] for i in (2, 3, 5))
    assert landscape == {
        "gloveDevices": {
            "dongleCount": 1,
            "gloves": [{"id": 11, "side": "Right", "batteryPercentage": 80}],
            "gloveCount": 1,
        },
        "settings": {"license": licence},
    }  # cut to gloveCount, the enum named
    assert ergonomics["data"] == [
        {
            "id": 11,
            "isUserID": False,
            "data": {"LeftFingerIndexMCPStretch": 10.0, "RightFingerIndexMCPStretch": 20.123457},
        }
    ]
    assert system == {"type": 3, "infoString": "hello", "infoUInt": 7}
    assert "raw device data may not come: your MANUS licence does not include Raw" in caplog.text
    assert "licence_raw_device_data" not in wire.bridge_warnings(messages[1])  # not read yet
    for raw in messages[2:]:
        assert "licence_raw_device_data" in wire.bridge_warnings(raw)
    publisher = ManusPublisher(handedness="right")
    for raw in messages[:6]:
        assert wire.decode(raw)[1] == {}
        assert publisher.keypoints_from_message(raw) == (None, [])


def test_a_lost_glove_is_reported_until_it_is_back_and_teleop_hears_it_once(
    monkeypatch, tmp_path, caplog
):
    """A glove that sends nothing is reported lost, in every message (a status message
    when nothing else goes out); teleop's publisher logs it once, and once when it's back."""
    monkeypatch.setattr(bridge, "LOST_AFTER_S", 0.3)
    monkeypatch.setattr(bridge, "STATUS_EVERY_S", 0.5)
    module, _ = _two_gloves(monkeypatch, _clip(1)[0])
    (tmp_path / "Calibration_left.mcal").write_bytes(b"left")
    (tmp_path / "Calibration_right.mcal").write_bytes(b"right")
    running = _Running(module, tmp_path)
    got = []
    try:
        running.first_with_both_hands()
        for _ in range(30):  # 0.6 s of the right glove alone (glove 11; 22 is the left)
            _fire(module, "raw_skeleton", skeletonsCount=1)
            time.sleep(0.02)
        _fire(module, "raw_skeleton", skeletonsCount=2)  # the left one is back
        time.sleep(0.8)  # then nothing at all: status messages
        while running.sub.poll(200):
            got.append(running.sub.recv())
    finally:
        running.stop()
    lost = [i for i, raw in enumerate(got) if "glove_22_lost" in wire.bridge_warnings(raw)]
    back = next(i for i, raw in enumerate(got) if len(wire.decode(raw)[1]) == 2)
    assert lost and lost[0] < back and back not in lost  # lost, then back
    status = [raw for raw in got[back:] if wire.sdk_data(raw)[0] == "status"]
    assert status, "no status message while nothing else went out"
    assert {"glove_11_lost", "glove_22_lost"} <= set(wire.bridge_warnings(status[-1]))

    publisher = ManusPublisher(handedness="right")
    caplog.clear()
    caplog.set_level("INFO", logger="orca_teleop.ingress.manus.publisher")
    for raw in got:
        publisher.update_warnings(raw)
    log = [r.getMessage() for r in caplog.records]
    assert sum("Glove bridge: No data from the left glove (22)" in m for m in log) == 2
    assert sum("Glove bridge, cleared: No data from the left glove (22)" in m for m in log) == 1
    assert set(publisher.warnings) == {"glove_11_lost", "glove_22_lost"}


def test_only_a_glove_silent_for_lost_after_s_is_lost_and_its_next_frame_clears_it(
    monkeypatch, caplog, tmp_path
):
    module, _ = _fake_manus_sdk(monkeypatch, {}, {})
    the_bridge = bridge.ManusBridge(calibration_dir=tmp_path)
    the_bridge.ffi = module.ffi
    rows = list(range(25))
    the_bridge._layouts = {11: (rows, "right", rows), 22: (rows, "left", rows)}
    the_bridge._calibration_tried = {11: time.monotonic(), 22: time.monotonic()}
    now = time.monotonic()
    the_bridge._last_seen = {11: (now, "right"), 22: (now - 2 * bridge.LOST_AFTER_S, "left")}
    caplog.set_level("INFO", logger="manus_bridge")
    for _ in range(3):
        the_bridge._check_gloves()
    assert list(the_bridge.warnings) == ["glove_22_lost"]
    info = module.ffi.new("RawSkeletonInfo*")
    hands, _ = the_bridge._hands({22: (rows, np.zeros((25, 7)), info)})
    # back, it tries its .mcal again (none in tmp_path: uncalibrated, said so)
    assert list(hands) == ["left"] and list(the_bridge.warnings) == ["uncalibrated_left"]
    log = [r.getMessage() for r in caplog.records]
    assert sum("No data from the left glove (22)" in m for m in log) == 1
    assert "The left glove (22) is back" in log


def test_two_gloves_on_one_side_send_the_first_and_say_so(monkeypatch, caplog):
    module, _ = _fake_manus_sdk(monkeypatch, {}, {})
    the_bridge = bridge.ManusBridge()
    the_bridge.ffi = module.ffi
    rows = list(range(25))
    the_bridge._layouts = {11: (rows, "right", rows), 33: (rows, "right", rows)}
    the_bridge._calibration_tried = {11: time.monotonic(), 33: time.monotonic()}
    info = module.ffi.new("RawSkeletonInfo*")
    first, second = np.zeros((25, 7)), np.ones((25, 7))
    hands, _ = the_bridge._hands({11: (rows, first, info), 33: (rows, second, info)})
    assert list(hands) == ["right"] and np.all(hands["right"] == 0)
    assert "same_side_right" in the_bridge.warnings
    the_bridge._hands({11: (rows, first, info)})
    assert "same_side_right" not in the_bridge.warnings


def test_a_hand_frame_that_waited_too_long_is_skipped_and_reported(monkeypatch):
    module, _ = _fake_manus_sdk(monkeypatch, {}, {})
    monkeypatch.setattr(bridge, "_STATS_EVERY_S", 0.0)
    the_bridge = bridge.ManusBridge()
    the_bridge.ffi = module.ffi
    rows = list(range(25))
    the_bridge._layouts = {11: (rows, "right", rows)}
    the_bridge._calibration_tried = {11: time.monotonic()}
    info = module.ffi.new("RawSkeletonInfo*")
    now = time.monotonic()
    for t_ns, queued in ((1, now - 1.0), (2, now)):
        gloves = {11: (rows, np.full((25, 7), float(t_ns)), info)}
        the_bridge._frames.put((t_ns, queued, "raw_skeleton", gloves))
    socket = _Socket()
    threading.Timer(0.3, the_bridge.stop).start()
    the_bridge.publish_loop(socket)
    sent = [wire.decode(raw) for raw in socket.sent if wire.sdk_data(raw)[0] != "status"]
    assert [t for t, _ in sent] == [2]
    assert "falling_behind" in the_bridge.warnings


def test_the_licence_is_checked_on_each_landscape_and_never_stops_a_stream(
    monkeypatch, tmp_path, caplog
):
    """A licence without SDK Integrated gives warnings but stops no stream the SDK gives;
    a later landscape with the full licence clears them."""
    module, _ = _two_gloves(monkeypatch, _clip(1)[0])
    caplog.set_level("INFO", logger="manus_bridge")

    def landscape(**licence):
        full = {"sdk": True, "raw": True, "integrated": True, "linuxTarget": True}
        devices = {"dongleCount": 1, "gloveCount": 0}
        _fire(module, "landscape", gloveDevices=devices, settings={"license": full | licence})

    running = _Running(module, tmp_path)
    try:
        running.first_with_both_hands()
        landscape(integrated=False)
        _fire(module, "raw_skeleton", skeletonsCount=2)  # the hands still go out
        _fire(module, "ergonomics", dataCount=0)  # not licensed: still sent
        _fire(module, "raw_device_data", rawDeviceDataCount=1)  # not licensed: still sent
        landscape()
        _fire(module, "raw_device_data", rawDeviceDataCount=1)
        messages = running.receive(6)
    finally:
        running.stop()
    streams = [wire.sdk_data(raw)[0] for raw in messages]
    assert streams == [
        *("landscape", "raw_skeleton", "ergonomics", "raw_device_data", "landscape"),
        "raw_device_data",
    ]
    lacking = {"licence_raw_skeleton", "licence_ergonomics", "licence_raw_device_data"}
    for raw in messages[1:4]:
        assert lacking <= set(wire.bridge_warnings(raw))
    assert len(wire.decode(messages[1])[1]) == 2
    assert not lacking & set(wire.bridge_warnings(messages[5]))
    assert "no_glove" in wire.bridge_warnings(messages[1])  # the landscapes list no glove
    assert "raw skeleton may not come: your MANUS licence does not include SDK" in (caplog.text)
    assert "ergonomics: the MANUS licence covers it now" in caplog.text


def test_teleop_logs_a_bridge_warning_once_and_its_clearing_once(caplog):
    publisher = ManusPublisher(handedness="right")
    hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(0.0))
    warned = wire.encode({"right": hand}, 0, warnings={"uncalibrated_left": "UNCALIBRATED left"})
    plain = wire.encode({"right": hand}, 0)
    caplog.set_level("INFO", logger="orca_teleop.ingress.manus.publisher")
    for raw in [plain] * 3 + [warned] * 100 + [plain] * 100:
        publisher.update_warnings(raw)
        if raw is warned:
            assert publisher.warnings == {"uncalibrated_left": "UNCALIBRATED left"}
    assert [r.getMessage() for r in caplog.records] == [
        "Glove bridge: UNCALIBRATED left",
        "Glove bridge, cleared: UNCALIBRATED left",
    ]
    assert publisher.warnings == {}


def test_the_fake_bridge_sends_warnings_that_teleop_logs_once(caplog):
    if not CLIP.exists():
        pytest.skip("tests/data/pinch_clip.npz is missing")
    probe_ctx, probe, address = _free_tcp()
    probe.close()
    probe_ctx.term()
    publisher = ManusPublisher("localhost:1", "right", address)  # its ZMQ side only
    receiver = threading.Thread(target=publisher._zmq_receiver, name="zmq-receiver")
    receiver.start()
    caplog.set_level("WARNING", logger="orca_teleop.ingress.manus.publisher")
    args = ["--landmarks", str(CLIP), "--rate", "500", "--bind", address]
    try:
        fake_bridge.main([*args, "--warning", "uncalibrated_left=Streaming UNCALIBRATED left"])
        time.sleep(0.2)
    finally:
        publisher.stop()
        receiver.join(timeout=5.0)
    assert publisher._fresh.is_set(), "no hand reached the publisher"
    assert publisher.warnings == {"uncalibrated_left": "Streaming UNCALIBRATED left"}
    log = [r.getMessage() for r in caplog.records if r.name.endswith("publisher")]
    assert log.count("Glove bridge: Streaming UNCALIBRATED left") == 1


def test_raw_device_data_goes_out_when_licensed(monkeypatch, tmp_path):
    module, _ = _two_gloves(monkeypatch, _clip(1)[0])
    running = _Running(module, tmp_path)
    try:
        running.first_with_both_hands()
        _fire(module, "raw_device_data", publishTime={"time": 9}, rawDeviceDataCount=2)
        (raw,) = running.receive(1)
    finally:
        running.stop()
    stream, data = wire.sdk_data(raw)
    assert stream == "raw_device_data" and data["publishTime"] == {"time": 9}
    assert [d["id"] for d in data["devices"]] == [100, 101]
    sensors = data["devices"][0]["sensorData"]
    assert len(sensors) == 2 and sensors[1]["position"]["x"] == 0.5  # cut to sensorCount


def test_a_missing_or_rejected_calibration_streams_uncalibrated_and_says_so(
    monkeypatch, tmp_path, caplog
):
    module, calls = _fake_manus_sdk(monkeypatch, {}, {})
    the_bridge = bridge.ManusBridge(calibration_dir=tmp_path)
    the_bridge.ffi, the_bridge.lib = module.ffi, module.lib
    the_bridge.enums = sys.modules["manus_sdk.generated._enums"]
    caplog.set_level("INFO", logger="manus_bridge")

    the_bridge._load_calibration(5, "left")  # no file
    assert "uncalibrated_left" in the_bridge.warnings and "CoreSdk_SetGloveCalibration" not in calls
    assert "Streaming UNCALIBRATED left glove" in caplog.text
    assert "Calibration_left.mcal" in caplog.text and "MANUS's SDK client" in caplog.text

    (tmp_path / "Calibration_right.mcal").write_bytes(b"too old")
    module.lib.calibration_result = 2  # VersionError
    caplog.clear()
    the_bridge._load_calibration(6, "right")
    assert "uncalibrated_right" in the_bridge.warnings and module.lib.calibrations == {
        6: b"too old"
    }
    assert "Streaming UNCALIBRATED right glove: the MANUS SDK rejected" in caplog.text
    assert "VersionError" in caplog.text

    module.lib.calibration_result = 1  # Success
    caplog.clear()
    the_bridge._load_calibration(7, "right")
    assert "uncalibrated_right" not in the_bridge.warnings  # a loaded .mcal clears it
    assert "Loaded the right glove's calibration" in caplog.text and "Success" in caplog.text


@pytest.mark.parametrize("lang", [None, "de_CH.UTF-8"])
def test_missing_locale_variables_are_set_before_the_sdk_starts(monkeypatch, caplog, lang):
    """The MANUS SDK aborts in CoreSdk_InitializeIntegrated without them (a bare
    environment: SSH, a service); set ones stay as they are."""
    module, calls = _fake_manus_sdk(monkeypatch, {}, {})  # it puts them back at the end
    for name in bridge._LOCALE_VARS:
        monkeypatch.delenv(name, raising=False)
    if lang:
        monkeypatch.setenv("LANG", lang)
    caplog.set_level("INFO", logger="manus_bridge")
    bridge.ManusBridge().start_sdk()
    expected = lang or "C.UTF-8"
    assert calls[0] == "CoreSdk_InitializeIntegrated"
    assert module.lib.locale == {
        "LANG": expected,
        "LC_ALL": expected,
        "LC_MESSAGES": expected,
        "LANGUAGE": "en_US:en",
    }
    assert "for the MANUS SDK, which aborts without them" in caplog.text
    full = {"LANG": "a", "LC_ALL": "b", "LC_MESSAGES": "c", "LANGUAGE": "d"}
    assert bridge.fill_locale(dict(full)) == []


class _Socket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send(self, raw: bytes) -> None:
        self.sent.append(raw)


def test_every_frame_goes_out_in_order(monkeypatch):
    """The publish loop sends each queued frame, in order, never merging or skipping one,
    calibrated glove or not."""
    module, _ = _fake_manus_sdk(monkeypatch, {}, {})
    the_bridge = bridge.ManusBridge()
    the_bridge.ffi = module.ffi
    rows = list(range(25))
    the_bridge._layouts = {11: (rows, "right", rows), 22: (rows, "left", rows)}
    the_bridge._calibration_tried = {11: time.monotonic(), 22: time.monotonic()}
    info = module.ffi.new("RawSkeletonInfo*")
    for t_ns in (500, 999, 1_000, 1_001, 2_000):
        nodes = np.full((25, 7), float(t_ns))
        gloves = {11: (rows, nodes, info), 22: (rows, nodes, info)}
        the_bridge._frames.put((t_ns, time.monotonic(), "raw_skeleton", gloves))
    socket = _Socket()
    threading.Timer(0.3, the_bridge.stop).start()
    the_bridge.publish_loop(socket)
    sent = [wire.decode(raw) for raw in socket.sent if wire.sdk_data(raw)[0] != "status"]
    assert [t for t, _ in sent] == [500, 999, 1_000, 1_001, 2_000]
    assert [sorted(hands) for _, hands in sent] == [["left", "right"]] * 5
    for t_ns, hands in sent:
        assert np.all(hands["left"] == t_ns)


def test_the_bridge_runs_without_orca_teleop():
    """On the glove computer it runs as a file next to conversion.py and wire.py."""
    folder = pathlib.Path(bridge.__file__).parent
    code = (
        f"import sys; sys.path.insert(0, {str(folder)!r}); import bridge; "
        "assert 'orca_teleop' not in sys.modules, 'imported orca_teleop'; "
        "print(bridge.DEFAULT_BIND)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=folder
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "tcp://127.0.0.1:2044"


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT to a child needs POSIX signals")
def test_ctrl_c_ends_the_stream_cleanly(caplog):
    """A stopped publisher lets its stream close before cancelling: a cancelled stream
    logs an error (with a traceback) in the ingress server on every stop."""
    import signal

    q, stop = queue.Queue(maxsize=50), threading.Event()
    server = IngressServer(q, stop, port=0)
    port = server.start()
    ctx, sock, address = _free_tcp()
    proc = subprocess.Popen(
        [sys.executable, "-m", "orca_teleop.ingress.manus.publisher"]
        + ["--server", f"localhost:{port}", "--glove-address", address],
        # A test run started in the background (``&``) passes "ignore SIGINT" on, and Python
        # then installs no Ctrl+C handler: give the child the terminal's default back.
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    try:
        deadline = time.monotonic() + 20.0
        hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(0.5))
        while q.empty() and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand}, time.time_ns()))
            time.sleep(0.05)
        assert not q.empty(), "no frame reached the ingress server"
        with caplog.at_level("ERROR", logger="orca_teleop.ingress.server"):
            proc.send_signal(signal.SIGINT)
            assert proc.wait(timeout=10) == 0
            time.sleep(0.3)
        errors = [r for r in caplog.records if r.name == "orca_teleop.ingress.server"]
        assert not errors, [r.getMessage() for r in errors]
    finally:
        if proc.poll() is None:
            proc.kill()
        sock.close()
        ctx.term()
        stop.set()
        server.stop()


def test_source_manus_runs_the_manus_publisher_on_the_given_bridge(monkeypatch):
    from orca_teleop import cli

    got = {}
    monkeypatch.setattr(cli, "run_manus_local", lambda **kwargs: got.update(kwargs))
    monkeypatch.setattr(cli, "run_local", None)  # the webcam must not start
    args = cli._common_parser("test").parse_args(
        ["--source", "manus", "--glove-address", "10.0.0.2:2044"]
    )
    sink = object()
    cli._launch(args, sink, "right")
    assert got["zmq_address"] == "10.0.0.2:2044" and got["handedness"] == "right"
    assert got["sink"] is sink
    cli._launch(args.__class__(**{**vars(args), "record": "out"}), sink, "right")
    assert got["record_dir"] == "out"
    with pytest.raises(SystemExit, match="drop --no-local"):  # no local reader to address
        cli._launch(args.__class__(**{**vars(args), "local": False}), sink, "right")
    remote = {}
    import orca_teleop.cli as cli_module

    real_run = cli_module.run
    cli_module.run = lambda **kwargs: remote.update(kwargs)
    try:
        glove = cli._common_parser("test").parse_args(["--source", "manus", "--no-local"])
        cli._resolve_retargeter(cli._common_parser("test"), glove, glove.source)
        cli._launch(glove, sink, "right")
    finally:
        cli_module.run = real_run
    assert remote["retargeter_config_path"].endswith("manus.yaml")  # glove frames from elsewhere
    webcam = cli._common_parser("test").parse_args(["--glove-address", "10.0.0.2:2044"])
    with pytest.raises(SystemExit, match="for --source manus"):
        cli._launch(webcam, sink, "right")
    plain = cli._common_parser("test").parse_args(["--source", "manus"])
    cli._launch(plain, sink, "right")
    assert got["zmq_address"] == DEFAULT_MANUS_ADDRESS


def test_more_than_25_nodes_use_the_first_25():
    nodes = np.random.default_rng(0).normal(size=(27, 3))
    assert np.array_equal(
        conversion.wire_to_keypoints(nodes), conversion.wire_to_keypoints(nodes[:25])
    )
    with pytest.raises(ValueError):
        conversion.wire_to_keypoints(nodes[:24])


def test_record_saves_the_glove_hands_in_the_webcam_recording_format(tmp_path):
    """--source manus --record DIR: every hand the publisher takes, as a recording that
    main's replay and fidelity tools load (no video, no image landmarks)."""
    from orca_teleop.ingress.landmark_recording import load_recording

    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    server = IngressServer(landmarks_q, stop, port=0)
    port = server.start()
    ctx, sock, address = _free_tcp()
    rec = tmp_path / "rec"
    publisher = ManusPublisher(f"localhost:{port}", "right", address, str(rec))
    thread = threading.Thread(target=publisher.run, name="manus-publisher")
    thread.start()
    hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(1.0))
    try:
        deadline = time.monotonic() + 10.0
        while landmarks_q.empty() and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand}, time.time_ns()))
            time.sleep(0.02)
    finally:
        publisher.stop()
        thread.join(timeout=5.0)
        stop.set()
        server.stop(grace=0.5)
        sock.close()
        ctx.term()
    got = load_recording(rec)
    assert got.meta["input"] == "glove" and got.meta["video"] is False
    assert len(got.world) >= 1 and len(got.capture_time_ns) == len(got.world)
    np.testing.assert_allclose(got.world[0], conversion.wire_to_keypoints(hand), atol=1e-6)
    assert not got.image.any() and got.handedness[0] == "right"
    assert not (rec / "video.mp4").exists()


def test_record_refuses_a_used_folder_before_the_hand_starts(tmp_path, capsys):
    from orca_teleop import cli

    used = tmp_path / "used"
    used.mkdir()
    (used / "meta.json").write_text("{}")
    with pytest.raises(SystemExit):
        cli._common_parser("test").parse_args(["--record", str(used)])
    assert "is not an empty folder" in capsys.readouterr().err
    args = cli._common_parser("test").parse_args(["--record", str(tmp_path / "new")])
    assert args.record == str(tmp_path / "new")


def test_a_disk_error_stops_only_the_recording_and_says_why(caplog):
    class Full:
        closed = False

        def add_time(self, t_ns):
            raise OSError(28, "No space left on device")

        def close(self):
            self.closed = True

    publisher = ManusPublisher(handedness="right")
    publisher._recorder = recorder = Full()
    publisher._record(np.zeros((21, 3)))
    assert publisher._recorder is None and recorder.closed
    assert "Recording stopped, teleop goes on" in caplog.text
    assert "No space left on device" in caplog.text


def test_the_recording_waits_for_the_hand_so_a_failed_start_leaves_no_folder(tmp_path):
    from orca_teleop.ingress.landmark_recording import load_recording

    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    server = IngressServer(landmarks_q, stop, port=0)
    port = server.start()
    ctx, sock, address = _free_tcp()
    rec = tmp_path / "rec"
    hand_ready = threading.Event()
    publisher = ManusPublisher(f"localhost:{port}", "right", address, str(rec), hand_ready)
    thread = threading.Thread(target=publisher.run, name="manus-publisher")
    thread.start()
    hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(1.0))

    def send_until_streamed():
        while not landmarks_q.empty():
            landmarks_q.get_nowait()
        deadline = time.monotonic() + 10.0
        while landmarks_q.empty() and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand}, time.time_ns()))
            time.sleep(0.02)
        assert not landmarks_q.empty()

    try:
        send_until_streamed()  # hands stream to teleop, but the hand is not ready yet
        assert not rec.exists()
        hand_ready.set()
        lines = rec / "landmarks.jsonl"
        deadline = time.monotonic() + 10.0
        while not (lines.exists() and lines.read_text()) and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand}, time.time_ns()))
            time.sleep(0.02)
    finally:
        publisher.stop()
        thread.join(timeout=5.0)
        stop.set()
        server.stop(grace=0.5)
        sock.close()
        ctx.term()
    assert len(load_recording(rec).world) >= 1
