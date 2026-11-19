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
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

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


def _free_tcp() -> tuple[object, object, str]:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind("tcp://127.0.0.1:*")
    return ctx, sock, sock.getsockopt_string(zmq.LAST_ENDPOINT)


def test_glove_frames_reach_the_ingress_server_as_landmarks():
    """Bridge socket -> ManusPublisher (ZMQ, then gRPC) -> IngressServer's queue."""
    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    server = IngressServer(landmarks_q, stop, port=0)
    port = server.start()
    ctx, sock, address = _free_tcp()
    publisher = ManusPublisher(f"localhost:{port}", "right", address)
    thread = threading.Thread(target=publisher.run, name="manus-publisher")
    thread.start()
    hand = fake_bridge.landmarks_to_wire(fake_bridge.made_up_hand(1.0))
    try:
        deadline = time.monotonic() + 10.0
        got = None
        while got is None and time.monotonic() < deadline:
            sock.send(wire.encode({"right": hand, "left": hand}, time.time_ns()))
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
    assert wire.zmq_address("192.168.64.5:2044") == "tcp://192.168.64.5:2044"
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
"""
# The struct each stream's callback gets.
_CALLBACK_STRUCTS = {"raw_skeleton": "SkeletonStreamInfo"}


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
            count[0] = 1
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
        probe = self.ctx.socket(zmq.PUB)
        probe.bind("tcp://127.0.0.1:*")
        address = probe.getsockopt_string(zmq.LAST_ENDPOINT)
        probe.close()  # a free port for the bridge
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

    def receive(self, n: int) -> list[bytes]:
        got = []
        while len(got) < n and self.sub.poll(2000):
            got.append(self.sub.recv())
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

    expected = (landmarks - landmarks[0]) @ fake_bridge._INTO_MANUS.T
    keypoints, sides = ManusPublisher(handedness="right").keypoints_from_message(raw)
    assert sides == ["left", "right"]
    np.testing.assert_allclose(keypoints, expected, atol=1e-6)
    left_kp, _ = ManusPublisher(handedness="left").keypoints_from_message(raw)
    mirrored = ((landmarks * [-1.0, 1.0, 1.0]) - landmarks[0] * [-1.0, 1.0, 1.0]) @ (
        fake_bridge._INTO_MANUS.T
    )
    np.testing.assert_allclose(left_kp, mirrored, atol=1e-6)

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


def test_a_missing_or_rejected_calibration_streams_uncalibrated_and_says_so(
    monkeypatch, tmp_path, caplog
):
    module, calls = _fake_manus_sdk(monkeypatch, {}, {})
    the_bridge = bridge.ManusBridge(calibration_dir=tmp_path)
    the_bridge.ffi, the_bridge.lib = module.ffi, module.lib
    the_bridge.enums = sys.modules["manus_sdk.generated._enums"]
    caplog.set_level("INFO", logger="manus_bridge")

    the_bridge._load_calibration(5, "left")  # no file
    assert the_bridge._calibrated_since[5] == 0 and "CoreSdk_SetGloveCalibration" not in calls
    assert "Streaming UNCALIBRATED left glove" in caplog.text
    assert "Calibration_left.mcal" in caplog.text and "MANUS's SDK client" in caplog.text

    (tmp_path / "Calibration_right.mcal").write_bytes(b"too old")
    module.lib.calibration_result = 2  # VersionError
    caplog.clear()
    the_bridge._load_calibration(6, "right")
    assert the_bridge._calibrated_since[6] == 0 and module.lib.calibrations == {6: b"too old"}
    assert "Streaming UNCALIBRATED right glove: the MANUS SDK rejected" in caplog.text
    assert "VersionError" in caplog.text

    module.lib.calibration_result = 1  # Success
    caplog.clear()
    before = time.time_ns()
    the_bridge._load_calibration(7, "right")
    assert the_bridge._calibrated_since[7] >= before
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


def test_every_frame_goes_out_except_those_from_before_the_calibration():
    """The publish loop sends each queued frame, in order, never merging or skipping one,
    except a calibrated glove's frames from before its .mcal was loaded."""
    the_bridge = bridge.ManusBridge()
    rows = list(range(25))
    the_bridge._layouts = {11: (rows, "right", rows), 22: (rows, "left", rows)}
    the_bridge._calibrated_since = {11: 1_000, 22: 0}  # the left one has no .mcal
    for t_ns in (500, 999, 1_000, 1_001, 2_000):
        nodes = np.full((25, 7), float(t_ns))
        the_bridge._frames.put((t_ns, {11: (rows, nodes), 22: (rows, nodes)}))
    socket = _Socket()
    threading.Timer(0.3, the_bridge.stop).start()
    the_bridge.publish_loop(socket)
    sent = [wire.decode(raw) for raw in socket.sent]
    assert [t for t, _ in sent] == [500, 999, 1_000, 1_001, 2_000]
    assert [sorted(hands) for _, hands in sent] == [["left"]] * 2 + [["left", "right"]] * 3
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
        + ["--server", f"localhost:{port}", "--glove-address", address]
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
