"""Tests for orca_teleop.pipeline, covering threading/queue plumbing only without
touching real hardware for CI compliance.
"""

import inspect
import queue
import threading
import time

import grpc
import numpy as np
import pytest
from conftest import CANONICAL_LANDMARK_SHAPE, plausible_hand_keypoints
from orca_core import OrcaJointPositions
from orca_core.hardware_hand import MockOrcaHand

from orca_teleop.constants import MOTION_NUM_STEPS
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.pipeline import (
    _SHUTDOWN,
    TeleopQueues,
    retargeter_worker,
    robot_worker,
    run,
    run_local,
)


def _make_queues(maxsize: int = 8) -> TeleopQueues:
    return TeleopQueues(
        landmarks_q=queue.Queue(maxsize=maxsize),
        actions_q=queue.Queue(maxsize=maxsize),
    )


def _make_landmark(handedness: str = "right") -> HandLandmarks:
    """Wrap the canonical plausible hand keypoints in a HandLandmarks."""
    return HandLandmarks(
        keypoints=plausible_hand_keypoints(),
        handedness=handedness,
        timestamp_ns=time.time_ns(),
    )


def _midpoint_action() -> OrcaJointPositions:
    """Build an OrcaJointPositions at every joint's ROM midpoint (degrees)."""
    roms = MockOrcaHand().config.joint_roms_dict
    return OrcaJointPositions({j: 0.5 * (lo + hi) for j, (lo, hi) in roms.items()})


def _wait_for(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def _start(target, *args, name: str | None = None) -> threading.Thread:
    t = threading.Thread(target=target, args=args, name=name, daemon=True)
    t.start()
    return t


def test_public_exports():
    from orca_teleop import (  # noqa: F401
        TeleopQueues as _PQ,
    )


def test_pipeline_queues_dataclass():
    q = _make_queues()
    assert isinstance(q.landmarks_q, queue.Queue)
    assert isinstance(q.actions_q, queue.Queue)


def test_run_signature_stable():
    sig = inspect.signature(run)
    assert "model_path" in sig.parameters


def test_pipeline_defaults_to_adaptive_retargeter():
    assert (
        inspect.signature(retargeter_worker).parameters["retargeter_backend"].default
        == "adaptive_analytical"
    )
    assert inspect.signature(run).parameters["retargeter_backend"].default == "adaptive_analytical"
    assert (
        inspect.signature(run_local).parameters["retargeter_backend"].default
        == "adaptive_analytical"
    )


def test_ingress_server_start_stop():
    """IngressServer starts and stops without error."""
    q = queue.Queue(maxsize=4)
    stop = threading.Event()
    server = IngressServer(q, stop, port=0)
    port = server.start()
    assert port > 0
    server.stop()


def test_ingress_server_receives_frames():
    """Frames streamed by a gRPC client end up on the landmarks queue."""
    n_frames = 5  # TODO: add sensitivity test to larger number of frames
    q = queue.Queue(maxsize=8)
    stop = threading.Event()
    server = IngressServer(q, stop, port=0)
    port = server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = hand_stream_pb2_grpc.HandStreamStub(channel)

    def gen_frames():
        for _ in range(n_frames):
            kp = np.random.randn(*CANONICAL_LANDMARK_SHAPE).astype(np.float32)
            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness="right",
                timestamp_ns=time.time_ns(),
                wrist_angle_degrees=12.5,
            )
            time.sleep(0.01)

    try:
        response = stub.StreamHandFrames(gen_frames())
        assert response.frames_received == n_frames

        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert len(items) > 0
        for item in items:
            assert isinstance(item, HandLandmarks)
            assert item.keypoints.shape == CANONICAL_LANDMARK_SHAPE
            assert item.handedness == "right"
            assert item.wrist_angle_degrees == pytest.approx(12.5)

    finally:
        channel.close()
        server.stop()


def test_ingress_server_drops_stale_on_full_queue():
    """When the queue is full, server drops oldest and enqueues latest."""
    n_frames = 5
    q = queue.Queue(maxsize=2)
    stop = threading.Event()
    server = IngressServer(q, stop, port=0)
    port = server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = hand_stream_pb2_grpc.HandStreamStub(channel)

    def gen_frames():
        for i in range(n_frames):
            kp = np.full(CANONICAL_LANDMARK_SHAPE, float(i), dtype=np.float32)
            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness="right",
                timestamp_ns=time.time_ns(),
            )
            time.sleep(0.01)

    response = stub.StreamHandFrames(gen_frames())
    assert response.frames_received == n_frames

    items = []
    while not q.empty():
        items.append(q.get_nowait())

    # Queue is bounded at 2, so we should have at most 2 items
    assert len(items) <= 2

    channel.close()
    server.stop()


def test_retargeter_forwards_shutdown_downstream(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())
    q = _make_queues()
    stop = threading.Event()
    q.landmarks_q.put(_SHUTDOWN)
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert q.actions_q.get_nowait() is _SHUTDOWN


def test_retargeter_emits_shutdown_on_stop_event(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())
    q = _make_queues()
    stop = threading.Event()
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    time.sleep(0.05)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    items = []
    while True:
        try:
            items.append(q.actions_q.get_nowait())
        except queue.Empty:
            break
    assert items[-1] is _SHUTDOWN


def test_robot_exits_on_shutdown_sentinel(patch_mock_hand):
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    q.actions_q.put(_SHUTDOWN)
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(patch_mock_hand) == 1  # the hand was constructed


def test_robot_sets_ready_after_init(patch_mock_hand):
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    hand = patch_mock_hand[0]
    assert hand.is_connected()
    stop.set()
    t.join(timeout=2.0)


def test_robot_consumes_orca_joint_positions(patch_mock_hand):
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    action = _midpoint_action()
    q.actions_q.put(action)
    q.actions_q.put(action)
    q.actions_q.put(_SHUTDOWN)
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_robot_accepts_in_rom_positions(patch_mock_hand):
    """An OrcaJointPositions built from the mock's own ROM midpoints must
    flow through without raising. Locks the units the retargeter must emit."""
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    q.actions_q.put(_midpoint_action())
    q.actions_q.put(_SHUTDOWN)
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)


class _FailingConnectHand:
    def __init__(self, model_path=None):
        self.init_called = False
        self.disconnected = False

    def connect(self):
        return False, "Connection failed"

    def init_joints(self, move_to_neutral=True):
        self.init_called = True

    def set_joint_positions(self, action):
        pass

    def disable_torque(self):
        pass

    def disconnect(self):
        self.disconnected = True


class _ExplodingHand:
    def __init__(self, model_path=None):
        self.disabled = False
        self.disconnected = False

    def connect(self):
        return True, "ok"

    def init_joints(self, move_to_neutral=True):
        pass

    def set_joint_positions(self, action):
        raise RuntimeError("OrcaHand.set_joint_positions() failed")

    def disable_torque(self):
        self.disabled = True

    def disconnect(self):
        self.disconnected = True


def test_robot_connect_failure_leaves_ready_clear(monkeypatch):
    instances: list[_FailingConnectHand] = []

    def factory(model_path=None):
        h = _FailingConnectHand()
        instances.append(h)
        return h

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)

    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert not ready.is_set()
    assert not instances[0].init_called


class _ExplodingHand:
    def __init__(self, model_path=None):
        self.disabled = False
        self.disconnected = False

    def connect(self):
        return True, "ok"

    def init_joints(self, move_to_neutral=True):
        pass

    def set_joint_positions(self, action):
        raise RuntimeError("OrcaHand.set_joint_positions() failed")

    def disable_torque(self):
        self.disabled = True

    def disconnect(self):
        self.disconnected = True


class _HomePoseHand:
    _NEUTRAL = {"wrist": -20.0, "thumb_mcp": 33.0}

    def __init__(self, model_path=None):
        self.joint_calls: list = []

    def connect(self):
        return True, "ok"

    def init_joints(self, move_to_neutral=True):
        pass

    def get_joint_position(self):
        return OrcaJointPositions.from_dict({"wrist": 0.0})

    def set_joint_positions(self, action, num_steps=1):
        self.joint_calls.append((action, num_steps))

    def set_zero_position(self):
        pass

    def disable_torque(self):
        pass

    def disconnect(self):
        pass

    @property
    def config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            joint_ids=list(self._NEUTRAL),
            type="right",
            neutral_position=dict(self._NEUTRAL),
        )


def test_orca_hand_sink_go_home(monkeypatch):
    from types import SimpleNamespace

    instances: list[_HomePoseHand] = []

    def factory(model_path=None):
        h = _HomePoseHand()
        instances.append(h)
        return h

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)
    monkeypatch.setattr(
        "orca_teleop.pipeline.CameraManager",
        lambda configs: SimpleNamespace(
            open=lambda: None,
            ensure_live=lambda: None,
            close=lambda: None,
            shapes={},
            capture=lambda: {},
        ),
    )

    from orca_teleop.pipeline import OrcaHandSink

    sink = OrcaHandSink(model_path="ignored")
    sink.connect()
    sink.go_home()
    sink.close()

    hand = instances[0]
    assert len(hand.joint_calls) == 1
    action, num_steps = hand.joint_calls[0]
    assert action.as_dict() == {"wrist": 0.0, "thumb_mcp": 33.0}
    assert num_steps == 5 * MOTION_NUM_STEPS


def test_orca_hand_sink_go_home_skipped_without_hardware(monkeypatch):
    from types import SimpleNamespace

    instances: list[_HomePoseHand] = []

    def factory(model_path=None):
        h = _HomePoseHand()
        instances.append(h)
        return h

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)
    monkeypatch.setattr(
        "orca_teleop.pipeline.CameraManager",
        lambda configs: SimpleNamespace(
            open=lambda: None,
            ensure_live=lambda: None,
            close=lambda: None,
            shapes={},
            capture=lambda: {},
        ),
    )

    from orca_teleop.pipeline import OrcaHandSink

    sink = OrcaHandSink(model_path="ignored", connect_hardware=False)
    sink.connect()
    sink.go_home()

    assert instances[0].joint_calls == []


def test_robot_finally_cleans_up_on_exception(monkeypatch):
    instances: list[_ExplodingHand] = []

    def factory(model_path=None):
        h = _ExplodingHand()
        instances.append(h)
        return h

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)

    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    q.actions_q.put(_midpoint_action())
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert instances[0].disabled and instances[0].disconnected


class _NoopRetargeter:
    """Stands in for the real retargeter, which needs the hand model on disk."""

    @classmethod
    def from_paths(cls, *_args, **_kwargs):
        return cls()

    def retarget(self, _target_pose):
        return None


def test_run_raises_on_connect_failure(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", _FailingConnectHand)
    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _NoopRetargeter)
    with pytest.raises(RuntimeError, match="failed to connect"):
        run("ignored", port=0)


def test_run_does_not_start_producers_if_robot_fails(monkeypatch):
    """Checks threads don't spawn if the robot fails to connect."""
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", _FailingConnectHand)
    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _NoopRetargeter)
    before = {t.name for t in threading.enumerate()}
    with pytest.raises(RuntimeError):
        run("ignored", port=0)
    time.sleep(0.05)
    after_names = {t.name for t in threading.enumerate() if t.is_alive()}
    leaked = after_names - before

    assert "retargeter" not in leaked


def test_run_starts_then_stops_cleanly(monkeypatch, patch_mock_hand):  # noqa: PT019
    """Interrupt the main-thread action loop after a few iterations and verify
    the retargeter thread and gRPC server shut down cleanly."""
    real_get = queue.Queue.get
    calls = {"n": 0}
    main_ident = threading.main_thread().ident

    def fake_get(self, *args, **kwargs):
        if threading.get_ident() == main_ident:
            calls["n"] += 1
            if calls["n"] >= 3:
                raise KeyboardInterrupt
        return real_get(self, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "get", fake_get)
    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _NoopRetargeter)

    run(None, port=0)
    time.sleep(0.05)
    alive = {t.name for t in threading.enumerate() if t.is_alive()}
    assert "retargeter" not in alive


def test_retargeter_forwards_joint_positions(monkeypatch):
    """retargeter_worker turns each HandLandmarks into an OrcaJointPositions action.

    Stubs Retargeter so the test only exercises plumbing — the real retargeter
    needs an OrcaHand model whose joint_ids match the URDF, which is an
    environment concern outside the scope of this test.
    """
    n_frames = 3

    class _StubRetargeter:
        # TODO: move to conftest.py
        def __init__(self):
            self._action = _midpoint_action()

        @classmethod
        def from_paths(cls, *_args, **_kwargs):
            return cls()

        def retarget(self, _target_pose):
            return self._action

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _StubRetargeter)
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())

    q = _make_queues()
    stop = threading.Event()
    for _ in range(n_frames):
        q.landmarks_q.put(_make_landmark())
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)
    items = []
    while True:
        try:
            items.append(q.actions_q.get_nowait())
        except queue.Empty:
            break
    actions = [x for x in items if x is not _SHUTDOWN]
    assert len(actions) > 0
    for action in actions:
        assert isinstance(action, OrcaJointPositions)


def test_retargeter_converts_metaquest_right_hand_landmarks(monkeypatch):
    captured = {}

    class _StubRetargeter:
        @classmethod
        def from_paths(cls, *_args, **_kwargs):
            return cls()

        def retarget(self, target_pose):
            captured["keypoints"] = target_pose.joint_positions.copy()
            return _midpoint_action()

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _StubRetargeter)

    keypoints = plausible_hand_keypoints()
    q = _make_queues()
    q.landmarks_q.put(
        HandLandmarks(
            keypoints=keypoints,
            handedness="right",
            timestamp_ns=time.time_ns(),
        )
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=retargeter_worker,
        args=(q, stop),
        kwargs={"landmark_source": "metaquest"},
        daemon=True,
    )
    thread.start()
    _wait_for(lambda: "keypoints" in captured)
    q.landmarks_q.put(_SHUTDOWN)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    expected = keypoints.copy()
    expected[:, 1] *= -1.0
    np.testing.assert_allclose(captured["keypoints"], expected)


def test_retargeter_preserves_webxr_landmarks_and_wrist_angle(monkeypatch):
    captured = {}

    class _StubRetargeter:
        @classmethod
        def from_paths(cls, *_args, **_kwargs):
            return cls()

        def retarget(self, target_pose):
            captured["keypoints"] = target_pose.joint_positions.copy()
            captured["wrist_angle_degrees"] = target_pose.wrist_angle_degrees
            return _midpoint_action()

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _StubRetargeter)

    keypoints = plausible_hand_keypoints()
    q = _make_queues()
    q.landmarks_q.put(
        HandLandmarks(
            keypoints=keypoints,
            handedness="right",
            timestamp_ns=time.time_ns(),
            wrist_angle_degrees=17.5,
        )
    )
    thread = threading.Thread(
        target=retargeter_worker,
        args=(q, threading.Event()),
        kwargs={"landmark_source": "webxr"},
        daemon=True,
    )
    thread.start()
    _wait_for(lambda: "keypoints" in captured)
    q.landmarks_q.put(_SHUTDOWN)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    np.testing.assert_allclose(captured["keypoints"], keypoints)
    assert captured["wrist_angle_degrees"] == pytest.approx(17.5)


def test_retargeter_worker_passes_backend_and_config(monkeypatch):
    calls = {}

    class _StubRetargeter:
        def __init__(self):
            self._action = _midpoint_action()

        @classmethod
        def from_paths(cls, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return cls()

        def retarget(self, _target_pose):
            return self._action

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _StubRetargeter)

    q = _make_queues()
    stop = threading.Event()
    q.landmarks_q.put(_make_landmark())
    t = _start(
        retargeter_worker,
        q,
        stop,
        "model.yaml",
        "hand.urdf",
        "adaptive_analytical",
        "retarget.yaml",
        name="retargeter",
    )
    time.sleep(0.2)
    stop.set()
    t.join(timeout=2.0)

    assert calls == {
        "args": ("model.yaml", "hand.urdf"),
        "kwargs": {
            "backend": "adaptive_analytical",
            "config_path": "retarget.yaml",
        },
    }


def test_retargeter_skips_none_actions(monkeypatch):
    """
    Tests ``None`` persists as a no-op for the retargeter, so that it is not enqueued.
    """

    class _StubRetargeter:
        # TODO: move to conftest.py
        @classmethod
        def from_paths(cls, *_args, **_kwargs):
            return cls()

        def retarget(self, _target_pose):
            return None

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _StubRetargeter)
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())

    q = _make_queues()
    stop = threading.Event()
    for _ in range(3):
        q.landmarks_q.put(_make_landmark())
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    time.sleep(0.2)
    stop.set()
    t.join(timeout=2.0)

    items = []
    while True:
        try:
            items.append(q.actions_q.get_nowait())
        except queue.Empty:
            break

    assert items == [_SHUTDOWN]
