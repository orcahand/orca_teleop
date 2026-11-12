"""Robustness of the live pipeline: broken frames, stale frames, tracking loss, start and quit."""

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import plausible_hand_keypoints
from orca_core import OrcaJointPositions

from orca_teleop.ingress.server import HandLandmarks, landmark_problem
from orca_teleop.pipeline import (
    _SHUTDOWN,
    MotionRamp,
    OrcaHandSink,
    TeleopQueues,
    _get_latest,
    _put_latest,
    _stop_when_dead,
    retargeter_worker,
    run,
)
from orca_teleop.retargeting.retargeter import TargetPose


def _broken_frames() -> dict[str, np.ndarray]:
    good = plausible_hand_keypoints()
    nan, inf, far = good.copy(), good.copy(), good.copy()
    nan[8, 0] = np.nan
    inf[4, 2] = np.inf
    far[8] = [2.0, 0.0, 0.0]  # a fingertip 2 m away
    return {"nan": nan, "inf": inf, "far": far, "zeros": np.zeros_like(good)}


@pytest.mark.parametrize(
    "name,reason",
    [("nan", "non-finite"), ("inf", "non-finite"), ("far", "out of reach"), ("zeros", "palm size")],
)
def test_broken_frames_are_named(name, reason):
    assert landmark_problem(_broken_frames()[name]) == reason


def test_a_hand_passes():
    assert landmark_problem(plausible_hand_keypoints()) is None
    assert landmark_problem(plausible_hand_keypoints(), wrist_angle_degrees=float("nan")) == (
        "non-finite"
    )


def test_target_pose_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        TargetPose(joint_positions=_broken_frames()["nan"])


class _RecordingRetargeter:
    """Answers every frame with the same command and remembers what it was given."""

    def __init__(self, delay_s: float = 0.0):
        self.seen: list[np.ndarray] = []
        self.resets = 0
        self.delay_s = delay_s
        roms = {"index_mcp": (0.0, 90.0), "wrist": (-20.0, 20.0)}
        self.action = OrcaJointPositions({j: 0.5 * (lo + hi) for j, (lo, hi) in roms.items()})

    def retarget(self, pose):
        assert np.all(np.isfinite(pose.joint_positions)), "a broken frame reached the solver"
        self.seen.append(pose.joint_positions.copy())
        time.sleep(self.delay_s)
        return self.action

    def reset_tracking(self):
        self.resets += 1


def _frame(keypoints: np.ndarray, stamp: int = 0) -> HandLandmarks:
    return HandLandmarks(keypoints=keypoints, handedness="right", timestamp_ns=stamp)


def _queues(size: int = 8) -> TeleopQueues:
    return TeleopQueues(landmarks_q=queue.Queue(maxsize=size), actions_q=queue.Queue(maxsize=size))


def _run_worker(q: TeleopQueues, retargeter, until, timeout: float = 3.0) -> None:
    stop = threading.Event()
    thread = threading.Thread(
        target=retargeter_worker,
        args=(q, stop),
        kwargs={"retargeter": retargeter},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_broken_frames_never_reach_the_solver_and_good_ones_still_do():
    q = _queues(size=32)
    retargeter = _RecordingRetargeter()
    good = plausible_hand_keypoints()
    for broken in _broken_frames().values():
        q.landmarks_q.put(_frame(broken))
        _run_worker(q, retargeter, until=q.landmarks_q.empty, timeout=1.0)
    assert retargeter.seen == []
    q.landmarks_q.put(_frame(good))
    _run_worker(q, retargeter, until=lambda: len(retargeter.seen) == 1)
    np.testing.assert_allclose(retargeter.seen[0], good)


def test_a_slow_retargeter_always_gets_the_newest_frame():
    """Frames arrive every 5 ms, the solver takes 40 ms: it must skip, never fall behind."""
    q = _queues()
    retargeter = _RecordingRetargeter(delay_s=0.04)
    stop = threading.Event()
    thread = threading.Thread(
        target=retargeter_worker, args=(q, stop), kwargs={"retargeter": retargeter}, daemon=True
    )
    thread.start()
    n = 60
    for i in range(n):
        kp = plausible_hand_keypoints()
        kp[:, 2] = i * 1e-4  # tag the frame with its number
        _put_latest(q.landmarks_q, _frame(kp))
        time.sleep(0.005)
    time.sleep(0.1)
    stop.set()
    thread.join(timeout=2.0)
    taken = [int(round(kp[0, 2] / 1e-4)) for kp in retargeter.seen]
    assert taken == sorted(taken)
    assert taken[-1] == n - 1  # the last frame is the one solved last
    assert len(taken) < n / 3  # stale frames were skipped, not queued up
    # each solve took the newest frame there was: no gap between two solves is
    # smaller than the frames that arrived while the previous one ran (except the
    # last: when the sender stops, its final frame may come right after)
    assert min(np.diff(taken[:-1])) >= 2


def test_get_latest_skips_old_items_but_never_a_shutdown():
    q: queue.Queue = queue.Queue()
    for i in range(5):
        q.put(i)
    assert _get_latest(q, 0.1) == (4, 4)
    for item in (1, _SHUTDOWN, 3):
        q.put(item)
    assert _get_latest(q, 0.1)[0] is _SHUTDOWN


def test_put_latest_keeps_the_new_item_when_full():
    q: queue.Queue = queue.Queue(maxsize=2)
    for i in range(5):
        _put_latest(q, i)
    assert [q.get_nowait(), q.get_nowait()] == [3, 4]


def _pose(value: float) -> OrcaJointPositions:
    return OrcaJointPositions({"index_mcp": value, "wrist": 0.0})


def test_ramp_eases_in_from_the_start_pose():
    ramp = MotionRamp(loss_timeout_s=0.15, ramp_s=0.25, start=_pose(0.0))
    target = _pose(90.0)
    values = [ramp.step(target, t).as_dict()["index_mcp"] for t in (0.0, 0.0625, 0.125, 0.25)]
    assert values[0] == 0.0
    assert 0.0 < values[1] < values[2] < 90.0
    assert values[2] == pytest.approx(45.0)  # smoothstep is half way at half time
    assert values[3] == 90.0


def test_a_slow_steady_stream_still_moves_the_hand():
    """At 5 fps every gap is over the loss timeout; the hand must follow anyway."""
    ramp = MotionRamp(loss_timeout_s=0.15, ramp_s=0.25, start=_pose(0.0))
    values = [ramp.step(_pose(90.0), 0.2 * k).as_dict()["index_mcp"] for k in range(5)]
    assert values[-1] == 90.0
    assert not ramp.lost(0.2 * 5)
    assert ramp.lost(0.2 * 4 + 1.0)  # a second without a frame is still a loss


def test_flickering_detection_cannot_train_the_ramp_to_miss_a_loss():
    """Frames every 0.35 s (each gap a loss) must not raise the timeout to ~1 s."""
    ramp = MotionRamp(loss_timeout_s=0.15, ramp_s=0.25, start=_pose(0.0))
    t = 0.0
    for _ in range(10):
        ramp.step(_pose(0.0), t)
        t += 0.35
    assert ramp.lost(t - 0.35 + 0.9)  # back after 0.9 s in a new pose: eases in, no jump


def test_ramp_tracks_directly_while_tracking_and_eases_back_after_a_loss():
    ramp = MotionRamp(loss_timeout_s=0.15, ramp_s=0.25)
    assert ramp.step(_pose(10.0), 0.0).as_dict()["index_mcp"] == 10.0  # no start pose: no ramp
    assert ramp.step(_pose(20.0), 0.03).as_dict()["index_mcp"] == 20.0
    assert not ramp.lost(0.1)
    assert ramp.lost(0.5)  # 470 ms without a frame: holding
    back = ramp.step(_pose(80.0), 0.5).as_dict()["index_mcp"]
    assert back == 20.0  # the return starts where the hand was held
    assert ramp.step(_pose(80.0), 0.625).as_dict()["index_mcp"] == pytest.approx(50.0)
    assert ramp.step(_pose(80.0), 0.75).as_dict()["index_mcp"] == 80.0


def test_tracking_loss_resets_the_solver_and_eases_back():
    q = _queues()
    retargeter = _RecordingRetargeter()
    stop = threading.Event()
    thread = threading.Thread(
        target=retargeter_worker,
        args=(q, stop),
        kwargs={"retargeter": retargeter, "start_positions": _pose(0.0)},
        daemon=True,
    )
    thread.start()
    q.landmarks_q.put(_frame(plausible_hand_keypoints()))
    first, _ = _get_latest(q.actions_q, 2.0)
    assert first.as_dict()["index_mcp"] == 0.0  # eases in from the hand's start pose
    resets_before_loss = retargeter.resets
    time.sleep(0.3)  # longer than the loss timeout
    q.landmarks_q.put(_frame(plausible_hand_keypoints()))
    _get_latest(q.actions_q, 2.0)
    stop.set()
    thread.join(timeout=2.0)
    assert retargeter.resets == resets_before_loss + 1


class _QuitHand:
    config = SimpleNamespace(joint_ids=["index_mcp", "wrist"])

    def __init__(self, model_path=None):
        self.calls: list[str] = []

    def connect(self):
        return True, "ok"

    def init_joints(self, move_to_neutral=True):
        self.calls.append("init")

    def get_joint_position(self):
        return _pose(12.0)

    def set_joint_positions(self, action, num_steps=1):
        self.calls.append("move")

    def set_zero_position(self):
        self.calls.append("zero")

    def disable_torque(self):
        self.calls.append("torque off")

    def disconnect(self):
        self.calls.append("disconnect")


def _no_cameras(monkeypatch):
    monkeypatch.setattr(
        "orca_teleop.pipeline.CameraManager",
        lambda configs: SimpleNamespace(
            open=lambda: None, ensure_live=lambda: None, close=lambda: None, shapes={}
        ),
    )


def test_quitting_does_not_move_the_hand(monkeypatch):
    hands: list[_QuitHand] = []
    monkeypatch.setattr(
        "orca_teleop.pipeline.OrcaHand", lambda mp=None: hands.append(_QuitHand()) or hands[-1]
    )
    _no_cameras(monkeypatch)
    sink = OrcaHandSink(model_path="ignored", move_to_neutral=False)
    sink.connect()
    assert sink.current_positions().as_dict()["index_mcp"] == 12.0
    sink.close()
    assert hands[0].calls == ["init", "torque off", "disconnect"]


class _BrokenRetargeter:
    @classmethod
    def from_paths(cls, *_args, **_kwargs):
        raise ValueError("bad retarget config")


class _NeverConnectSink:
    retarget_model_path = None

    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def current_positions(self):
        return None

    def run_loop(self, actions_q, stop_event):
        pass

    def close(self):
        pass


def test_a_broken_retargeter_fails_before_the_hand_connects(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _BrokenRetargeter)
    sink = _NeverConnectSink()
    with pytest.raises(ValueError, match="bad retarget config"):
        run(None, sink=sink, port=0)
    assert not sink.connected


def test_a_busy_port_fails_before_the_hand_connects(monkeypatch):
    from orca_teleop.ingress.server import IngressServer

    class _Noop:
        @classmethod
        def from_paths(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _Noop)
    busy = IngressServer(queue.Queue(), threading.Event(), port=0)
    port = busy.start()
    sink = _NeverConnectSink()
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            run(None, sink=sink, port=port)
    finally:
        busy.stop()
    assert not sink.connected


def test_a_dead_publisher_ends_the_session():
    stop = threading.Event()
    process = SimpleNamespace(is_alive=lambda: False, exitcode=1)
    _stop_when_dead(process, stop, "publisher")
    assert stop.wait(2.0)


class _ViewerHand(_QuitHand):
    """``fail_reads``: "stale" as orca_core 0.13 fails (cached values, last_read_ok
    False), "raise" as an older one might."""

    def __init__(self, model_path=None, fail_reads: str | None = None):
        super().__init__(model_path)
        self.reads = 0
        self.fail_reads = fail_reads
        self.last_read_ok = True

    def get_joint_position(self):
        self.reads += 1
        self.last_read_ok = self.fail_reads is None
        if self.fail_reads == "raise":
            raise OSError("bus timeout")
        return _pose(0.0 if self.fail_reads else 12.0)


class _Viewer:
    def __init__(self):
        self.commanded_seen: list[OrcaJointPositions] = []
        self.measured_seen: list[OrcaJointPositions] = []

    def commanded(self, positions):
        self.commanded_seen.append(positions)

    def measured(self, positions, read_ms=None):
        self.measured_seen.append(positions)


def _run_sink(monkeypatch, hand, seconds: float, commands: int, **sink_kwargs):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    _no_cameras(monkeypatch)
    viewer = _Viewer()
    sink = OrcaHandSink(model_path="ignored", move_to_neutral=False, viewer=viewer, **sink_kwargs)
    sink.connect()
    actions: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    thread = threading.Thread(target=sink.run_loop, args=(actions, stop), daemon=True)
    thread.start()
    for k in range(commands):
        actions.put(_pose(float(k)))
        time.sleep(seconds / max(commands, 1))
    time.sleep(seconds if commands == 0 else 0.05)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    return viewer


def test_the_viewer_sees_every_command_and_regular_motor_reads(monkeypatch):
    hand = _ViewerHand()
    viewer = _run_sink(monkeypatch, hand, seconds=0.5, commands=10, readback_hz=20.0)
    assert [p.as_dict()["index_mcp"] for p in viewer.commanded_seen] == [
        float(k) for k in range(10)
    ]
    assert 6 <= hand.reads <= 16  # about 20 a second over ~0.55 s
    assert all(p.as_dict()["index_mcp"] == 12.0 for p in viewer.measured_seen)


def test_motor_reads_continue_while_no_command_arrives(monkeypatch):
    hand = _ViewerHand()
    viewer = _run_sink(monkeypatch, hand, seconds=0.4, commands=0, readback_hz=20.0)
    assert viewer.commanded_seen == []
    assert hand.reads >= 4  # the viewer shows the hand's pose before the first frame too


@pytest.mark.parametrize("failure", ["stale", "raise"])
def test_a_failing_motor_read_does_not_stop_teleop(monkeypatch, failure):
    hand = _ViewerHand(fail_reads=failure)
    viewer = _run_sink(monkeypatch, hand, seconds=0.3, commands=5, readback_hz=20.0)
    assert len(viewer.commanded_seen) == 5
    assert hand.calls.count("move") == 5
    assert viewer.measured_seen == [None] * 3  # shown as no reading, never as stale values
    assert hand.reads == 3  # three in a row: then it waits a second before trying again


def test_one_failed_read_does_not_pause_the_reads(monkeypatch):
    class _OneMiss(_ViewerHand):
        def get_joint_position(self):
            self.fail_reads = "stale" if self.reads == 1 else None
            return super().get_joint_position()

    hand = _OneMiss()
    viewer = _run_sink(monkeypatch, hand, seconds=0.3, commands=0, readback_hz=20.0)
    assert viewer.measured_seen[1] is None
    assert len(viewer.measured_seen) >= 5  # reading went on at 20 a second


def test_a_dry_run_never_reads_the_motors(monkeypatch):
    hand = _ViewerHand()
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    _no_cameras(monkeypatch)
    viewer = _Viewer()
    sink = OrcaHandSink(model_path="ignored", connect_hardware=False, viewer=viewer)
    sink.connect()
    actions: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    thread = threading.Thread(target=sink.run_loop, args=(actions, stop), daemon=True)
    thread.start()
    actions.put(_pose(5.0))
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2.0)
    assert len(viewer.commanded_seen) == 1 and hand.reads == 0 and hand.calls == []


class _ClosingSink(_NeverConnectSink):
    def __init__(self):
        super().__init__()
        self.closed = False

    def run_loop(self, actions_q, stop_event):
        stop_event.wait(3.0)  # with no stop after 3 s, the test fails instead of hanging

    def close(self):
        time.sleep(0.2)  # a slow torque-off: the second Ctrl+C lands in the middle
        self.closed = True


def test_two_ctrl_c_in_a_row_still_close_the_hand(monkeypatch):
    """Under `uv run` one Ctrl+C arrives twice; the second must not cut the shutdown short."""
    import os
    import signal

    class _Noop:
        @classmethod
        def from_paths(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _Noop)
    stray: list[int] = []

    def ours(*_):  # catches any Ctrl+C run() lets through
        stray.append(1)

    previous = signal.signal(signal.SIGINT, ours)
    try:

        def press_twice():
            time.sleep(0.3)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.1)  # the second lands while the sink is closing
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=press_twice, daemon=True).start()
        sink = _ClosingSink()
        run(None, sink=sink, port=0)
        assert sink.closed
        assert stray == []
        assert signal.getsignal(signal.SIGINT) is ours  # run() put the old handler back
    finally:
        signal.signal(signal.SIGINT, previous)


class _NoopRetargeter:
    @classmethod
    def from_paths(cls, *_args, **_kwargs):
        return cls()


def _hand_sink(monkeypatch, hand) -> OrcaHandSink:
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _NoopRetargeter)
    _no_cameras(monkeypatch)
    return OrcaHandSink(model_path="ignored", move_to_neutral=False)


def test_a_bad_start_read_refuses_to_start_and_turns_torque_off(monkeypatch):
    """A failed first read would make up the start pose; the hand must not move from it."""
    hand = _ViewerHand(fail_reads="stale")
    sink = _hand_sink(monkeypatch, hand)
    with pytest.raises(RuntimeError, match="Could not read the hand's pose"):
        run(None, sink=sink, port=0)
    assert hand.reads == 3
    assert "move" not in hand.calls
    assert hand.calls[-2:] == ["torque off", "disconnect"]


def test_ctrl_c_during_start_up_still_turns_torque_off(monkeypatch):
    class _InterruptedHand(_QuitHand):
        def init_joints(self, move_to_neutral=True):
            self.calls.append("torque on")
            raise KeyboardInterrupt  # Ctrl+C right after init switched the torque on

    hand = _InterruptedHand()
    sink = _hand_sink(monkeypatch, hand)
    run(None, sink=sink, port=0)  # returns: Ctrl+C is a normal way to stop
    assert hand.calls == ["torque on", "torque off", "disconnect"]


def test_a_hand_that_never_connected_is_not_touched_on_close(monkeypatch):
    class _NoBus(_QuitHand):
        def connect(self):
            return False, "no port"

    hand = _NoBus()
    sink = _hand_sink(monkeypatch, hand)
    with pytest.raises(RuntimeError, match="failed to connect"):
        run(None, sink=sink, port=0)
    assert hand.calls == []


def _logged_run(monkeypatch, tmp_path, hand, commands=(), seconds=0.2, **sink_kwargs):
    """Run the sink with a joint log, send ``commands``, return the logged events."""
    import json

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    _no_cameras(monkeypatch)
    log = tmp_path / "joints.jsonl"
    sink = OrcaHandSink(
        model_path="ignored", move_to_neutral=False, joint_log=str(log), **sink_kwargs
    )
    sink.connect()
    actions: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    thread = threading.Thread(target=sink.run_loop, args=(actions, stop), daemon=True)
    thread.start()
    for action in commands:
        actions.put(action)
        time.sleep(seconds / len(commands))
    if not commands:
        time.sleep(seconds)
    stop.set()
    thread.join(timeout=2.0)
    sink.close()
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_the_joint_log_has_every_command_and_read(monkeypatch, tmp_path):
    import numpy as np

    poses = [OrcaJointPositions({"index_mcp": np.float32(k), "wrist": 0.0}) for k in range(5)]
    events = _logged_run(monkeypatch, tmp_path, _ViewerHand(), poses, seconds=0.3, readback_hz=20.0)
    assert "start" in events[0]
    commanded = [e["commanded"]["index_mcp"] for e in events if "commanded" in e]
    measured = [e for e in events if "measured" in e]
    assert commanded == [0.0, 1.0, 2.0, 3.0, 4.0]  # numpy floats written as plain numbers
    assert len(measured) >= 3 and all(e["measured"]["index_mcp"] == 12.0 for e in measured)
    assert all(e["t"] > 1e9 for e in events)  # wall-clock seconds


def test_a_failed_read_is_logged_as_none(monkeypatch, tmp_path):
    events = _logged_run(monkeypatch, tmp_path, _ViewerHand(fail_reads="stale"), log_current=True)
    measured = [e["measured"] for e in events if "measured" in e]
    assert 1 <= len(measured) <= 3 and all(m is None for m in measured)
    assert not any("motors" in e for e in events)  # no current read after a failed one


class _CurrentHand(_ViewerHand):
    config = SimpleNamespace(joint_ids=["index_mcp", "wrist"], motor_ids=[3, 17])

    def __init__(self, stale_state: bool = False):
        super().__init__()
        self.stale_state = stale_state

    def get_motor_state(self):
        import numpy as np

        self.last_read_ok = not self.stale_state
        return SimpleNamespace(
            position=np.zeros(2), velocity=np.array([0.5, 0.0]), current=np.array([120.0, -6.5])
        )


def test_motor_currents_are_logged_by_motor_id_when_asked(monkeypatch, tmp_path):
    events = _logged_run(monkeypatch, tmp_path, _CurrentHand(), readback_hz=20.0, log_current=True)
    reads = [e for e in events if "measured" in e]
    assert len(reads) >= 2
    assert all(
        e["motors"] == {"current_ma": {"3": 120.0, "17": -6.5}, "velocity": {"3": 0.5, "17": 0.0}}
        for e in reads
    )


def test_a_stale_current_read_is_logged_as_none(monkeypatch, tmp_path):
    hand = _CurrentHand(stale_state=True)
    events = _logged_run(monkeypatch, tmp_path, hand, readback_hz=20.0, log_current=True)
    reads = [e for e in events if "measured" in e]
    assert reads and all(e["motors"] is None for e in reads)
    assert all(e["measured"] is not None for e in reads)  # the joint read itself was fine


def test_currents_are_not_read_unless_asked(monkeypatch, tmp_path):
    events = _logged_run(monkeypatch, tmp_path, _CurrentHand(), readback_hz=20.0)
    assert not any("motors" in e for e in events)


def test_the_joint_log_stamps_reads_and_commands_when_they_start(monkeypatch, tmp_path):
    """Lag is measured from these times: a read is stamped when it begins, not when its
    line is written after it (and after the current read)."""

    class _SlowHand(_CurrentHand):
        def __init__(self):
            super().__init__()
            self.read_starts: list[float] = []
            self.sent: list[float] = []

        def get_joint_position(self):
            self.read_starts.append(time.time())
            time.sleep(0.03)
            return super().get_joint_position()

        def get_motor_state(self):
            time.sleep(0.03)
            return super().get_motor_state()

        def set_joint_positions(self, action, num_steps=1):
            self.sent.append(time.time())
            time.sleep(0.02)

    hand = _SlowHand()
    events = _logged_run(
        monkeypatch, tmp_path, hand, [_pose(1.0)], readback_hz=5.0, log_current=True
    )
    reads = [e["t"] for e in events if "measured" in e]
    assert reads == pytest.approx(hand.read_starts[: len(reads)], abs=0.002)
    (commanded,) = [e["t"] for e in events if "commanded" in e]
    assert commanded == pytest.approx(hand.sent[0], abs=0.002)


def test_the_joint_log_never_overwrites_and_fails_before_the_hand(monkeypatch, tmp_path):
    log = tmp_path / "joints.jsonl"
    log.write_text("an earlier measurement\n")
    hand = _QuitHand()
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    _no_cameras(monkeypatch)
    sink = OrcaHandSink(model_path="ignored", joint_log=str(log))
    with pytest.raises(RuntimeError, match="exists"):
        sink.connect()
    assert hand.calls == [] and log.read_text() == "an earlier measurement\n"


def test_the_log_start_line_says_how_it_was_made(monkeypatch, tmp_path):
    (start,) = [
        e["start"] for e in _logged_run(monkeypatch, tmp_path, _ViewerHand()) if "start" in e
    ]
    assert start["log_current"] is False and start["orca_core"] and start["model_path"] == "ignored"


class _FailingCloseHand(_QuitHand):
    def disable_torque(self):
        self.calls.append("torque off")
        raise OSError("port gone")


def test_close_disconnects_even_when_the_torque_off_fails(monkeypatch):
    hand = _FailingCloseHand()
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    _no_cameras(monkeypatch)
    sink = OrcaHandSink(model_path="ignored", move_to_neutral=False)
    sink.connect()
    sink.close()  # logs the error, doesn't raise
    assert hand.calls == ["init", "torque off", "disconnect"]  # disconnect retries it


def test_a_failing_camera_close_does_not_skip_the_torque_off(monkeypatch):
    def broken():
        raise RuntimeError("camera stuck")

    hand = _QuitHand()
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: hand)
    monkeypatch.setattr(
        "orca_teleop.pipeline.CameraManager",
        lambda configs: SimpleNamespace(
            open=lambda: None, ensure_live=lambda: None, close=broken, shapes={}
        ),
    )
    sink = OrcaHandSink(model_path="ignored", move_to_neutral=False)
    sink.connect()
    sink.close()
    assert hand.calls == ["init", "torque off", "disconnect"]


def test_a_doubled_ctrl_c_during_start_up_still_closes_the_hand(monkeypatch):
    """Under `uv run`: the first Ctrl+C stops init, the second lands in the shutdown."""
    import os
    import signal

    class _PressedHand(_QuitHand):
        def init_joints(self, move_to_neutral=True):
            self.calls.append("torque on")
            os.kill(os.getpid(), signal.SIGINT)  # e.g. stopping a calibration
            time.sleep(1.0)  # interrupted before this ends

        def disable_torque(self):
            os.kill(os.getpid(), signal.SIGINT)  # uv's copy of the same Ctrl+C
            self.calls.append("torque off")

    hand = _PressedHand()
    sink = _hand_sink(monkeypatch, hand)
    previous = signal.getsignal(signal.SIGINT)
    run(None, sink=sink, port=0)  # returns: no KeyboardInterrupt escapes
    assert hand.calls == ["torque on", "torque off", "disconnect"]
    assert signal.getsignal(signal.SIGINT) is previous


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_sigterm_and_sighup_close_the_hand(monkeypatch, name):
    """A closed terminal (SIGHUP) or a kill (SIGTERM) must not leave the torque on."""
    import os
    import signal

    sig = getattr(signal, name)
    monkeypatch.setattr("orca_teleop.pipeline.Retargeter", _NoopRetargeter)
    threading.Timer(0.3, lambda: os.kill(os.getpid(), sig)).start()
    sink = _ClosingSink()
    previous = signal.getsignal(sig)
    run(None, sink=sink, port=0)
    assert sink.closed
    assert signal.getsignal(sig) is previous


def test_log_current_without_a_log_file_is_refused(monkeypatch):
    from orca_teleop.cli import teleop_hand

    monkeypatch.setattr("orca_teleop.cli.OrcaHandSink", None)  # must fail before any sink
    with pytest.raises(SystemExit):
        teleop_hand(["--log-current"])


def _publisher(monkeypatch):
    """A MediaPipe publisher without the model (no camera, no server)."""
    import mediapipe as mp

    from orca_teleop.ingress.mediapipe.publisher import MediaPipePublisher

    monkeypatch.setattr(
        mp.tasks.vision.HandLandmarker, "create_from_options", lambda options: object()
    )
    return MediaPipePublisher(handedness="right")


def _result(value: float):
    points = [SimpleNamespace(x=value, y=0.0, z=0.0) for _ in range(21)]
    return SimpleNamespace(
        hand_landmarks=[points],
        hand_world_landmarks=[points],
        handedness=[[SimpleNamespace(category_name="Right", score=0.9)]],
    )


def test_the_publisher_sends_the_newest_result_once_and_wakes_at_once(monkeypatch):
    publisher = _publisher(monkeypatch)
    frames = publisher._frame_generator()
    publisher._on_result(_result(1.0), None, 1)  # before the generator waits: kept
    assert next(frames).keypoints[0] == 1.0
    publisher._on_result(_result(2.0), None, 2)
    publisher._on_result(_result(3.0), None, 3)
    assert next(frames).keypoints[0] == 3.0  # newest wins, sent once

    got: list[tuple[float, float]] = []
    waiter = threading.Thread(
        target=lambda: got.append((next(frames).keypoints[0], time.perf_counter())), daemon=True
    )
    waiter.start()
    time.sleep(0.15)
    assert not got  # nothing new: no repeat of 3.0
    sent = time.perf_counter()
    publisher._on_result(_result(4.0), None, 4)
    waiter.join(1.0)
    assert got and got[0][0] == 4.0
    assert got[0][1] - sent < 0.05  # woken by the result, not by a poll or timeout
