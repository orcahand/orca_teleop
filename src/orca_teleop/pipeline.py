"""Teleop pipeline: ingress (gRPC) -> retargeter -> robot control.

Architecture:

    Typical deployment spans two machines, with gRPC as the only network boundary:

        TELEOP MACHINE                                      ROBOT MACHINE
    +---------------------------+              +--------------------------------------+
    |         publisher         |    gRPC      |            teleop pipeline           |
    |  (MediaPipe, Manus, etc.) +------------->|  +---------------+                   |
    |                           |  HandFrame   |  | IngressServer |                   |
    +---------------------------+              |  +-------+-------+                   |
                                               |          | landmarks_q               |
                                               |          v                           |
                                               |  +---------------+                   |
                                               |  |  retargeter   |                   |
                                               |  +-------+-------+                   |
                                               |          | actions_q                 |
                                               |          v                           |
                                               |  +---------------+                   |
                                               |  |     robot     |                   |
                                               |  +---------------+                   |
                                               +--------------------------------------+

The ingress is a gRPC server streaming ``HandFrame`` from a generic publisher (MediaPipe webcam,
Manus glove, VisionPro, replay file, ...).
Publishers are standalone scripts that know nothing about the robot, they just stream
``(21, 3)`` hand landmarks over the network.

The retargeter and robot stages run as threads on the robot-side machine.
"""

import importlib.metadata
import json
import logging
import queue
import signal
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

# ``OpenCVCameraConfig`` is also re-exported here for callers that historically
# imported it from ``orca_teleop.pipeline``.
from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
from orca_teleop.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HAND,
    DEFAULT_PORT,
    HEARTBEAT_INTERVAL,
    JOIN_TIMEOUT,
    LOSS_TIMEOUT_MAX_S,
    LOSS_TIMEOUT_S,
    MOTION_NUM_STEPS,
    QUEUES_MAXSIZE,
    RAMP_S,
    READ_FAILS_IN_ROW,
    READ_RETRY_S,
)
from orca_teleop.ingress.server import HandLandmarks, IngressServer, landmark_problem
from orca_teleop.retargeting.retargeter import Retargeter, RetargeterBackend, TargetPose

logger = logging.getLogger(__name__)

_SHUTDOWN = object()
LandmarkSource = Literal["mediapipe", "metaquest", "webxr"]


def _shutdown_queue(q: "queue.Queue[Any]") -> None:
    """Signal a downstream worker to stop, without blocking."""
    _put_latest(q, _SHUTDOWN)


def _put_latest(q: "queue.Queue[Any]", item: Any) -> None:
    """Enqueue without blocking; when full, drop the oldest item, never the new one."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def _get_latest(q: "queue.Queue[Any]", timeout: float) -> tuple[Any, int]:
    """The newest queued item (or ``_SHUTDOWN`` if one is queued) and how many were skipped.

    Raises ``queue.Empty`` after ``timeout`` with nothing queued.
    """
    item = q.get(timeout=timeout)
    skipped = 0
    while item is not _SHUTDOWN:
        try:
            newer = q.get_nowait()
        except queue.Empty:
            break
        item, skipped = newer, skipped + 1
    return item, skipped


class MotionRamp:
    """Hold on tracking loss, then ease back to the targets (DD3).

    The hand keeps its last command while no frame arrives. The first command, and
    the first after a loss (see ``lost``), starts a ramp: for
    ``ramp_s`` the output eases from the pose sent last (or ``start``, the hand's
    pose before teleop) to the live targets, so neither the start nor a tracking
    return jumps. Built in teleop for now; moves to orca_core with the control loop.
    """

    def __init__(
        self,
        loss_timeout_s: float = LOSS_TIMEOUT_S,
        ramp_s: float = RAMP_S,
        start: OrcaJointPositions | None = None,
    ) -> None:
        self.loss_timeout_s = float(loss_timeout_s)
        self.ramp_s = float(ramp_s)
        self._sent: dict[str, float] | None = start.as_dict() if start is not None else None
        self._last_t: float | None = None
        self._gaps: deque[float] = deque(maxlen=9)  # recent frame intervals, s
        self._ramp_t0 = 0.0
        self._ramp_from: dict[str, float] | None = None

    def lost(self, now: float) -> bool:
        """True before the first target and after a gap longer than both the loss timeout
        and three usual frame intervals: a slow but steady stream (a few fps) is still
        tracking, or every frame would restart the ramp and the hand would never move.
        Capped at LOSS_TIMEOUT_MAX_S, so flickering detection (whose gaps are losses
        themselves) can't train it to miss a real loss."""
        if self._last_t is None:
            return True
        usual = float(np.median(self._gaps)) if self._gaps else 0.0
        timeout = max(self.loss_timeout_s, min(3.0 * usual, LOSS_TIMEOUT_MAX_S))
        return now - self._last_t > timeout

    def step(self, target: OrcaJointPositions, now: float) -> OrcaJointPositions:
        if self.lost(now):
            self._ramp_t0, self._ramp_from = now, self._sent
        if self._last_t is not None:
            self._gaps.append(now - self._last_t)
        self._last_t = now
        goal = target.as_dict()
        s = min(1.0, (now - self._ramp_t0) / self.ramp_s) if self.ramp_s > 0 else 1.0
        if self._ramp_from is None or s >= 1.0:
            self._ramp_from = None
            out = goal
        else:
            ease = s * s * (3.0 - 2.0 * s)
            out = {
                j: v
                if self._ramp_from.get(j) is None
                else self._ramp_from[j] + ease * (v - self._ramp_from[j])
                for j, v in goal.items()
            }
        self._sent = out
        return OrcaJointPositions(out)


@dataclass
class TeleopQueues:
    landmarks_q: "queue.Queue[HandLandmarks]"
    actions_q: "queue.Queue[OrcaJointPositions | object]"


@dataclass
class SinkObservation:
    """One synchronized observation produced by a sink.

    ``joint_state`` is proprioception aligned with the sink's ``joint_ids``;
    ``images`` maps each camera name to an RGB ``(H, W, 3)`` uint8 frame.
    """

    joint_state: np.ndarray
    images: dict[str, np.ndarray]


class RobotSink(ABC):
    """Pluggable consumer of ``OrcaJointPositions``.

    The sink owns the main-thread loop, routing actions to a sink (real robot
    or sim environment), which implements its own run_loop.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def run_loop(
        self,
        actions_q: "queue.Queue[OrcaJointPositions | object]",
        stop_event: threading.Event,
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def current_positions(self) -> OrcaJointPositions | None:
        """The pose the sink holds before teleop starts, if known; the start ramp begins there."""
        return None


class RecordableSink(RobotSink):
    """A sink that owns and composes the full observation for recording.

    Beyond driving the robot, a recordable sink is the single source of truth
    for its observation — proprioception plus any cameras. Those cameras are the
    sink's own concern: real cameras it opens from configs, and/or intrinsic
    sources only it can produce (e.g. a sim's MuJoCo render). This keeps the
    recorder a thin wrapper that merely packages ``get_observation()`` into the
    target dataset format.

    Contract:
      - ``joint_ids``: names for the proprioception/action vectors.
      - ``camera_shapes``: ``{name: (H, W, C)}`` for every image the observation
        contains (known before recording so the dataset schema can be built).
      - ``get_observation()``: one synchronized :class:`SinkObservation`.
      - ``dispatch_action()``: send a commanded action to the robot.
    """

    @property
    @abstractmethod
    def joint_ids(self) -> list[str]: ...

    @property
    @abstractmethod
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]: ...

    @abstractmethod
    def get_observation(self) -> SinkObservation: ...

    @abstractmethod
    def dispatch_action(self, action: OrcaJointPositions) -> None: ...

    @abstractmethod
    def home_position(self) -> OrcaJointPositions:
        """Target joint pose used as the idle reference between episodes."""

    @abstractmethod
    def go_home(self) -> None:
        """Move the hand to the recording rest pose between episodes."""


class OrcaHandSink(RecordableSink):
    """Default sink: streams actions to a physical ``OrcaHand``.

    Resolves ``OrcaHand`` via the module-level attribute so tests that
    monkeypatch ``orca_teleop.pipeline.OrcaHand`` still intercept construction.

    With a ``viewer`` (``orca_teleop.hand_viewer.HandViewer``) or a ``joint_log``
    file, every command goes to it too, and the loop reads the motors' positions
    ``readback_hz`` times a second, between commands, so one thread owns the bus.
    The log is JSON lines: a ``start`` line, then ``{"t", "commanded"}`` per command
    (degrees, as sent: before orca_core clamps them to the joint ranges) and
    ``{"t", "measured", "read_ms"}`` per read (``measured`` null if it failed). ``t`` is
    ``time.time()`` in seconds when the command was sent or the read began. With
    ``log_current``, a good read also logs ``"motors": {"current_ma", "velocity"}``
    (velocity in rad/s) per motor id, from a second bus read right after (about 4 ms;
    orca_core has no public call giving joints and currents in one). The log never
    overwrites a file: an existing one fails at ``connect``, before the hand.
    """

    def __init__(
        self,
        model_path: str | None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
        connect_hardware: bool = True,
        move_to_neutral: bool = True,
        viewer: Any | None = None,
        readback_hz: float = 10.0,
        joint_log: str | None = None,
        log_current: bool = False,
    ) -> None:
        self._model_path = model_path
        self._hand = OrcaHand(model_path)
        self._connect_hardware = connect_hardware
        self._move_to_neutral = move_to_neutral
        self.viewer = viewer
        self._readback_hz = float(readback_hz)
        self._read_failures = 0
        self._failed_in_row = 0
        self._read_error = ""  # why the last read failed
        self._joint_log_path = joint_log
        self._log_current = log_current
        self._joint_log: Any | None = None
        self._connected = False  # the bus is open: init may have switched the torque on
        self._cameras = CameraManager(camera_configs or [])

    def connect(self) -> None:
        if self._joint_log_path is not None:
            # Before the hand, so a bad path fails with the torque off; "x" so an
            # earlier measurement is never overwritten.
            try:
                self._joint_log = open(self._joint_log_path, "x", buffering=1)
            except FileExistsError:
                raise RuntimeError(
                    f"Joint log {self._joint_log_path} exists; give a new file name"
                ) from None
        if self._connect_hardware:
            success, message = self._hand.connect()
            if not success:
                raise RuntimeError(f"Robot failed to connect: {message}")
            self._connected = True
            self._hand.init_joints(move_to_neutral=self._move_to_neutral)
        else:
            logger.warning(
                "OrcaHandSink: hardware connection skipped (connect_hardware=False) — "
                "joint state is reported as zeros and actions are NOT dispatched. Useful "
                "for validating the recording/vision pipeline without a physical hand."
            )
        self._cameras.open()
        self._cameras.ensure_live()

    @property
    def joint_ids(self) -> list[str]:
        return list(self._hand.config.joint_ids)

    @property
    def handedness(self) -> str:
        """Physical hand side declared by the loaded OrcaHand config."""
        return str(self._hand.config.type)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        return self._cameras.shapes

    def get_observation(self) -> SinkObservation:
        return SinkObservation(
            joint_state=self._read_joint_state(),
            images=self._cameras.capture(),
        )

    def _read_joint_state(self) -> np.ndarray:
        if not self._connect_hardware:
            return np.zeros(len(self.joint_ids), dtype=np.float32)
        return self._hand.get_joint_position().as_array(self.joint_ids).astype(np.float32)

    def dispatch_action(self, action: OrcaJointPositions) -> None:
        if not self._connect_hardware:
            return
        self._hand.set_joint_positions(action, num_steps=MOTION_NUM_STEPS)

    def go_home(self) -> None:
        if not self._connect_hardware:
            return

        # Homing to neutral position with 0 wrist
        self._hand.set_joint_positions(self.home_position(), num_steps=5 * MOTION_NUM_STEPS)

    def home_position(self) -> OrcaJointPositions:
        positions = dict(self._hand.config.neutral_position)
        positions["wrist"] = 0.0
        return OrcaJointPositions(positions)

    def current_positions(self) -> OrcaJointPositions | None:
        """The hand's pose before teleop starts; the start ramp begins there.

        Never from a failed read: orca_core 0.13 then returns the motors' cached
        positions (zeros before the first good read), a far-off pose the first
        commands would jump from. So it tries 3 times, then refuses to start.
        """
        if not self._connect_hardware:
            return None
        for _ in range(3):
            positions, ok = self._read_hand()
            if ok:
                return positions
        raise RuntimeError(
            f"Could not read the hand's pose 3 times ({self._read_error}); "
            "not starting, so nothing jumps"
        )

    def _read_hand(self) -> tuple[OrcaJointPositions | None, bool]:
        """The motors' joint positions, and whether the read was fresh."""
        try:
            positions = self._hand.get_joint_position()
        except Exception as err:  # e.g. the port is gone; a missed reply sets last_read_ok
            self._read_error = f"{type(err).__name__}: {err}"
            return None, False
        if getattr(self._hand, "last_read_ok", True):
            return positions, True
        absent = list(getattr(self._hand, "absent_motor_ids", None) or [])
        self._read_error = f"no reply from motors {absent}" if absent else "no reply"
        return None, False

    def run_loop(
        self,
        actions_q: "queue.Queue[OrcaJointPositions | object]",
        stop_event: threading.Event,
    ) -> None:
        assert self._hand is not None, "connect() must be called before run_loop()"
        if self._joint_log is not None:
            self._log(
                start={
                    "joint_ids": self.joint_ids,
                    "readback_hz": self._readback_hz,
                    "log_current": self._log_current,
                    "orca_core": _version("orca_core"),
                    "model_path": self._model_path,
                }
            )
        wanted = self.viewer is not None or self._joint_log is not None
        reading = wanted and self._connect_hardware and self._readback_hz > 0
        period = 1.0 / self._readback_hz if reading else HEARTBEAT_INTERVAL
        next_read = time.monotonic()
        while not stop_event.is_set():
            wait = HEARTBEAT_INTERVAL
            if reading:
                wait = min(wait, max(0.0, next_read - time.monotonic()))
            try:
                action, _ = _get_latest(actions_q, wait)
            except queue.Empty:
                action = None
            if action is _SHUTDOWN:
                break
            if action is not None:
                assert isinstance(action, OrcaJointPositions)
                sent_t = time.time()
                self.dispatch_action(action)
                if self.viewer is not None:
                    self.viewer.commanded(action)
                self._log(t=sent_t, commanded=_plain(action.as_dict()))
            if reading and time.monotonic() >= next_read:
                ok = self._read_measured()
                self._failed_in_row = 0 if ok else self._failed_in_row + 1
                # Timed from the read's end, so a slow read (a silent motor costs
                # ~50 ms) can't take every turn from the commands.
                pause = self._failed_in_row >= READ_FAILS_IN_ROW
                next_read = time.monotonic() + (READ_RETRY_S if pause else period)

    def _read_measured(self) -> bool:
        """Read the motors for the viewer and the log; a failed read shows as none, not as
        stale values."""
        read_t, t0 = time.time(), time.perf_counter()
        positions, ok = self._read_hand()
        read_ms = (time.perf_counter() - t0) * 1000
        if not ok:
            self._read_failures += 1
            if self._read_failures in (1, 10, 100) or self._read_failures % 1000 == 0:
                logger.warning(
                    "Reading the motors failed (%d so far): %s",
                    self._read_failures,
                    self._read_error,
                )
        if self.viewer is not None:
            self.viewer.measured(positions, read_ms=read_ms)
        event: dict[str, Any] = {
            "t": read_t,
            "measured": None if positions is None else _plain(positions.as_dict()),
            "read_ms": read_ms,
        }
        if ok and self._log_current and self._joint_log is not None:
            event["motors"] = self._read_motors()
        self._log(**event)
        return ok

    def _read_motors(self) -> dict[str, dict[str, float]] | None:
        """Currents (mA) and velocities by motor id, or None if the read failed."""
        try:
            state = self._hand.get_motor_state()
        except Exception as err:
            logger.debug("Motor state read raised: %s", err)
            return None
        if not getattr(self._hand, "last_read_ok", True):
            return None
        ids = [str(m) for m in self._hand.config.motor_ids]
        return {
            "current_ma": dict(zip(ids, map(float, state.current), strict=True)),
            "velocity": dict(zip(ids, map(float, state.velocity), strict=True)),
        }

    def _log(self, **event: Any) -> None:
        """One JSON line; ``t`` defaults to now."""
        if self._joint_log is None:
            return
        try:
            self._joint_log.write(json.dumps({"t": time.time(), **event}) + "\n")
        except OSError as err:  # e.g. a full disk: stop logging, keep teleop running
            logger.warning("Joint log stopped: %s", err)
            self._joint_log = None

    def close(self) -> None:
        """Stop without commanding a move: torque goes off where the hand is."""
        if self._hand is None:
            return
        try:
            if self._connected:
                try:
                    unconfirmed = self._hand.disable_torque() or []
                finally:  # also after a failed torque-off: disconnect retries it, closes the port
                    self._connected = False
                    self._hand.disconnect()
                if unconfirmed:
                    logger.warning(
                        "Motors %s did not confirm torque off and may still hold; "
                        "hand disconnected",
                        list(unconfirmed),
                    )
                else:
                    logger.info("Torque off where the hand is; hand disconnected")
        except Exception:
            logger.exception("OrcaHandSink.close() encountered an error")
        finally:
            self._hand = None
            try:
                self._cameras.close()
            except Exception:
                logger.exception("Closing the cameras failed")
            if self._joint_log is not None:
                self._joint_log.close()
                self._joint_log = None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _plain(values: dict) -> dict[str, float | None]:
    """Joint values as plain floats (numpy floats aren't all JSON-serializable)."""
    return {k: None if v is None else float(v) for k, v in values.items()}


def retargeter_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    model_path: str | None = None,
    urdf_path: str | None = None,
    retargeter_backend: RetargeterBackend = "adaptive_analytical",
    retargeter_config_path: str | None = None,
    landmarks_viz: Any | None = None,
    landmark_source: LandmarkSource = "mediapipe",
    retargeter: Any | None = None,
    start_positions: OrcaJointPositions | None = None,
) -> None:
    """Consume ``HandLandmarks`` from the gRPC ingress, retarget, push to actions_q.

    Uses ``retargeter`` if given (``run`` builds it before touching the hand), else
    builds one from model_path and urdf_path. For each incoming ``HandLandmarks``
    (21, 3) it takes the newest queued frame, drops frames that can't be a hand,
    wraps the keypoints in a ``TargetPose`` and calls ``retargeter.retarget()`` —
    the retargeter handles normalization, auto-scale calibration and the URDF-frame
    transform internally, and returns ``None`` during calibration. Commands pass a
    ``MotionRamp``: the hand holds while tracking is lost and eases back when it
    returns, and the first command eases in from ``start_positions``.
    """
    if landmark_source not in ("mediapipe", "metaquest", "webxr"):
        raise ValueError(
            f"Unsupported landmark source {landmark_source!r}; expected "
            "'mediapipe', 'metaquest', or 'webxr'."
        )

    _LOG_EVERY = 30
    _t_retarget_ms: list[float] = []
    _latency_ms: list[float] = []
    _t_window_start: float = time.perf_counter()
    dropped: dict[str, int] = {}
    stale = 0

    try:
        if retargeter is None:
            try:
                retargeter = Retargeter.from_paths(
                    model_path,
                    urdf_path,
                    backend=retargeter_backend,
                    config_path=retargeter_config_path,
                )
            except Exception:
                logger.exception("Retargeter init failed; shutting down worker.")
                return
        ramp = MotionRamp(start=start_positions)

        while not stop_event.is_set():
            try:
                item, skipped = _get_latest(queues.landmarks_q, HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if item is _SHUTDOWN:
                break
            stale += skipped

            if not isinstance(item, HandLandmarks):
                raise ValueError(f"Expected instance of HandLandmarks, got {type(item)}")

            keypoints = item.keypoints
            if landmark_source == "metaquest":
                from orca_teleop.ingress.metaquest.landmarks import (
                    retargeter_landmarks_from_quest,
                )

                keypoints = retargeter_landmarks_from_quest(keypoints, item.handedness)

            problem = landmark_problem(keypoints, item.wrist_angle_degrees)
            if problem is not None:
                dropped[problem] = dropped.get(problem, 0) + 1
                continue

            if landmarks_viz is not None:
                landmarks_viz.put(keypoints, item.handedness)

            now = time.monotonic()
            if ramp.lost(now) and hasattr(retargeter, "reset_tracking"):
                retargeter.reset_tracking()  # a stale filter or warm start would drag the return

            t_retarget_start = time.perf_counter()
            try:
                # After the Quest convention adapter, all ingress types are in
                # the canonical MediaPipe-style frame consumed by both hand
                # retargeter backends.
                target_pose = TargetPose(
                    joint_positions=keypoints,
                    source="mediapipe",
                    wrist_angle_degrees=item.wrist_angle_degrees,
                )
                action = retargeter.retarget(target_pose)
            except (AssertionError, ValueError):
                dropped["degenerate"] = dropped.get("degenerate", 0) + 1
                continue
            t_retarget_end = time.perf_counter()

            _t_retarget_ms.append((t_retarget_end - t_retarget_start) * 1e3)
            if action is None:
                continue

            _put_latest(queues.actions_q, ramp.step(action, now))
            _latency_ms.append((time.time_ns() - item.timestamp_ns) / 1e6)

            if len(_t_retarget_ms) >= _LOG_EVERY:
                num_samples = len(_t_retarget_ms)
                elapsed_s = time.perf_counter() - _t_window_start
                avg_retarget_ms = sum(_t_retarget_ms) / num_samples
                fps = num_samples / elapsed_s

                logger.info(
                    "Retargeter | %.1f fps | retarget %.2f ms | capture->command %s | "
                    "skipped %d stale, dropped %s",
                    fps,
                    avg_retarget_ms,
                    f"{float(np.median(_latency_ms)):.0f} ms" if _latency_ms else "-",
                    stale,
                    dropped or "none",
                )
                _t_retarget_ms.clear()
                _latency_ms.clear()
                _t_window_start = time.perf_counter()
    finally:
        _shutdown_queue(queues.actions_q)


def robot_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    ready_event: threading.Event,
    model_path: str,
) -> None:
    """Consume OrcaJointPositions and stream them to the OrcaHand."""
    hand: OrcaHand | None = None
    try:
        hand = OrcaHand(model_path)
        success, message = hand.connect()
        if not success:
            logger.error("Robot worker: failed to connect: %s", message)
            return
        hand.init_joints()
        ready_event.set()

        while not stop_event.is_set():
            try:
                action, _ = _get_latest(queues.actions_q, HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if action is _SHUTDOWN:
                break
            assert isinstance(action, OrcaJointPositions)
            hand.set_joint_positions(action, num_steps=MOTION_NUM_STEPS)
    except Exception as e:
        logger.exception("Robot worker error: %s", e)
    finally:
        if hand is not None:
            try:
                hand.disable_torque()
                hand.disconnect()
            except Exception:
                pass


def run(
    model_path: str | None = None,
    urdf_path: str | None = None,
    port: int = DEFAULT_PORT,
    sink: RobotSink | None = None,
    visualize_landmarks: bool = False,
    retargeter_backend: RetargeterBackend = "adaptive_analytical",
    retargeter_config_path: str | None = None,
    landmark_source: LandmarkSource = "mediapipe",
    stop_event: threading.Event | None = None,
) -> None:
    """Start the full teleop pipeline:
    - gRPC-ingress -> retargeter -> robot consumer

    Start-up fails before the hand is touched: the retargeter is built and the
    ingress port bound first, then the sink connects.

    The robot-side machine runs this function. A publisher running on *any*
    machine (same host, a laptop across the room, etc.) connects via gRPC and
    streams hand-landmark frames. The main thread is handed to the sink's ``run_loop``.

    Args:
        model_path: Path to the OrcaHand model directory. ``None`` uses the
            default model bundled with ``orca_core``.
        urdf_path: Path to the hand URDF file. ``None`` resolves automatically
            from the ``orcahand_description`` package. Used for retargeting.
        port: TCP port for the gRPC ingress server, which streams HandLandmarks to the retargeter.
        sink: Consumer of retargeted joint positions. Defaults to
            ``OrcaHandSink(model_path)``, i.e. a physical OrcaHand.
        retargeter_backend: Retargeting implementation. ``"adaptive_analytical"``
            is the default Wuji-style Orca-native backend; ``"rmsprop"`` keeps
            the historical fingertip key-vector backend available.
        retargeter_config_path: Optional YAML config for the adaptive backend.
        landmark_source: Landmark convention received over gRPC. Live ``webxr``
            frames are already in the canonical 21-point layout; legacy
            ``metaquest`` HTS/Unity replays still need their historical mirror.
        stop_event: Set it to end the session (e.g. when a local publisher dies).
    """
    if sink is None:
        sink = OrcaHandSink(model_path)
    if model_path is None:
        sink_model_path = getattr(sink, "retarget_model_path", None)
        if sink_model_path:
            model_path = sink_model_path
            logger.info("Using sink-provided retargeter model: %s", model_path)

    retargeter = Retargeter.from_paths(
        model_path, urdf_path, backend=retargeter_backend, config_path=retargeter_config_path
    )

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = stop_event if stop_event is not None else threading.Event()

    ingress_server = IngressServer(queues.landmarks_q, stop_event, port=port)
    ingress_server.start()
    landmarks_viz = None
    retargeter_thread = None
    signals = _StopSignals(stop_event)
    # From connect() on, the torque may be on: whatever fails or is interrupted
    # after that (init, the start read, a signal), the finally turns it off.
    try:
        sink.connect()
        if visualize_landmarks:
            from orca_teleop.ingress.visualizer import HandLandmarkVisualizer

            landmarks_viz = HandLandmarkVisualizer()
            landmarks_viz.start()

        start_positions = sink.current_positions()
        retargeter_thread = threading.Thread(
            target=retargeter_worker,
            args=(queues, stop_event),
            kwargs=dict(
                model_path=model_path,
                urdf_path=urdf_path,
                retargeter_backend=retargeter_backend,
                retargeter_config_path=retargeter_config_path,
                landmarks_viz=landmarks_viz,
                landmark_source=landmark_source,
                retargeter=retargeter,
                start_positions=start_positions,
            ),
            name="retargeter",
        )
        retargeter_thread.start()

        signals.starting = False  # from here a signal only sets stop_event
        sink.run_loop(queues.actions_q, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        signals.starting = False  # a signal now must not interrupt the shutdown
        stop_event.set()
        sink.close()  # first: torque off at once, not after the server's grace period
        ingress_server.stop()
        if retargeter_thread is not None:
            retargeter_thread.join(timeout=JOIN_TIMEOUT)
        if landmarks_viz is not None:
            landmarks_viz.stop()
        signals.restore()


class _StopSignals:
    """Ctrl+C, SIGTERM and SIGHUP (a closed terminal) all end ``run()`` the same way, once.

    Without this, SIGTERM and SIGHUP kill the process with the torque on. The first
    signal sets ``stop_event``; while ``starting`` it also raises KeyboardInterrupt,
    so a blocking start-up step (a port picker, init, a calibration) stops. Every
    later signal is ignored: under ``uv run`` one Ctrl+C reaches the process twice,
    and a second KeyboardInterrupt in the middle of the shutdown could skip the
    torque-off. Handlers live in the main thread only; elsewhere this does nothing.
    """

    NAMES = ("SIGINT", "SIGTERM", "SIGHUP")

    def __init__(self, stop_event: threading.Event) -> None:
        self.starting = True
        self._stop_event = stop_event
        self._received = False
        self._previous: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for name in self.NAMES:
                if hasattr(signal, name):
                    sig = getattr(signal, name)
                    self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum: int, frame: Any) -> None:
        if self._received:
            return
        self._received = True
        name = "Ctrl+C" if signum == signal.SIGINT else signal.Signals(signum).name
        logger.info("%s: stopping", name)
        self._stop_event.set()
        if self.starting:
            raise KeyboardInterrupt

    def restore(self) -> None:
        for sig, previous in self._previous.items():
            signal.signal(sig, signal.SIG_DFL if previous is None else previous)
        self._previous = {}


def _stop_when_dead(process: Any, stop_event: threading.Event, what: str) -> threading.Thread:
    """Watch a publisher process; if it exits (e.g. its camera failed), end the session."""

    def watch() -> None:
        while not stop_event.wait(0.2):
            if not process.is_alive():
                logger.error("%s exited (code %s); stopping teleop.", what, process.exitcode)
                stop_event.set()
                return

    thread = threading.Thread(target=watch, name=f"watch-{what}", daemon=True)
    thread.start()
    return thread


def _mediapipe_publisher(
    port: int,
    handedness: str,
    confidence: float,
    show_video: bool,
    camera_index: int = 0,
    record_dir: str | None = None,
) -> None:
    """Entry point for the MediaPipe publisher."""
    from orca_teleop.ingress.mediapipe.publisher import MediaPipePublisher

    server_address = f"localhost:{port}"
    deadline = time.monotonic() + 10.0

    # Wait until the ingress server is actually accepting connections
    while True:
        try:
            with socket.create_connection(tuple(server_address.split(":")), timeout=0.5):
                break
        except OSError as err:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ingress server on {server_address} did not become ready"
                ) from err
            time.sleep(0.1)

    publisher = MediaPipePublisher(
        server_address=server_address,
        handedness=handedness,
        confidence=confidence,
        show_video=show_video,
        camera_index=camera_index,
        record_dir=record_dir,
    )
    publisher.run()


def _metaquest_publisher(
    port: int,
    handedness: str,
    quest_host: str,
    quest_port: int,
    fps: int,
    wrist_enabled: bool = True,
    wrist_scale: float = 1.0,
    reset_event: Any | None = None,
) -> None:
    """Entry point for the repository-owned Quest WebXR publisher subprocess."""
    from orca_teleop.ingress.metaquest.publisher import MetaQuestPublisher

    server_address = f"localhost:{port}"
    deadline = time.monotonic() + 10.0

    while True:
        try:
            with socket.create_connection(tuple(server_address.split(":")), timeout=0.5):
                break
        except OSError as err:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ingress server on {server_address} did not become ready"
                ) from err
            time.sleep(0.1)

    publisher = MetaQuestPublisher(
        server_address=server_address,
        handedness=handedness,
        quest_host=quest_host,
        quest_port=quest_port,
        fps=fps,
        wrist_enabled=wrist_enabled,
        wrist_scale=wrist_scale,
        reset_event=reset_event,
    )
    publisher.run()


def run_local(
    model_path: str | None = None,
    urdf_path: str | None = None,
    port: int = DEFAULT_PORT,
    handedness: str = DEFAULT_HAND,
    confidence: float = DEFAULT_CONFIDENCE,
    show_video: bool = False,
    sink: RobotSink | None = None,
    visualize_landmarks: bool = False,
    retargeter_backend: RetargeterBackend = "adaptive_analytical",
    retargeter_config_path: str | None = None,
    camera_index: int = 0,
    record_dir: str | None = None,
) -> None:
    """Run ``run()`` plus a local MediaPipe publisher for one-command teleop.
    Useful for prototyping.
    """
    import multiprocessing

    # Start the publisher in a child process so the webcam doesn't fight with main thread
    publisher_process = multiprocessing.Process(
        target=_mediapipe_publisher,
        args=(port, handedness, confidence, show_video, camera_index, record_dir),
        name="mediapipe-publisher",
        daemon=True,
    )

    publisher_process.start()
    logger.info(
        "Local MediaPipe publisher started (pid=%d, hand=%s)",
        publisher_process.pid,
        handedness,
    )

    stop_event = threading.Event()
    _stop_when_dead(publisher_process, stop_event, "MediaPipe publisher")
    try:
        run(
            model_path=model_path,
            urdf_path=urdf_path,
            port=port,
            sink=sink,
            visualize_landmarks=visualize_landmarks,
            retargeter_backend=retargeter_backend,
            retargeter_config_path=retargeter_config_path,
            stop_event=stop_event,
        )
    finally:
        stop_event.set()
        if publisher_process.is_alive():
            publisher_process.terminate()
        publisher_process.join(timeout=3.0)


def _manus_publisher(
    port: int,
    handedness: str,
    zmq_address: str,
) -> None:
    """Entry point for the Manus publisher subprocess."""
    from orca_teleop.ingress.manus.publisher import ManusPublisher

    server_address = f"localhost:{port}"
    deadline = time.monotonic() + 10.0

    while True:
        try:
            with socket.create_connection(tuple(server_address.split(":")), timeout=0.5):
                break
        except OSError as err:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ingress server on {server_address} did not become ready"
                ) from err
            time.sleep(0.1)

    publisher = ManusPublisher(
        server_address=server_address,
        handedness=handedness,
        zmq_address=zmq_address,
    )
    publisher.run()


def run_manus_local(
    model_path: str | None = None,
    urdf_path: str | None = None,
    port: int = DEFAULT_PORT,
    handedness: str = DEFAULT_HAND,
    zmq_address: str = "tcp://127.0.0.1:2044",
    sink: RobotSink | None = None,
    visualize_landmarks: bool = False,
) -> None:
    """Run ``run()`` plus a local Manus ZMQ→gRPC publisher for one-command teleop."""
    import multiprocessing

    print(
        "\033[93mPrerequisite: run 'manus-client run' in a separate terminal before "
        "starting the Manus teleop pipeline. The Manus SDK client must be streaming "
        "glove data over ZMQ for this pipeline to receive frames.\033[0m"
    )

    publisher_process = multiprocessing.Process(
        target=_manus_publisher,
        args=(port, handedness, zmq_address),
        name="manus-publisher",
        daemon=True,
    )

    publisher_process.start()
    logger.info(
        "Local Manus publisher started (pid=%d, hand=%s, zmq=%s)",
        publisher_process.pid,
        handedness,
        zmq_address,
    )

    try:
        run(
            model_path=model_path,
            urdf_path=urdf_path,
            port=port,
            sink=sink,
            visualize_landmarks=visualize_landmarks,
        )
    finally:
        if publisher_process.is_alive():
            publisher_process.terminate()
        publisher_process.join(timeout=3.0)


def run_metaquest_local(
    model_path: str | None = None,
    urdf_path: str | None = None,
    port: int = DEFAULT_PORT,
    handedness: str = DEFAULT_HAND,
    quest_host: str = "0.0.0.0",
    quest_port: int = 8765,
    quest_fps: int = 30,
    wrist_enabled: bool = True,
    wrist_scale: float = 1.0,
    sink: RobotSink | None = None,
    visualize_landmarks: bool = False,
    retargeter_backend: RetargeterBackend = "adaptive_analytical",
    retargeter_config_path: str | None = None,
) -> None:
    """Run ``run()`` plus a local Quest WebXR publisher for one-command teleop."""
    import multiprocessing

    print(
        "\033[93mOpen the printed Quest WebXR URL in Quest Browser, then enter VR and "
        "enable hand tracking. Live WebXR frames use the canonical landmark layout "
        "(retargeter landmark_source='webxr').\033[0m"
    )

    ctx = multiprocessing.get_context("spawn")
    publisher_process = ctx.Process(
        target=_metaquest_publisher,
        args=(
            port,
            handedness,
            quest_host,
            quest_port,
            quest_fps,
            wrist_enabled,
            wrist_scale,
            None,
        ),
        name="metaquest-webxr-publisher",
        daemon=True,
    )

    publisher_process.start()
    logger.info(
        "Local MetaQuest WebXR publisher started (pid=%d, hand=%s, http=%s:%d)",
        publisher_process.pid,
        handedness,
        quest_host,
        quest_port,
    )

    try:
        run(
            model_path=model_path,
            urdf_path=urdf_path,
            port=port,
            sink=sink,
            visualize_landmarks=visualize_landmarks,
            retargeter_backend=retargeter_backend,
            retargeter_config_path=retargeter_config_path,
            landmark_source="webxr",
        )
    finally:
        if publisher_process.is_alive():
            publisher_process.terminate()
        publisher_process.join(timeout=3.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Orca Hand teleoperation pipeline. "
        "Launches a publisher and the full retargeting pipeline.",
    )
    parser.add_argument("--model_path", default=None, help="OrcaHand model directory")
    parser.add_argument("--urdf_path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--hand", default="right", choices=["left", "right"], help="Hand to track (default: right)"
    )
    parser.add_argument(
        "--source",
        default="mediapipe",
        choices=["mediapipe", "manus"],
        help="Input source (default: mediapipe)",
    )
    parser.add_argument(
        "--zmq-address",
        default="tcp://127.0.0.1:2044",
        help="ZMQ address for Manus C++ client (default: tcp://127.0.0.1:2044)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument(
        "--visualize-landmarks",
        action="store_true",
        help="Open a live 3D matplotlib window showing hand keypoints",
    )
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retarget-config",
        default=None,
        help="YAML config for --retargeter adaptive_analytical",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.source == "manus":
        run_manus_local(
            args.model_path,
            args.urdf_path,
            port=args.port,
            handedness=args.hand,
            zmq_address=args.zmq_address,
            visualize_landmarks=args.visualize_landmarks,
        )
    else:
        run_local(
            args.model_path,
            args.urdf_path,
            port=args.port,
            handedness=args.hand,
            confidence=args.confidence,
            show_video=args.show_video,
            visualize_landmarks=args.visualize_landmarks,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
        )
