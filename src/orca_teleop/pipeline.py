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

import logging
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
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
    MOTION_NUM_STEPS,
    QUEUES_MAXSIZE,
)
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.retargeting.retargeter import Retargeter, RetargeterBackend, TargetPose

logger = logging.getLogger(__name__)

_SHUTDOWN = object()
LandmarkSource = Literal["mediapipe", "metaquest", "webxr"]


def _shutdown_queue(q: "queue.Queue[Any]") -> None:
    """Signal a downstream worker to stop, without blocking."""
    try:
        q.put_nowait(_SHUTDOWN)
    except queue.Full:
        pass


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
    """

    def __init__(
        self,
        model_path: str | None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
        connect_hardware: bool = True,
        move_to_neutral: bool = True,
    ) -> None:
        self._model_path = model_path
        self._hand = OrcaHand(model_path)
        self._connect_hardware = connect_hardware
        self._move_to_neutral = move_to_neutral
        self._cameras = CameraManager(camera_configs or [])

    def connect(self) -> None:
        if self._connect_hardware:
            success, message = self._hand.connect()
            if not success:
                raise RuntimeError(f"Robot failed to connect: {message}")
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

    def run_loop(
        self,
        actions_q: "queue.Queue[OrcaJointPositions | object]",
        stop_event: threading.Event,
    ) -> None:
        assert self._hand is not None, "connect() must be called before run_loop()"
        while not stop_event.is_set():
            try:
                action = actions_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if action is _SHUTDOWN:
                break
            assert isinstance(action, OrcaJointPositions)
            self.dispatch_action(action)

    def close(self) -> None:
        if self._hand is None:
            return
        try:
            self._cameras.close()
            if self._connect_hardware:
                self._hand.set_zero_position()
                self._hand.disable_torque()
                self._hand.disconnect()
        except Exception:
            logger.exception("OrcaHandSink.close() encountered an error")
        finally:
            self._hand = None


def retargeter_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    model_path: str | None = None,
    urdf_path: str | None = None,
    retargeter_backend: RetargeterBackend = "adaptive_analytical",
    retargeter_config_path: str | None = None,
    landmarks_viz: Any | None = None,
    landmark_source: LandmarkSource = "mediapipe",
) -> None:
    """Consume ``HandLandmarks`` from the gRPC ingress, retarget, push to actions_q.

    Builds a Retargeter from model_path and urdf_path, then for each incoming
    ``HandLandmarks`` (21, 3) wraps the raw keypoints in a ``TargetPose`` and
    calls ``retargeter.retarget()`` — the retargeter handles MANO normalization,
    auto-scale calibration, and the URDF-frame transform internally. During the
    calibration window it may return ``None``, in which case no robot command is
    enqueued yet.
    """
    if landmark_source not in ("mediapipe", "metaquest", "webxr"):
        raise ValueError(
            f"Unsupported landmark source {landmark_source!r}; expected "
            "'mediapipe', 'metaquest', or 'webxr'."
        )

    _LOG_EVERY = 30
    _t_retarget_ms: list[float] = []
    _t_window_start: float = time.perf_counter()

    try:
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

        while not stop_event.is_set():
            try:
                item = queues.landmarks_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if item is _SHUTDOWN:
                break

            if not isinstance(item, HandLandmarks):
                raise ValueError(f"Expected instance of HandLandmarks, got {type(item)}")

            keypoints = item.keypoints
            if landmark_source == "metaquest":
                from orca_teleop.ingress.metaquest.landmarks import (
                    retargeter_landmarks_from_quest,
                )

                keypoints = retargeter_landmarks_from_quest(keypoints, item.handedness)

            if landmarks_viz is not None:
                landmarks_viz.put(keypoints, item.handedness)

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
                logger.debug("Skipping degenerate landmark frame.")
                continue
            t_retarget_end = time.perf_counter()

            _t_retarget_ms.append((t_retarget_end - t_retarget_start) * 1e3)
            if action is None:
                continue

            try:
                queues.actions_q.put_nowait(action)
            except queue.Full:
                pass

            if len(_t_retarget_ms) >= _LOG_EVERY:
                num_samples = len(_t_retarget_ms)
                elapsed_s = time.perf_counter() - _t_window_start
                avg_retarget_ms = sum(_t_retarget_ms) / num_samples
                fps = num_samples / elapsed_s

                logger.info(
                    "Retargeter | %.1f fps | retarget %.2f ms",
                    fps,
                    avg_retarget_ms,
                )
                _t_retarget_ms.clear()
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
                action = queues.actions_q.get(timeout=HEARTBEAT_INTERVAL)
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
) -> None:
    """Start the full teleop pipeline:
    - gRPC-ingress -> retargeter -> robot consumer

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
    """
    if sink is None:
        sink = OrcaHandSink(model_path)

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()

    landmarks_viz = None
    if visualize_landmarks:
        from orca_teleop.ingress.visualizer import HandLandmarkVisualizer

        landmarks_viz = HandLandmarkVisualizer()
        landmarks_viz.start()

    sink.connect()
    if model_path is None:
        sink_model_path = getattr(sink, "retarget_model_path", None)
        if sink_model_path:
            model_path = sink_model_path
            logger.info("Using sink-provided retargeter model: %s", model_path)

    ingress_server = IngressServer(queues.landmarks_q, stop_event, port=port)
    ingress_server.start()

    retargeter_thread = threading.Thread(
        target=retargeter_worker,
        args=(
            queues,
            stop_event,
            model_path,
            urdf_path,
            retargeter_backend,
            retargeter_config_path,
            landmarks_viz,
            landmark_source,
        ),
        name="retargeter",
    )
    retargeter_thread.start()

    try:
        sink.run_loop(queues.actions_q, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ingress_server.stop()
        retargeter_thread.join(timeout=JOIN_TIMEOUT)
        sink.close()
        if landmarks_viz is not None:
            landmarks_viz.stop()


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

    try:
        run(
            model_path=model_path,
            urdf_path=urdf_path,
            port=port,
            sink=sink,
            visualize_landmarks=visualize_landmarks,
            retargeter_backend=retargeter_backend,
            retargeter_config_path=retargeter_config_path,
        )
    finally:
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
