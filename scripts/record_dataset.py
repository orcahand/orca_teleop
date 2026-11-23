"""Record teleoperated episodes into a LeRobotDataset.

Record — per episode:

  - teleop consumer thread: owns ``actions_q``, mirrors the latest command, and
    dispatches to the hand at teleop rate (independent of recording)
  - main thread: sample the sink on a fixed ``--fps`` clock (default 30 Hz).
    Each tick snapshots the mirrored teleop command and current observations
    into one dataset row. Recording timing does not drive robot dispatch.
  - recorder thread: blocks on ``rec_q``, calls ``dataset.add_frame(...)`` for
    every dict, and ``dataset.save_episode()`` when it sees the SAVE sentinel
  - episodes after the first begin with ``go_home()`` (reset to the recording home
    pose), rest (default 5 s) with no teleop dispatch, then sensor warmup and
    recording; episode 1 starts without homing; SPACE saves and advances; at
    the end: optional push to Hub

Add cameras:

  Cameras are opened with OpenCV. On macOS a plugged-in / paired iPhone shows up
  as a regular video device (through Continuity Camera) via the AVFoundation
  backend, so it can be recorded like any other camera. First discover its
  index::

      python scripts/record_dataset.py --list-cameras

  then pass ``--camera NAME:INDEX[:WIDTHxHEIGHT][@FPS]`` (repeatable). Omitted
  resolution/fps default to 640x480@30. To validate the camera pipeline
  *without* the physical hand, add ``--stub`` (state/action are dummy, but real
  video is recorded).

Example usage:

.. code-block:: bash

    HF_USERNAME=your_username python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-teleop \\
        --task "pick up the block" \\
        --num-episodes 5 --episode-seconds 20 \\
        --camera front:0 --camera wrist:1 \\
        --push-to-hub

    # Discover which index the iPhone (Continuity Camera) landed on
    python scripts/record_dataset.py --list-cameras

    # Camera-only smoke test with the iPhone, no hand required
    python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-iphone-test \\
        --task "wave at the camera" --stub \\
        --camera iphone:1:1280x720@30 --num-episodes 1 --episode-seconds 5

    # Record using a simulated hand
    python scripts/record_dataset.py --backend sim --repo-id $HF_USERNAME/orca-sim \\
        --task "pick up the block" --num-episodes 5

    # Teleop the REAL hand with Manus gloves, ending each episode by pressing SPACE
    # (prerequisite: the glove bridge streaming glove data; README, MANUS gloves)
    python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-manus \\
        --task "pick up the block" --backend hardware \\
        --local --source manus --hand right \\
        --episode-end space --num-episodes 20 --camera front:0

    # Teleop and record the REAL hand from the repository-owned Quest WebXR page
    python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-quest-hardware \\
        --task "pick up the block" --backend hardware \\
        --local --source metaquest --hand right \\
        --episode-end space --num-episodes 20 --camera iphone:1
"""

import argparse
import logging
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.cameras import (
    OpenCVCameraConfig,
    parse_camera_spec,
    print_available_cameras,
    with_recording_defaults,
)
from orca_teleop.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HAND,
    DEFAULT_MANUS_ADDRESS,
    DEFAULT_PORT,
    JOIN_TIMEOUT,
    QUEUES_MAXSIZE,
    SIM_RENDER_MODE,
)
from orca_teleop.ingress.server import IngressServer
from orca_teleop.pipeline import (
    _SHUTDOWN,
    OrcaHandSink,
    RecordableSink,
    TeleopQueues,
    _manus_publisher,
    _mediapipe_publisher,
    _metaquest_publisher,
    retargeter_worker,
)
from orca_teleop.recording import (
    TeleopActionMirror,
    drain_actions_queue,
    teleop_consumer_loop,
    wait_for_teleop_mirror_ready,
)
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)


MODEL_PATH = (
    "/Users/fracapuano/Documents/orca_core/orca_core/models/v2/orcahand_touch_right/config.yaml"
)
_DEFAULT_FPS = 30
_DEFAULT_NUM_EPISODES = 1
_DEFAULT_EPISODE_SECONDS = 30.0
_DEFAULT_REST_SECONDS = 5.0

# Sentinels pushed onto rec_q between episodes — the recorder either flushes
# (save) or drops (discard) the in-progress episode buffer, then keeps draining.
_SAVE_EPISODE = object()
_DISCARD_EPISODE = object()


class _KeyboardController:
    """Read single keypresses from an interactive terminal on a background thread.

    Puts the TTY in cbreak mode (so keys are delivered without waiting for Enter,
    while Ctrl+C still raises ``KeyboardInterrupt``) and maps:

      - ``space`` -> ``advance_event`` (stop recording, save episode, then start
        the next episode with go_home + rest)
      - ``q`` / ``Q`` / ``Esc`` -> ``stop_event`` (end the whole recording session)

    The terminal state is always restored on ``stop()`` or thread exit.
    """

    def __init__(self, stop_event: threading.Event, advance_event: threading.Event) -> None:
        self._stop_event = stop_event
        self._advance_event = advance_event
        self._thread: threading.Thread | None = None
        self._old_attrs = None
        self._fd: int | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--episode-end space/both requires an interactive terminal, but stdin "
                "is not a TTY. Run in a real terminal or use --episode-end timer."
            )
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # cbreak keeps ISIG on, so Ctrl+C still works
        self._thread = threading.Thread(target=self._loop, name="keyboard-controller", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import select

        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                ch = os.read(self._fd, 1)
                if not ch:
                    continue
                if ch == b" ":
                    self._advance_event.set()
                elif ch in (b"q", b"Q", b"\x1b"):
                    self._stop_event.set()
                    self._advance_event.set()  # unblock the episode loop immediately
        finally:
            self._restore()

    def _restore(self) -> None:
        if self._old_attrs is not None and self._fd is not None:
            import termios

            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except Exception:
                pass
            self._old_attrs = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore()


def _default_lerobot_root(repo_id: str) -> Path:
    """Mirror lerobot's default cache location."""
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base) / repo_id
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _parse_camera_configs(specs: list[str]) -> list[OpenCVCameraConfig]:
    return [with_recording_defaults(parse_camera_spec(spec)) for spec in specs]


def _stub_action_publisher(
    actions_q: "queue.Queue",
    stop_event: threading.Event,
    joint_ids: list[str],
    fps: float,
) -> None:
    """Push small random-walk ``OrcaJointPositions`` onto actions_q (dev only).

    These actions are *recorded* by the main loop but, in stub mode, are
    deliberately not dispatched to the device — see ``main()``.
    """
    rng = np.random.default_rng()
    n = len(joint_ids)
    pose = np.zeros(n, dtype=float)
    period = 1.0 / float(fps)
    while not stop_event.is_set():
        pose += rng.uniform(-0.2, 0.2, size=n)
        pose = np.clip(pose, -3.0, 3.0)
        action = OrcaJointPositions.from_dict(dict(zip(joint_ids, pose, strict=True)))
        try:
            actions_q.put_nowait(action)
        except queue.Full:
            try:
                actions_q.get_nowait()
            except queue.Empty:
                pass
            try:
                actions_q.put_nowait(action)
            except queue.Full:
                pass
        time.sleep(period)


def _recorder_loop(
    dataset,
    rec_q: "queue.Queue",
    episode_finalized_q: "queue.Queue[bool]",
) -> None:
    """Drain rec_q. Frames are dicts; save/discard sentinels finalize; ``None`` exits."""
    while True:
        item = rec_q.get()
        if item is None:
            break
        if item is _SAVE_EPISODE:
            succeeded = False
            try:
                dataset.save_episode()
                logger.info("Episode saved.")
                succeeded = True
            except Exception:
                logger.exception("Failed to save episode")
            finally:
                episode_finalized_q.put(succeeded)
            continue
        if item is _DISCARD_EPISODE:
            succeeded = False
            try:
                dataset.clear_episode_buffer()
                logger.warning("Episode discarded (incomplete / sensor failure).")
                succeeded = True
            except Exception:
                logger.exception("Failed to discard episode buffer")
            finally:
                episode_finalized_q.put(succeeded)
            continue
        try:
            dataset.add_frame(item)
        except Exception:
            logger.exception("Failed to add frame")


def _main_record(argv: list[str]) -> None:
    # Short-circuit camera discovery before argparse enforces required flags
    if "--list-cameras" in argv:
        print_available_cameras()
        return

    parser = argparse.ArgumentParser(
        description="Record teleoperated episodes into a LeRobotDataset.",
    )
    parser.add_argument("--repo-id", required=True, help="LeRobotDataset repo id")
    parser.add_argument("--task", required=True, help="Task description for every episode")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=_DEFAULT_NUM_EPISODES,
        help=f"Number of episodes to record (default: {_DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=_DEFAULT_EPISODE_SECONDS,
        help=(
            "Length of each episode in seconds. Only used for --episode-end timer/both "
            f"(default: {_DEFAULT_EPISODE_SECONDS})"
        ),
    )
    parser.add_argument(
        "--episode-end",
        choices=["timer", "space", "both"],
        default="timer",
        help=(
            "How each episode ends: 'timer' after --episode-seconds (default), "
            "'space' when you press the SPACE key (no time limit), or 'both' "
            "(whichever comes first). For space/both, press 'q'/Esc to end the "
            "whole session. Requires an interactive terminal."
        ),
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=_DEFAULT_REST_SECONDS,
        help="Pause after go_home() at the start of each episode after the first "
        "episode (default: {_DEFAULT_REST_SECONDS})",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Camera spec NAME[:INDEX][:WIDTHxHEIGHT][@FPS], repeatable "
            "(e.g. front:0, iphone:1 — defaults to 640x480@30 when omitted). "
            "Cameras become part of the sink's observation; with --backend sim "
            "they're recorded alongside the sim's MuJoCo render. "
            "Use --list-cameras to discover indices."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera indices (with resolution) and exit.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After recording, push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe any existing dataset at the same root before creating.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root (defaults to lerobot's HF cache).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=_DEFAULT_FPS,
        help=f"Fixed dataset sampling rate in Hz (default: {_DEFAULT_FPS})",
    )
    parser.add_argument("--model-path", default=MODEL_PATH, help="OrcaHand model directory")
    parser.add_argument("--urdf-path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--backend",
        choices=["hardware", "sim"],
        default="hardware",
        help="Backend to record against (default: hardware).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"gRPC ingress port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also launch a local publisher (see --source) for one-command recording.",
    )
    parser.add_argument(
        "--source",
        choices=["mediapipe", "manus", "metaquest"],
        default="mediapipe",
        help=(
            "Input publisher and landmark convention: 'mediapipe' webcam (default), "
            "'manus' gloves (requires the glove bridge), or 'metaquest' WebXR in "
            "Quest Browser. With --local the matching publisher is launched here; "
            "without --local, launch it separately and point it at this ingress."
        ),
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default=DEFAULT_HAND,
        help=f"Which hand the local publisher streams (default: {DEFAULT_HAND}).",
    )
    parser.add_argument(
        "--glove-address",
        default=DEFAULT_MANUS_ADDRESS,
        help=f"The MANUS glove bridge for --source manus, [tcp://]HOST:PORT "
        f"(default: {DEFAULT_MANUS_ADDRESS}).",
    )
    parser.add_argument(
        "--quest-host",
        default="0.0.0.0",
        help="Quest WebXR HTTP/WebSocket bind host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--quest-port",
        type=int,
        default=8765,
        help="Quest WebXR HTTP/WebSocket port (default: 8765).",
    )
    parser.add_argument(
        "--quest-fps",
        type=int,
        default=_DEFAULT_FPS,
        help=f"Meta Quest gRPC publish-rate cap (default: {_DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--quest-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive relative wrist flexion from the WebXR wrist pose (default: enabled).",
    )
    parser.add_argument(
        "--quest-wrist-scale",
        type=float,
        default=1.0,
        help="Scale applied to relative WebXR wrist flexion (default: 1.0).",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show the local MediaPipe webcam feed with landmarks.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Dev only: no hardware required. Skip ingress + retargeter, record random "
            "(undispatched) actions. Combine with --camera to test camera recording alone."
        ),
    )
    args = parser.parse_args(argv)
    if args.local and args.stub:
        parser.error("--local cannot be combined with --stub.")
    if args.source == "manus" and not args.local:
        logger.warning(
            "--source manus has no effect without --local; run the Manus publisher "
            "yourself (python -m orca_teleop.ingress.manus.publisher) or add --local."
        )
    if args.episode_end in ("space", "both") and not sys.stdin.isatty():
        parser.error("--episode-end space/both needs an interactive terminal.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # TODO: Use constants key names from lerobot for frame key names

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()

    # Sinks own and composes the full observation
    camera_configs = _parse_camera_configs(args.camera)
    if args.backend == "sim":
        from orca_teleop.sim import OrcaHandSimSink

        sink = OrcaHandSimSink(
            render_mode=SIM_RENDER_MODE,
            camera_configs=camera_configs,
        )
    else:
        sink = OrcaHandSink(
            model_path=args.model_path,
            camera_configs=camera_configs,
            connect_hardware=not args.stub,
        )
        if sink.handedness != args.hand:
            parser.error(
                f"--hand {args.hand!r} does not match the loaded physical OrcaHand "
                f"model ({sink.handedness!r}). Pass the matching --model-path or --hand."
            )
    sink.connect()
    assert isinstance(sink, RecordableSink), "Sinks must be recordable to collect data"

    joint_ids = list(sink.joint_ids)
    n_joints = len(joint_ids)

    # LeRobotDataset features schema
    features = {
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": joint_ids},
        "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_ids},
    }
    for cam_name, shape in sink.camera_shapes.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    target_root = args.root if args.root is not None else _default_lerobot_root(args.repo_id)
    if args.overwrite and target_root.exists():
        logger.info("--overwrite: removing existing dataset at %s", target_root)
        shutil.rmtree(target_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
    )
    logger.info("Dataset root: %s", dataset.root)

    rec_q: queue.Queue = queue.Queue(maxsize=64)
    episode_finalized_q: queue.Queue[bool] = queue.Queue(maxsize=1)

    ingress_server: IngressServer | None = None
    retargeter_thread: threading.Thread | None = None
    stub_thread: threading.Thread | None = None
    publisher_process = None
    quest_reset_event = None

    if args.stub:
        logger.info(
            "STUB mode — no hardware required; ingress + retargeter bypassed, random "
            "actions recorded but NOT dispatched. Cameras still record real video."
        )
        stub_thread = threading.Thread(
            target=_stub_action_publisher,
            args=(queues.actions_q, stop_event, joint_ids, args.fps),
            name="stub-action-publisher",
        )
        stub_thread.start()
    else:
        if args.local:
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            if args.source == "manus":
                publisher_process = ctx.Process(
                    target=_manus_publisher,
                    args=(args.port, args.hand, args.glove_address),
                    name="manus-publisher",
                    daemon=True,
                )
            elif args.source == "metaquest":
                quest_reset_event = ctx.Event()
                publisher_process = ctx.Process(
                    target=_metaquest_publisher,
                    args=(
                        args.port,
                        args.hand,
                        args.quest_host,
                        args.quest_port,
                        args.quest_fps,
                        args.quest_wrist,
                        args.quest_wrist_scale,
                        quest_reset_event,
                    ),
                    name="metaquest-webxr-publisher",
                    daemon=True,
                )
            else:
                publisher_process = ctx.Process(
                    target=_mediapipe_publisher,
                    args=(args.port, args.hand, DEFAULT_CONFIDENCE, args.show_video),
                    name="mediapipe-publisher",
                    daemon=True,
                )
            publisher_process.start()
            logger.info(
                "Local %s publisher started (pid=%d, hand=%s)",
                args.source,
                publisher_process.pid,
                args.hand,
            )
        ingress_server = IngressServer(queues.landmarks_q, stop_event, port=args.port)
        ingress_server.start()
        retargeter_thread = threading.Thread(
            target=retargeter_worker,
            args=(queues, stop_event, args.model_path, args.urdf_path),
            kwargs={"landmark_source": ("webxr" if args.source == "metaquest" else "mediapipe")},
            name="retargeter",
        )
        retargeter_thread.start()

    recorder_thread = threading.Thread(
        target=_recorder_loop,
        args=(dataset, rec_q, episode_finalized_q),
        name="dataset-recorder",
    )
    recorder_thread.start()

    action_mirror = TeleopActionMirror()
    dispatch_enabled = threading.Event()
    teleop_thread = threading.Thread(
        target=teleop_consumer_loop,
        args=(queues.actions_q,),
        kwargs={
            "mirror": action_mirror,
            "stop_event": stop_event,
            "shutdown_sentinel": _SHUTDOWN,
            "dispatch_action": None if args.stub else sink.dispatch_action,
            "dispatch_enabled": dispatch_enabled,
        },
        name="teleop-consumer",
        daemon=True,
    )
    teleop_thread.start()

    use_keyboard = args.episode_end in ("space", "both")
    use_timer = args.episode_end in ("timer", "both")
    advance_event = threading.Event()
    keyboard: _KeyboardController | None = None
    if use_keyboard:
        keyboard = _KeyboardController(stop_event, advance_event)
        keyboard.start()

    if use_keyboard:
        logger.info(
            "Recording up to %d episode(s). From episode 2 onward: go_home()"
            "%s before recording. Press SPACE to save and advance, 'q'/Esc to finish.",
            args.num_episodes,
            f", then rest {args.rest_seconds:.1f}s" if args.rest_seconds > 0 else "",
        )
    else:
        logger.info(
            "Recording %d episode(s) of ~%.1fs each — each episode waits until teleop + "
            "cameras are healthy before capturing. Ctrl+C to abort.",
            args.num_episodes,
            args.episode_seconds,
        )

    try:
        for ep_idx in range(args.num_episodes):
            if stop_event.is_set():
                break
            logger.info("=== Episode %d / %d ===", ep_idx + 1, args.num_episodes)
            advance_event.clear()
            if quest_reset_event is not None:
                quest_reset_event.set()

            if ep_idx > 0 and not args.stub:
                logger.info("Resetting hand to home position...")
                try:
                    sink.go_home()
                except Exception:
                    logger.exception("Failed to move hand to home position; continuing anyway.")

            if ep_idx > 0 and args.rest_seconds > 0 and not stop_event.is_set():
                logger.info("Resting %.1fs before recording...", args.rest_seconds)
                stop_event.wait(args.rest_seconds)

            dispatch_enabled.clear()
            action_mirror.reset()
            drained = drain_actions_queue(
                queues.actions_q,
                stop_event=stop_event,
                shutdown_sentinel=_SHUTDOWN,
            )
            if drained:
                logger.debug(
                    "Discarded %d stale teleop command(s) before episode %d.",
                    drained,
                    ep_idx + 1,
                )
            if stop_event.is_set():
                break

            ready = wait_for_teleop_mirror_ready(
                get_observation=sink.get_observation,
                mirror=action_mirror,
                stop_event=stop_event,
            )
            if not ready:
                logger.info(
                    "Stopped before episode %d — waiting for sensors ended without readiness.",
                    ep_idx + 1,
                )
                break

            logger.info("Sensors ready — recording episode %d at %d Hz.", ep_idx + 1, args.fps)
            dispatch_enabled.set()
            ep_end = time.perf_counter() + args.episode_seconds
            n_frames = 0
            episode_ok = True
            ticker = RateTicker(dt=1.0 / float(args.fps))

            while not stop_event.is_set():
                if use_keyboard and advance_event.is_set():
                    break
                if use_timer and time.perf_counter() >= ep_end:
                    break

                action = action_mirror.snapshot()
                if action is None:
                    ticker.tick()
                    continue

                action_arr = action.as_array(joint_ids).astype(np.float32)

                try:
                    observation = sink.get_observation()
                except Exception:
                    logger.exception(
                        "Observation capture failed; discarding episode and aborting session."
                    )
                    episode_ok = False
                    stop_event.set()
                    break

                frame = {
                    "observation.state": observation.joint_state,
                    "action": action_arr,
                    "task": args.task,
                }
                for cam_name, img in observation.images.items():
                    frame[f"observation.images.{cam_name}"] = img

                try:
                    rec_q.put_nowait(frame)
                    n_frames += 1
                except queue.Full:
                    logger.debug("rec_q full; dropping frame")

                ticker.tick()

            dispatch_enabled.clear()
            logger.info("Episode %d captured %d frames.", ep_idx + 1, n_frames)
            if episode_ok and n_frames > 0:
                logger.info("Finalizing episode %d before continuing...", ep_idx + 1)
                rec_q.put(_SAVE_EPISODE)
                if not episode_finalized_q.get():
                    logger.error("Stopping because episode %d could not be saved.", ep_idx + 1)
                    break
            elif n_frames > 0:
                logger.info("Discarding episode %d before continuing...", ep_idx + 1)
                rec_q.put(_DISCARD_EPISODE)
                if not episode_finalized_q.get():
                    logger.error("Stopping because episode %d could not be discarded.", ep_idx + 1)
                    break
            elif not episode_ok:
                logger.error("Episode %d produced no usable frames; not saving.", ep_idx + 1)

    except KeyboardInterrupt:
        logger.info("Interrupted — finalizing dataset.")

    finally:
        stop_event.set()
        if keyboard is not None:
            keyboard.stop()  # restore terminal mode early
        rec_q.put(None)  # drain sentinel — recorder finishes pending items first

        if ingress_server is not None:
            ingress_server.stop()
        if publisher_process is not None and publisher_process.is_alive():
            publisher_process.terminate()
        if publisher_process is not None:
            publisher_process.join(timeout=3.0)
        if retargeter_thread is not None:
            retargeter_thread.join(timeout=JOIN_TIMEOUT)
        if stub_thread is not None:
            stub_thread.join(timeout=JOIN_TIMEOUT)
        teleop_thread.join(timeout=JOIN_TIMEOUT)

        recorder_thread.join()

        try:
            num_eps = dataset.num_episodes
        except Exception:
            num_eps = "?"
        logger.info("Dataset now contains %s episode(s) at %s", num_eps, dataset.root)

        if args.push_to_hub:
            try:
                logger.info("Pushing %s to the Hugging Face Hub...", args.repo_id)
                dataset.push_to_hub()
                logger.info("Push complete.")
            except Exception:
                logger.exception("Failed to push to hub")

        sink.close()


def main() -> None:
    _main_record(sys.argv[1:])


if __name__ == "__main__":
    main()
