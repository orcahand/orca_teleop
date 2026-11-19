"""Replay the Manus MANO dataset through the sim teleop stack.

This is a dummy Manus publisher path: it consumes the Hugging Face dataset,
streams frames over the same gRPC ingress used by live publishers, retargets
with the adaptive analytical backend, and renders the simulated ORCA hand.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import queue
import threading
import time

from orca_teleop.constants import JOIN_TIMEOUT, QUEUES_MAXSIZE
from orca_teleop.ingress.manus.dummy_publisher import (
    MANUS_MANO_REPO_ID,
    ManusDummyPublisher,
)
from orca_teleop.ingress.server import DEFAULT_PORT, IngressServer
from orca_teleop.pipeline import TeleopQueues, retargeter_worker
from orca_teleop.retargeting.config import resolve
from orca_teleop.sim import OrcaHandSimSink

logger = logging.getLogger(__name__)


def _publisher_process(
    server_address: str,
    repo_id: str,
    split: str,
    rate_hz: float | None,
    speed: float,
    loop: bool,
    start_index: int,
    max_frames: int | None,
    handedness: str | None,
    log_level: str,
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    publisher = ManusDummyPublisher(
        server_address=server_address,
        repo_id=repo_id,
        split=split,
        rate_hz=rate_hz,
        speed=speed,
        loop=loop,
        start_index=start_index,
        max_frames=max_frames,
        handedness=handedness,
    )
    publisher.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the Manus MANO Hugging Face dataset in orca_sim via the "
            "adaptive analytical retargeter."
        )
    )
    parser.add_argument("--model_path", default=None, help="OrcaHand model directory")
    parser.add_argument("--urdf_path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--env",
        default="right",
        choices=["left", "right"],
        help="orca_sim env variant (default: right)",
    )
    parser.add_argument(
        "--version", default=None, help="orca_sim embodiment version, e.g. 'v1' or 'v2'"
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array"],
        help="MuJoCo render mode (default: human)",
    )
    parser.add_argument("--repo-id", default=MANUS_MANO_REPO_ID, help="Hugging Face dataset repo")
    parser.add_argument("--split", default="train", help="Dataset split to replay")
    parser.add_argument("--rate-hz", type=float, default=None, help="Override replay rate")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop the dataset until interrupted")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset row to replay")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum rows to replay")
    parser.add_argument(
        "--hand",
        default=None,
        choices=["left", "right"],
        help="Override dataset handedness before publishing",
    )
    parser.add_argument(
        "--retargeter-config",
        default=None,
        help="Retargeter config (default: the glove's, manus.yaml)",
    )
    parser.add_argument(
        "--hold-after-finish",
        type=float,
        default=2.0,
        help="Seconds to keep rendering the last pose after a non-looping replay",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the sim open after a non-looping replay instead of exiting",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    try:
        args.retargeter, args.retargeter_config = resolve(None, args.retargeter_config, "manus")
    except (OSError, ValueError) as err:
        parser.error(str(err))

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sink = OrcaHandSimSink(
        env_name=args.env,
        version=args.version,
        render_mode=args.render_mode,
    )
    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()

    publisher: multiprocessing.Process | None = None
    ingress_server: IngressServer | None = None
    retargeter_thread: threading.Thread | None = None

    try:
        sink.connect()
        model_path = args.model_path
        if model_path is None and sink.retarget_model_path:
            model_path = sink.retarget_model_path
            logger.info("Using sink-provided retargeter model: %s", model_path)

        server_address = f"localhost:{args.port}"
        mp_context = multiprocessing.get_context("spawn")
        publisher = mp_context.Process(
            target=_publisher_process,
            args=(
                server_address,
                args.repo_id,
                args.split,
                args.rate_hz,
                args.speed,
                args.loop,
                args.start_index,
                args.max_frames,
                args.hand,
                args.log_level,
            ),
            name="manus-dataset-publisher",
            daemon=True,
        )
        publisher.start()
        logger.info(
            "Started Manus dataset publisher pid=%d repo=%s split=%s",
            publisher.pid,
            args.repo_id,
            args.split,
        )

        ingress_server = IngressServer(queues.landmarks_q, stop_event, port=args.port)
        ingress_server.start()

        retargeter_thread = threading.Thread(
            target=retargeter_worker,
            args=(
                queues,
                stop_event,
                model_path,
                args.urdf_path,
                args.retargeter,
                str(args.retargeter_config),
            ),
            name="retargeter",
        )
        retargeter_thread.start()

        if not args.loop and not args.keep_open:

            def stop_after_replay() -> None:
                assert publisher is not None
                publisher.join()
                logger.info(
                    "Dataset replay finished; holding last pose for %.2fs",
                    args.hold_after_finish,
                )
                time.sleep(max(0.0, args.hold_after_finish))
                stop_event.set()

            threading.Thread(
                target=stop_after_replay,
                name="publisher-watch",
                daemon=True,
            ).start()

        sink.run_loop(queues.actions_q, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if ingress_server is not None:
            ingress_server.stop()
        if retargeter_thread is not None:
            retargeter_thread.join(timeout=JOIN_TIMEOUT)
        if publisher is not None and publisher.is_alive():
            publisher.terminate()
            publisher.join(timeout=3.0)
        sink.close()


if __name__ == "__main__":
    main()
