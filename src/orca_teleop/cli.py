"""Console entry points for the teleoperation pipeline.

Every path these take resolves by default: the model from orca_core, the URDF
from the orcahand_description checkout, and the retarget config from this
package. A bare ``orca-teleop-hand`` is therefore enough to drive a connected
hand, and ``orca-teleop-urdf`` the same pipeline into a kinematic viewer.
"""

from __future__ import annotations

import argparse
import logging

from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import OrcaHandSink, run, run_local

logger = logging.getLogger("orca_teleop")


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_path", default=None, help="OrcaHand model config")
    parser.add_argument(
        "--urdf_path", default=None, help="Hand URDF the retargeter solves against"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--hand",
        default=None,
        choices=["left", "right"],
        help="Hand side to track (default: whichever side the model declares)",
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch a local MediaPipe publisher (default: on; --no-local to wait for one)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show the webcam feed")
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)"
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="DIR",
        help="Record the webcam session (raw frames + landmarks) into DIR for offline replay",
    )
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retarget-config", default=None, help="YAML config for adaptive_analytical"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _launch(args, sink, handedness: str) -> None:
    common = dict(
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        port=args.port,
        sink=sink,
        retargeter_backend=args.retargeter,
        retargeter_config_path=args.retarget_config,
    )
    if args.record and not args.local:
        raise SystemExit("--record records the local webcam publisher; drop --no-local")
    if args.local:
        run_local(handedness=handedness, confidence=args.confidence,
                  show_video=args.show_video, camera_index=args.camera_index,
                  record_dir=args.record, **common)
    else:
        run(**common)


def teleop_hand(argv: list[str] | None = None) -> None:
    """Teleoperate a physical ORCA hand."""
    parser = _common_parser("Teleoperate a physical ORCA hand from a webcam.")
    parser.add_argument(
        "--home",
        action="store_true",
        help=(
            "Let initialisation drive the hand to its neutral pose. Off by default: "
            "that move completes quickly and can stall tendons from a far pose."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full stack but never connect or dispatch to the motors",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    sink = OrcaHandSink(
        model_path=args.model_path,
        connect_hardware=not args.dry_run,
        move_to_neutral=args.home,
    )
    handedness = args.hand or sink.handedness
    if handedness != sink.handedness:
        parser.error(
            f"Model is a {sink.handedness} hand but --hand is {handedness}; "
            "the publisher would track the wrong hand."
        )
    if not args.dry_run and not args.home:
        logger.info("Hand is not pre-positioned; it holds its pose until the first frame")
    _launch(args, sink, handedness)


def teleop_urdf(argv: list[str] | None = None) -> None:
    """Teleoperate a kinematic URDF viewer instead of hardware."""
    from orca_teleop.urdf_viz import OrcaHandUrdfVizSink

    parser = _common_parser("Teleoperate a kinematic ORCA URDF viewer from a webcam.")
    parser.add_argument(
        "--version", default="v2", help="Embodiment version for default paths (default: v2)"
    )
    parser.add_argument(
        "--rate-hz", type=float, default=90.0, help="Viewer update rate (default: 90)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser tab automatically"
    )
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    handedness = args.hand or "right"
    sink = OrcaHandUrdfVizSink(
        hand_type=handedness,
        version=args.version,
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        rate_hz=args.rate_hz,
        open_browser=not args.no_browser,
    )
    _launch(args, sink, handedness)
