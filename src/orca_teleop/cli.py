"""Console entry points for the teleoperation pipeline.

Every path these take resolves by default: the model from orca_core, the URDF
from the orcahand_description checkout, and the retarget config from this
package. A bare ``orca-teleop-hand`` is therefore enough to drive a connected
hand, and ``orca-teleop-urdf`` the same pipeline into a kinematic viewer.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging

from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import OrcaHandSink, run, run_local

logger = logging.getLogger("orca_teleop")


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_path", default=None, help="OrcaHand model config")
    parser.add_argument("--urdf_path", default=None, help="Hand URDF the retargeter solves against")
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
        run_local(
            handedness=handedness,
            confidence=args.confidence,
            show_video=args.show_video,
            camera_index=args.camera_index,
            record_dir=args.record,
            **common,
        )
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
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Also show the hand model in the browser: commanded and measured pose (needs viser)",
    )
    parser.add_argument(
        "--viewer-port", type=int, default=8098, help="Viewer web page port (default: 8098)"
    )
    parser.add_argument(
        "--readback-hz",
        type=float,
        default=10.0,
        help="Motor position reads per second for the viewer and log (default: 10; 0 = none)",
    )
    parser.add_argument(
        "--log-joints",
        default=None,
        metavar="FILE",
        help="Log commanded and measured joints with timestamps (JSON lines) for lag analysis",
    )
    parser.add_argument(
        "--log-current",
        action="store_true",
        help="With --log-joints, also log motor currents and velocities (a second read, ~4 ms)",
    )
    args = parser.parse_args(argv)
    if args.log_current and not args.log_joints:
        parser.error("--log-current needs --log-joints FILE to log into")
    _configure_logging(args.log_level)
    logger.info("orca_core %s", importlib.metadata.version("orca_core"))

    sink = OrcaHandSink(
        model_path=args.model_path,
        connect_hardware=not args.dry_run,
        move_to_neutral=args.home,
        readback_hz=args.readback_hz,
        joint_log=args.log_joints,
        log_current=args.log_current,
    )
    handedness = args.hand or sink.handedness
    if handedness != sink.handedness:
        parser.error(
            f"Model is a {sink.handedness} hand but --hand is {handedness}; "
            "the publisher would track the wrong hand."
        )
    viewer = None
    if args.viewer:
        from orca_teleop.hand_viewer import HandViewer
        from orca_teleop.utils import default_urdf_path, infer_model_version

        urdf = args.urdf_path or default_urdf_path(handedness, infer_model_version(args.model_path))
        viewer = HandViewer(urdf, handedness, sink.joint_ids, port=args.viewer_port)
        sink.viewer = viewer
        logger.info("Viewer: open %s", viewer.url)
    if not args.dry_run and not args.home:
        logger.info("Hand is not pre-positioned; it holds its pose until the first frame")
    try:
        _launch(args, sink, handedness)
    finally:
        if viewer is not None:
            viewer.close()


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
