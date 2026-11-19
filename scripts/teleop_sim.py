"""Teleoperate the simulated ORCA hand via the same ingress + retargeter
stack used for the physical robot. The only difference from
``pipeline.run()`` is the sink: instead of streaming joint targets to an
``OrcaHand``, ``SimSink`` steps an ``orca_sim`` Gymnasium env at its native
render rate.

Examples:
    # Robot machine: start the pipeline against the sim and wait for a
    # remote publisher to connect over gRPC.
    python scripts/teleop_sim.py --env right

    # Laptop: one-command teleop with a local MediaPipe webcam publisher.
    python scripts/teleop_sim.py --env right --local --show-video
"""

import argparse
import logging

from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import run, run_local
from orca_teleop.sim import OrcaHandSimSink


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orca Hand teleoperation pipeline against orca_sim."
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also launch a local MediaPipe publisher for one-command teleop",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Hand to track in the local publisher (default: right)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retargeter-config",
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

    sink = OrcaHandSimSink(
        env_name=args.env,
        version=args.version,
        render_mode=args.render_mode,
    )

    if args.local:
        run_local(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            handedness=args.hand,
            confidence=args.confidence,
            show_video=args.show_video,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retargeter_config,
        )
    else:
        run(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retargeter_config,
        )


if __name__ == "__main__":
    main()
