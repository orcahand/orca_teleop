"""Teleoperate the simulated ORCA hand via the Manus glove pipeline.

Start the glove bridge (``ingress/manus/bridge.py``, or ``fake_bridge`` without
gloves) first, then run this script. It launches a MANUS ZMQ→gRPC publisher and the
full retargeting pipeline against an ``orca_sim`` Gymnasium env.

Examples:
    # Default: right hand sim, expects the glove bridge on tcp://127.0.0.1:2044
    python scripts/manus_test.py

    # Left hand, custom ZMQ address
    python scripts/manus_test.py --hand left --zmq-address tcp://192.168.1.10:2044

    # Headless rendering
    python scripts/manus_test.py --render-mode rgb_array
"""

import argparse
import logging

# Force torch native extensions to load before MuJoCo creates its OpenGL
# context.  The retargeter runs in a thread and imports torch lazily;
# if that first import happens after GL init the triton/dynamo native
# extensions segfault.
import torch._dynamo  # noqa: F401

from orca_teleop.constants import DEFAULT_PORT
from orca_teleop.pipeline import run_manus_local
from orca_teleop.sim import OrcaHandSimSink


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manus glove teleoperation pipeline against orca_sim."
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
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Hand to track (default: right)",
    )
    parser.add_argument(
        "--zmq-address",
        default="tcp://127.0.0.1:2044",
        help="ZMQ address for Manus C++ client (default: tcp://127.0.0.1:2044)",
    )
    parser.add_argument(
        "--visualize-landmarks",
        action="store_true",
        help="Open a live 3D matplotlib window showing hand keypoints",
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

    run_manus_local(
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        port=args.port,
        handedness=args.hand,
        zmq_address=args.zmq_address,
        sink=sink,
        visualize_landmarks=args.visualize_landmarks,
    )


if __name__ == "__main__":
    main()
