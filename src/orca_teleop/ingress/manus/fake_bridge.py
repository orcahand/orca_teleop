"""Stand in for the MANUS glove bridge: send ``GloveState`` frames on ZMQ, without gloves.

Two kinds of hand, sent as ``bridge.py`` sends hands (it has no other SDK streams), so
the publisher and everything after it can't tell the difference:

    python -m orca_teleop.ingress.manus.fake_bridge                       # made-up hand
    python -m orca_teleop.ingress.manus.fake_bridge --landmarks tests/data/pinch_clip.npz

- made up: a right (or ``--hand left``) hand that slowly closes and opens every finger;
- ``--landmarks``: 21-landmark clips (an ``.npz`` with ``world`` ``(N, 21, 3)``, as in
  ``tests/data``) turned into a glove skeleton turned into MANUS's frame (Z up), at
  ``--rate``.

Then ``orca-teleop-urdf --source manus`` (or ``orca-teleop-hand --source manus
--dry-run``) shows the robot following it.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Iterator

import numpy as np

from orca_teleop.ingress.manus import conversion, wire

logger = logging.getLogger(__name__)

# A proper rotation (no mirror) from the landmarks' frame into a Z-up frame, so the
# replayed hand is not in the frame it came in: the retargeter must not care.
_INTO_MANUS = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def landmarks_to_wire(landmarks: np.ndarray) -> np.ndarray:
    """``(21, 3)`` landmarks -> ``(25, 3)`` wire nodes; each long finger's CMC (which the
    landmarks lack and the publisher drops) a third of the way from wrist to MCP."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    nodes = np.zeros((conversion.NUM_NODES, 3))
    nodes[conversion.MANUS_TO_MANO] = landmarks
    for finger in ("index", "middle", "ring", "pinky"):
        mcp = landmarks[conversion.MANO_LANDMARK_NAMES.index(f"{finger}_mcp")]
        nodes[conversion.MANUS_NODE_NAMES.index(f"{finger}_cmc")] = (
            landmarks[0] + (mcp - landmarks[0]) / 3
        )
    return nodes


def made_up_hand(t_s: float, side: str = "right", period_s: float = 4.0) -> np.ndarray:
    """``(21, 3)`` landmarks of a hand closing and opening every finger, in metres.

    Built in the retargeter's palm frame (x towards the thumb side, y wrist to fingers,
    z out of the palm for a right hand); a left hand is the mirror image.
    """
    closed = 0.5 - 0.5 * math.cos(2 * math.pi * t_s / period_s)  # 0 open .. 1 closed
    kp = np.zeros((21, 3))
    fingers = {  # MCP (x, y), segment lengths, joint flexion when closed (rad)
        "index": ((0.025, 0.09), (0.045, 0.025, 0.02), (1.3, 1.6, 1.0)),
        "middle": ((0.0, 0.095), (0.05, 0.03, 0.022), (1.3, 1.6, 1.0)),
        "ring": ((-0.022, 0.088), (0.045, 0.028, 0.021), (1.3, 1.6, 1.0)),
        "pinky": ((-0.042, 0.078), (0.035, 0.022, 0.019), (1.3, 1.6, 1.0)),
    }
    for finger, ((x, y), lengths, flex) in fingers.items():
        i = conversion.MANO_LANDMARK_NAMES.index(f"{finger}_mcp")
        point, angle = np.array([x, y, 0.0]), 0.0
        kp[i] = point
        for k, (length, bend) in enumerate(zip(lengths, flex, strict=True)):
            angle += closed * bend
            point = point + length * np.array([0.0, math.cos(angle), math.sin(angle)])
            kp[i + 1 + k] = point
    # Thumb: from the CMC out along the palm's side, folding over the palm when closed.
    cmc = np.array([0.025, 0.025, 0.012])
    kp[1] = cmc
    point = cmc
    direction = np.array([0.6, 0.8, 0.0])
    for k, length in enumerate((0.045, 0.032, 0.028)):
        fold = closed * (0.5 + 0.2 * k)
        towards_palm = np.array([-math.sin(fold), 0.0, math.sin(fold)])
        d = direction * math.cos(fold) + towards_palm
        point = point + length * d / np.linalg.norm(d)
        kp[2 + k] = point
    if side == "left":
        kp[:, 0] *= -1
    return kp


def _from_landmarks(clip: np.ndarray, rate_hz: float) -> Iterator[tuple[float, np.ndarray]]:
    for i, landmarks in enumerate(clip):
        if np.all(np.isfinite(landmarks)):  # a clip's tracking gaps are NaN; gloves send nothing
            yield i / rate_hz, landmarks_to_wire(landmarks @ _INTO_MANUS.T)


def frames(args: argparse.Namespace) -> Iterator[tuple[float, dict[str, np.ndarray]]]:
    """``(seconds since start, {side: (25, 3) wire nodes})`` for one pass."""
    if args.landmarks:
        clip = np.load(args.landmarks)["world"]
        for t_s, nodes in _from_landmarks(clip, args.rate):
            yield t_s, {args.hand: nodes}
    else:
        for i in range(int(args.rate * 8)):  # two closes and opens
            t_s = i / args.rate
            yield t_s, {args.hand: landmarks_to_wire(made_up_hand(t_s, args.hand) @ _INTO_MANUS.T)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send made-up MANUS glove frames.")
    parser.add_argument("--landmarks", default=None, help=".npz with world (N, 21, 3) landmarks")
    parser.add_argument("--hand", default="right", choices=wire.SIDES, help="the hand to send")
    parser.add_argument("--rate", type=float, default=90.0, help="frames/s (default: 90)")
    parser.add_argument("--loop", action="store_true", help="start over at the end")
    parser.add_argument(
        "--bind", default=f"tcp://127.0.0.1:{wire.DEFAULT_PORT}", help="ZMQ address to publish on"
    )
    args = parser.parse_args(argv)
    args.bind = wire.zmq_address(args.bind)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")

    import zmq

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    logger.info("Sending fake glove frames on %s", args.bind)
    sent = 0
    try:
        while True:
            start = time.monotonic()
            for t_s, hands in frames(args):
                time.sleep(max(0.0, start + t_s - time.monotonic()))
                socket.send(wire.encode(hands, time.time_ns(), sent))
                sent += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Sent %d frames", sent)
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
