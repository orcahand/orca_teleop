"""Replay a recording as if it were a live publisher.

Streams the recorded landmarks over gRPC at the pace they were captured, so the
whole robot side (ingress, retargeter, sink) runs exactly as with a camera. Takes a
``--record`` folder (``landmarks.jsonl``) or a fidelity corpus file (``.npz`` with
``world_tri``, ``world_front`` and ``capture_ns``, as ``scripts/fidelity_corpus.py``
writes it; the triangulated landmarks unless ``--front``). Gaps where nothing was
detected stay gaps, so tracking loss behaves as it did live.

    # terminal 1: the robot side, waiting for a publisher
    uv run --extra teleop orca-teleop-urdf --no-local
    # terminal 2
    uv run --no-sync python -m orca_teleop.ingress.replay_publisher recordings/thumb-01 --loop
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import grpc
import numpy as np

from orca_teleop.constants import DEFAULT_PORT
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc

logger = logging.getLogger(__name__)


def load_frames(path: str | pathlib.Path, front: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks (n, 21, 3) and their capture times (n,) in seconds from the first."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        world = data["world_front"] if front else data["world_tri"]
        times = data["capture_ns"]
    else:
        from orca_teleop.ingress.landmark_recording import load_recording

        rec = load_recording(path)
        world, times = rec.world, rec.capture_time_ns[rec.frame]
    times = (np.asarray(times, dtype=np.int64) - int(times[0])) / 1e9
    return np.asarray(world, dtype=np.float32), times


def frames(world: np.ndarray, times: np.ndarray, handedness: str, speed: float, loop: bool):
    """HandFrames paced like the recording; frames with no hand are skipped."""
    while True:
        start = time.monotonic()
        for kp, t in zip(world, times, strict=True):
            delay = start + t / speed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if not np.all(np.isfinite(kp)):
                continue
            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(), handedness=handedness, timestamp_ns=time.time_ns()
            )
        if not loop:
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", help="a --record folder or a corpus .npz")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--hand", default="right", choices=["left", "right"])
    ap.add_argument("--front", action="store_true", help="corpus: the front camera's landmarks")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1)")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")

    world, times = load_frames(args.recording, args.front)
    logger.info("Replaying %d frames (%.0f s) to localhost:%d", len(world), times[-1], args.port)
    channel = grpc.insecure_channel(f"localhost:{args.port}")
    grpc.channel_ready_future(channel).result(timeout=10.0)
    stub = hand_stream_pb2_grpc.HandStreamStub(channel)
    try:
        summary = stub.StreamHandFrames(frames(world, times, args.hand, args.speed, args.loop))
        logger.info("Done: %d frames received", summary.frames_received)
    except KeyboardInterrupt:
        pass
    finally:
        channel.close()


if __name__ == "__main__":
    main()
