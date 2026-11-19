"""Replay Manus-derived MANO landmarks as teleop ``HandFrame`` messages.

This module intentionally keeps the Manus CSV conversion next to the replay
publisher. The Hugging Face dataset already stores converted ``(21, 3)``
keypoints. The live glove path (bridge.py, publisher.py) maps the SDK's raw
skeleton itself, in conversion.py.
"""

from __future__ import annotations

import argparse
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import numpy as np

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc

logger = logging.getLogger(__name__)

MANUS_MANO_REPO_ID = "fracapuano/manus-mano-poses"

MANO_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Manus has an extra non-thumb CMC joint. The teleop stack consumes the
# MediaPipe/MANO-style 21-point surface, so those CMCs are intentionally skipped.
MANUS_POSITION_JOINTS_FOR_MANO: tuple[str, ...] = (
    "Hand",
    "Thumb_CMC",
    "Thumb_MCP",
    "Thumb_DIP",
    "Thumb_TIP",
    "Index_MCP",
    "Index_PIP",
    "Index_DIP",
    "Index_TIP",
    "Middle_MCP",
    "Middle_PIP",
    "Middle_DIP",
    "Middle_TIP",
    "Ring_MCP",
    "Ring_PIP",
    "Ring_DIP",
    "Ring_TIP",
    "Pinky_MCP",
    "Pinky_PIP",
    "Pinky_DIP",
    "Pinky_TIP",
)


@dataclass(frozen=True)
class ReplayFrame:
    keypoints: np.ndarray
    handedness: str
    timestamp_ns: int
    timestamp_s: float
    frame_index: int


def manus_unity_positions_to_mano_keypoints(positions_cm: np.ndarray) -> np.ndarray:
    """Convert selected Manus Unity-style position joints to teleop landmarks.

    Args:
        positions_cm: Either ``(21, 3)`` or ``(N, 21, 3)`` Manus position values
            in centimeters, ordered as ``MANUS_POSITION_JOINTS_FOR_MANO``.

    Returns:
        Wrist-relative meters in the MediaPipe/MANO 21-landmark order expected
        by the current retargeter path.
    """
    positions = np.asarray(positions_cm, dtype=np.float32)
    single_frame = positions.ndim == 2
    if single_frame:
        positions = positions[None, ...]

    if positions.ndim != 3 or positions.shape[1:] != (21, 3):
        raise ValueError(
            "positions_cm must have shape (21, 3) or (N, 21, 3); " f"got {positions.shape}"
        )

    relative_cm = positions - positions[:, [0], :]
    keypoints = np.empty_like(relative_cm, dtype=np.float32)
    keypoints[..., 0] = -relative_cm[..., 0] / 100.0
    keypoints[..., 1] = relative_cm[..., 2] / 100.0
    keypoints[..., 2] = -relative_cm[..., 1] / 100.0
    return keypoints[0] if single_frame else keypoints


def extract_mano_keypoints_from_manus_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract converted keypoints and timestamps from a Manus CSV export."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading Manus CSV exports requires `pandas`.") from exc

    df = pd.read_csv(csv_path)
    positions = []
    missing_columns = []
    for joint_name in MANUS_POSITION_JOINTS_FOR_MANO:
        cols = [f"{joint_name}_Position_{axis}" for axis in "XYZ"]
        missing_columns.extend([col for col in cols if col not in df.columns])
        if not any(col not in df.columns for col in cols):
            positions.append(df[cols].to_numpy(dtype=np.float32))

    if missing_columns:
        raise ValueError(f"Manus CSV is missing expected position columns: {missing_columns}")

    positions_cm = np.stack(positions, axis=1)
    keypoints = manus_unity_positions_to_mano_keypoints(positions_cm)
    timestamps_ns = np.rint(
        df["Elapsed_Time_In_Milliseconds"].to_numpy(dtype=np.float64) * 1_000_000
    ).astype(np.int64)
    return keypoints, timestamps_ns


class ManusDummyPublisher:
    """Replay Manus-derived MANO landmarks as teleop ``HandFrame`` messages."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        repo_id: str = MANUS_MANO_REPO_ID,
        split: str = "train",
        *,
        rate_hz: float | None = None,
        speed: float = 1.0,
        loop: bool = False,
        start_index: int = 0,
        max_frames: int | None = None,
        handedness: str | None = None,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self._server_address = server_address
        self._repo_id = repo_id
        self._split = split
        self._rate_hz = rate_hz
        self._speed = float(speed)
        self._loop = loop
        self._start_index = int(start_index)
        self._max_frames = max_frames
        self._handedness = handedness
        self._connect_timeout_s = float(connect_timeout_s)

        if self._rate_hz is not None and self._rate_hz <= 0:
            raise ValueError("rate_hz must be positive when provided")
        if self._speed <= 0:
            raise ValueError("speed must be positive")
        if self._start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self._max_frames is not None and self._max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")

    def run(self) -> int:
        """Connect to the ingress server and stream frames. Returns frames sent."""
        frames = self._load_frames()
        self._wait_for_server()

        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        try:
            summary = stub.StreamHandFrames(self._frame_generator(frames))
            logger.info("Dataset publisher completed: sent %d frames", summary.frames_received)
            return int(summary.frames_received)
        finally:
            channel.close()

    def _load_frames(self) -> list[ReplayFrame]:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ManusDatasetPublisher requires the Hugging Face `datasets` package."
            ) from exc

        dataset = load_dataset(self._repo_id, split=self._split)
        stop_index = (
            len(dataset) if self._max_frames is None else self._start_index + self._max_frames
        )
        rows = dataset.select(range(self._start_index, min(stop_index, len(dataset))))
        frames = [self._row_to_frame(row, ordinal=i) for i, row in enumerate(rows)]
        if not frames:
            raise ValueError(
                f"No frames selected from {self._repo_id!r} split={self._split!r} "
                f"start_index={self._start_index} max_frames={self._max_frames}"
            )

        logger.info(
            "Loaded %d frames from %s[%s] (first=%.3fs last=%.3fs)",
            len(frames),
            self._repo_id,
            self._split,
            frames[0].timestamp_s,
            frames[-1].timestamp_s,
        )
        return frames

    def _row_to_frame(self, row: dict[str, Any], ordinal: int) -> ReplayFrame:
        keypoints = np.asarray(row["keypoints"], dtype=np.float32)
        if keypoints.shape != (21, 3):
            raise ValueError(f"Expected row keypoints shape (21, 3), got {keypoints.shape}")
        if not np.isfinite(keypoints).all():
            raise ValueError("Dataset row contains non-finite keypoint values")

        handedness = self._handedness or str(row.get("handedness", "right")).lower()
        if handedness not in ("left", "right"):
            raise ValueError(f"Invalid handedness {handedness!r}")

        timestamp_ns = int(row.get("timestamp_ns", time.time_ns()))
        timestamp_s = float(row.get("timestamp_s", timestamp_ns / 1e9))
        frame_index = int(row.get("frame", ordinal))
        return ReplayFrame(
            keypoints=keypoints,
            handedness=handedness,
            timestamp_ns=timestamp_ns,
            timestamp_s=timestamp_s,
            frame_index=frame_index,
        )

    def _frame_generator(self, frames: list[ReplayFrame]):
        sent = 0
        while True:
            previous_timestamp_s: float | None = None
            for frame in frames:
                if previous_timestamp_s is not None:
                    if self._rate_hz is None:
                        delay_s = max(0.0, frame.timestamp_s - previous_timestamp_s)
                    else:
                        delay_s = 1.0 / self._rate_hz
                    time.sleep(delay_s / self._speed)

                previous_timestamp_s = frame.timestamp_s
                sent += 1
                yield hand_stream_pb2.HandFrame(
                    keypoints=frame.keypoints.ravel().tolist(),
                    handedness=frame.handedness,
                    timestamp_ns=frame.timestamp_ns,
                )

            if not self._loop:
                logger.info("Finished one dataset replay pass (%d frames yielded)", sent)
                return

    def _wait_for_server(self) -> None:
        host, port_str = self._server_address.rsplit(":", 1)
        deadline = time.monotonic() + self._connect_timeout_s
        while True:
            try:
                with socket.create_connection((host, int(port_str)), timeout=0.5):
                    return
            except OSError as err:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Ingress server on {self._server_address} did not become ready"
                    ) from err
                time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the Manus MANO Hugging Face dataset into an orca_teleop server.",
    )
    parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--repo-id", default=MANUS_MANO_REPO_ID, help="Hugging Face dataset repo")
    parser.add_argument("--split", default="train", help="Dataset split to replay")
    parser.add_argument("--rate-hz", type=float, default=None, help="Override replay rate")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop the selected frames forever")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset row to replay")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum rows to replay")
    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default=None,
        help="Override handedness",
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

    publisher = ManusDummyPublisher(
        server_address=args.server,
        repo_id=args.repo_id,
        split=args.split,
        rate_hz=args.rate_hz,
        speed=args.speed,
        loop=args.loop,
        start_index=args.start_index,
        max_frames=args.max_frames,
        handedness=args.hand,
    )
    publisher.run()


if __name__ == "__main__":
    main()
