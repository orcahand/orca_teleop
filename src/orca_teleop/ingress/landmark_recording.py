"""Record webcam sessions (raw frames + MediaPipe landmarks) and load them back.

A recording is a folder:

    meta.json         format, version, camera and MediaPipe settings
    video.mp4         every captured frame, in capture order
    frames.csv        frame index -> capture time (ns, ``time.time_ns()``)
    landmarks.jsonl   one line per landmark result the publisher sent:
                      frame index, world landmarks (m, 21x3), image landmarks
                      (normalized x, y and relative z, 21x3), handedness, score

Frames without a line in ``landmarks.jsonl`` had no accepted hand. Replaying the
world landmarks reproduces exactly what the retargeter received.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FORMAT = "orca_teleop.webcam_recording"
VERSION = 1


class LandmarkRecorder:
    """Writes one recording folder. Safe to call from the capture and MediaPipe threads."""

    def __init__(self, path: str | Path, meta: dict, fps: float) -> None:
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f"Recording folder {self.path} is not empty")
        self.path.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._lock = threading.Lock()
        self._video: cv2.VideoWriter | None = None
        self._n_frames = 0
        self._n_landmarks = 0
        (self.path / "meta.json").write_text(
            json.dumps({"format": FORMAT, "version": VERSION, **meta}, indent=2)
        )
        self._frames = open(self.path / "frames.csv", "w", newline="")
        self._frames_csv = csv.writer(self._frames)
        self._frames_csv.writerow(["frame", "capture_time_ns"])
        self._landmarks = open(self.path / "landmarks.jsonl", "w")

    def add_frame(self, frame_bgr: np.ndarray, capture_time_ns: int) -> int:
        """Store one captured frame; returns its index."""
        with self._lock:
            if self._video is None:
                height, width = frame_bgr.shape[:2]
                self._video = cv2.VideoWriter(
                    str(self.path / "video.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps,
                    (width, height),
                )
            self._video.write(frame_bgr)
            index = self._n_frames
            self._frames_csv.writerow([index, capture_time_ns])
            self._n_frames += 1
            return index

    def add_landmarks(
        self, frame: int, world: np.ndarray, image: np.ndarray, handedness: str, score: float
    ) -> None:
        line = {
            "frame": int(frame),
            "world": np.round(np.asarray(world, dtype=float), 6).tolist(),
            "image": np.round(np.asarray(image, dtype=float), 6).tolist(),
            "handedness": handedness,
            "score": round(float(score), 4),
            "result_time_ns": time.time_ns(),
        }
        with self._lock:
            self._landmarks.write(json.dumps(line) + "\n")
            self._n_landmarks += 1

    def close(self) -> tuple[int, int]:
        """Finish the files; returns (frames, landmark results)."""
        with self._lock:
            if self._video is not None:
                self._video.release()
            self._frames.close()
            self._landmarks.close()
            return self._n_frames, self._n_landmarks


@dataclass(frozen=True)
class Recording:
    path: Path
    meta: dict
    capture_time_ns: np.ndarray  # (n_frames,)
    frame: np.ndarray  # (n,) frame index of each landmark result
    world: np.ndarray  # (n, 21, 3) metres, as sent to the retargeter
    image: np.ndarray  # (n, 21, 3)
    handedness: list[str]
    score: np.ndarray  # (n,)

    @property
    def video_path(self) -> Path:
        return self.path / "video.mp4"


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError(f"{path} is not a version-{VERSION} {FORMAT} folder")
    with open(path / "frames.csv") as f:
        rows = list(csv.DictReader(f))
    capture = np.array([int(r["capture_time_ns"]) for r in rows], dtype=np.int64)
    lines = [
        json.loads(line) for line in (path / "landmarks.jsonl").read_text().splitlines() if line
    ]
    return Recording(
        path=path,
        meta=meta,
        capture_time_ns=capture,
        frame=np.array([d["frame"] for d in lines], dtype=np.int64),
        world=np.array([d["world"] for d in lines], dtype=np.float64).reshape(-1, 21, 3),
        image=np.array([d["image"] for d in lines], dtype=np.float64).reshape(-1, 21, 3),
        handedness=[d["handedness"] for d in lines],
        score=np.array([d["score"] for d in lines], dtype=np.float64),
    )
