"""Record several webcams at once, every frame with its capture time, for offline comparison.

Each camera gets its own folder in the session, in the ``--record`` format
(video.mp4, frames.csv, meta.json, and an empty landmarks.jsonl that
``scripts/landmarks_from_video.py`` fills). Capture times share one clock, so
frames can be paired across cameras. No hand, robot or ingress server is needed.

    uv run --no-sync python scripts/record_cameras.py recordings/thumb-2cam \\
        --camera brio:0 --camera mac:1

    # per camera NAME:INDEX[:WIDTHxHEIGHT][@FPS], e.g. --camera brio:0:1280x720@30

A preview window shows every camera; press q in it (or Ctrl+C) to stop, or pass
--seconds. Capture time is when OpenCV hands over the frame; each camera's own
delay before that is unknown and may differ between cameras by a few tens of ms.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

from orca_teleop.cameras import parse_camera_spec
from orca_teleop.ingress.landmark_recording import LandmarkRecorder


class CameraRecorder(threading.Thread):
    """Reads one camera as fast as it delivers and stores every frame."""

    def __init__(self, spec: str, folder: pathlib.Path, handedness: str) -> None:
        super().__init__(daemon=True)
        self.config = parse_camera_spec(spec)
        self.name = f"record-{self.config.name}"
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.config.index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {spec!r}")
        if self.config.width and self.config.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        size = [
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        fps = self.cap.get(cv2.CAP_PROP_FPS) or self.config.fps or 30.0
        self.recorder = LandmarkRecorder(
            folder,
            meta={
                "handedness": handedness,
                "camera_name": self.config.name,
                "camera_index": self.config.index,
                "camera_size": size,
                "capture_fps": fps,
                "recorder": "scripts/record_cameras.py",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            fps=fps,
        )
        self.stop_event = threading.Event()
        self.latest: np.ndarray | None = None
        self.frames = 0
        self.t_first = self.t_last = 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            t = time.time_ns()
            if not ok:
                time.sleep(0.002)
                continue
            self.recorder.add_frame(frame, t)
            self.latest = frame
            self.frames += 1
            self.t_first = self.t_first or t
            self.t_last = t

    def close(self) -> str:
        self.stop_event.set()
        self.join(timeout=2.0)
        self.cap.release()
        frames, _ = self.recorder.close()
        span = (self.t_last - self.t_first) / 1e9
        fps = (frames - 1) / span if span > 0 else 0.0
        return f"{self.config.name}: {frames} frames, {fps:.1f} fps measured"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=pathlib.Path, help="new folder for this recording")
    ap.add_argument("--camera", action="append", required=True, help="NAME:INDEX[:WxH][@FPS]")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    if args.session.exists() and any(args.session.iterdir()):
        raise SystemExit(f"{args.session} is not empty")
    names = [parse_camera_spec(s).name for s in args.camera]
    if len(set(names)) != len(names):
        raise SystemExit("give every camera its own NAME")
    args.session.mkdir(parents=True, exist_ok=True)
    (args.session / "session.json").write_text(
        json.dumps({"cameras": names, "hand": args.hand}, indent=2)
    )

    cams = [
        CameraRecorder(s, args.session / n, args.hand)
        for s, n in zip(args.camera, names, strict=True)
    ]
    for c in cams:
        c.start()
    print(
        f"Recording {', '.join(names)} into {args.session}; "
        "press q in the preview or Ctrl+C to stop"
    )
    t0 = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            if args.no_preview:
                time.sleep(0.05)
                continue
            tiles = [cv2.resize(c.latest, (480, 270)) for c in cams if c.latest is not None]
            if tiles:
                cv2.imshow("record_cameras (q to stop)", np.hstack(tiles))
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            print(c.close())
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
