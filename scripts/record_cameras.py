"""Record several webcams at once, every frame with its capture time, for offline comparison.

Each camera gets its own folder in the session, in the ``--record`` format
(video.mp4, frames.csv, meta.json, and an empty landmarks.jsonl that
``scripts/landmarks_from_video.py`` fills). Capture times share one clock, so
frames can be paired across cameras. No hand, robot or ingress server is needed.

    # Check placement first: preview only, nothing saved
    uv run --no-sync python scripts/record_cameras.py --check --camera brio:1 --camera side:0

    uv run --no-sync python scripts/record_cameras.py recordings/thumb-2cam \\
        --camera brio:1 --camera side:0

    # per camera NAME:INDEX[:WIDTHxHEIGHT][@FPS], e.g. --camera brio:1:1280x720@30

The preview shows every camera with MediaPipe's landmarks drawn on it (only on
the preview, never on the recorded frames), so you can see that each camera
finds the hand. Press q in it (or Ctrl+C) to stop, or pass --seconds. Capture
time is when OpenCV hands over the frame; each camera's own delay before that
is unknown and may differ between cameras by a few tens of ms.
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

MODEL = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src/orca_teleop/ingress/mediapipe/hand_landmarker.task"
)
BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


class CameraRecorder(threading.Thread):
    """Reads one camera as fast as it delivers and stores every frame."""

    def __init__(self, spec: str, folder: pathlib.Path | None, handedness: str) -> None:
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
        self.recorder = (
            None
            if folder is None
            else LandmarkRecorder(
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
            if self.recorder is not None:
                self.recorder.add_frame(frame, t)
            self.latest = frame
            self.frames += 1
            self.t_first = self.t_first or t
            self.t_last = t

    def close(self) -> str:
        self.stop_event.set()
        self.join(timeout=2.0)
        self.cap.release()
        if self.recorder is not None:
            self.recorder.close()
        return f"{self.config.name}: {self.frames} frames, {self.fps():.1f} fps measured"

    def fps(self) -> float:
        span = (self.t_last - self.t_first) / 1e9
        return (self.frames - 1) / span if span > 0 else 0.0


def preview_tile(cam: CameraRecorder, landmarker, t_ms: int) -> np.ndarray:
    """The camera's latest frame at 640x360, with the hand MediaPipe finds drawn on it."""
    frame = cv2.resize(cam.latest, (640, 360))
    found = "no landmarks"
    if landmarker is not None:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms
        )
        found = "NO HAND"
        if result.hand_landmarks:
            pts = [(int(p.x * 640), int(p.y * 360)) for p in result.hand_landmarks[0]]
            for a, b in BONES:
                cv2.line(frame, pts[a], pts[b], (230, 73, 128) if b <= 4 else (0, 220, 0), 2)
            found = f"hand {result.handedness[0][0].score:.2f}"
    color = (0, 0, 255) if found == "NO HAND" else (255, 255, 255)
    cv2.putText(
        frame,
        f"{cam.config.name} (index {cam.config.index})  {cam.fps():.0f} fps  {found}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=pathlib.Path, nargs="?", help="new folder for this recording")
    ap.add_argument("--check", action="store_true", help="preview only, save nothing")
    ap.add_argument(
        "--no-landmarks", action="store_true", help="don't run MediaPipe in the preview"
    )
    ap.add_argument("--camera", action="append", required=True, help="NAME:INDEX[:WxH][@FPS]")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    names = [parse_camera_spec(s).name for s in args.camera]
    if len(set(names)) != len(names):
        raise SystemExit("give every camera its own NAME")
    if args.check:
        folders = [None] * len(names)
    else:
        if args.session is None:
            raise SystemExit("give a folder to record into, or --check to only preview")
        if args.session.exists() and any(args.session.iterdir()):
            raise SystemExit(f"{args.session} is not empty")
        args.session.mkdir(parents=True, exist_ok=True)
        (args.session / "session.json").write_text(
            json.dumps({"cameras": names, "hand": args.hand}, indent=2)
        )
        folders = [args.session / n for n in names]

    cams = [CameraRecorder(s, f, args.hand) for s, f in zip(args.camera, folders, strict=True)]
    landmarkers = [None] * len(cams)
    if not args.no_preview and not args.no_landmarks:
        import mediapipe as mp

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        landmarkers = [mp.tasks.vision.HandLandmarker.create_from_options(options) for _ in cams]
    for c in cams:
        c.start()
    what = "Previewing" if args.check else f"Recording into {args.session}:"
    print(f"{what} {', '.join(names)}; press q in the preview or Ctrl+C to stop", flush=True)
    t0 = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            if args.no_preview:
                time.sleep(0.05)
                continue
            t_ms = int((time.monotonic() - t0) * 1000) + 1
            tiles = [
                preview_tile(c, lm, t_ms)
                for c, lm in zip(cams, landmarkers, strict=True)
                if c.latest is not None
            ]
            if len(tiles) == len(cams):
                view = np.hstack(tiles)
                elapsed = time.monotonic() - t0
                label = (
                    "PREVIEW ONLY"
                    if args.check
                    else f"REC {int(elapsed) // 60}:{int(elapsed) % 60:02d}"
                )
                cv2.putText(
                    view,
                    label,
                    (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255) if args.check else (0, 0, 255),
                    2,
                )
                cv2.imshow("record_cameras (q to stop)", view)
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
