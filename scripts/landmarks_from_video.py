"""Run MediaPipe's hand landmarker on recorded video, offline and on every frame.

Fills ``landmarks.jsonl`` of a folder written by ``scripts/record_cameras.py``
(or by ``--record``, with --overwrite). Same model file as the live publisher, in
VIDEO mode: deterministic, nothing dropped. The folder then works with
``scripts/fidelity_inspector.py`` and ``scripts/multiview_landmarks.py``.

    uv run --no-sync python scripts/landmarks_from_video.py \\
        recordings/thumb-2cam/brio recordings/thumb-2cam/mac
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import mediapipe as mp
import numpy as np

from orca_teleop.ingress.landmark_recording import load_recording

MODEL = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src/orca_teleop/ingress/mediapipe/hand_landmarker.task"
)


def extract(folder: pathlib.Path, confidence: float, overwrite: bool) -> None:
    out = folder / "landmarks.jsonl"
    if out.exists() and out.stat().st_size > 0 and not overwrite:
        raise SystemExit(f"{out} already has landmarks; pass --overwrite to replace them")
    rec = load_recording(folder)
    capture = rec.capture_time_ns
    hand = rec.meta.get("handedness", "right")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    cap = cv2.VideoCapture(str(folder / "video.mp4"))
    lines, other_side, last_ms, i = [], 0, -1, 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = int(capture[i] // 1_000_000) if i < len(capture) else last_ms + 33
            t_ms = max(t_ms, last_ms + 1)  # VIDEO mode needs strictly increasing times
            last_ms = t_ms
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = landmarker.detect_for_video(image, t_ms)
            if result.hand_landmarks:
                label = result.handedness[0][0].category_name.lower()
                other_side += label != hand
                lines.append(
                    {
                        "frame": i,
                        "world": np.round(
                            [[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], 6
                        ).tolist(),
                        "image": np.round(
                            [[p.x, p.y, p.z] for p in result.hand_landmarks[0]], 6
                        ).tolist(),
                        "handedness": label,
                        "score": round(float(result.handedness[0][0].score), 4),
                    }
                )
            i += 1
    cap.release()
    out.write_text("".join(json.dumps(line) + "\n" for line in lines))
    print(
        f"{folder}: {i} frames, hand found in {len(lines)} ({len(lines) / max(i, 1):.0%}), "
        f"labelled as the other hand in {other_side}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", type=pathlib.Path)
    ap.add_argument("--confidence", type=float, default=0.7, help="as the publisher (default 0.7)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing landmarks")
    args = ap.parse_args()
    for folder in args.folders:
        extract(folder, args.confidence, args.overwrite)


if __name__ == "__main__":
    main()
