"""Compare two cameras' landmarks of the same session, or average them into one recording.

Both folders come from ``scripts/record_cameras.py`` + ``scripts/landmarks_from_video.py``.
Frames are paired by capture time. Each view's hand is put in the hand's own
frame (as the retargeter does) and scaled to the same palm size, so what is left
is disagreement about the hand's shape, landmark by landmark.

    # How much do the views disagree? (mm at the average palm size)
    uv run --no-sync python scripts/multiview_landmarks.py compare \\
        recordings/s/brio recordings/s/mac

    # Average both views into a new recording the inspector can score
    uv run --no-sync python scripts/multiview_landmarks.py fuse \\
        recordings/s/brio recordings/s/mac --out recordings/s/fused
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

import numpy as np

from orca_teleop.ingress.landmark_recording import FORMAT, VERSION, load_recording
from orca_teleop.retargeting.utils import get_normalized_local_manohand_joint_pos

NAMES = ["wrist"] + [
    f"{finger}_{joint}"
    for finger, joints in [
        ("thumb", ["cmc", "mcp", "ip", "tip"]),
        ("index", ["mcp", "pip", "dip", "tip"]),
        ("middle", ["mcp", "pip", "dip", "tip"]),
        ("ring", ["mcp", "pip", "dip", "tip"]),
        ("pinky", ["mcp", "pip", "dip", "tip"]),
    ]
    for joint in joints
]


def palm_size(world: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(world[[5, 9, 13, 17]] - world[0], axis=1)))


def pairs(a, b, max_skew_ms: float):
    """Index pairs (i into a, j into b) whose capture times are closest and within the skew."""
    ta = a.capture_time_ns[a.frame]
    tb = b.capture_time_ns[b.frame]
    order = np.argsort(tb)
    tb_sorted = tb[order]
    out = []
    for i, t in enumerate(ta):
        k = np.searchsorted(tb_sorted, t)
        cands = [c for c in (k - 1, k) if 0 <= c < len(tb_sorted)]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(tb_sorted[c] - t))
        if abs(tb_sorted[c] - t) <= max_skew_ms * 1e6:
            out.append((i, int(order[c])))
    return out


def aligned(a, b, i: int, j: int):
    """Both views of one moment in the hand frame, both at their average palm size (m)."""
    ha = get_normalized_local_manohand_joint_pos(a.world[i], "mediapipe")
    hb = get_normalized_local_manohand_joint_pos(b.world[j], "mediapipe")
    sa, sb = palm_size(a.world[i]), palm_size(b.world[j])
    s = 0.5 * (sa + sb)
    return ha * (s / sa), hb * (s / sb), sb / sa


def compare(a, b, max_skew_ms: float) -> None:
    pp = pairs(a, b, max_skew_ms)
    if not pp:
        raise SystemExit("no frames pair up; check that both recordings overlap in time")
    d, ratios = [], []
    for i, j in pp:
        ha, hb, r = aligned(a, b, i, j)
        d.append(np.linalg.norm(ha - hb, axis=1) * 1000)
        ratios.append(r)
    d = np.array(d)
    print(
        f"{len(pp)} paired frames (of {len(a.frame)} / {len(b.frame)} with a hand); "
        f"palm size ratio b/a median {np.median(ratios):.3f}"
    )
    print(f"{'landmark':12s} {'median mm':>10s} {'p90 mm':>8s}")
    for k, name in enumerate(NAMES):
        if k == 0 or name.endswith(("_tip", "thumb_cmc", "thumb_mcp", "thumb_ip")):
            print(f"{name:12s} {np.median(d[:, k]):10.1f} {np.percentile(d[:, k], 90):8.1f}")
    print(f"{'all 21':12s} {np.median(d):10.1f} {np.percentile(d, 90):8.1f}")


def fuse(a, b, max_skew_ms: float, out: pathlib.Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(a.meta, format=FORMAT, version=VERSION, fused_from=[str(a.path), str(b.path)])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(a.path / "frames.csv", out / "frames.csv")
    os.symlink(os.path.relpath(a.video_path, out), out / "video.mp4")
    lines = []
    for i, j in pairs(a, b, max_skew_ms):
        ha, hb, _ = aligned(a, b, i, j)
        lines.append(
            {
                "frame": int(a.frame[i]),
                "world": np.round(0.5 * (ha + hb), 6).tolist(),
                "image": a.image[i].tolist(),
                "handedness": a.handedness[i],
                "score": float(min(a.score[i], b.score[j])),
            }
        )
    (out / "landmarks.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(f"Wrote {len(lines)} fused frames to {out} (video and timing from {a.path.name})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["compare", "fuse"])
    ap.add_argument("a", type=pathlib.Path, help="first camera's folder (fuse keeps its video)")
    ap.add_argument("b", type=pathlib.Path, help="second camera's folder")
    ap.add_argument("--max-skew-ms", type=float, default=20.0, help="pair frames this close")
    ap.add_argument("--out", type=pathlib.Path, help="fuse: new folder for the result")
    args = ap.parse_args()
    a, b = load_recording(args.a), load_recording(args.b)
    if args.action == "compare":
        compare(a, b, args.max_skew_ms)
    else:
        if args.out is None:
            raise SystemExit("fuse needs --out")
        fuse(a, b, args.max_skew_ms, args.out)


if __name__ == "__main__":
    main()
