"""Replay a webcam recording through the retargeter and see how well each finger is matched.

Feeds the recorded landmarks to the same retargeter the pipeline builds, frame by
frame (auto-scale calibration and smoothing included), and measures, per finger,
how far the commanded robot pose is from the targets the solver aims at. The
thumb comes first everywhere. Record a session with ``--record DIR`` on
``orca-teleop-hand``, ``orca-teleop-urdf`` or the MediaPipe publisher.

    # Score only (prints a summary; --save-summary writes it as JSON)
    uv run --no-sync python scripts/fidelity_inspector.py recordings/thumb-01 \\
        --model-path ~/.orca_ui/hands/orcahand-right/config.yaml --no-viewer

    # Same, with other fingertip points (e.g. docs/tools/tip_points.yaml from the tip picker)
    uv run --no-sync python scripts/fidelity_inspector.py recordings/thumb-01 \\
        --model-path ~/.orca_ui/hands/orcahand-right/config.yaml --no-viewer \\
        --tip-points docs/tools/tip_points.yaml

    # Score + 3D viewer with a timeline at http://localhost:8091
    uv run --no-sync --with viser==1.1.1 python scripts/fidelity_inspector.py recordings/thumb-01 \\
        --model-path ~/.orca_ui/hands/orcahand-right/config.yaml

What is compared (all in the robot's frame, anchored at the robot's palm point,
exactly as the solver's loss sees it):

  tip error     robot palm->tip vector vs the target's, cm
  tip angle     robot last-segment direction (tip - dip) vs the human's, degrees
  pinch gap     robot thumb-finger distance minus the human's (scaled), cm;
                only frames where the human gap is under 5 cm
  at limit      share of frames each thumb joint sits within 1 deg of its range end
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import threading
import time

import numpy as np

from orca_teleop.ingress.landmark_recording import load_recording
from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.retargeter import FINGERS, Retargeter, TargetPose

PINCH_GAP_CM = 5.0
LIMIT_TOL_DEG = 1.0
FINGER_COLORS = {
    "thumb": (230, 73, 128),
    "index": (34, 139, 230),
    "middle": (64, 192, 87),
    "ring": (250, 176, 5),
    "pinky": (132, 94, 247),
}


def use_tip_points(r, path: str) -> None:
    """Swap the retargeter's fingertip offsets for picked points (last-segment frame, metres)."""
    import yaml

    points = yaml.safe_load(pathlib.Path(path).read_text())["points"]
    for finger, frame_name in zip(FINGERS, r._frame_map.tip, strict=True):
        r._frame_offsets[r._computed_frame_names.index(frame_name)] = np.array(
            points[finger]["xyz_m"], dtype=np.float64
        )
    r._build_target_alignment_params()  # the auto-scale reference depends on the tips


def replay(rec, retargeter) -> dict:
    """Run every landmark frame through the retargeter; return per-frame arrays."""
    r = retargeter
    n = len(rec.frame)
    out = {
        "calibrating": np.zeros(n, dtype=bool),
        "q_full": np.zeros((n, r._robot.nq)),
        "q_deg": np.full((n, len(r._finger_joint_ids)), np.nan),
        "tip_err_cm": np.full((n, 5), np.nan),
        "solver_tip_err_cm": np.full((n, 5), np.nan),
        "dir_err_deg": np.full((n, 5), np.nan),
        "human_gap_cm": np.full((n, 4), np.nan),
        "robot_gap_cm": np.full((n, 4), np.nan),
        "palm_m": np.zeros((n, 3)),
        "targets_m": np.zeros((n, 5, 3, 3)),  # finger, (pip, dip, tip), xyz
        "robot_pts_m": np.zeros((n, 5, 3, 3)),
    }
    for k in range(n):
        pose = TargetPose(joint_positions=rec.world[k], source="mediapipe", wrist_angle_degrees=0.0)
        try:
            action = r.retarget(pose)
        except (AssertionError, ValueError):  # the pipeline skips degenerate frames too
            out["calibrating"][k] = True
            continue
        if action is None:
            out["calibrating"][k] = True
            continue

        # The solver's targets, computed by the retargeter's own steps (as in retarget()).
        normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(
            pose.joint_positions, pose.source
        )
        if r._rotation_matrix is not None:
            normalized = normalized @ r._rotation_matrix.T
        kp = r._target_keypoints_in_urdf_frame(normalized)
        target_tip = r._compute_tip_vectors(kp)
        target_dir = r._compute_tip_dirs(kp)
        target_full = r._compute_full_hand_vectors(kp).reshape(3, 5, 3)  # (pip, dip, tip)

        joints = action.as_dict()
        q_deg = np.array([joints[j] for j in r._finger_joint_ids])
        for key, q_phys in (
            ("tip_err_cm", np.deg2rad(q_deg)),
            ("solver_tip_err_cm", r._last_qpos_phys),
        ):
            q_full = r._full_model_qpos(q_phys)
            pos, _ = r._robot.compute_positions_and_jacobians(
                q_full, r._computed_frame_indices, r._frame_offsets
            )
            pos_cm = pos * 100.0
            palm_cm = r._robot_palm_from_positions_cm(pos_cm)
            tip = pos_cm[r._tip_indices]
            out[key][k] = np.linalg.norm((tip - palm_cm) - target_tip, axis=1)
            if key == "tip_err_cm":  # the commanded pose is the one shown and scored further
                out["q_full"][k], out["q_deg"][k] = q_full, q_deg
                dip, pip = pos_cm[r._dip_indices], pos_cm[r._pip_indices]
                d = tip - dip
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
                cos = np.clip(np.sum(d * target_dir, axis=1), -1.0, 1.0)
                out["dir_err_deg"][k] = np.degrees(np.arccos(cos))
                out["robot_gap_cm"][k] = np.linalg.norm(tip[1:] - tip[0], axis=1)
                out["palm_m"][k] = palm_cm / 100.0
                out["targets_m"][k] = (palm_cm + target_full).transpose(1, 0, 2) / 100.0
                out["robot_pts_m"][k] = np.stack([pip, dip, tip], axis=1) / 100.0
        tips_kp = kp[[4, 8, 12, 16, 20]] * 100.0
        out["human_gap_cm"][k] = np.linalg.norm(tips_kp[1:] - tips_kp[0], axis=1)
    return out


def summarize(rec, r, res) -> dict:
    ok = ~res["calibrating"]
    pct = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else None  # noqa: E731
    fingers = {}
    for i, f in enumerate(FINGERS):
        row = {
            "tip_err_cm": {
                "median": pct(res["tip_err_cm"][ok, i], 50),
                "p90": pct(res["tip_err_cm"][ok, i], 90),
            },
            "solver_tip_err_cm": {"median": pct(res["solver_tip_err_cm"][ok, i], 50)},
            "dir_err_deg": {
                "median": pct(res["dir_err_deg"][ok, i], 50),
                "p90": pct(res["dir_err_deg"][ok, i], 90),
            },
        }
        if i > 0:
            near = ok & (res["human_gap_cm"][:, i - 1] < PINCH_GAP_CM)
            diff = res["robot_gap_cm"][near, i - 1] - res["human_gap_cm"][near, i - 1]
            row["pinch_gap_err_cm"] = {
                "frames": int(near.sum()),
                "median": pct(diff, 50),
                "p90_abs": pct(np.abs(diff), 90),
            }
        fingers[f] = row
    lo, hi = np.rad2deg(r._lower_phys), np.rad2deg(r._upper_phys)
    q = res["q_deg"][ok]
    at_limit = {}
    for j, jid in enumerate(r._finger_joint_ids):
        if jid.startswith("thumb"):
            at_lo = float(np.mean(q[:, j] <= lo[j] + LIMIT_TOL_DEG)) if len(q) else None
            at_hi = float(np.mean(q[:, j] >= hi[j] - LIMIT_TOL_DEG)) if len(q) else None
            at_limit[jid] = {
                "range_deg": [float(lo[j]), float(hi[j])],
                "at_lower": at_lo,
                "at_upper": at_hi,
            }
    return {
        "recording": str(rec.path),
        "frames_with_hand": int(len(rec.frame)),
        "frames_scored": int(ok.sum()),
        "mano_scale": float(r._mano_scale),
        "fingers": fingers,
        "thumb_at_limit": at_limit,
    }


def print_summary(s: dict) -> None:
    print(
        f"\n{s['recording']}: {s['frames_scored']} of {s['frames_with_hand']} hand frames scored "
        f"(auto-scale {s['mano_scale']:.3f})"
    )
    fmt = lambda v: "   -" if v is None else f"{v:5.2f}"  # noqa: E731
    print(f"{'finger':7s} {'tip err cm':>16s} {'tip angle deg':>16s} {'pinch gap cm':>22s}")
    head = ("median", "p90", "median", "p90")
    print(
        f"{'':7s} "
        + "".join(f"{h:>8s}" for h in head)
        + f" {'frames':>7s}{'median':>8s}{'p90|.|':>7s}"
    )
    for f, row in s["fingers"].items():
        pg = row.get("pinch_gap_err_cm")
        pinch = f"{pg['frames']:7d}{fmt(pg['median']):>8s}{fmt(pg['p90_abs']):>7s}" if pg else ""
        print(
            f"{f:7s} {fmt(row['tip_err_cm']['median']):>8s}{fmt(row['tip_err_cm']['p90']):>8s} "
            f"{fmt(row['dir_err_deg']['median']):>8s}{fmt(row['dir_err_deg']['p90']):>8s} {pinch}"
        )
    print("thumb joints at a range end (share of frames):")
    for jid, v in s["thumb_at_limit"].items():
        print(
            f"  {jid:10s} range {v['range_deg'][0]:6.1f}..{v['range_deg'][1]:5.1f}  "
            f"lower {v['at_lower']:.0%}  upper {v['at_upper']:.0%}"
        )


def _read_stl(path: pathlib.Path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    tri = np.frombuffer(
        data,
        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        count=n,
        offset=84,
    )["v"].reshape(-1, 3)
    verts, faces = np.unique(tri, axis=0, return_inverse=True)
    return verts, faces.reshape(-1, 3).astype(np.uint32)


def _rot_to_wxyz(R: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    q = pin.Quaternion(R)
    return np.array([q.w, q.x, q.y, q.z])


def serve_viewer(rec, r, res, summary: dict, port: int) -> None:
    import cv2
    import pinocchio as pin
    import viser

    urdf = pathlib.Path(r.urdf_path)
    desc_parent = next(p for p in urdf.parents if p.name == "orcahand_description").parent
    model, _, vmodel = pin.buildModelsFromUrdf(str(urdf), package_dirs=[str(desc_parent)])
    if model.nq != r._robot.nq:
        raise SystemExit("viewer model and retargeter model disagree")
    data, vdata = model.createData(), pin.GeometryData(vmodel)

    server = viser.ViserServer(host="127.0.0.1", port=port, label="Fidelity inspector")
    meshes = []
    for g in vmodel.geometryObjects:
        if g.parentJoint == 0:  # forearm and tower: never move here (wrist held at 0)
            meshes.append(None)
            continue
        v, f = _read_stl(pathlib.Path(g.meshPath))
        meshes.append(
            server.scene.add_mesh_simple(
                f"/robot/{g.name}",
                v * np.asarray(g.meshScale),
                f,
                color=(200, 200, 205),
                opacity=0.55,
            )
        )
    target_lines = {
        f: server.scene.add_line_segments(
            f"/targets/{f}",
            np.zeros((3, 2, 3)),
            FINGER_COLORS[f],
            thickness=0.0015,
            thickness_units="world",
        )
        for f in FINGERS
    }
    target_pts = {
        f: server.scene.add_point_cloud(
            f"/targets/{f}_pts",
            np.zeros((3, 3)),
            FINGER_COLORS[f],
            point_size=0.004,
            point_shape="circle",
            precision="float32",
        )
        for f in FINGERS
    }
    robot_pts = {
        f: server.scene.add_point_cloud(
            f"/robot_pts/{f}",
            np.zeros((3, 3)),
            (40, 40, 40),
            point_size=0.003,
            point_shape="square",
            precision="float32",
        )
        for f in FINGERS
    }
    gaps = server.scene.add_line_segments(
        "/errors", np.zeros((15, 2, 3)), (220, 30, 30), thickness=0.0006, thickness_units="world"
    )

    gui = server.gui
    gui.add_markdown(
        "Coloured = the solver's targets from your hand (thumb: MCP, IP, TIP landmarks; "
        "fingers: PIP, DIP, TIP), dark squares = the robot points matched to them, "
        "red = the gap. All anchored at the robot's palm point, as the solver sees them."
    )
    slider = gui.add_slider("Frame", 0, len(rec.frame) - 1, 1, 0)
    playing = gui.add_checkbox("Play", False)
    speed = gui.add_slider("Speed", 0.25, 4.0, 0.25, 1.0)
    only_thumb = gui.add_checkbox("Thumb only", False)
    info = gui.add_markdown("")
    has_video = rec.video_path.exists()
    image = (
        gui.add_image(np.zeros((180, 320, 3), dtype=np.uint8), label="Webcam")
        if has_video
        else None
    )
    with gui.add_folder("Whole recording", expand_by_default=False):
        rows = [
            "| finger | tip err cm (med/p90) | angle deg (med) | pinch gap cm (med) |",
            "|---|---|---|---|",
        ]
        for f, row in summary["fingers"].items():
            pg = row.get("pinch_gap_err_cm") or {}
            fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
            rows.append(
                f"| {f} | {fmt(row['tip_err_cm']['median'])} / {fmt(row['tip_err_cm']['p90'])} | "
                f"{fmt(row['dir_err_deg']['median'])} | {fmt(pg.get('median'))} |"
            )
        gui.add_markdown("\n".join(rows))

    cap = cv2.VideoCapture(str(rec.video_path)) if has_video else None
    lock = threading.Lock()

    def show(k: int) -> None:
        with lock:
            calibrating = bool(res["calibrating"][k])
            q = res["q_full"][k]
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, vmodel, vdata, q)
            for m, M in zip(meshes, vdata.oMg, strict=True):
                if m is not None:
                    m.position, m.wxyz = M.translation, _rot_to_wxyz(M.rotation)
            seg_err = []
            for i, f in enumerate(FINGERS):
                visible = not calibrating and (f == "thumb" or not only_thumb.value)
                t, rp = res["targets_m"][k, i], res["robot_pts_m"][k, i]
                chain = np.vstack([res["palm_m"][k], t])
                target_lines[f].points = np.stack([chain[:-1], chain[1:]], axis=1)
                target_pts[f].points, robot_pts[f].points = t, rp
                for h in (target_lines[f], target_pts[f], robot_pts[f]):
                    h.visible = visible
                seg_err.append(np.stack([t, rp], axis=1) if visible else np.zeros((3, 2, 3)))
            gaps.points = np.concatenate(seg_err)
            if calibrating:
                info.content = f"**Frame {k}**: calibrating hand scale (hold the hand open)"
            else:
                q_deg = dict(zip(r._finger_joint_ids, res["q_deg"][k], strict=True))
                thumb = ", ".join(
                    f"{j[6:]} {q_deg[j]:.0f}°" for j in r._finger_joint_ids if j.startswith("thumb")
                )
                gaps_txt = ", ".join(
                    f"{f} {res['robot_gap_cm'][k, i]:.1f}/{res['human_gap_cm'][k, i]:.1f}"
                    for i, f in enumerate(FINGERS[1:])
                )
                info.content = (
                    f"**Frame {k}** (video frame {rec.frame[k]}, score {rec.score[k]:.2f})  \n"
                    f"Thumb tip error **{res['tip_err_cm'][k, 0]:.2f} cm**, "
                    f"angle **{res['dir_err_deg'][k, 0]:.0f}°**  \n"
                    f"Thumb joints: {thumb}  \n"
                    f"Thumb–finger gap robot/human cm: {gaps_txt}  \n"
                    "Tip error cm: "
                    + ", ".join(f"{f} {res['tip_err_cm'][k, i]:.2f}" for i, f in enumerate(FINGERS))
                )
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(rec.frame[k]))
                ok, frame = cap.read()
                if ok:
                    h, w = frame.shape[:2]
                    pts = (rec.image[k, :, :2] * [w, h]).astype(int)
                    for a, b in [
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
                    ]:
                        color = (128, 73, 230) if b <= 4 else (0, 220, 0)
                        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
                    scale = 480 / w
                    small = cv2.resize(frame, (480, int(h * scale)))
                    image.image = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    slider.on_update(lambda _e: show(int(slider.value)))
    only_thumb.on_update(lambda _e: show(int(slider.value)))

    @server.on_client_connect
    def _(client):
        client.camera.near = 0.001
        palm = (
            res["palm_m"][~res["calibrating"]].mean(axis=0)
            if (~res["calibrating"]).any()
            else np.zeros(3)
        )
        axes = (
            r._urdfhand_rot_matrix
        )  # columns: across the knuckles, towards the fingers, palm normal
        client.camera.up_direction = tuple(axes[:, 1])
        client.camera.position = tuple(palm + 0.2 * axes[:, 2] + 0.05 * axes[:, 1])
        client.camera.look_at = tuple(palm + 0.05 * axes[:, 1])

    show(0)
    print(f"\nViewer: http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    fps = float(rec.meta.get("capture_fps", 30.0))
    while True:
        if playing.value:
            slider.value = (int(slider.value) + 1) % len(rec.frame)
            time.sleep(1.0 / (fps * float(speed.value)))
        else:
            time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", help="folder written by --record")
    ap.add_argument("--model-path", default=None, help="the hand's orca_core config.yaml")
    ap.add_argument("--urdf-path", default=None, help="URDF to solve on (default: as the pipeline)")
    ap.add_argument(
        "--retarget-config", default=None, help="adaptive_analytical YAML (default: tuned)"
    )
    ap.add_argument(
        "--tip-points",
        default=None,
        metavar="YAML",
        help="replace FINGERTIP_OFFSETS with points saved by the tip picker",
    )
    ap.add_argument("--save-summary", default=None, metavar="JSON", help="write the summary here")
    ap.add_argument("--no-viewer", action="store_true", help="print the summary and exit")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    rec = load_recording(args.recording)
    retargeter = Retargeter.from_paths(
        args.model_path,
        args.urdf_path,
        backend="adaptive_analytical",
        config_path=args.retarget_config,
    )
    if args.tip_points:
        use_tip_points(retargeter, args.tip_points)
    t0 = time.perf_counter()
    res = replay(rec, retargeter)
    print(f"Replayed {len(rec.frame)} frames in {time.perf_counter() - t0:.1f} s")
    summary = summarize(rec, retargeter, res)
    summary["retarget_config"] = args.retarget_config or "default (tuned)"
    summary["tip_points"] = args.tip_points or "FINGERTIP_OFFSETS"
    summary["urdf"] = retargeter.urdf_path
    print_summary(summary)
    if args.save_summary:
        pathlib.Path(args.save_summary).write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {args.save_summary}")
    if not args.no_viewer:
        serve_viewer(rec, retargeter, res, summary, args.port)


if __name__ == "__main__":
    main()
