"""Standalone diagnostic for the retargeter.

Live-plots, side by side, the hand pose extracted from the ingress and the
per-finger key vectors fed into the IK against those reconstructed from the
URDF after the optimizer step. Useful for iterating on the retargeter without
running the robot — and for deciding whether teleop mismatch is a retargeting
problem or a landmark/sensor problem.

How to read the plot
--------------------
  - Left panel looks like your hand? → source landmarks are usable?
  - Right panel:
    - dashed == solid → retargeting is healthy
    - dashed != solid → retargeter / config / kinematics problem

Two ways to run:

    # Default adaptive_analytical backend with ready-to-use mediapipe publisher:
    python scripts/retargeter_diagnostics.py --with-mediapipe --show-video \\
        --model-path /path/to/orcahand_touch_right

    # Legacy key-vector backend (has live hyperparam sliders):
    python scripts/retargeter_diagnostics.py --retargeter rmsprop --with-mediapipe

    # Bring your own publisher:
    python scripts/retargeter_diagnostics.py
    python -m orca_teleop.ingress.mediapipe.publisher --server localhost:50051

Layout: a single matplotlib window with two panels.

  - Left:  the normalized hand landmarks (21 joints), wrist + bones colored
           per finger so the pose is always visible.
  - Right: the five palm-to-tip key vectors as arrows from the palm origin —
           dashed for the target (from the ingress) and solid for the URDF
           reconstruction, color-coded to match the left panel.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import queue
import threading
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3D

from orca_teleop.constants import HEARTBEAT_INTERVAL, QUEUES_MAXSIZE
from orca_teleop.ingress import DEFAULT_PORT, HandLandmarks, IngressServer
from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.config import DEFAULT_CONFIGS, resolve
from orca_teleop.retargeting.retargeter import FINGERS, Retargeter, RetargeterBackend, TargetPose

logger = logging.getLogger(__name__)

# Per-finger color, shared between the landmark panel and the key-vector panel.
_FINGER_COLORS: dict[str, str] = {
    "thumb": "tab:red",
    "index": "tab:orange",
    "middle": "tab:green",
    "ring": "tab:blue",
    "pinky": "tab:purple",
}

# MediaPipe 21-keypoint indices for the bones we draw on the landmark panel.
# Each chain starts at the wrist (0) and walks out to the fingertip.
_FINGER_BONE_INDICES: dict[str, list[int]] = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

_LANDMARK_LIM = 0.12  # axis half-range (m) for the landmark panel
_KV_LIM = 0.12  # axis half-range (m) for the key-vector panel

# ---------- Live-tunable hyperparameters --------------------------------------
# Defaults mirror the values baked into the Retargeter at construction time so
# the sliders start at the same point as a vanilla pipeline run.
_DEFAULT_LR = 25.0
_DEFAULT_N_STEPS = 2
_DEFAULT_FINGER_COEFFS: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
_DEFAULT_ABD_REG_WEIGHT = 1e-6
_ABD_JOINT_NAMES: tuple[str, ...] = ("index_abd", "middle_abd", "ring_abd", "pinky_abd")


@dataclass
class _HyperParams:
    lr: float
    n_steps: int
    finger_coeffs: tuple[float, float, float, float, float]
    abd_reg_weight: float


def _default_hyperparams() -> _HyperParams:
    return _HyperParams(
        lr=_DEFAULT_LR,
        n_steps=_DEFAULT_N_STEPS,
        finger_coeffs=_DEFAULT_FINGER_COEFFS,
        abd_reg_weight=_DEFAULT_ABD_REG_WEIGHT,
    )


class _HyperParamHolder:
    """Thread-safe holder for live-tunable retargeter hyperparameters."""

    def __init__(self, initial: _HyperParams) -> None:
        self._lock = threading.Lock()
        self._hp = initial

    def set(self, hp: _HyperParams) -> None:
        with self._lock:
            self._hp = hp

    def get(self) -> _HyperParams:
        with self._lock:
            return self._hp


def _make_dynamic_loss(hp_holder: _HyperParamHolder):
    """IK loss that re-reads finger coefficients from the holder each call.

    The Retargeter's frozen config holds a reference to this closure, so simply
    pushing new values into ``hp_holder`` is enough to take effect on the very
    next optimizer step — no need to rebuild the Retargeter.
    """

    def loss(target: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        coeffs = torch.tensor(
            hp_holder.get().finger_coeffs, dtype=target.dtype, device=target.device
        )
        diffs_sq = torch.norm(target - robot, dim=-1) ** 2
        return (coeffs * diffs_sq).sum()

    return loss


@dataclass
class _Snapshot:
    landmarks: np.ndarray  # (21, 3) normalized MANO joints
    target_kvs: np.ndarray  # (5, 3) palm-to-tip vectors from the ingress (m)
    robot_kvs: np.ndarray  # (5, 3) palm-to-tip vectors from URDF FK (m)
    tip_error_cm: np.ndarray  # (5,) per-finger ||robot - target|| in cm
    backend: str


def _tip_errors_cm(target_kvs_m: np.ndarray, robot_kvs_m: np.ndarray) -> np.ndarray:
    return np.linalg.norm(robot_kvs_m - target_kvs_m, axis=1) * 100.0


def _compute_rmsprop_diagnostic_snapshot(
    retargeter: Retargeter, target_pose: TargetPose
) -> _Snapshot | None:
    """After ``retarget()``, recompute the values needed by the diagnostic.

    Mirrors the URDF-frame transform from ``Retargeter.retarget`` and runs FK
    on the latest optimized joint angles to recover the URDF-side key vectors.
    Returns ``None`` while auto-scale calibration is still in progress.
    """
    cfg = retargeter.config
    if not retargeter._calibration_done:
        return None

    joints = np.asarray(target_pose.joint_positions, dtype=float)
    normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(
        joints, target_pose.source
    )
    scaled = normalized * retargeter._mano_scale
    in_urdf_frame = scaled @ cfg.urdfhand_rot_matrix.T + cfg.urdfhand_center
    joint_t = torch.tensor(in_urdf_frame, dtype=torch.float32, device=cfg.device)
    mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(
        joint_t, list(FINGERS), target_pose.source
    )
    target_kvs = torch.stack([(mano_ft[f] - mano_palm).squeeze(0) for f in FINGERS])

    with torch.no_grad():
        urdf_angles = torch.zeros(cfg.chain.n_joints, device=cfg.device)
        urdf_angles[cfg.finger_reorder_indices] = retargeter._joint_angles / (180.0 / np.pi)
        root = torch.zeros(1, 3, device=cfg.device)
        fingertips, palm = retargeter_utils.extract_orca_fingertips_and_palm(
            cfg.chain,
            urdf_angles,
            cfg.optimization_frames,
            cfg.hand_type,
            list(FINGERS),
            root,
            fingertip_offsets=cfg.fingertip_offsets,
        )
        robot_kvs = torch.stack(
            [kv.squeeze(0) for kv in retargeter_utils.get_keyvectors(fingertips, palm)]
        )

    target_np = target_kvs.detach().cpu().numpy()
    robot_np = robot_kvs.detach().cpu().numpy()
    return _Snapshot(
        landmarks=np.asarray(normalized, dtype=float),
        target_kvs=target_np,
        robot_kvs=robot_np,
        tip_error_cm=_tip_errors_cm(target_np, robot_np),
        backend="rmsprop",
    )


def _compute_adaptive_diagnostic_snapshot(
    retargeter: Any, target_pose: TargetPose
) -> _Snapshot | None:
    """Compare adaptive tip-vector targets against FK of the commanded qpos."""
    from orca_teleop.retargeting.adaptive_analytical import M_TO_CM

    if not retargeter._calibration_done:
        return None

    qpos_phys = retargeter._filter._state
    if qpos_phys is None:
        qpos_phys = retargeter._last_qpos_phys
    if qpos_phys is None:
        return None

    normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(
        target_pose.joint_positions,
        target_pose.source,
    )
    if retargeter._rotation_matrix is not None:
        normalized = normalized @ retargeter._rotation_matrix.T

    keypoints = retargeter._target_keypoints_in_urdf_frame(normalized)
    # Same tip-vector objective the adaptive solver minimizes (cm → m for the plot).
    target_kvs = retargeter._compute_tip_vectors(keypoints) / M_TO_CM

    positions, _ = retargeter._robot.compute_positions_and_jacobians(
        retargeter._full_model_qpos(np.asarray(qpos_phys, dtype=np.float64)),
        retargeter._computed_frame_indices,
        retargeter._frame_offsets,
    )
    palm = retargeter._robot_palm_from_positions(positions)
    robot_kvs = positions[retargeter._tip_indices] - palm

    return _Snapshot(
        landmarks=np.asarray(normalized, dtype=float),
        target_kvs=np.asarray(target_kvs, dtype=float),
        robot_kvs=np.asarray(robot_kvs, dtype=float),
        tip_error_cm=_tip_errors_cm(target_kvs, robot_kvs),
        backend="adaptive_analytical",
    )


class _LatestSnapshot:
    """Thread-safe holder for the most recent diagnostic snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap: _Snapshot | None = None

    def set(self, snap: _Snapshot) -> None:
        with self._lock:
            self._snap = snap

    def get(self) -> _Snapshot | None:
        with self._lock:
            return self._snap


def _retargeter_loop(
    landmarks_q: queue.Queue[HandLandmarks],
    stop_event: threading.Event,
    snapshot_holder: _LatestSnapshot,
    hp_holder: _HyperParamHolder,
    model_path: str | None,
    urdf_path: str | None,
    backend: RetargeterBackend = "adaptive_analytical",
    retargeter_config_path: str | None = None,
) -> None:
    try:
        if backend == "rmsprop":
            retargeter = Retargeter.from_paths(
                model_path,
                urdf_path,
                backend="rmsprop",
                ik_loss=_make_dynamic_loss(hp_holder),
            )
        else:
            retargeter = Retargeter.from_paths(
                model_path,
                urdf_path,
                backend="adaptive_analytical",
                config_path=retargeter_config_path,
            )
    except Exception:
        logger.exception("Retargeter init failed; diagnostic loop exiting.")
        stop_event.set()
        return

    logger.info("Diagnostic retargeter ready (backend=%s)", backend)

    abd_indices: list[int] = []
    if backend == "rmsprop":
        # Patch _optimize so the number of IK steps per frame is read live from
        # the hyperparam holder. The Retargeter calls ``self._optimize(target_kvs)``
        # internally; assigning a plain function as an instance attribute shadows
        # the bound method without re-binding ``self``.
        _orig_optimize = retargeter._optimize

        def _patched_optimize(target_kvs: torch.Tensor) -> np.ndarray:
            return _orig_optimize(target_kvs, n_steps=int(hp_holder.get().n_steps))

        retargeter._optimize = _patched_optimize  # type: ignore[method-assign]

        abd_indices = [
            retargeter.config.finger_joint_ids.index(jid)
            for jid in _ABD_JOINT_NAMES
            if jid in retargeter.config.finger_joint_ids
        ]
        if not abd_indices:
            logger.warning(
                "No abductor joints found in finger_joint_ids; the abd-reg slider "
                "will have no effect on this hand model."
            )

    frames = 0
    while not stop_event.is_set():
        try:
            item = landmarks_q.get(timeout=HEARTBEAT_INTERVAL)
        except queue.Empty:
            continue
        if not isinstance(item, HandLandmarks):
            continue

        if backend == "rmsprop":
            # Sync live-tunable hyperparameters into the retargeter before each
            # optimizer pass. lr/regularizer-weight live on mutable state we can
            # poke directly; the loss coefficients are read from the holder inside
            # the dynamic IK loss closure, so they need no per-frame sync here.
            hp = hp_holder.get()
            for pg in retargeter._optimizer.param_groups:
                pg["lr"] = float(hp.lr)
            if abd_indices:
                with torch.no_grad():
                    retargeter._regularizer_weights[abd_indices] = float(hp.abd_reg_weight)

        try:
            target_pose = TargetPose(joint_positions=item.keypoints, source="mediapipe")
            retargeter.retarget(target_pose)
        except (AssertionError, ValueError):
            logger.debug("Skipping degenerate landmark frame.")
            continue

        if backend == "rmsprop":
            snap = _compute_rmsprop_diagnostic_snapshot(retargeter, target_pose)
        else:
            snap = _compute_adaptive_diagnostic_snapshot(retargeter, target_pose)
        if snap is not None:
            snapshot_holder.set(snap)
            frames += 1
            if frames == 1 or frames % 60 == 0:
                mean_err = float(np.mean(snap.tip_error_cm))
                per_finger = ", ".join(
                    f"{f}={e:.1f}" for f, e in zip(FINGERS, snap.tip_error_cm, strict=True)
                )
                logger.info(
                    "Tip fit residual mean=%.1f cm [%s] (frame %d)",
                    mean_err,
                    per_finger,
                    frames,
                )


@dataclass
class _PlotArtists:
    bone_lines: dict[str, Line3D]
    target_kv_lines: dict[str, Line3D]
    robot_kv_lines: dict[str, Line3D]


def _make_figure() -> tuple[plt.Figure, _PlotArtists]:
    fig = plt.figure(figsize=(11, 5))
    fig.suptitle(
        "Retargeter diagnostics  —  left: landmarks  |  "
        "right: tip vectors (dashed = target, solid = URDF)  |  "
        "calibrating…"
    )

    ax_landmarks = fig.add_subplot(1, 2, 1, projection="3d")
    ax_kvs = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, title in (
        (ax_landmarks, "hand landmarks (normalized)"),
        (ax_kvs, "palm-to-tip key vectors"),
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax_landmarks.set_xlim(-_LANDMARK_LIM, _LANDMARK_LIM)
    ax_landmarks.set_ylim(-_LANDMARK_LIM, _LANDMARK_LIM)
    ax_landmarks.set_zlim(-_LANDMARK_LIM, _LANDMARK_LIM)
    ax_kvs.set_xlim(-_KV_LIM, _KV_LIM)
    ax_kvs.set_ylim(-_KV_LIM, _KV_LIM)
    ax_kvs.set_zlim(-_KV_LIM, _KV_LIM)

    bone_lines: dict[str, Line3D] = {}
    target_kv_lines: dict[str, Line3D] = {}
    robot_kv_lines: dict[str, Line3D] = {}

    for finger in FINGERS:
        color = _FINGER_COLORS[finger]
        (bone,) = ax_landmarks.plot(
            [],
            [],
            [],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=finger,
        )
        bone_lines[finger] = bone

        (lt,) = ax_kvs.plot(
            [],
            [],
            [],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{finger} target",
        )
        (lr,) = ax_kvs.plot(
            [],
            [],
            [],
            color=color,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
        target_kv_lines[finger] = lt
        robot_kv_lines[finger] = lr

    ax_landmarks.legend(loc="upper left", fontsize="x-small")
    fig.tight_layout()
    return fig, _PlotArtists(bone_lines, target_kv_lines, robot_kv_lines)


def _animate(
    _frame: int,
    holder: _LatestSnapshot,
    artists: _PlotArtists,
    fig: plt.Figure,
) -> list[Line3D]:
    flat: list[Line3D] = (
        list(artists.bone_lines.values())
        + list(artists.target_kv_lines.values())
        + list(artists.robot_kv_lines.values())
    )
    snap = holder.get()
    if snap is None:
        return flat

    landmarks = snap.landmarks
    for finger, indices in _FINGER_BONE_INDICES.items():
        pts = landmarks[indices]  # (5, 3)
        artists.bone_lines[finger].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])

    origin = np.zeros(3)
    for i, finger in enumerate(FINGERS):
        tgt = snap.target_kvs[i]
        rob = snap.robot_kvs[i]
        artists.target_kv_lines[finger].set_data_3d(
            [origin[0], tgt[0]], [origin[1], tgt[1]], [origin[2], tgt[2]]
        )
        artists.robot_kv_lines[finger].set_data_3d(
            [origin[0], rob[0]], [origin[1], rob[1]], [origin[2], rob[2]]
        )

    mean_err = float(np.mean(snap.tip_error_cm))
    per_finger = "  ".join(
        f"{f[0].upper()}={e:.1f}" for f, e in zip(FINGERS, snap.tip_error_cm, strict=True)
    )
    # Rough guide: <1.5 cm mean tip residual ⇒ IK is fitting; larger means
    # retargeter can't realize the landmark tip targets on this URDF.
    health = "IK healthy" if mean_err < 1.5 else "IK struggling"
    fig.suptitle(
        f"{snap.backend}  —  tip residual mean {mean_err:.1f} cm ({health})  |  "
        f"{per_finger}  |  dashed=target  solid=URDF"
    )
    return flat


def _make_slider_figure(hp_holder: _HyperParamHolder) -> plt.Figure:
    """Build a separate matplotlib window with sliders for the hyperparameters.

    Slider callbacks run on the main GUI thread; the retargeter loop reads
    from ``hp_holder`` on its worker thread, so the lock inside the holder is
    the only synchronization needed.
    """
    fig = plt.figure(figsize=(5.5, 6.8))
    try:
        fig.canvas.manager.set_window_title("Retargeter hyperparameters")
    except Exception:
        pass
    fig.suptitle("Retargeter hyperparameters", fontsize=11)

    # (label, vmin, vmax, vinit, valstep, valfmt)
    rows: list[tuple[str, float, float, float, float | None, str]] = [
        ("lr", 0.1, 100.0, _DEFAULT_LR, None, "%.2f"),
        ("n_steps", 1, 8, _DEFAULT_N_STEPS, 1, "%d"),
        ("coef thumb", 0.0, 5.0, _DEFAULT_FINGER_COEFFS[0], None, "%.2f"),
        ("coef index", 0.0, 5.0, _DEFAULT_FINGER_COEFFS[1], None, "%.2f"),
        ("coef middle", 0.0, 5.0, _DEFAULT_FINGER_COEFFS[2], None, "%.2f"),
        ("coef ring", 0.0, 5.0, _DEFAULT_FINGER_COEFFS[3], None, "%.2f"),
        ("coef pinky", 0.0, 5.0, _DEFAULT_FINGER_COEFFS[4], None, "%.2f"),
        ("abd reg log10", -8.0, -2.0, float(np.log10(_DEFAULT_ABD_REG_WEIGHT)), None, "%.1f"),
    ]
    n = len(rows)
    top = 0.92
    bottom = 0.06
    row_h = (top - bottom) / n

    sliders: list[Slider] = []
    for i, (label, vmin, vmax, vinit, vstep, vfmt) in enumerate(rows):
        ax = fig.add_axes([0.34, top - (i + 1) * row_h + 0.012, 0.58, row_h * 0.55])
        kwargs: dict = {"valinit": vinit, "valfmt": vfmt}
        if vstep is not None:
            kwargs["valstep"] = vstep
        sliders.append(Slider(ax, label, vmin, vmax, **kwargs))

    s_lr, s_steps, s_th, s_ix, s_md, s_rg, s_pk, s_abd = sliders

    def _on_change(_val: float) -> None:
        hp_holder.set(
            _HyperParams(
                lr=float(s_lr.val),
                n_steps=int(s_steps.val),
                finger_coeffs=(
                    float(s_th.val),
                    float(s_ix.val),
                    float(s_md.val),
                    float(s_rg.val),
                    float(s_pk.val),
                ),
                abd_reg_weight=float(10.0**s_abd.val),
            )
        )

    for s in sliders:
        s.on_changed(_on_change)

    # Keep slider refs alive for the lifetime of the figure (otherwise GC).
    fig._diagnostic_sliders = sliders  # type: ignore[attr-defined]
    return fig


def _start_mediapipe_publisher(
    port: int, handedness: str, confidence: float, show_video: bool
) -> multiprocessing.Process:
    """Spawn the MediaPipe publisher in a child process.

    Reuses ``pipeline._mediapipe_publisher``, which already handles waiting
    for the gRPC server to come up before connecting. When ``show_video`` is
    set the publisher pops its own OpenCV window with the live MediaPipe
    landmark overlay.
    """
    from orca_teleop.pipeline import _mediapipe_publisher

    proc = multiprocessing.Process(
        target=_mediapipe_publisher,
        args=(port, handedness, confidence, show_video),
        name="mediapipe-publisher",
        daemon=True,
    )
    proc.start()
    logger.info("Local MediaPipe publisher started (pid=%d, hand=%s)", proc.pid, handedness)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="OrcaHand model directory")
    parser.add_argument("--urdf-path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--retargeter",
        default=None,
        choices=sorted(DEFAULT_CONFIGS),
        help="Retargeter, with its webcam config (default: the config's type)",
    )
    parser.add_argument(
        "--retargeter-config",
        default=None,
        help="Retargeter config instead of the default; its `type:` names the retargeter",
    )
    parser.add_argument(
        "--with-mediapipe",
        action="store_true",
        help="Spawn a local MediaPipe publisher in a child process.",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Handedness for the spawned MediaPipe publisher (default: right)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="MediaPipe confidence (default: 0.7)",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Pop the MediaPipe webcam window (only with --with-mediapipe).",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    try:
        args.retargeter, args.retargeter_config = resolve(args.retargeter, args.retargeter_config)
    except (OSError, ValueError) as err:
        parser.error(str(err))
    args.retargeter_config = str(args.retargeter_config)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    landmarks_q: queue.Queue[HandLandmarks] = queue.Queue(maxsize=QUEUES_MAXSIZE)
    stop_event = threading.Event()
    snapshot_holder = _LatestSnapshot()
    hp_holder = _HyperParamHolder(_default_hyperparams())

    server = IngressServer(landmarks_q, stop_event, port=args.port)
    server.start()

    worker = threading.Thread(
        target=_retargeter_loop,
        args=(
            landmarks_q,
            stop_event,
            snapshot_holder,
            hp_holder,
            args.model_path,
            args.urdf_path,
            args.retargeter,
            args.retargeter_config,
        ),
        name="retargeter-diagnostic",
        daemon=True,
    )
    worker.start()

    publisher_proc: multiprocessing.Process | None = None
    if args.with_mediapipe:
        publisher_proc = _start_mediapipe_publisher(
            args.port, args.hand, args.confidence, args.show_video
        )
    elif args.show_video:
        logger.warning("--show-video has no effect without --with-mediapipe; ignoring.")

    fig, artists = _make_figure()
    # Live hyperparam sliders only apply to the RMSprop optimizer.
    _slider_fig = _make_slider_figure(hp_holder) if args.retargeter == "rmsprop" else None
    _anim = FuncAnimation(
        fig,
        _animate,
        fargs=(snapshot_holder, artists, fig),
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        stop_event.set()
        server.stop()
        worker.join(timeout=2.0)
        if publisher_proc is not None and publisher_proc.is_alive():
            publisher_proc.terminate()
            publisher_proc.join(timeout=3.0)
        del _slider_fig, _anim


if __name__ == "__main__":
    main()
