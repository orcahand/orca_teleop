"""Orca-native port of Wuji's adaptive analytical retargeting strategy."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import WRIST_MOTOR_IDX
from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.constants import (
    CALIBRATION_FRAMES,
    MANO_TO_URDF_TRANSLATION,
    TIP_LANDMARK_OFFSET_M,
)
from orca_teleop.retargeting.retargeter import FINGERS, TargetPose
from orca_teleop.utils import default_urdf_path, infer_model_version

M_TO_CM = 100.0
MANO_OFFSET_CM = np.array([0.0, 0.0, 1.5], dtype=np.float64)
logger = logging.getLogger(__name__)

_MP_TIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
_MP_PIP_INDICES = np.array([2, 6, 10, 14, 18], dtype=np.int64)
_MP_DIP_INDICES = np.array([3, 7, 11, 15, 19], dtype=np.int64)
# The landmarks matched to the robot's (pip, dip) frames and the start of each tip
# direction. "legacy" matches the fingers' PIP/DIP landmarks to the robot's MCP/PIP
# joints, one joint too far out. "anatomical" matches MCP/PIP landmarks to MCP/PIP
# joints, and points the robot's last segment (PIP joint -> tip) like the human's
# PIP -> TIP, since that one rigid segment spans the human's last two phalanges.
_LANDMARK_MAPS = {
    "legacy": (_MP_PIP_INDICES, _MP_DIP_INDICES, _MP_DIP_INDICES),
    "anatomical": (
        np.array([2, 5, 9, 13, 17], dtype=np.int64),
        np.array([3, 6, 10, 14, 18], dtype=np.int64),
        np.array([3, 6, 10, 14, 18], dtype=np.int64),
    ),
}
_MP_THUMB_CMC = 1
# Thumb segments whose direction can be matched: CMC->MCP (metacarpal) and MCP->IP (proximal).
_MP_THUMB_SEGMENTS = np.array([[1, 2], [2, 3]], dtype=np.int64)
_MAX_PINCH_ALPHA = 0.7

_OPERATOR2MANO_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_OPERATOR2MANO_LEFT = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _optional_imports():
    try:
        import nlopt
        import pinocchio as pin
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "backend='adaptive_analytical' requires optional dependencies. Install the "
            "project with its adaptive extra, e.g. `uv sync --extra adaptive`."
        ) from exc
    return nlopt, pin, yaml


def _default_config_path() -> Path:
    return (
        resources.files("orca_teleop.retargeting")
        / "configs"
        / "adaptive_analytical_orca_tuned.yaml"
    )


def _huber_loss(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))


def _huber_grad(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, x, delta * np.sign(x))


def _direction_terms(
    vecs: np.ndarray, j_vecs: np.ndarray, targets: np.ndarray, weights: np.ndarray, delta: float
) -> tuple[float, np.ndarray]:
    """Loss and gradient for pointing robot vectors like unit ``targets``.

    ``vecs`` (k, 3) in cm with Jacobians ``j_vecs`` (k, 3, n); each term is
    ``weight * huber(|unit(vec) - target|)``.
    """
    norms = np.linalg.norm(vecs, axis=1) + 1e-8
    units = vecs / norms[:, None]
    diffs = units - targets
    dists = np.linalg.norm(diffs, axis=1)
    along = diffs / (dists[:, None] + 1e-8)
    # d|unit - target|/dvec = along @ (I - unit unit^T) / norm
    through = (along - np.sum(along * units, axis=1, keepdims=True) * units) / norms[:, None]
    coeff = weights * _huber_grad(dists, delta)
    grad = np.einsum("k,ki,kij->j", coeff, through, j_vecs)
    return float(np.sum(weights * _huber_loss(dists, delta))), grad


def _pinch_targets(
    human_vectors_cm: np.ndarray, contact_cm: float, near_cm: float, far_cm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Thumb-pad to finger-pad target vectors and weights for the four fingers.

    The human's tip landmarks sit inside the fingers, so at contact they are still
    ``contact_cm`` apart: the target gap is the human's minus that, never below zero,
    so a human pinch closes the robot's pads. The term weighs 1 up to ``near_cm`` and
    fades out by ``far_cm``; further apart, shape terms alone place the fingers.
    """
    gaps = np.linalg.norm(human_vectors_cm, axis=1)
    target_gap = np.maximum(gaps - contact_cm, 0.0)
    targets = human_vectors_cm / (gaps[:, None] + 1e-8) * target_gap[:, None]
    weights = np.clip((far_cm - gaps) / max(far_cm - near_cm, 1e-8), 0.0, 1.0)
    return targets, weights


def _estimate_mediapipe_frame(keypoints: np.ndarray) -> np.ndarray:
    if keypoints.shape != (21, 3):
        raise ValueError(f"Expected MediaPipe keypoints with shape (21, 3), got {keypoints.shape}")

    points = keypoints[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]

    x_axis = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Degenerate MediaPipe hand frame: wrist-middle axis collapsed")
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Degenerate MediaPipe hand frame: palm normal collapsed")
    z_axis = z_axis / z_norm

    if np.sum(z_axis * (centered[1] - centered[2])) < 0:
        normal *= -1.0
        z_axis *= -1.0
    return np.stack([x_axis, normal, z_axis], axis=1)


def _transform_mediapipe_keypoints(keypoints: np.ndarray, hand_type: str) -> np.ndarray:
    centered = np.asarray(keypoints, dtype=np.float64) - keypoints[0:1, :]
    wrist_frame = _estimate_mediapipe_frame(centered)
    operator2mano = _OPERATOR2MANO_RIGHT if hand_type == "right" else _OPERATOR2MANO_LEFT
    return centered @ wrist_frame @ operator2mano


def _rotation_matrix_xyz(rotation_degrees: dict[str, float]) -> np.ndarray:
    rx = math.radians(float(rotation_degrees.get("x", 0.0)))
    ry = math.radians(float(rotation_degrees.get("y", 0.0)))
    rz = math.radians(float(rotation_degrees.get("z", 0.0)))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _thumb_shape_from_base(
    full_hand_vectors: np.ndarray,
    keypoints: np.ndarray,
    thumb_scaling: np.ndarray,
    robot_thumb_base_cm: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Re-anchor the thumb's shape targets (MCP, IP, tip) at the thumb base.

    The full-hand targets are measured from the palm point. The robot's thumb
    base sits about 2 cm from where the human thumb base lands, so palm-measured
    thumb targets make the solver over-bend the thumb MCP and bend the IP
    backwards. Here each thumb target is the human offset from the thumb CMC
    landmark, applied at the robot's thumb base (``robot_thumb_base_cm``, from
    the robot palm point). That shifts the whole thumb by the same ~2 cm, which
    breaks pinches, so ``weight`` blends between the palm-measured target (0)
    and the base-measured one (1).
    """
    vectors = full_hand_vectors.copy()
    landmarks = (_MP_PIP_INDICES[0], _MP_DIP_INDICES[0], _MP_TIP_INDICES[0])
    for row, landmark, scale in zip((0, 5, 10), landmarks, thumb_scaling, strict=True):
        offset = (keypoints[landmark] - keypoints[_MP_THUMB_CMC]) * scale * M_TO_CM
        vectors[row] += weight * (offset + robot_thumb_base_cm - vectors[row])
    return vectors


def _squeeze_pinches(
    keypoints: np.ndarray, squeeze_cm: float, full_below_cm: float, start_below_cm: float
) -> np.ndarray:
    """Pull the thumb tip and a fingertip together when the human pinches them.

    Copying the human's tip-landmark gap doesn't make the robot's fingertips
    touch: the landmarks sit inside the fingers, and where the robot's surfaces
    meet depends on how its fingertips turn. Below ``start_below_cm`` the gap
    shrinks by up to ``squeeze_cm`` (all of it from ``full_below_cm`` down),
    split between both tips and never past zero.
    """
    out = keypoints.copy()
    thumb = keypoints[_MP_TIP_INDICES[0]]
    for tip in _MP_TIP_INDICES[1:]:
        offset = keypoints[tip] - thumb
        gap_cm = float(np.linalg.norm(offset)) * M_TO_CM
        ramp = (start_below_cm - gap_cm) / (start_below_cm - full_below_cm)
        shift_cm = min(squeeze_cm * float(np.clip(ramp, 0.0, 1.0)), gap_cm)
        if shift_cm <= 0.0:
            continue
        step = offset / np.linalg.norm(offset) * (0.5 * shift_cm / M_TO_CM)
        out[_MP_TIP_INDICES[0]] += step
        out[tip] -= step
    return out


@dataclass(frozen=True)
class _FrameMap:
    palm: str
    base: list[str]
    pip: list[str]
    dip: list[str]
    tip: list[str]
    # Point matched to the human TIP landmark, in each tip frame (m).
    tip_offset: list[np.ndarray]
    pad: list[str] | None  # where pinches touch; the pinch term needs them


class _OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) on joint targets: smooth when still, quick when moving.

    The cutoff rises from ``min_cutoff_hz`` with the joint's (smoothed) speed in rad/s
    times ``beta``. Frames are taken ``1 / rate_hz`` apart.
    """

    def __init__(
        self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0, rate_hz: float = 30.0
    ) -> None:
        if min_cutoff_hz <= 0 or d_cutoff_hz <= 0 or rate_hz <= 0 or beta < 0:
            raise ValueError("one_euro needs positive cutoffs and rate, and beta >= 0")
        self.min_cutoff_hz, self.beta, self.d_cutoff_hz = min_cutoff_hz, beta, d_cutoff_hz
        self.dt = 1.0 / rate_hz
        self._state: np.ndarray | None = None
        self._speed: np.ndarray | None = None

    def _alpha(self, cutoff_hz: np.ndarray | float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / self.dt)

    def next(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._state is None:
            self._state, self._speed = x.copy(), np.zeros_like(x)
            return x.copy()
        speed = (x - self._state) / self.dt
        self._speed = self._speed + self._alpha(self.d_cutoff_hz) * (speed - self._speed)
        cutoff = self.min_cutoff_hz + self.beta * np.abs(self._speed)
        self._state = self._state + self._alpha(cutoff) * (x - self._state)
        return self._state.copy()

    def reset(self) -> None:
        self._state = self._speed = None


def _make_filter(cfg: dict[str, Any]) -> _LPFilter | _OneEuroFilter:
    """``filter: {type: one_euro, ...}`` or, by default, the low-pass with ``lp_alpha``."""
    spec = cfg.get("filter") or {}
    kind = spec.get("type", "lowpass")
    if kind == "one_euro":
        return _OneEuroFilter(
            float(spec.get("min_cutoff_hz", 1.0)),
            float(spec.get("beta", 0.5)),
            float(spec.get("d_cutoff_hz", 1.0)),
            float(spec.get("rate_hz", 30.0)),
        )
    if kind != "lowpass":
        raise ValueError(f"filter.type must be lowpass or one_euro, got {kind!r}")
    return _LPFilter(float(cfg.get("lp_alpha", 0.15)))


class _LPFilter:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"lp_alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    def next(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(np.float64, copy=True)
        else:
            self._state = self._state + self.alpha * (x - self._state)
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


class AdaptiveAnalyticalRetargeter:
    """Wuji-style adaptive retargeter adapted to OrcaHand's kinematics and API."""

    def __init__(
        self,
        *,
        hand_config_path: str | None,
        urdf_path: str | None,
        config_path: str | None,
    ) -> None:
        self._nlopt, self._pin, yaml = _optional_imports()

        self._hand = OrcaHand(hand_config_path)
        self.hand_type = self._hand.config.type
        if self.hand_type not in ("left", "right"):
            raise ValueError(f"hand.config.type must be 'left' or 'right', got {self.hand_type!r}")
        self.model_version = infer_model_version(getattr(self._hand.config, "model_path", None))

        if urdf_path is None:
            urdf_path = default_urdf_path(self.hand_type, self.model_version)
        self.urdf_path = str(urdf_path)

        cfg_path = Path(config_path) if config_path is not None else _default_config_path()
        with open(cfg_path) as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

        self._retarget_cfg = self.config.get("retarget", {})
        self._frame_map = self._load_frame_map(self.config)
        self._joint_aliases = (
            self.config.get("joint_name_aliases", {}).get(self.hand_type, {}) or {}
        )

        self._robot = _PinocchioOrcaRobot(self.urdf_path, self._pin)
        self._build_joint_mapping()
        self._build_frame_indices()
        self._build_retarget_params()

        self._filter = _make_filter(self._retarget_cfg)
        self._neutral_qpos_phys = self._build_neutral_qpos()
        self._build_target_alignment_params()
        self._mano_scale: float = 1.0
        self._calibration_mags: list[np.ndarray] = []
        self._calibration_done: bool = False
        self._last_qpos_phys: np.ndarray | None = None

    @classmethod
    def from_paths(
        cls,
        model_path: str | None = None,
        urdf_path: str | None = None,
        config_path: str | None = None,
    ) -> AdaptiveAnalyticalRetargeter:
        return cls(hand_config_path=model_path, urdf_path=urdf_path, config_path=config_path)

    def reset(self) -> None:
        self.reset_tracking()
        self._calibration_mags.clear()
        self._calibration_done = False
        self._mano_scale = 1.0

    def reset_tracking(self) -> None:
        """Forget the previous pose (filter state, warm start); keep the hand-size calibration."""
        self._filter.reset()
        self._last_qpos_phys = None

    def retarget(self, target_pose: TargetPose) -> OrcaJointPositions | None:
        if target_pose.source != "mediapipe":
            raise ValueError(
                "Adaptive analytical backend currently expects MediaPipe-format "
                f"(21, 3) keypoints; got source={target_pose.source!r}."
            )

        normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(
            target_pose.joint_positions,
            target_pose.source,
        )
        if self._rotation_matrix is not None:
            normalized = normalized @ self._rotation_matrix.T

        if not self._calibration_done:
            self._ingest_calibration_frame(normalized, target_pose.source)
            return None

        keypoints = self._target_keypoints_in_urdf_frame(normalized)

        qpos_phys = self._filter.next(self._solve(keypoints))
        if not np.all(np.isfinite(qpos_phys)):
            logger.warning("Retargeter produced a non-finite pose; resetting its state")
            self.reset_tracking()
            return None

        finger_phys_deg = np.rad2deg(qpos_phys)
        wrist_phys_deg = float(
            np.clip(target_pose.wrist_angle_degrees, self._wrist_lower_deg, self._wrist_upper_deg)
        )
        wrist_signed = wrist_phys_deg if self.hand_type == "left" else -wrist_phys_deg

        return OrcaJointPositions(
            {
                **dict(zip(self._finger_joint_ids, finger_phys_deg, strict=True)),
                self._wrist_joint_id: wrist_signed,
            }
        )

    def _load_frame_map(self, config: dict[str, Any]) -> _FrameMap:
        versioned_cfg = config.get("robot_frames_by_version", {}).get(self.model_version, {})
        side_cfg = versioned_cfg.get(self.hand_type)
        if side_cfg is None:
            side_cfg = config.get("robot_frames", {}).get(self.hand_type)
        if not side_cfg:
            raise ValueError(f"robot_frames.{self.hand_type} is required in retarget config")
        fingers_cfg = side_cfg.get("fingers", {})
        missing = [finger for finger in FINGERS if finger not in fingers_cfg]
        if missing:
            raise ValueError(f"Missing robot frame config for fingers: {missing}")

        def read_frame(finger: str, key: str) -> str:
            value = fingers_cfg[finger].get(key)
            if not value:
                raise ValueError(
                    f"robot_frames.{self.hand_type}.fingers.{finger}.{key} is required"
                )
            return str(value)

        def read_offset(finger: str) -> np.ndarray:
            offset = np.asarray(
                fingers_cfg[finger].get("tip_offset_m", TIP_LANDMARK_OFFSET_M), float
            )
            if offset.shape != (3,):
                raise ValueError(
                    f"robot_frames.{self.hand_type}.fingers.{finger}.tip_offset_m must be [x, y, z]"
                )
            return offset

        return _FrameMap(
            palm=str(side_cfg["palm"]),
            base=[read_frame(finger, "base") for finger in FINGERS],
            pip=[read_frame(finger, "pip") for finger in FINGERS],
            dip=[read_frame(finger, "dip") for finger in FINGERS],
            tip=[read_frame(finger, "tip") for finger in FINGERS],
            tip_offset=[read_offset(finger) for finger in FINGERS],
            pad=(
                [read_frame(finger, "pad") for finger in FINGERS]
                if all("pad" in fingers_cfg[finger] for finger in FINGERS)
                else None
            ),
        )

    def _build_joint_mapping(self) -> None:
        joint_ids = list(self._hand.config.joint_ids)
        physical_lower, physical_upper = map(
            list, zip(*self._hand.config.joint_roms_dict.values(), strict=True)
        )
        wrist_idx = joint_ids.index("wrist") if "wrist" in joint_ids else WRIST_MOTOR_IDX

        self._wrist_joint_id = joint_ids[wrist_idx]
        self._wrist_lower_deg = float(physical_lower[wrist_idx])
        self._wrist_upper_deg = float(physical_upper[wrist_idx])

        self._finger_joint_ids = [jid for i, jid in enumerate(joint_ids) if i != wrist_idx]
        lower_deg = np.array(
            [v for i, v in enumerate(physical_lower) if i != wrist_idx], dtype=np.float64
        )
        upper_deg = np.array(
            [v for i, v in enumerate(physical_upper) if i != wrist_idx], dtype=np.float64
        )
        self._lower_phys = np.deg2rad(lower_deg)
        self._upper_phys = np.deg2rad(upper_deg)

        model_joint_names = self._robot.dof_joint_names
        self._model_q_indices: list[int] = []
        missing = []
        for jid in self._finger_joint_ids:
            urdf_suffix = self._joint_aliases.get(jid, jid)
            urdf_joint_name = f"{self.hand_type}_{urdf_suffix}"
            try:
                self._model_q_indices.append(model_joint_names.index(urdf_joint_name))
            except ValueError:
                missing.append(urdf_joint_name)
        if missing:
            raise ValueError(
                "URDF is missing joints required by the Orca config: "
                f"{missing}. Check joint_name_aliases in the retarget config."
            )
        self._model_q_indices_np = np.array(self._model_q_indices, dtype=np.int64)

    def _build_frame_indices(self) -> None:
        self._base_frame_names = list(self._frame_map.base)
        names = (
            [self._frame_map.palm]
            + self._base_frame_names
            + self._frame_map.tip
            + self._frame_map.pip
            + self._frame_map.dip
            + (self._frame_map.pad or [])
        )
        self._computed_frame_names = list(dict.fromkeys(names))
        missing = [name for name in self._computed_frame_names if not self._robot.has_frame(name)]
        if missing:
            raise ValueError(
                f"{self.urdf_path} has no frames {missing}. The v2 retarget config needs "
                "orcahand_description with the <side>_<digit>_tip frames (the v2.1 model, "
                "on its main); for other models check robot_frames in the config."
            )

        self._computed_frame_indices = [
            self._robot.get_frame_index(name) for name in self._computed_frame_names
        ]
        self._palm_index = self._computed_frame_names.index(self._frame_map.palm)
        self._base_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._base_frame_names],
            dtype=np.int64,
        )
        self._thumb_base_index = int(self._base_indices[0])
        self._pinky_base_index = int(self._base_indices[-1])
        self._tip_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._frame_map.tip], dtype=np.int64
        )
        self._pip_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._frame_map.pip], dtype=np.int64
        )
        self._dip_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._frame_map.dip], dtype=np.int64
        )
        self._pad_indices = (
            np.array(
                [self._computed_frame_names.index(n) for n in self._frame_map.pad], dtype=np.int64
            )
            if self._frame_map.pad
            else self._tip_indices
        )
        self._frame_offsets = [np.zeros(3, dtype=np.float64) for _ in self._computed_frame_names]
        for offset, frame_idx in zip(self._frame_map.tip_offset, self._tip_indices, strict=True):
            self._frame_offsets[frame_idx] = offset

        # Each last segment's axis point: on the dip frame's z axis (the segment's own
        # axis), as far out as the tip point. The tip point sits on the skin, off that axis
        # (about 1 cm towards the palm on v2 fingers), so a straight robot finger "points"
        # PIP -> tip some 13 degrees palmwards; matching a straight human finger with that
        # vector bends the robot's PIP backwards. Directions use the axis point instead.
        zero = np.zeros(self._robot.nq, dtype=np.float64)
        positions, _ = self._robot.compute_positions_and_jacobians(
            zero, self._computed_frame_indices, self._frame_offsets
        )
        rotations = [
            self._robot.frame_rotation(self._computed_frame_indices[i]) for i in self._dip_indices
        ]
        self._axis_indices = []
        for rotation, dip, tip in zip(rotations, self._dip_indices, self._tip_indices, strict=True):
            reach = float((rotation.T @ (positions[tip] - positions[dip]))[2])
            self._axis_indices.append(len(self._computed_frame_indices))
            self._computed_frame_indices.append(self._computed_frame_indices[dip])
            self._computed_frame_names.append(f"{self._computed_frame_names[dip]}+axis")
            self._frame_offsets.append(np.array([0.0, 0.0, reach]))
        self._axis_indices = np.array(self._axis_indices, dtype=np.int64)

    def _build_retarget_params(self) -> None:
        cfg = self._retarget_cfg
        self._huber_delta = float(cfg.get("huber_delta_cm", 2.0))
        self._huber_delta_dir = float(cfg.get("huber_delta_dir", 0.5))
        self._norm_delta = float(cfg.get("norm_delta", 50.0))
        self._w_pos = float(cfg.get("w_pos", 1.0))
        self._w_dir = float(cfg.get("w_dir", 10.0))
        self._w_full_hand = float(cfg.get("w_full_hand", 1.0))
        self._scaling = float(cfg.get("scaling", 1.0))
        self._thumb_shape_from_base = bool(cfg.get("thumb_shape_from_base", False))

        # Per-finger tip-direction weight applied outside the pinch blend, which is
        # zero unless pinching and so leaves distal joints unconstrained in orientation.
        unblended_cfg = cfg.get("w_dir_unblended", {})
        self._w_dir_unblended = np.array(
            [float(unblended_cfg.get(finger, 0.0)) for finger in FINGERS], dtype=np.float64
        )

        segment_cfg = cfg.get("segment_scaling", {})
        self._segment_scaling = np.ones((len(FINGERS), 3), dtype=np.float64)
        for i, finger in enumerate(FINGERS):
            if finger not in segment_cfg:
                continue
            values = np.asarray(segment_cfg[finger], dtype=np.float64)
            if values.shape != (3,):
                raise ValueError(f"segment_scaling.{finger} must be a length-3 list")
            self._segment_scaling[i] = values

        # Direction weights for the thumb's metacarpal and proximal segments. Position
        # targets alone leave the base joints (cmc, abd) free to make up for the
        # robot's thumb base sitting ~2 cm from where the human's lands.
        thumb_dir_cfg = cfg.get("w_thumb_segment_dir", {})
        self._w_thumb_segment_dir = np.array(
            [float(thumb_dir_cfg.get(k, 0.0)) for k in ("metacarpal", "proximal")]
        )
        # The same for the four fingers: proximal (MCP -> PIP joint) and distal
        # (PIP joint -> tip) segments, matched to the human's MCP->PIP and PIP->TIP.
        finger_dir_cfg = cfg.get("w_finger_segment_dir", {})
        self._w_finger_segment_dir = np.array(
            [float(finger_dir_cfg.get(k, 0.0)) for k in ("proximal", "distal")]
        )

        landmarks = str(cfg.get("finger_landmarks", "legacy"))
        if landmarks not in _LANDMARK_MAPS:
            raise ValueError(f"finger_landmarks must be one of {sorted(_LANDMARK_MAPS)}")
        self._lm_pip, self._lm_dip, self._lm_dir_from = _LANDMARK_MAPS[landmarks]
        full_hand_cfg = cfg.get("full_hand_weights", [1.0, 1.0, 1.0])
        self._full_hand_weights = np.asarray(full_hand_cfg, dtype=np.float64)
        if self._full_hand_weights.shape != (3,):
            raise ValueError("full_hand_weights must be [pip, dip, tip]")

        self._distal_on_axis = str(cfg.get("distal_direction", "tip")) == "axis"

        pinch = cfg.get("pinch", {})
        self._pinch_weight = float(pinch.get("weight", 0.0))
        self._pinch_contact_cm = float(pinch.get("contact_cm", 1.5))
        self._pinch_near_cm = float(pinch.get("near_cm", 3.0))
        self._pinch_far_cm = float(pinch.get("far_cm", 5.0))
        self._pinch_delta_cm = float(pinch.get("huber_delta_cm", 1.0))

        squeeze_cfg = cfg.get("pinch_squeeze", {})
        self._pinch_squeeze_cm = float(squeeze_cfg.get("cm", 0.0))
        self._pinch_squeeze_full_below_cm = float(squeeze_cfg.get("full_below_cm", 1.5))
        self._pinch_squeeze_start_below_cm = float(squeeze_cfg.get("start_below_cm", 2.5))

        pinch_cfg = cfg.get("pinch_thresholds_cm", {})
        self._pinch_d1 = np.array(
            [pinch_cfg.get(f, {}).get("d1", 2.0) for f in FINGERS[1:]], dtype=np.float64
        )
        self._pinch_d2 = np.array(
            [pinch_cfg.get(f, {}).get("d2", 5.0) for f in FINGERS[1:]], dtype=np.float64
        )

        rotation = cfg.get("mediapipe_rotation", {})
        if any(float(rotation.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z")):
            self._rotation_matrix = _rotation_matrix_xyz(rotation)
        else:
            self._rotation_matrix = None

    def _build_target_alignment_params(self) -> None:
        """Build the same human-to-URDF frame alignment used by the RMSprop backend."""
        zero_qpos = np.zeros(self._robot.nq, dtype=np.float64)
        positions, _ = self._robot.compute_positions_and_jacobians(
            zero_qpos,
            self._computed_frame_indices,
            self._frame_offsets,
        )
        base_pos = {
            finger: positions[index]
            for finger, index in zip(FINGERS, self._base_indices, strict=True)
        }
        palm_pos = positions[self._palm_index]
        center, rot_matrix = retargeter_utils.get_hand_center_and_rotation(
            thumb_base=base_pos["thumb"],
            index_base=base_pos["index"],
            middle_base=base_pos["middle"],
            ring_base=base_pos["ring"],
            pinky_base=base_pos["pinky"],
            wrist=palm_pos,
        )
        self._urdfhand_center = np.asarray(center, dtype=np.float64).reshape(3)
        self._urdfhand_rot_matrix = np.asarray(rot_matrix, dtype=np.float64)

        neutral_positions, _ = self._robot.compute_positions_and_jacobians(
            self._full_model_qpos(self._neutral_qpos_phys),
            self._computed_frame_indices,
            self._frame_offsets,
        )
        tip_pos = neutral_positions[self._tip_indices]
        palm = self._robot_palm_from_positions(neutral_positions)
        self._robot_thumb_base_cm = (neutral_positions[self._thumb_base_index] - palm) * M_TO_CM
        urdf_keyvectors = tip_pos - palm
        self._urdf_keyvector_mags = 0.9 * np.linalg.norm(urdf_keyvectors, axis=1)

    def _ingest_calibration_frame(self, normalized_joints: np.ndarray, source: str) -> None:
        if not self._calibration_mags:
            logger.info(
                "Calibrating hand scale over the next %d frames; hold your hand open "
                "and fully in view. Nothing is retargeted until this completes.",
                CALIBRATION_FRAMES,
            )
        joint_t = torch.tensor(normalized_joints, dtype=torch.float32)
        fingertips, palm = retargeter_utils.extract_mano_fingertips_and_palm(
            joint_t,
            list(FINGERS),
            source,
        )
        keyvectors = retargeter_utils.get_keyvectors(fingertips, palm)
        mags = np.array([kv.detach().cpu().norm().item() for kv in keyvectors])
        self._calibration_mags.append(mags)

        if len(self._calibration_mags) < CALIBRATION_FRAMES:
            return

        all_mags = np.array(self._calibration_mags)
        median_mano = np.median(all_mags, axis=0)
        ratios = self._urdf_keyvector_mags / np.clip(median_mano, 1e-6, None)
        self._mano_scale = float(np.median(ratios))
        self._calibration_done = True
        self._last_qpos_phys = None
        self._filter.reset()
        logger.info(
            "Adaptive retargeter auto-scale calibrated: mano_scale=%.4f (per-finger ratios=%s)",
            self._mano_scale,
            np.round(ratios, 3).tolist(),
        )

    def _target_keypoints_in_urdf_frame(self, normalized_joints: np.ndarray) -> np.ndarray:
        scaled = normalized_joints * self._mano_scale
        keypoints = (
            scaled @ self._urdfhand_rot_matrix.T
            + self._urdfhand_center
            + np.array(MANO_TO_URDF_TRANSLATION)
        )
        if self._pinch_squeeze_cm > 0.0:
            keypoints = _squeeze_pinches(
                keypoints,
                self._pinch_squeeze_cm,
                self._pinch_squeeze_full_below_cm,
                self._pinch_squeeze_start_below_cm,
            )
        return keypoints

    def _solve(self, keypoints: np.ndarray) -> np.ndarray:
        init_qpos = self._initial_qpos()
        reg_qpos = self._last_qpos_phys

        alphas = self._compute_pinch_alpha(keypoints)
        target_tip_vectors = self._compute_tip_vectors(keypoints)
        target_tip_dirs = self._compute_tip_dirs(keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(keypoints)
        thumb_vectors = keypoints[_MP_THUMB_SEGMENTS[:, 1]] - keypoints[_MP_THUMB_SEGMENTS[:, 0]]
        target_thumb_dirs = thumb_vectors / (
            np.linalg.norm(thumb_vectors, axis=1, keepdims=True) + 1e-8
        )
        finger_vectors = np.stack(
            [
                keypoints[self._lm_dip[1:]] - keypoints[self._lm_pip[1:]],  # proximal
                keypoints[_MP_TIP_INDICES[1:]] - keypoints[self._lm_dip[1:]],  # distal
            ]
        )
        target_finger_dirs = finger_vectors / (
            np.linalg.norm(finger_vectors, axis=2, keepdims=True) + 1e-8
        )
        human_pinch = (
            (keypoints[_MP_TIP_INDICES[1:]] - keypoints[_MP_TIP_INDICES[0]])
            * self._scaling
            * M_TO_CM
        )
        pinch_targets, pinch_weights = _pinch_targets(
            human_pinch, self._pinch_contact_cm, self._pinch_near_cm, self._pinch_far_cm
        )

        opt = self._nlopt.opt(self._nlopt.LD_SLSQP, len(self._finger_joint_ids))
        opt.set_maxeval(int(self._retarget_cfg.get("maxeval", 50)))
        opt.set_ftol_abs(float(self._retarget_cfg.get("ftol_abs", 1e-4)))
        opt.set_lower_bounds(self._lower_phys.tolist())
        opt.set_upper_bounds(self._upper_phys.tolist())

        def objective(x, grad_out):
            loss, grad = self._loss_and_grad(
                np.asarray(x, dtype=np.float64),
                target_tip_vectors,
                target_tip_dirs,
                target_full_hand_vectors,
                target_thumb_dirs,
                alphas,
                reg_qpos,
                target_finger_dirs,
                pinch_targets,
                pinch_weights * self._pinch_weight,
            )
            if grad_out.size > 0:
                grad_out[:] = grad
            return float(loss)

        opt.set_min_objective(objective)
        try:
            result = np.asarray(opt.optimize(init_qpos.tolist()), dtype=np.float64)
        except (RuntimeError, self._nlopt.RoundoffLimited, self._nlopt.ForcedStop):
            result = init_qpos  # nlopt's own errors aren't RuntimeErrors; keep the last pose
        result = np.clip(result, self._lower_phys, self._upper_phys)
        self._last_qpos_phys = result.copy()
        return result

    def _initial_qpos(self) -> np.ndarray:
        if self._last_qpos_phys is not None:
            init = self._last_qpos_phys
        else:
            init = self._neutral_qpos_phys
        return np.clip(init, self._lower_phys, self._upper_phys)

    def _build_neutral_qpos(self) -> np.ndarray:
        neutral_deg = np.array(
            [
                float(self._hand.config.neutral_position.get(jid, 0.0))
                for jid in self._finger_joint_ids
            ],
            dtype=np.float64,
        )
        neutral_rad = np.deg2rad(neutral_deg)
        return np.clip(neutral_rad, self._lower_phys, self._upper_phys)

    def _full_model_qpos(self, qpos_phys: np.ndarray) -> np.ndarray:
        qpos = np.zeros(self._robot.nq, dtype=np.float64)
        qpos[self._model_q_indices_np] = qpos_phys
        return qpos

    def _compute_pinch_alpha(self, keypoints: np.ndarray) -> np.ndarray:
        thumb_tip = keypoints[_MP_TIP_INDICES[0]]
        finger_tips = keypoints[_MP_TIP_INDICES[1:]]
        distances = np.linalg.norm(finger_tips - thumb_tip, axis=1) * M_TO_CM
        alphas_4 = np.clip(
            (self._pinch_d2 - distances) / (self._pinch_d2 - self._pinch_d1 + 1e-8),
            0.0,
            _MAX_PINCH_ALPHA,
        )
        return np.concatenate([[float(np.max(alphas_4))], alphas_4])

    def _target_palm(self, keypoints: np.ndarray) -> np.ndarray:
        return 0.5 * (keypoints[1] + keypoints[17])

    def _compute_tip_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        palm = self._target_palm(keypoints)
        return ((keypoints[_MP_TIP_INDICES] - palm) * self._scaling * M_TO_CM).astype(np.float64)

    def _compute_tip_dirs(self, keypoints: np.ndarray) -> np.ndarray:
        vectors = keypoints[_MP_TIP_INDICES] - keypoints[self._lm_dir_from]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def _compute_full_hand_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        palm = self._target_palm(keypoints)
        pip_vectors = (keypoints[self._lm_pip] - palm) * self._segment_scaling[:, 0:1]
        dip_vectors = (keypoints[self._lm_dip] - palm) * self._segment_scaling[:, 1:2]
        tip_vectors = (keypoints[_MP_TIP_INDICES] - palm) * self._segment_scaling[:, 2:3]
        vectors = np.vstack([pip_vectors, dip_vectors, tip_vectors]) * M_TO_CM
        if self._thumb_shape_from_base:
            # Fade back to the palm-measured targets as the thumb nears a finger, so
            # pinches land as before.
            weight = 1.0 - self._compute_pinch_alpha(keypoints)[0] / _MAX_PINCH_ALPHA
            vectors = _thumb_shape_from_base(
                vectors, keypoints, self._segment_scaling[0], self._robot_thumb_base_cm, weight
            )
        return vectors

    def _robot_palm_from_positions(self, positions_m: np.ndarray) -> np.ndarray:
        return (
            0.5 * (positions_m[self._thumb_base_index] + positions_m[self._pinky_base_index])
            - MANO_OFFSET_CM / M_TO_CM
        )

    def _robot_palm_from_positions_cm(self, positions_cm: np.ndarray) -> np.ndarray:
        return (
            0.5 * (positions_cm[self._thumb_base_index] + positions_cm[self._pinky_base_index])
            - MANO_OFFSET_CM
        )

    def _loss_and_grad(
        self,
        qpos_phys: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        target_thumb_dirs: np.ndarray,
        alphas: np.ndarray,
        reg_qpos: np.ndarray | None,
        target_finger_dirs: np.ndarray | None = None,
        pinch_targets: np.ndarray | None = None,
        pinch_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        qpos_full = self._full_model_qpos(qpos_phys)
        positions, jacobians = self._robot.compute_positions_and_jacobians(
            qpos_full,
            self._computed_frame_indices,
            self._frame_offsets,
        )
        positions_cm = positions * M_TO_CM
        jacobians_cm = jacobians[:, :, self._model_q_indices_np] * M_TO_CM

        palm_pos = self._robot_palm_from_positions_cm(positions_cm)
        tip_pos = positions_cm[self._tip_indices]
        pip_pos = positions_cm[self._pip_indices]
        dip_pos = positions_cm[self._dip_indices]

        j_palm = 0.5 * (jacobians_cm[self._thumb_base_index] + jacobians_cm[self._pinky_base_index])
        j_tip = jacobians_cm[self._tip_indices]
        j_pip = jacobians_cm[self._pip_indices]
        j_dip = jacobians_cm[self._dip_indices]

        total_grad = np.zeros_like(qpos_phys, dtype=np.float64)

        robot_tip_vec = tip_pos - palm_pos
        diff_pos = robot_tip_vec - target_tip_vectors
        dist_pos = np.linalg.norm(diff_pos, axis=1)
        loss_tip_pos = _huber_loss(dist_pos, self._huber_delta)
        grad_pos = _huber_grad(dist_pos, self._huber_delta)
        diff_normed_pos = diff_pos / (dist_pos[:, None] + 1e-8)

        coeff = alphas * self._w_pos * grad_pos
        total_grad += np.einsum("k,ki,kij->j", coeff, diff_normed_pos, j_tip - j_palm)

        # Last segment's direction: blended with the pinch alpha, plus an unblended part.
        distal = self._axis_indices if self._distal_on_axis else self._tip_indices
        loss_tip_dir, grad_tip_dir = _direction_terms(
            positions_cm[distal] - dip_pos,
            jacobians_cm[distal] - j_dip,
            target_tip_dirs,
            alphas * self._w_dir + self._w_dir_unblended,
            self._huber_delta_dir,
        )
        total_grad += grad_tip_dir

        target_pip = target_full_hand_vectors[:5]
        target_dip = target_full_hand_vectors[5:10]
        target_tip = target_full_hand_vectors[10:15]

        diff_pip = (pip_pos - palm_pos) - target_pip
        diff_dip = (dip_pos - palm_pos) - target_dip
        diff_tip = (tip_pos - palm_pos) - target_tip

        dist_pip = np.linalg.norm(diff_pip, axis=1)
        dist_dip = np.linalg.norm(diff_dip, axis=1)
        dist_tip = np.linalg.norm(diff_tip, axis=1)

        loss_pip = _huber_loss(dist_pip, self._huber_delta)
        loss_dip = _huber_loss(dist_dip, self._huber_delta)
        loss_tip_full = _huber_loss(dist_tip, self._huber_delta)

        grad_pip = _huber_grad(dist_pip, self._huber_delta)
        grad_dip = _huber_grad(dist_dip, self._huber_delta)
        grad_tip_full = _huber_grad(dist_tip, self._huber_delta)

        diff_normed_pip = diff_pip / (dist_pip[:, None] + 1e-8)
        diff_normed_dip = diff_dip / (dist_dip[:, None] + 1e-8)
        diff_normed_tip = diff_tip / (dist_tip[:, None] + 1e-8)

        w_pip, w_dip, w_tip = self._full_hand_weights
        loss_full_hand = (w_pip * loss_pip + w_dip * loss_dip + w_tip * loss_tip_full) / 3.0
        grad_coeff = (1.0 - alphas) * self._w_full_hand / 3.0
        total_grad += np.einsum(
            "k,ki,kij->j", grad_coeff * w_pip * grad_pip, diff_normed_pip, j_pip - j_palm
        )
        total_grad += np.einsum(
            "k,ki,kij->j", grad_coeff * w_dip * grad_dip, diff_normed_dip, j_dip - j_palm
        )
        total_grad += np.einsum(
            "k,ki,kij->j", grad_coeff * w_tip * grad_tip_full, diff_normed_tip, j_tip - j_palm
        )

        loss_full = self._w_full_hand * loss_full_hand
        total_loss = float(np.sum(alphas * self._w_pos * loss_tip_pos + (1.0 - alphas) * loss_full))
        total_loss += loss_tip_dir

        thumb_segments = (
            (self._thumb_base_index, self._pip_indices[0]),
            (self._pip_indices[0], self._dip_indices[0]),
        )
        # Segment directions: the thumb's metacarpal and proximal, the fingers' proximal
        # and distal. Each gives way to a pinch, like the full-hand targets.
        starts, ends, targets, weights = [], [], [], []
        for weight, (start, end), target in zip(
            self._w_thumb_segment_dir, thumb_segments, target_thumb_dirs, strict=True
        ):
            if weight != 0.0:
                starts.append(start), ends.append(end), targets.append(target)
                weights.append(weight * (1.0 - alphas[0]))
        if target_finger_dirs is not None:
            finger_segments = (
                (self._pip_indices[1:], self._dip_indices[1:]),
                (self._dip_indices[1:], distal[1:]),
            )
            for weight, (seg_starts, seg_ends), seg_targets in zip(
                self._w_finger_segment_dir, finger_segments, target_finger_dirs, strict=True
            ):
                if weight != 0.0:
                    starts.extend(seg_starts), ends.extend(seg_ends), targets.extend(seg_targets)
                    weights.extend(weight * (1.0 - alphas[1:]))
        if starts:
            loss, grad = _direction_terms(
                positions_cm[ends] - positions_cm[starts],
                jacobians_cm[ends] - jacobians_cm[starts],
                np.asarray(targets),
                np.asarray(weights),
                self._huber_delta_dir,
            )
            total_loss += loss
            total_grad += grad

        if pinch_weights is not None and np.any(pinch_weights > 0.0):
            on = np.flatnonzero(pinch_weights > 0.0)
            thumb_pad = self._pad_indices[0]
            pads = self._pad_indices[1:][on]
            w = pinch_weights[on]
            diff = positions_cm[pads] - positions_cm[thumb_pad] - pinch_targets[on]
            dist = np.linalg.norm(diff, axis=1)
            total_loss += float(np.sum(w * _huber_loss(dist, self._pinch_delta_cm)))
            coeff = w * _huber_grad(dist, self._pinch_delta_cm)
            total_grad += np.einsum(
                "k,ki,kij->j",
                coeff,
                diff / (dist[:, None] + 1e-8),
                jacobians_cm[pads] - jacobians_cm[thumb_pad],
            )

        if reg_qpos is not None:
            diff_reg = qpos_phys - reg_qpos
            total_loss += self._norm_delta * float(np.sum(diff_reg**2))
            total_grad += 2.0 * self._norm_delta * diff_reg

        return total_loss, total_grad


class _PinocchioOrcaRobot:
    def __init__(self, urdf_path: str, pin_module) -> None:
        self._pin = pin_module
        self.model = self._pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._plan_key: tuple | None = None
        self._plan_value: tuple | None = None
        if self.model.nq != self.model.nv:
            raise NotImplementedError("Adaptive analytical backend requires one-DoF joints only")

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def dof_joint_names(self) -> list[str]:
        return [name for i, name in enumerate(self.model.names) if self.model.nqs[i] > 0]

    def has_frame(self, name: str) -> bool:
        return self.model.getFrameId(name) < self.model.nframes

    def frame_rotation(self, frame_index: int) -> np.ndarray:
        """World rotation of a frame after the last ``compute_positions_and_jacobians``."""
        self._pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[frame_index].rotation)

    def get_frame_index(self, name: str) -> int:
        idx = self.model.getFrameId(name)
        if idx >= self.model.nframes:
            raise RuntimeError(f"Frame {name!r} not found in URDF")
        return int(idx)

    def compute_positions_and_jacobians(
        self,
        qpos: np.ndarray,
        frame_indices: list[int],
        frame_offsets: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World positions (n, 3) of the frames' points and their linear Jacobians (n, 3, nv).

        One C++ pass (``computeJointJacobians`` also runs the forward kinematics), then
        numpy: a point p fixed to joint j's subtree moves at ``v_j + w_j x p`` per unit of
        each supporting joint's velocity (``data.J`` is in the world frame). Assumes each
        joint's columns of ``data.J`` are its own motion: no mimic joints in the model.
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        self._pin.computeJointJacobians(self.model, self.data, qpos)
        n = len(frame_indices)
        if frame_offsets is None:
            frame_offsets = [np.zeros(3, dtype=np.float64) for _ in frame_indices]
        # Rebuilt only when the frame list or offsets change (the solver repeats them).
        key = (tuple(frame_indices), np.asarray(frame_offsets, dtype=np.float64).tobytes())
        if key != self._plan_key:
            self._plan_key, self._plan_value = key, self._plan(frame_indices, frame_offsets)
        joints, which, local, mask = self._plan_value
        rot = np.empty((len(joints), 3, 3))
        trans = np.empty((len(joints), 3))
        for k, j in enumerate(joints):
            placement = self.data.oMi[j]
            rot[k] = placement.rotation
            trans[k] = placement.translation
        positions = np.einsum("nij,nj->ni", rot[which], local) + trans[which]
        jac = np.asarray(self.data.J)  # (6, nv): linear velocity at the world origin, angular
        lin, ang = jac[:3], jac[3:]
        cross = np.empty((n, 3, lin.shape[1]))
        px, py, pz = positions[:, 0:1], positions[:, 1:2], positions[:, 2:3]
        cross[:, 0] = ang[1] * pz - ang[2] * py
        cross[:, 1] = ang[2] * px - ang[0] * pz
        cross[:, 2] = ang[0] * py - ang[1] * px
        jacobians = (lin[None] + cross) * mask[:, None, :]
        return positions, jacobians

    def _plan(self, frame_indices, frame_offsets):
        """Per frame list: parent joints, each point in its parent joint's frame, support mask."""
        parents = [int(self.model.frames[i].parentJoint) for i in frame_indices]
        joints = sorted(set(parents))
        which = np.array([joints.index(p) for p in parents], dtype=np.int64)
        local = np.empty((len(frame_indices), 3))
        mask = np.zeros((len(frame_indices), self.model.nv))
        for k, (idx, offset, parent) in enumerate(
            zip(frame_indices, frame_offsets, parents, strict=True)
        ):
            placement = self.model.frames[idx].placement
            local[k] = placement.translation + placement.rotation @ np.asarray(offset, float)
            for j in self.model.supports[parent]:
                if self.model.nvs[j] > 0:
                    start = self.model.idx_vs[j]
                    mask[k, start : start + self.model.nvs[j]] = 1.0
        return joints, which, local, mask
