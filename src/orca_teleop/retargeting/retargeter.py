import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from os import PathLike
from typing import Literal

import numpy as np
import pytorch_kinematics as pk
import torch
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import WRIST_MOTOR_IDX
from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.constants import (
    CALIBRATION_FRAMES,
    DEFAULT_LOSS_COEFFS,
    MANO_TO_URDF_TRANSLATION,
)
from orca_teleop.utils import default_urdf_path, infer_model_version

logger = logging.getLogger(__name__)

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
RetargeterBackend = Literal["rmsprop", "adaptive_analytical"]


IKLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_DEFAULT_LR: float = 5.0
_DEFAULT_JOINT_REGULARIZERS: tuple[tuple[str, float, float], ...] = (
    ("index_abd", 0.0, 1e-6),
    ("middle_abd", 0.0, 1e-6),
    ("ring_abd", 0.0, 1e-6),
    ("pinky_abd", 0.0, 1e-6),
)
# Keep the historical overall regularization magnitude after normalizing the
# per-joint weights to sum to 1.0.
_DEFAULT_REGULARIZATION_WEIGHT: float = sum(weight for _, _, weight in _DEFAULT_JOINT_REGULARIZERS)


def get_device(best: bool = False) -> str:
    if not best:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def weighted_vector_loss(
    coeffs: tuple[float, ...] = DEFAULT_LOSS_COEFFS,
) -> IKLossFn:
    """Return an IK loss function: weighted sum of squared distances between
    the five target and robot palm-to-fingertip key vectors.
    """
    coeffs_tensor: torch.Tensor | None = None

    def loss(target: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        nonlocal coeffs_tensor
        if coeffs_tensor is None or coeffs_tensor.device != target.device:
            coeffs_tensor = torch.tensor(coeffs, dtype=target.dtype, device=target.device)
        diffs_sq = torch.norm(target - robot, dim=-1) ** 2  # (5,)
        return (coeffs_tensor * diffs_sq).sum()

    return loss


def _normalize_regularizer_weights(weights: torch.Tensor) -> torch.Tensor:
    """Normalize non-negative regularizer weights to sum to 1 when present."""
    if torch.any(weights < 0):
        raise ValueError("joint_regularizers weights must be non-negative")

    total = weights.sum()
    if total.item() == 0.0:
        return weights
    return weights / total


@dataclass(frozen=True)
class TargetPose:
    """Raw hand-pose input to the retargeter.

    The retargeter is responsible for normalizing the joints into the URDF
    base frame, applying the auto-scale calibration, and computing key
    vectors — none of that can happen in the ingress layer because it
    requires URDF knowledge and stateful calibration.
    """

    joint_positions: np.ndarray
    source: str = "mediapipe"
    wrist_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        joints = np.array(self.joint_positions, dtype=float, copy=True)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError(
                f"TargetPose.joint_positions must have shape (N, 3); got {joints.shape}"
            )
        if not np.all(np.isfinite(joints)):
            raise ValueError("TargetPose.joint_positions must be finite")
        joints.setflags(write=False)
        object.__setattr__(self, "joint_positions", joints)
        object.__setattr__(self, "wrist_angle_degrees", float(self.wrist_angle_degrees))


@dataclass(frozen=True)
class RetargeterConfig:
    """Immutable configuration for a Retargeter.

    Fields without defaults are derived from the hand model and URDF and must
    be provided (use ``from_paths`` to build them automatically). Fields with
    defaults are optimizer hyperparameters that can be freely tuned.
    """

    chain: object  # pk.Chain
    hand_type: str
    finger_joint_ids: list[str]
    finger_urdf_joint_ids: list[str]
    finger_reorder_indices: list[int]
    finger_limits_lower_urdf: torch.Tensor  # degrees; the model's zero is the hand's
    finger_limits_upper_urdf: torch.Tensor
    wrist_joint_id: str
    wrist_urdf_joint_id: str
    wrist_limit_lower: float
    wrist_limit_upper: float
    optimization_frames: object
    urdfhand_center: np.ndarray  # MANO frame → URDF base frame translation
    urdfhand_rot_matrix: np.ndarray  # MANO frame → URDF base frame rotation
    fingertip_offsets: dict[str, torch.Tensor]
    urdf_keyvector_mags: np.ndarray  # neutral-pose magnitudes for autoscale
    device: str

    lr: float = _DEFAULT_LR
    ik_loss: IKLossFn = field(default_factory=weighted_vector_loss)
    regularization_weight: float = _DEFAULT_REGULARIZATION_WEIGHT
    joint_regularizers: tuple[tuple[str, float, float], ...] = field(
        default=_DEFAULT_JOINT_REGULARIZERS
    )

    @classmethod
    def from_paths(
        cls,
        hand_config_path: PathLike | None = None,
        urdf_path: PathLike | None = None,
        *,
        lr: float = _DEFAULT_LR,
        ik_loss: IKLossFn | None = None,
        regularization_weight: float = _DEFAULT_REGULARIZATION_WEIGHT,
        joint_regularizers: tuple[tuple[str, float, float], ...] = _DEFAULT_JOINT_REGULARIZERS,
    ) -> "RetargeterConfig":
        device = get_device()

        hand = OrcaHand(hand_config_path)
        hand_type = hand.config.type
        if hand_type not in ("left", "right"):
            raise ValueError(
                f"hand.config.type must be 'left' or 'right'. Check {hand_config_path}"
            )

        if urdf_path is None:
            urdf_path = default_urdf_path(hand_type, infer_model_version(hand_config_path))
        if not os.path.exists(urdf_path):
            raise ValueError(f"URDF file not found at {urdf_path}")

        import pytorch_kinematics.urdf_parser_py.xml_reflection as _xmlr
        import pytorch_kinematics.urdf_parser_py.xml_reflection.core as _urdf_core

        with open(urdf_path) as f:
            urdf_text = f.read()
        # v1 URDF uses "thumb_pip" for the joint that v2 config calls "thumb_cmc".
        urdf_text = urdf_text.replace("thumb_pip", "thumb_cmc")
        _orig_core_on_error = _urdf_core.on_error
        _orig_xmlr_on_error = _xmlr.on_error
        _urdf_core.on_error = lambda _: None
        _xmlr.on_error = lambda _: None
        try:
            chain = pk.build_chain_from_urdf(urdf_text).to(device=device)
        finally:
            _urdf_core.on_error = _orig_core_on_error
            _xmlr.on_error = _orig_xmlr_on_error

        joint_ids = list(hand.config.joint_ids)
        urdf_joint_ids = [f"{hand_type}_{jid}" for jid in joint_ids]
        physical_lower, physical_upper = map(
            list, zip(*hand.config.joint_roms_dict.values(), strict=False)
        )

        wrist_idx = joint_ids.index("wrist") if "wrist" in joint_ids else WRIST_MOTOR_IDX

        # Split wrist from finger joints — wrist is a passthrough, not optimized
        wrist_joint_id = joint_ids[wrist_idx]
        wrist_urdf_joint_id = urdf_joint_ids[wrist_idx]
        finger_joint_ids = [jid for i, jid in enumerate(joint_ids) if i != wrist_idx]
        finger_urdf_joint_ids = [jid for i, jid in enumerate(urdf_joint_ids) if i != wrist_idx]
        finger_lower_phys = np.array(
            [v for i, v in enumerate(physical_lower) if i != wrist_idx], dtype=float
        )
        finger_upper_phys = np.array(
            [v for i, v in enumerate(physical_upper) if i != wrist_idx], dtype=float
        )

        urdf_joint_names = chain.get_joint_parameter_names()
        if set(urdf_joint_ids) != set(urdf_joint_names):
            in_config_not_urdf = set(urdf_joint_ids) - set(urdf_joint_names)
            in_urdf_not_config = set(urdf_joint_names) - set(urdf_joint_ids)
            raise AssertionError(
                f"Joint name mismatch between config and URDF.\n"
                f"  In config but not URDF: {sorted(in_config_not_urdf)}\n"
                f"  In URDF but not config: {sorted(in_urdf_not_config)}\n"
                f"  URDF path: {urdf_path}"
            )
        all_reorder_indices = [urdf_joint_names.index(name) for name in urdf_joint_ids]
        finger_reorder_indices = [
            idx for i, idx in enumerate(all_reorder_indices) if i != wrist_idx
        ]

        root = torch.zeros(1, 3, device=device)
        try:
            urdfhand_center, urdfhand_rot_matrix, optimization_frames = (
                retargeter_utils.get_urdf_model_params(chain, hand_type, list(FINGERS), root)
            )
        except KeyError as err:  # a frame the model doesn't have
            raise ValueError(
                f"{urdf_path} has no frame {err}. The rmsprop retargeter needs "
                "orcahand_description with the <side>_<digit>_tip frames (the v2.1 model, "
                "on its main)."
            ) from None

        fingertip_offsets = retargeter_utils.get_fingertip_offset_tensors(list(FINGERS), device)

        # URDF key-vector magnitudes at the zero pose. Used as the target scale
        # for autocalibration: mano_scale = urdf_mag / mano_mag.
        neutral_angles = torch.zeros(chain.n_joints, device=device)
        with torch.no_grad():
            urdf_fingertips, urdf_palm = retargeter_utils.extract_orca_fingertips_and_palm(
                chain,
                neutral_angles,
                optimization_frames,
                hand_type,
                list(FINGERS),
                root,
                fingertip_offsets=fingertip_offsets,
            )
            urdf_keyvectors = retargeter_utils.get_keyvectors(urdf_fingertips, urdf_palm)
            urdf_keyvector_mags = 0.9 * np.array(
                [kv.detach().cpu().norm().item() for kv in urdf_keyvectors]
            )

        return cls(
            chain=chain,
            hand_type=hand_type,
            finger_joint_ids=finger_joint_ids,
            finger_urdf_joint_ids=finger_urdf_joint_ids,
            finger_reorder_indices=finger_reorder_indices,
            finger_limits_lower_urdf=torch.tensor(finger_lower_phys, device=device),
            finger_limits_upper_urdf=torch.tensor(finger_upper_phys, device=device),
            wrist_joint_id=wrist_joint_id,
            wrist_urdf_joint_id=wrist_urdf_joint_id,
            wrist_limit_lower=physical_lower[WRIST_MOTOR_IDX],
            wrist_limit_upper=physical_upper[WRIST_MOTOR_IDX],
            optimization_frames=optimization_frames,
            urdfhand_center=urdfhand_center,
            urdfhand_rot_matrix=urdfhand_rot_matrix,
            fingertip_offsets=fingertip_offsets,
            urdf_keyvector_mags=urdf_keyvector_mags,
            device=device,
            lr=lr,
            ik_loss=ik_loss if ik_loss is not None else weighted_vector_loss(),
            regularization_weight=regularization_weight,
            joint_regularizers=joint_regularizers,
        )


class Retargeter:
    """Maps a TargetPose (raw 3D hand joints) to Orca Hand joint angles.

    Pipeline per call:
        1. Source-specific preprocess (mediapipe / avp)
        2. Normalize joints to a local hand frame
        3. (First N frames) auto-scale calibration: collect MANO key-vector
           magnitudes, then set ``mano_scale`` so MANO ≈ metres
        4. Scale → rotate into URDF base frame → compute key vectors
        5. RMSprop IK (2 steps) in URDF degrees, clamped to URDF-space limits

    Typical usage::

        retargeter = Retargeter.from_paths(model_path, urdf_path)
        joint_angles = retargeter.retarget(target_pose)
    """

    def __init__(self, config: RetargeterConfig) -> None:
        self.config = config

        self._joint_angles = torch.zeros(
            len(config.finger_urdf_joint_ids), device=config.device, requires_grad=True
        )
        self._optimizer = torch.optim.RMSprop([self._joint_angles], lr=config.lr)
        self._root = torch.zeros(1, 3, device=config.device)

        n_fingers = len(config.finger_joint_ids)
        self._regularizer_zeros = torch.zeros(n_fingers, device=config.device)
        self._regularizer_weights = torch.zeros(n_fingers, device=config.device)
        for joint_id, zero_val, weight in config.joint_regularizers:
            idx = config.finger_joint_ids.index(joint_id)
            self._regularizer_zeros[idx] = zero_val
            self._regularizer_weights[idx] = weight
        self._regularizer_weights = _normalize_regularizer_weights(self._regularizer_weights)

        # Auto-scale calibration state
        self._mano_scale: float = 1.0
        self._calibration_mags: list[np.ndarray] = []
        self._calibration_done: bool = False

    @classmethod
    def from_paths(
        cls,
        model_path: str | None = None,
        urdf_path: str | None = None,
        *,
        backend: RetargeterBackend = "adaptive_analytical",
        config_path: str | None = None,
        **kwargs,
    ) -> "Retargeter":
        if backend == "adaptive_analytical":
            if kwargs:
                raise ValueError(
                    "Adaptive analytical retargeter does not accept RMSprop-specific "
                    f"options: {sorted(kwargs)}"
                )
            from orca_teleop.retargeting.adaptive_analytical import (
                AdaptiveAnalyticalRetargeter,
            )

            return AdaptiveAnalyticalRetargeter.from_paths(
                model_path=model_path,
                urdf_path=urdf_path,
                config_path=config_path,
            )
        if backend != "rmsprop":
            raise ValueError(
                f"Unknown retargeter backend {backend!r}; expected 'rmsprop' or "
                "'adaptive_analytical'."
            )
        if config_path is not None:
            raise ValueError("config_path is only supported by backend='adaptive_analytical'")
        return cls(RetargeterConfig.from_paths(model_path, urdf_path, **kwargs))

    def _ik_loss(self, target_key_vectors: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        urdf_angles = torch.zeros(cfg.chain.n_joints, device=cfg.device)
        urdf_angles[cfg.finger_reorder_indices] = self._joint_angles / (180.0 / np.pi)

        fingertips, palm = retargeter_utils.extract_orca_fingertips_and_palm(
            cfg.chain,
            urdf_angles,
            cfg.optimization_frames,
            cfg.hand_type,
            list(FINGERS),
            self._root,
            fingertip_offsets=cfg.fingertip_offsets,
        )
        robot_kvs = torch.stack(
            [kv.squeeze(0) for kv in retargeter_utils.get_keyvectors(fingertips, palm)]
        )

        matching = cfg.ik_loss(target_key_vectors, robot_kvs)
        regularization = torch.sum(
            self._regularizer_weights * (self._joint_angles - self._regularizer_zeros) ** 2
        )
        return matching + cfg.regularization_weight * regularization

    def _optimize(self, target_key_vectors: torch.Tensor, n_steps: int = 2) -> np.ndarray:
        for _ in range(n_steps):
            self._optimizer.zero_grad()
            self._ik_loss(target_key_vectors).backward()
            self._optimizer.step()
            with torch.no_grad():
                self._joint_angles.clamp_(
                    self.config.finger_limits_lower_urdf,
                    self.config.finger_limits_upper_urdf,
                )
        return self._joint_angles.detach().cpu().numpy()

    def _ingest_calibration_frame(self, normalized_joints: np.ndarray, source: str) -> None:
        cfg = self.config
        joint_t = torch.tensor(normalized_joints, dtype=torch.float32, device=cfg.device)
        ft, palm = retargeter_utils.extract_mano_fingertips_and_palm(joint_t, list(FINGERS), source)
        kvs = retargeter_utils.get_keyvectors(ft, palm)
        mags = np.array([kv.detach().cpu().norm().item() for kv in kvs])
        self._calibration_mags.append(mags)

        if len(self._calibration_mags) >= CALIBRATION_FRAMES:
            all_mags = np.array(self._calibration_mags)
            median_mano = np.median(all_mags, axis=0)
            ratios = cfg.urdf_keyvector_mags / np.clip(median_mano, 1e-6, None)
            self._mano_scale = float(np.median(ratios))
            self._calibration_done = True
            logger.info(
                "Retargeter auto-scale calibrated: mano_scale=%.4f (per-finger ratios=%s)",
                self._mano_scale,
                np.round(ratios, 3).tolist(),
            )
            with torch.no_grad():
                self._joint_angles.zero_()
            self._optimizer = torch.optim.RMSprop([self._joint_angles], lr=cfg.lr)

    def retarget(self, target_pose: TargetPose) -> OrcaJointPositions | None:
        """Map a TargetPose to OrcaJointPositions ready to send to the robot.

        Returns ``None`` during the calibration window so the pipeline can keep
        the hand at whatever pose ``init_joints()`` established until the scale
        estimate is ready.
        """
        cfg = self.config
        source = target_pose.source

        # Step 1: source-specific raw preprocessing
        if source == "mediapipe":
            joints = np.asarray(target_pose.joint_positions, dtype=float)
        elif source == "avp":
            joints = np.asarray(target_pose.joint_positions, dtype=float)
        else:
            raise ValueError(f"Unsupported source: {source!r}")

        # Step 2: normalize into the local canonical hand frame
        normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(joints, source)

        wrist_phys_deg = float(
            np.clip(target_pose.wrist_angle_degrees, cfg.wrist_limit_lower, cfg.wrist_limit_upper)
        )

        # Step 3: calibration window — collect frames only, emit nothing yet
        if not self._calibration_done:
            self._ingest_calibration_frame(normalized, source)
            return None

        # Step 4: scale → rotate into URDF base frame → key vectors
        scaled = normalized * self._mano_scale
        in_urdf_frame = (
            scaled @ cfg.urdfhand_rot_matrix.T
            + cfg.urdfhand_center
            + np.array(MANO_TO_URDF_TRANSLATION)
        )
        joint_t = torch.tensor(in_urdf_frame, dtype=torch.float32, device=cfg.device)
        mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(
            joint_t, list(FINGERS), source
        )
        target_kvs = torch.stack([(mano_ft[f] - mano_palm).squeeze(0) for f in FINGERS])

        # Step 5: IK in URDF degrees, clamped to URDF-space limits
        finger_urdf_deg = self._optimize(target_kvs)

        # Step 6: assemble the action (the model's joint zero is the hand's)
        finger_phys_deg = finger_urdf_deg
        wrist_signed = wrist_phys_deg if cfg.hand_type == "left" else -wrist_phys_deg
        return OrcaJointPositions(
            {
                **dict(zip(cfg.finger_joint_ids, finger_phys_deg, strict=True)),
                cfg.wrist_joint_id: wrist_signed,
            }
        )
