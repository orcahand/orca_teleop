import numpy as np
import torch

from orca_teleop.retargeting.constants import TIP_LANDMARK_OFFSET_M

MANO_OFFSET = torch.tensor([0, 0, 0.015])


def get_hand_center_and_rotation(
    thumb_base: np.ndarray,
    index_base: np.ndarray,
    middle_base: np.ndarray,
    ring_base: np.ndarray,
    pinky_base: np.ndarray,
    wrist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hand center and a right-handed rotation matrix from finger bases.

    Axes:
        x — ring-base to index-base (lateral)
        y — wrist to middle-base (proximal)
        z — cross(x, y), points out of the palm for a right hand
    """
    hand_center = (thumb_base + pinky_base) / 2
    if wrist is None:
        wrist = hand_center

    y_axis = middle_base - wrist
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = index_base - ring_base
    x_axis -= (x_axis @ y_axis.T) * y_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    rot_matrix = np.concatenate(
        (x_axis.reshape(1, 3), y_axis.reshape(1, 3), z_axis.reshape(1, 3)), axis=0
    ).T
    assert np.allclose(
        rot_matrix @ rot_matrix.T, np.eye(3), atol=1e-6
    ), "Rotation matrix is not orthogonal"
    return hand_center, rot_matrix


def get_fingertip_urdf_name(hand_type: str, finger: str) -> str:
    """The v2 model's fingertip frame: on the skin at the end of the fingertip."""
    return f"{hand_type}_{finger}_tip"


def get_finger_base_urdf_name(hand_type: str, finger: str) -> str:
    """The part that turns at the finger's first joint (spread; thumb: CMC)."""
    return f"{hand_type}_thumb_tp" if finger == "thumb" else f"{hand_type}_{finger}_ap"


def get_palm_urdf_name(hand_type: str) -> str:
    return f"{hand_type}_carpals"


def get_fingertip_offset_tensors(fingers: list[str], device: str) -> dict[str, torch.Tensor]:
    """The point matched to MediaPipe's TIP landmark, in each fingertip frame (m)."""
    return {f: torch.tensor([list(TIP_LANDMARK_OFFSET_M)], device=device) for f in fingers}


def get_keyvectors(fingertips: dict[str, torch.Tensor], palm: torch.Tensor) -> list[torch.Tensor]:
    """Return palm-to-fingertip vectors for all five fingers."""
    return [
        fingertips["thumb"] - palm,
        fingertips["index"] - palm,
        fingertips["middle"] - palm,
        fingertips["ring"] - palm,
        fingertips["pinky"] - palm,
    ]


def extract_orca_fingertips_and_palm(
    chain,
    urdfhand_joint_angles: torch.Tensor,
    optimization_frames: list[int],
    hand_type: str,
    fingers: list[str],
    root: torch.Tensor,
    fingertip_offsets: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run FK and return fingertip positions and palm for the Orca Hand.

    If *fingertip_offsets* is provided, each finger's tip is that point in the
    finger's tip frame instead of the frame's origin.
    """
    chain_transforms = chain.forward_kinematics(
        urdfhand_joint_angles, frame_indices=optimization_frames
    )
    fingertips = {}
    for finger in fingers:
        offset_pt = (
            fingertip_offsets[finger]
            if fingertip_offsets is not None and finger in fingertip_offsets
            else root
        )
        fingertips[finger] = chain_transforms[
            get_fingertip_urdf_name(hand_type, finger)
        ].transform_points(offset_pt)

    thumb_base = chain_transforms[get_finger_base_urdf_name(hand_type, "thumb")].transform_points(
        root
    )
    pinky_base = chain_transforms[get_finger_base_urdf_name(hand_type, "pinky")].transform_points(
        root
    )
    palm = torch.mean(torch.cat([thumb_base, pinky_base], dim=0), dim=0, keepdim=True)

    palm = palm - MANO_OFFSET.to(palm.device)
    return fingertips, palm


def get_urdf_model_params(
    chain,
    hand_type: str,
    fingers: list[str],
    root_point: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One-time init: compute hand geometry and FK frame indices for optimization."""
    transforms = chain.forward_kinematics(torch.zeros(chain.n_joints, device=root_point.device))
    finger_base_positions = {
        f"{finger}_base": transforms[get_finger_base_urdf_name(hand_type, finger)]
        .transform_points(root_point)
        .cpu()
        .numpy()
        for finger in fingers
    }
    wrist_pos = transforms[get_palm_urdf_name(hand_type)].transform_points(root_point).cpu().numpy()
    center, rot_matrix = get_hand_center_and_rotation(**finger_base_positions, wrist=wrist_pos)

    urdf_frame_names = [
        get_finger_base_urdf_name(hand_type, "thumb"),
        get_finger_base_urdf_name(hand_type, "pinky"),
    ] + [get_fingertip_urdf_name(hand_type, finger) for finger in fingers]
    optimization_frames = chain.get_frame_indices(*urdf_frame_names)

    return center, rot_matrix, optimization_frames


def get_mano_joints_dict(joints: np.ndarray | torch.Tensor, source: str) -> dict:
    """Split a flat joint array into a per-finger dict based on source convention."""
    if source == "mediapipe":
        return {
            "wrist": joints[0, :],
            "thumb": joints[1:5, :],
            "index": joints[5:9, :],
            "middle": joints[9:13, :],
            "ring": joints[13:17, :],
            "pinky": joints[17:21, :],
        }
    elif source == "avp":
        return {
            "wrist": joints[1, :],
            "thumb": joints[2:6, :],
            "index": joints[6:10, :],
            "middle": joints[10:14, :],
            "ring": joints[14:18, :],
            "pinky": joints[18:22, :],
        }
    else:
        raise ValueError(f"Unsupported source: {source!r}")


def extract_mano_fingertips_and_palm(
    joints: torch.Tensor,
    fingers: list[str],
    source: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Extract fingertips and palm position from a MANO joint tensor."""
    joints_dict = get_mano_joints_dict(joints, source)
    fingertips, bases = {}, {}
    for finger in fingers:
        finger_joints = joints_dict[finger]
        fingertips[finger] = finger_joints[[-1], :]
        bases[finger] = finger_joints[[0], :]

    palm = torch.mean(torch.cat([bases["thumb"], bases["pinky"]], dim=0), dim=0, keepdim=True)

    return fingertips, palm


def get_normalized_local_manohand_joint_pos(joint_pos: np.ndarray, source: str) -> np.ndarray:
    """Express joint positions in the hand's local canonical frame.

    Removes global translation and rotates so the canonical hand axes align
    with the convention used by ``get_hand_center_and_rotation``.
    """
    joint_dict = get_mano_joints_dict(joint_pos, source)

    hand_center, hand_rot = get_hand_center_and_rotation(
        thumb_base=joint_dict["thumb"][0],
        index_base=joint_dict["index"][0],
        middle_base=joint_dict["middle"][0],
        ring_base=joint_dict["ring"][0],
        pinky_base=joint_dict["pinky"][0],
        wrist=joint_dict["wrist"],
    )

    joint_pos = joint_pos - hand_center
    joint_pos = joint_pos @ hand_rot
    return joint_pos
