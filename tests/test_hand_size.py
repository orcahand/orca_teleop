"""retarget.hand_size: bone_lengths scales at the first frame, whatever the pose."""

import numpy as np
import pytest
from test_tracking_and_palm import _solver, _world


def _frame():
    return next(f for f in _world(60) if np.all(np.isfinite(f)))


def _bend(frame, base, angle_deg):
    """The finger's points after ``base`` rotated about ``base``: same bones, new pose."""
    a = np.radians(angle_deg)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    out = frame.copy()
    out[base + 1 : base + 4] = (frame[base + 1 : base + 4] - frame[base]) @ rot.T + frame[base]
    return out


def _scale(r, frame):
    from orca_teleop.retargeting.retargeter import TargetPose

    r.reset()
    q = r.retarget(TargetPose(joint_positions=frame))
    return q, r._mano_scale


def test_bone_lengths_scale_at_the_first_frame(tmp_path):
    from test_fidelity import _model_or_skip

    _model_or_skip()
    q, scale = _scale(_solver(tmp_path, hand_size="bone_lengths"), _frame())
    assert q is not None and scale != 1.0


def test_bone_length_scale_does_not_change_with_the_pose(tmp_path):
    from test_fidelity import _model_or_skip

    _model_or_skip()
    r = _solver(tmp_path, hand_size="bone_lengths")
    frame = _frame()
    bent = _bend(_bend(frame, 5, 70), 9, -40)  # index and middle bent at the MCP
    rotated = frame @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T + [0.1, -0.2, 0.3]
    scales = [_scale(r, f)[1] for f in (frame, bent, rotated, frame * 2.0)]
    np.testing.assert_allclose(scales[:3], scales[0], rtol=1e-9)
    np.testing.assert_allclose(scales[3], scales[0] / 2.0, rtol=1e-9)


def test_key_vectors_stay_the_default(tmp_path):
    from test_fidelity import _model_or_skip

    _model_or_skip()
    r = _solver(tmp_path, hand_size=None)
    assert _scale(r, _frame())[0] is None  # still collecting its 30 frames


def test_an_unknown_hand_size_is_an_error(tmp_path):
    from test_fidelity import _model_or_skip

    _model_or_skip()
    with pytest.raises(ValueError, match="hand_size"):
        _solver(tmp_path, hand_size="tape_measure")


def test_a_hand_shaped_like_the_robot_gets_the_robot_size_factor(tmp_path, monkeypatch):
    """Human points placed on the robot's own frames: every finger's ratio is the factor, so
    each robot point pairs with its human point, finger by finger."""
    from test_fidelity import _model_or_skip

    from orca_teleop.retargeting.adaptive_analytical import ROBOT_SIZE_FACTOR

    _model_or_skip()
    r = _solver(tmp_path, hand_size="bone_lengths", palm_offset_cm=0.0)
    pos, _ = r._robot.compute_positions_and_jacobians(
        r._full_model_qpos(r._neutral_qpos_phys), r._computed_frame_indices, r._frame_offsets
    )
    frame = np.zeros((21, 3))
    frame[0] = pos[r._palm_index]
    for finger in range(5):
        points = (r._base_indices, r._pip_indices, r._dip_indices, r._tip_indices)
        for k, index in enumerate(points):
            frame[1 + 4 * finger + k] = pos[index[finger]]
    ratios = []
    monkeypatch.setattr(r, "_finish_calibration", lambda rs, method: ratios.append(rs))
    r._scale_from_bone_lengths(frame, "mediapipe")
    np.testing.assert_allclose(ratios[0], ROBOT_SIZE_FACTOR, rtol=1e-9)
