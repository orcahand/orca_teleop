"""No 2 cm landmark shift in the solve, the last pose kept through a tracking loss, and the
robot palm point's offset as retarget.palm_offset_cm."""

import numpy as np
import yaml
from test_fidelity import CLIP, _model_or_skip


def _solver(tmp_path, name="cfg", **retarget):
    """The default config with ``retarget`` keys set (``None``: the key left out)."""
    from orca_teleop.retargeting.config import default_config_path, load_config
    from orca_teleop.retargeting.retargeter import Retargeter

    cfg = load_config(default_config_path("adaptive_analytical"))
    cfg["retarget"].update(retarget)
    cfg["retarget"] = {k: v for k, v in cfg["retarget"].items() if v is not None}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return Retargeter.from_paths(None, None, config_path=str(path))


def _run(r, frames):
    from orca_teleop.retargeting.retargeter import TargetPose

    r.reset()
    out = []
    for frame in frames:
        if np.all(np.isfinite(frame)):
            q = r.retarget(TargetPose(joint_positions=frame))
            if q is not None:
                out.append([float(v) for v in q.as_dict().values()])
    return np.array(out)


def _world(n):
    return np.load(CLIP)["world"].astype(np.float64)[:n]


def test_palm_offset_default_is_bit_identical(tmp_path):
    _model_or_skip()
    world = _world(120)
    plain = _run(_solver(tmp_path, "plain"), world)
    no_key = _run(_solver(tmp_path, "nokey", palm_offset_cm=None), world)  # the code default
    assert plain.shape == no_key.shape and len(plain) > 50
    assert np.array_equal(plain, no_key)


def test_palm_offset_moves_the_robot_palm_point(tmp_path):
    _model_or_skip()
    a = _solver(tmp_path, "a")
    b = _solver(tmp_path, "b", palm_offset_cm=0.0)
    positions = np.random.default_rng(0).normal(size=(len(a._computed_frame_indices), 3))
    moved = b._robot_palm_from_positions(positions) - a._robot_palm_from_positions(positions)
    np.testing.assert_allclose(moved, [0.0, 0.0, 0.015], atol=1e-12)


def test_keypoints_are_not_shifted(tmp_path):
    """The keypoints go into the URDF frame by rotation, scale and the hand centre only."""
    _model_or_skip()
    r = _solver(tmp_path)
    kp = r._target_keypoints_in_urdf_frame(np.zeros((21, 3)))
    np.testing.assert_allclose(kp[0], r._urdfhand_center, atol=1e-15)


def test_a_tracking_loss_keeps_the_last_pose_and_reset_forgets_it(tmp_path):
    _model_or_skip()
    from orca_teleop.retargeting.retargeter import TargetPose

    r = _solver(tmp_path)
    for frame in _world(80):
        if np.all(np.isfinite(frame)):
            r.retarget(TargetPose(joint_positions=frame))
    last = r._last_qpos_phys.copy()
    r.reset_tracking()
    np.testing.assert_array_equal(r._last_qpos_phys, last)
    np.testing.assert_array_equal(r._initial_qpos(), last)
    assert r._calibration_done
    r.reset()
    assert r._last_qpos_phys is None and not r._calibration_done


def test_a_non_finite_pose_is_dropped_and_the_next_frame_solves(tmp_path):
    """reset_tracking keeps the pose, so the non-finite path must drop it itself."""
    _model_or_skip()
    from orca_teleop.retargeting.retargeter import TargetPose

    r = _solver(tmp_path)
    frames = [f for f in _world(80) if np.all(np.isfinite(f))]
    for frame in frames[:-2]:
        r.retarget(TargetPose(joint_positions=frame))
    solve = r._solve

    def nan_solve(keypoints):  # as when the loss turns non-finite: _solve stores what it returns
        bad = np.full_like(solve(keypoints), np.nan)
        r._last_qpos_phys = bad.copy()
        return bad

    r._solve = nan_solve
    assert r.retarget(TargetPose(joint_positions=frames[-2])) is None
    assert r._last_qpos_phys is None
    r._solve = solve
    assert r.retarget(TargetPose(joint_positions=frames[-1])) is not None
