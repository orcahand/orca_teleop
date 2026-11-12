"""The adaptive solver's hand-built kinematics and gradient, against independent references."""

import pathlib

import numpy as np
import pytest
from test_fidelity import CLIP, _model_or_skip


@pytest.fixture(scope="module", params=["right", "left"])
def solver(request):
    _model_or_skip()
    from orca_core import OrcaJointPositions

    from orca_teleop.retargeting.retargeter import Retargeter

    saved = OrcaJointPositions._default_joint_ids  # see test_fidelity.clip_replay
    model = None  # orca_core's v2 right hand
    if request.param == "left":
        import orca_core

        model = str(pathlib.Path(orca_core.__file__).parent / "models/v2/orcahand-left/config.yaml")
    yield request.param, Retargeter.from_paths(model, None)
    OrcaJointPositions._default_joint_ids = saved


def _random_q(model, rng):
    lo, hi = np.array(model.lowerPositionLimit), np.array(model.upperPositionLimit)
    return rng.uniform(np.where(np.isfinite(lo), lo, -np.pi), np.where(np.isfinite(hi), hi, np.pi))


def test_point_jacobians_match_pinocchio(solver):
    """Every frame of the model, with offsets, against pinocchio's own frame Jacobian."""
    import pinocchio as pin

    _side, r = solver
    robot, rng = r._robot, np.random.default_rng(0)
    model, data = robot.model, robot.model.createData()
    frames = list(range(model.nframes))
    for _ in range(5):
        q = _random_q(model, rng)
        offsets = [rng.normal(0, 0.01, 3) for _ in frames]
        positions, jacobians = robot.compute_positions_and_jacobians(q, frames, offsets)
        pin.framesForwardKinematics(model, data, q)
        for k, f in enumerate(frames):
            jac = pin.computeFrameJacobian(model, data, q, f, pin.LOCAL_WORLD_ALIGNED)
            r_off = data.oMf[f].rotation @ offsets[k]
            np.testing.assert_allclose(positions[k], data.oMf[f].translation + r_off, atol=1e-12)
            expected = jac[:3] + np.cross(jac[3:].T, r_off).T  # v + w x r
            np.testing.assert_allclose(jacobians[k], expected, atol=1e-12)
            np.testing.assert_allclose(robot.frame_rotation(f), data.oMf[f].rotation, atol=1e-12)


def test_the_gradient_matches_finite_differences(solver):
    """Targets from real solves on the pinch clip (pinch term on), at random poses."""
    from orca_teleop.retargeting.retargeter import TargetPose

    side, r = solver
    world = np.load(CLIP)["world"].astype(np.float64)
    if side == "left":
        world = world * np.array([-1.0, 1.0, 1.0])
    captured = []
    original = r._loss_and_grad

    def capture(x, *args):
        captured.append(args)
        return original(x, *args)

    r._loss_and_grad = capture
    try:
        r.reset()
        for frame in world[:200]:  # the first pinch starts at frame 128
            if np.all(np.isfinite(frame)):
                r.retarget(TargetPose(joint_positions=frame))
    finally:
        del r._loss_and_grad  # back to the method
    with_pinch = [a for a in captured if a[-1] is not None and np.any(a[-1] > 0)]
    assert with_pinch, "the clip should switch the pinch term on"
    rng, h = np.random.default_rng(1), 1e-6
    lo, hi = r._lower_phys, r._upper_phys
    for args in (captured[40], with_pinch[0], with_pinch[-1]):
        q = rng.uniform(lo, hi)
        _, grad = original(q, *args)
        fd = np.array(
            [
                (original(q + h * e, *args)[0] - original(q - h * e, *args)[0]) / (2 * h)
                for e in np.eye(len(q))
            ]
        )
        assert np.abs(fd - grad).max() <= 1e-5 * max(1.0, np.abs(grad).max())
