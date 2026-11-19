"""Tests for orca_teleop.retargeting.retargeter."""

import numpy as np
import pytest
import torch
from conftest import KEYVECTORS_SHAPE, plausible_hand_keypoints

from orca_teleop.retargeting.config import default_config_path
from orca_teleop.retargeting.retargeter import (
    Retargeter,
    RetargeterConfig,
    TargetPose,
    _normalize_regularizer_weights,
    weighted_vector_loss,
)


def _mediapipe_pose() -> np.ndarray:
    return plausible_hand_keypoints()


def test_target_pose_valid_construction():
    pose = TargetPose(joint_positions=_mediapipe_pose())
    assert pose.joint_positions.shape == (21, 3)
    assert pose.source == "mediapipe"


def test_target_pose_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        TargetPose(joint_positions=np.zeros((21,)))
    with pytest.raises(ValueError, match="shape"):
        TargetPose(joint_positions=np.zeros((21, 4)))


def test_target_pose_joint_positions_are_immutable():
    pose = TargetPose(joint_positions=_mediapipe_pose())
    with pytest.raises(ValueError, match="read-only"):
        pose.joint_positions[0, 0] = 99.0


def test_target_pose_wrist_angle_coerced_to_float():
    pose = TargetPose(joint_positions=_mediapipe_pose(), wrist_angle_degrees=10)
    assert isinstance(pose.wrist_angle_degrees, float)


def test_target_pose_wrist_angle_defaults_to_zero():
    pose = TargetPose(joint_positions=_mediapipe_pose())
    assert pose.wrist_angle_degrees == 0.0


def test_target_pose_input_array_is_copied():
    arr = _mediapipe_pose()
    pose = TargetPose(joint_positions=arr)
    arr[0, 0] = 999.0
    assert pose.joint_positions[0, 0] == 0.0


def test_weighted_vector_loss_zero_when_identical():
    loss_fn = weighted_vector_loss()
    kvs = torch.rand(KEYVECTORS_SHAPE)
    assert loss_fn(kvs, kvs).item() == pytest.approx(0.0, abs=1e-6)


def test_weighted_vector_loss_positive_when_different():
    loss_fn = weighted_vector_loss()
    target = torch.zeros(KEYVECTORS_SHAPE)
    robot = torch.ones(KEYVECTORS_SHAPE)
    assert loss_fn(target, robot).item() > 0.0


def test_weighted_vector_loss_increases_with_distance():
    loss_fn = weighted_vector_loss()
    target = torch.zeros(KEYVECTORS_SHAPE)
    small_error = loss_fn(target, torch.full(KEYVECTORS_SHAPE, 0.1))
    large_error = loss_fn(target, torch.full(KEYVECTORS_SHAPE, 1.0))
    assert small_error.item() < large_error.item()


def test_weighted_vector_loss_gradients_flow():
    loss_fn = weighted_vector_loss()
    robot = torch.zeros(KEYVECTORS_SHAPE, requires_grad=True)
    loss = loss_fn(torch.ones(KEYVECTORS_SHAPE), robot)
    loss.backward()
    assert robot.grad is not None
    assert not torch.all(robot.grad == 0)


def test_weighted_vector_loss_zero_coefficient_silences_finger():
    loss_fn = weighted_vector_loss(coeffs=(1.0, 0.0, 0.0, 0.0, 0.0))
    target = torch.zeros(KEYVECTORS_SHAPE)
    robot = torch.zeros(KEYVECTORS_SHAPE)
    robot[1:] = 100.0
    assert loss_fn(target, robot).item() == pytest.approx(0.0, abs=1e-6)


def test_weighted_vector_loss_higher_coefficient_increases_loss():
    target = torch.zeros(KEYVECTORS_SHAPE)
    robot = torch.ones(KEYVECTORS_SHAPE)
    low = weighted_vector_loss(coeffs=(1.0, 1.0, 1.0, 1.0, 1.0))(target, robot)
    high = weighted_vector_loss(coeffs=(10.0, 1.0, 1.0, 1.0, 1.0))(target, robot)
    assert high.item() > low.item()


def test_retargeter_config_default_ik_loss_produces_scalar_tensor():
    default_factory = RetargeterConfig.__dataclass_fields__["ik_loss"].default_factory
    loss_fn = default_factory()
    result = loss_fn(torch.zeros(KEYVECTORS_SHAPE), torch.zeros(KEYVECTORS_SHAPE))
    assert isinstance(result, torch.Tensor)
    assert result.shape == ()


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([2.0, 3.0, 5.0], [0.2, 0.3, 0.5]),
        ([1.0, 1.0, 1.0, 1.0], [0.25, 0.25, 0.25, 0.25]),
        ([1.0, 3.0], [0.25, 0.75]),
    ],
)
def test_normalize_regularizer_weights_sums_to_one(weights, expected):
    weights = torch.tensor(weights)
    normalized = _normalize_regularizer_weights(weights)
    assert normalized.sum().item() == pytest.approx(1.0)
    assert normalized.tolist() == pytest.approx(expected)


def test_normalize_regularizer_weights_keeps_all_zero_vector():
    weights = torch.zeros(4)
    normalized = _normalize_regularizer_weights(weights)
    assert torch.equal(normalized, weights)


def test_normalize_regularizer_weights_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_regularizer_weights(torch.tensor([1.0, -1.0]))


def test_retargeter_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown retargeter backend"):
        Retargeter.from_paths(backend="not-a-backend")  # type: ignore[arg-type]


def test_retargeter_dispatches_adaptive_backend(monkeypatch, tmp_path):
    from orca_teleop.retargeting import adaptive_analytical

    config = tmp_path / "retarget.yaml"
    config.write_text("type: adaptive_analytical\n")

    calls = {}

    class _StubAdaptive:
        @classmethod
        def from_paths(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    monkeypatch.setattr(adaptive_analytical, "AdaptiveAnalyticalRetargeter", _StubAdaptive)

    retargeter = Retargeter.from_paths(
        "model.yaml",
        "hand.urdf",
        backend="adaptive_analytical",
        config_path=str(config),
    )

    assert isinstance(retargeter, _StubAdaptive)
    assert calls == {
        "model_path": "model.yaml",
        "urdf_path": "hand.urdf",
        "config_path": str(config),
    }


def test_retargeter_defaults_to_adaptive_backend(monkeypatch):
    from orca_teleop.retargeting import adaptive_analytical

    calls = {}

    class _StubAdaptive:
        @classmethod
        def from_paths(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    monkeypatch.setattr(adaptive_analytical, "AdaptiveAnalyticalRetargeter", _StubAdaptive)

    retargeter = Retargeter.from_paths("model.yaml", "hand.urdf")

    assert isinstance(retargeter, _StubAdaptive)
    assert calls == {
        "model_path": "model.yaml",
        "urdf_path": "hand.urdf",
        "config_path": default_config_path("adaptive_analytical"),
    }
