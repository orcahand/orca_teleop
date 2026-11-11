import numpy as np
from conftest import plausible_hand_keypoints

from orca_teleop.retargeting.adaptive_analytical import _squeeze_pinches, _thumb_shape_from_base


def test_thumb_targets_are_measured_from_the_thumb_base():
    kp = plausible_hand_keypoints().astype(float)
    vectors = np.arange(45, dtype=float).reshape(15, 3)  # rows: pip x5, dip x5, tip x5 (cm)
    base = np.array([3.0, -1.0, -1.0])
    scaling = np.array([0.9, 1.0, 1.1])
    out = _thumb_shape_from_base(vectors, kp, scaling, base, weight=1.0)

    for row, landmark, scale in ((0, 2, 0.9), (5, 3, 1.0), (10, 4, 1.1)):
        np.testing.assert_allclose(out[row], (kp[landmark] - kp[1]) * scale * 100.0 + base)
    others = [i for i in range(15) if i not in (0, 5, 10)]
    np.testing.assert_array_equal(out[others], vectors[others])
    np.testing.assert_array_equal(vectors, np.arange(45, dtype=float).reshape(15, 3))


def test_weight_blends_from_the_palm_targets():
    kp = plausible_hand_keypoints().astype(float)
    vectors = np.arange(45, dtype=float).reshape(15, 3)
    base = np.array([3.0, -1.0, -1.0])
    scaling = np.ones(3)
    full = _thumb_shape_from_base(vectors, kp, scaling, base, weight=1.0)

    np.testing.assert_array_equal(_thumb_shape_from_base(vectors, kp, scaling, base, 0.0), vectors)
    half = _thumb_shape_from_base(vectors, kp, scaling, base, weight=0.5)
    np.testing.assert_allclose(half, 0.5 * (vectors + full))


def _gaps_cm(kp):
    return np.linalg.norm(kp[[8, 12, 16, 20]] - kp[4], axis=1) * 100.0


def _one_finger_near(gap_cm: float) -> np.ndarray:
    kp = np.zeros((21, 3))
    kp[[12, 16, 20]] = [[0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [0.0, -0.1, 0.0]]  # 10 cm away
    kp[8] = [gap_cm / 100.0, 0.0, 0.0]  # index tip; thumb tip at the origin
    return kp


def test_squeeze_closes_only_near_pinches():
    for gap, expected in ((1.0, 0.5), (2.0, 1.75), (3.0, 3.0), (0.3, 0.0)):
        kp = _one_finger_near(gap)
        out = _squeeze_pinches(kp, squeeze_cm=0.5, full_below_cm=1.5, start_below_cm=2.5)
        assert abs(_gaps_cm(out)[0] - expected) < 1e-9, (gap, _gaps_cm(out)[0])
        np.testing.assert_allclose(out[4] + out[8], kp[4] + kp[8])  # both tips move equally
        others = [k for k in range(21) if k not in (4, 8)]
        np.testing.assert_array_equal(out[others], kp[others])


def test_squeeze_off_or_far_changes_nothing():
    kp = plausible_hand_keypoints().astype(float)
    np.testing.assert_array_equal(_squeeze_pinches(kp, 0.0, 1.5, 2.5), kp)
