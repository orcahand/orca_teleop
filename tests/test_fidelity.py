"""Fidelity: the scoring functions, and a regression test that replays a recorded clip.

The clip (``tests/data/pinch_clip.npz``) is 700 frames of two-camera triangulated
landmarks from a session of pinches to each finger, with the frames where both
camera pictures show the thumb touching a fingertip. The replay needs the v2 hand
model with tip and pad frames (ORCAHAND_DESCRIPTION_DIR); without it those tests skip.
"""

import pathlib

import numpy as np
import pytest

from orca_teleop import fidelity
from orca_teleop.retargeting.adaptive_analytical import (
    _LPFilter,
    _OneEuroFilter,
    _pinch_targets,
)

CLIP = pathlib.Path(__file__).parent / "data" / "pinch_clip.npz"


def _straight_hand() -> np.ndarray:
    """A flat right hand in metres, palm in the z = 0 plane: each finger straight along
    its metacarpal (wrist -> knuckle) line, the thumb out on the +x side."""
    w = np.zeros((21, 3))
    for i, x in enumerate((0.03, 0.01, -0.01, -0.03)):  # index .. pinky
        knuckle = np.array([x, 0.08, 0.0])
        along = knuckle / np.linalg.norm(knuckle)
        for k in range(4):
            w[5 + 4 * i + k] = knuckle + 0.03 * k * along
    for k in range(4):
        w[1 + k] = [0.04 + 0.02 * k, 0.02 + 0.01 * k, 0.0]
    return w


def test_human_angles_of_a_straight_and_a_bent_finger():
    straight = _straight_hand()
    bent = straight.copy()
    # bend the index 90 degrees at its PIP (landmark 6), towards the palm (-z)
    bent[7] = bent[6] + [0.0, 0.0, -0.03]
    bent[8] = bent[6] + [0.0, 0.0, -0.06]
    angles = fidelity.human_joint_angles(np.stack([straight, bent]))
    assert angles["index_pip"][0] == pytest.approx(0.0, abs=0.01)
    assert angles["index_pip"][1] == pytest.approx(90.0, abs=0.01)
    assert angles["middle_pip"][1] == pytest.approx(0.0, abs=0.01)


def test_spread_is_positive_towards_the_thumb():
    hand = _straight_hand()  # the thumb is on the +x side
    spread = hand.copy()
    angle = 0.3
    turn = np.array(
        [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    for k in (6, 7, 8):  # turn the index's phalanges about its knuckle, towards +x
        spread[k] = hand[5] + turn @ (hand[k] - hand[5])
    assert spread[8, 0] > hand[8, 0]
    angles = fidelity.human_joint_angles(np.stack([hand, spread]))
    assert angles["index_abd"][0] == pytest.approx(0.0, abs=0.01)
    assert angles["index_abd"][1] == pytest.approx(np.degrees(angle), abs=0.01)


def test_distal_gap_is_between_the_last_phalanges():
    hand = _straight_hand()
    # lay the thumb's last phalanx 2 cm above the index's PIP->TIP part
    hand[3] = hand[6] + [0.0, 0.0, 0.02]
    hand[4] = hand[8] + [0.0, 0.0, 0.02]
    gaps = fidelity.human_distal_gaps_cm(hand[None])[0]
    assert gaps[0] == pytest.approx(2.0, abs=1e-6)
    assert np.all(np.diff(gaps) > 0)  # further fingers are further away


def test_best_lag_finds_a_shift():
    t = np.arange(300)
    human = np.sin(t / 15.0)
    robot = np.roll(human, 3)
    robot[:3] = human[0]
    assert fidelity._best_lag(human, robot) == pytest.approx(3.0, abs=0.2)


def test_pinch_targets_close_at_contact_and_fade_out():
    vectors = np.array([[1.2, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [5.0, 0, 0]])
    targets, weights = _pinch_targets(vectors, contact_cm=1.5, near_cm=2.5, far_cm=4.0)
    np.testing.assert_allclose(targets[:, 0], [0.0, 0.5, 1.5, 3.5])
    np.testing.assert_allclose(weights, [1.0, 1.0, 2 / 3, 0.0])


def test_one_euro_is_still_when_still_and_quick_when_moving():
    one_euro, low_pass = _OneEuroFilter(0.4, 2.0), _LPFilter(0.25)
    for _ in range(10):
        assert one_euro.next(np.array([0.3])) == pytest.approx(0.3)
        low_pass.next(np.array([0.3]))
    # a 0.5 rad step: after 3 frames (100 ms) One Euro has covered more of it
    for _ in range(3):
        quick, slow = one_euro.next(np.array([0.8]))[0], low_pass.next(np.array([0.8]))[0]
    assert quick - 0.3 > slow - 0.3


def _model_or_skip():
    try:
        from orca_teleop.utils import default_urdf_path

        urdf = default_urdf_path("right")
    except (FileNotFoundError, ValueError) as err:
        pytest.skip(f"no v2 hand model: {err}")
    if 'name="right_index_tip"' not in pathlib.Path(urdf).read_text():
        pytest.skip("the hand model has no tip/pad frames (needs orcahand_description v2.1)")
    return urdf


def _replay(retargeter, world: np.ndarray) -> np.ndarray:
    from orca_teleop.retargeting.retargeter import TargetPose

    joint_ids = retargeter._finger_joint_ids
    q = np.full((len(world), len(joint_ids)), np.nan)
    for k, frame in enumerate(world):
        if not np.all(np.isfinite(frame)):
            continue
        action = retargeter.retarget(TargetPose(joint_positions=frame))
        if action is not None:
            q[k] = action.as_array(joint_ids)
    return q


@pytest.fixture(scope="module")
def clip_replay():
    urdf = _model_or_skip()
    from orca_core import OrcaJointPositions

    from orca_teleop.retargeting.retargeter import Retargeter

    # Building a hand registers orca_core's joint names class-wide; undo that when the
    # module is done, as conftest does per test, or later tests' made-up names raise.
    saved = OrcaJointPositions._default_joint_ids
    data = np.load(CLIP)
    retargeter = Retargeter.from_paths(None, None)  # orca_core's v2 right hand, default config
    q = _replay(retargeter, data["world"])
    gaps = fidelity.FingertipGaps(urdf, "right", list(retargeter._finger_joint_ids))
    yield data, retargeter, q, gaps
    OrcaJointPositions._default_joint_ids = saved


def _score(data, retargeter, q, gaps) -> dict:
    robot_gap = np.full((len(q), 4), np.nan)
    for k in np.flatnonzero(np.all(np.isfinite(q), axis=1)):
        robot_gap[k] = gaps(q[k])
    world = data["world"]
    return fidelity.score_session(
        fidelity.SessionResult(
            name="pinch_clip",
            joint_ids=list(retargeter._finger_joint_ids),
            lower_deg=np.rad2deg(retargeter._lower_phys),
            upper_deg=np.rad2deg(retargeter._upper_phys),
            q_deg=q,
            solve_ms=np.zeros(len(q)),
            human=fidelity.human_joint_angles(world),
            human_gap_cm=fidelity.human_distal_gaps_cm(world),
            contact=data["contact"],
            robot_gap_mm=robot_gap,
        )
    )


def test_every_finger_pinches(clip_replay):
    score = _score(*clip_replay)
    for finger, row in score["pinch"].items():
        assert row["success_pct"] >= 80.0, (finger, row)  # F4's gate
        assert row["false_touch_pct"] <= 5.0, (finger, row)


def test_the_hand_follows_the_curl_without_jitter(clip_replay):
    score = _score(*clip_replay)
    flex = [v["corr"] for k, v in score["curl"].items() if k.endswith(("_mcp", "_pip"))]
    assert np.mean(flex) >= 0.8
    assert score["jitter_deg"] <= 0.2
    lags = [v["lag_frames"] for v in score["curl"].values() if v["lag_frames"] is not None]
    assert np.median(lags) <= 2.0


def test_no_finger_bends_backwards_while_the_human_is_straight(clip_replay):
    score = _score(*clip_replay)
    for finger in ("index", "middle", "ring", "pinky"):
        assert score["at_lower_pct"][f"{finger}_pip"] <= 10.0, score["at_lower_pct"]


def test_a_replay_repeats_exactly(clip_replay):
    data, retargeter, q, _ = clip_replay
    retargeter.reset()
    again = _replay(retargeter, data["world"])
    np.testing.assert_array_equal(np.isnan(q), np.isnan(again))
    np.testing.assert_array_equal(q[np.isfinite(q)], again[np.isfinite(again)])


def test_swapping_the_thumb_base_joints_fails(clip_replay):
    """The regression test notices a thumb mix-up like the one the old alias table had."""
    data, retargeter, q, gaps = clip_replay
    ids = list(retargeter._finger_joint_ids)
    swapped = q.copy()
    a, b = ids.index("thumb_cmc"), ids.index("thumb_abd")
    swapped[:, [a, b]] = swapped[:, [b, a]]
    lo, hi = np.rad2deg(retargeter._lower_phys), np.rad2deg(retargeter._upper_phys)
    swapped = np.clip(swapped, lo, hi)
    score = _score(data, retargeter, swapped, gaps)
    assert min(row["success_pct"] for row in score["pinch"].values()) < 80.0
