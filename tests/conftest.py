"""Shared fixtures for orca_teleop tests."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from orca_core import OrcaJointPositions
from orca_core.hardware_hand import MockOrcaHand

CANONICAL_LANDMARK_SHAPE = (21, 3)
KEYVECTORS_SHAPE = (5, 3)


def plausible_hand_keypoints() -> np.ndarray:
    """Canonical 21-point MediaPipe hand layout used across test modules.

    Finger bases are arranged in a fan and fingertips are extended outward so
    that the hand-center / rotation computation doesn't hit degenerate
    (zero-norm) axes.  Matches the hand's approximate neutral pose.

    MediaPipe layout: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle,
                      13-16=ring, 17-20=pinky.
    """
    kp = np.zeros((21, 3), dtype=np.float32)
    kp[0] = [0.0, 0.0, 0.0]
    # thumb
    kp[1] = [0.03, 0.02, 0.0]
    kp[2] = [0.05, 0.04, 0.0]
    kp[3] = [0.06, 0.06, 0.0]
    kp[4] = [0.07, 0.08, 0.0]
    # index
    kp[5] = [0.02, 0.06, 0.0]
    kp[6] = [0.02, 0.09, 0.0]
    kp[7] = [0.02, 0.11, 0.0]
    kp[8] = [0.02, 0.13, 0.0]
    # middle
    kp[9] = [0.00, 0.07, 0.0]
    kp[10] = [0.00, 0.10, 0.0]
    kp[11] = [0.00, 0.12, 0.0]
    kp[12] = [0.00, 0.14, 0.0]
    # ring
    kp[13] = [-0.02, 0.06, 0.0]
    kp[14] = [-0.02, 0.09, 0.0]
    kp[15] = [-0.02, 0.11, 0.0]
    kp[16] = [-0.02, 0.13, 0.0]
    # pinky
    kp[17] = [-0.04, 0.05, 0.0]
    kp[18] = [-0.04, 0.07, 0.0]
    kp[19] = [-0.04, 0.09, 0.0]
    kp[20] = [-0.04, 0.10, 0.0]
    return kp


@pytest.fixture
def patch_mock_hand(monkeypatch):
    """Make `robot_worker` build a `MockOrcaHand` instead of a real one.

    Returns the list of instances created during the test (usually one).
    """
    created: list[MockOrcaHand] = []

    def factory(model_path=None):
        hand = MockOrcaHand()
        created.append(hand)
        return hand

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)
    return created


@pytest.fixture(autouse=True)
def _scratch_orca_home(monkeypatch, tmp_path_factory):
    """Point ORCA_HOME at a temporary folder: orca_core keeps each hand's files
    (calibration, joint ranges) there, so no test reads or writes the real ~/.orca."""
    monkeypatch.setenv("ORCA_HOME", str(tmp_path_factory.mktemp("orca-home")))


@pytest.fixture(autouse=True)
def _isolate_registered_joint_names():
    """Undo ``OrcaJointPositions.register_joint_names`` between tests.

    orca_core keeps the registration in a class variable, so one test building a
    hand makes every later test's synthetic joint names raise. Without this the
    suite passes or fails depending on execution order.
    """
    saved = OrcaJointPositions._default_joint_ids
    yield
    OrcaJointPositions._default_joint_ids = saved


@pytest.fixture(autouse=True)
def _no_thread_leaks():
    """Fail any test that leaks a non-daemon thread it spawned."""
    before = {t.ident for t in threading.enumerate()}
    yield
    leaked = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
    # Give stragglers a brief moment to wind down before declaring a leak.
    for t in leaked:
        t.join(timeout=1.0)
    still_alive = [t.name for t in leaked if t.is_alive()]
    assert not still_alive, f"leaked threads: {still_alive}"
