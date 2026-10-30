import numpy as np
import pytest
from conftest import plausible_hand_keypoints

from orca_teleop.ingress.landmark_recording import LandmarkRecorder, load_recording


def test_recording_round_trip(tmp_path):
    rec_dir = tmp_path / "rec"
    recorder = LandmarkRecorder(rec_dir, meta={"handedness": "right"}, fps=30.0)
    world = plausible_hand_keypoints()
    for i in range(3):
        index = recorder.add_frame(np.zeros((48, 64, 3), dtype=np.uint8), capture_time_ns=1000 + i)
        assert index == i
        if i != 1:  # frame 1 had no hand
            recorder.add_landmarks(index, world + i, np.full((21, 3), 0.5), "right", 0.97)
    assert recorder.close() == (3, 2)

    rec = load_recording(rec_dir)
    assert rec.meta["handedness"] == "right"
    assert rec.capture_time_ns.tolist() == [1000, 1001, 1002]
    assert rec.frame.tolist() == [0, 2]
    np.testing.assert_allclose(rec.world[1], world + 2, atol=1e-6)
    assert rec.handedness == ["right", "right"]
    assert rec.video_path.exists()


def test_recorder_refuses_a_non_empty_folder(tmp_path):
    (tmp_path / "old.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        LandmarkRecorder(tmp_path, meta={}, fps=30.0)
    assert (tmp_path / "old.txt").read_text() == "keep me"
