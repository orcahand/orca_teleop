"""The hand viewer's joint table (drawing is checked by eye: it needs a browser)."""

from orca_teleop.hand_viewer import _joint_table


def test_the_table_flags_joints_the_hand_does_not_follow():
    ids = ["index_pip", "thumb_cmc", "wrist", "index_mcp"]
    cmd = {"index_pip": 80.0, "thumb_cmc": 10.0, "wrist": -0.3, "index_mcp": 40.0}
    meas = {"index_pip": 55.0, "thumb_cmc": 12.0, "wrist": 0.0, "index_mcp": 40.4}
    rows = _joint_table(ids, cmd, meas).split("\n")[2:]
    assert rows == [  # wrist first, then the thumb; no "-0°"
        "| wrist | 0° | 0° | +0° |",
        "| thumb_cmc | 10° | 12° | +2° |",
        "| index_pip | 80° | 55° | **-25°** |",  # more than 5 degrees off: marked
        "| index_mcp | 40° | 40° | +0° |",
    ]


def test_the_table_without_motor_reads():
    rows = _joint_table(["index_pip"], {"index_pip": 30.0}, None).split("\n")
    assert rows[2] == "| index_pip | 30° | - | - |"


def test_a_nan_reading_shows_as_missing_instead_of_failing():
    rows = _joint_table(["index_pip"], {"index_pip": 30.0}, {"index_pip": float("nan")})
    assert rows.split("\n")[2] == "| index_pip | 30° | - | - |"
