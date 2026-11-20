"""Picking the retargeter and its config: per input, with an override, and the file's type."""

import pytest

from orca_teleop.retargeting.config import (
    DEFAULT_CONFIGS,
    config_type,
    default_config_path,
    load_config,
    resolve,
)


def _write(path, text):
    path.write_text(text)
    return path


def test_each_retargeter_gets_its_file_for_the_input():
    assert resolve(None, None, "mediapipe") == (
        "adaptive_analytical",
        default_config_path("adaptive_analytical"),
    )
    name, path = resolve(None, None, "manus")
    assert (name, path.name) == ("adaptive_analytical", "manus.yaml")
    name, path = resolve("rmsprop", None, "manus")
    assert (name, path.name) == ("rmsprop", "rmsprop.yaml")


def test_an_input_without_its_own_file_gets_the_default():
    assert resolve(None, None, "metaquest") == resolve(None, None, "mediapipe")


def test_every_packaged_file_names_its_retargeter():
    for name, files in DEFAULT_CONFIGS.items():
        for source in files:
            assert config_type(default_config_path(name, source)) == name


def test_a_file_picks_its_retargeter_and_a_matching_flag_is_fine(tmp_path):
    cfg = _write(tmp_path / "r.yaml", "type: rmsprop\n")
    assert resolve(None, str(cfg)) == ("rmsprop", cfg)
    assert resolve("rmsprop", str(cfg)) == ("rmsprop", cfg)


def test_a_flag_that_disagrees_with_the_file_stops(tmp_path):
    cfg = _write(tmp_path / "r.yaml", "type: rmsprop\n")
    with pytest.raises(ValueError, match="--retargeter adaptive_analytical disagrees"):
        resolve("adaptive_analytical", str(cfg))


def test_a_file_without_a_type_stops(tmp_path):
    cfg = _write(tmp_path / "r.yaml", "retarget: {maxeval: 10}\n")
    with pytest.raises(ValueError, match="must name its retargeter"):
        resolve(None, str(cfg))


def test_unknown_retargeter_names_the_known_ones():
    with pytest.raises(ValueError, match="Unknown retargeter backend 'nope'"):
        resolve("nope", None)


def test_base_merges_mappings_and_replaces_other_values(tmp_path):
    _write(
        tmp_path / "base.yaml",
        "type: adaptive_analytical\nretarget: {a: 1, b: {c: 2, d: 3}, e: [1]}\n",
    )
    cfg = _write(tmp_path / "over.yaml", "base: base.yaml\nretarget: {b: {c: 5}, e: [2, 3]}\n")
    assert load_config(cfg) == {
        "type": "adaptive_analytical",
        "retarget": {"a": 1, "b": {"c": 5, "d": 3}, "e": [2, 3]},
    }


def test_a_base_loop_stops(tmp_path):
    _write(tmp_path / "a.yaml", "base: b.yaml\n")
    _write(tmp_path / "b.yaml", "base: a.yaml\n")
    with pytest.raises(ValueError, match="loop"):
        load_config(tmp_path / "a.yaml")


def test_the_glove_differs_from_the_webcam_only_where_it_says():
    glove = load_config(default_config_path("adaptive_analytical", "manus"))
    webcam = load_config(default_config_path("adaptive_analytical"))
    assert glove["retarget"].pop("palm_offset_cm") == 0.0
    webcam["retarget"].pop("palm_offset_cm")
    assert glove["retarget"].pop("hand_size") == "bone_lengths"
    assert webcam["retarget"].pop("hand_size") == "key_vectors"
    assert glove == webcam


def test_from_paths_checks_the_file_type(tmp_path):
    from orca_teleop.retargeting.retargeter import Retargeter

    cfg = _write(tmp_path / "r.yaml", "type: rmsprop\n")
    with pytest.raises(ValueError, match="is a `type: rmsprop` config, not adaptive_analytical"):
        Retargeter.from_paths(backend="adaptive_analytical", config_path=str(cfg))


def test_rmsprop_takes_no_settings_from_its_file(tmp_path):
    from orca_teleop.retargeting.retargeter import Retargeter

    cfg = _write(tmp_path / "r.yaml", "type: rmsprop\nlr: 1.0\n")
    with pytest.raises(ValueError, match="rmsprop reads only `type:`"):
        Retargeter.from_paths(backend="rmsprop", config_path=str(cfg))


def test_the_cli_stops_on_a_mismatch_before_anything_starts(tmp_path, capsys):
    from orca_teleop import cli

    cfg = _write(tmp_path / "r.yaml", "type: rmsprop\n")
    parser = cli._common_parser("test")
    args = parser.parse_args(
        ["--retargeter", "adaptive_analytical", "--retargeter-config", str(cfg)]
    )
    with pytest.raises(SystemExit):
        cli._resolve_retargeter(parser, args, "mediapipe")
    assert "disagrees" in capsys.readouterr().err


def test_the_cli_fills_in_the_input_default():
    from orca_teleop import cli

    parser = cli._common_parser("test")
    args = parser.parse_args([])
    cli._resolve_retargeter(parser, args, "manus")
    assert args.retargeter == "adaptive_analytical"
    assert args.retargeter_config.name == "manus.yaml"


def test_a_broken_yaml_file_is_a_clear_error(tmp_path):
    cfg = _write(tmp_path / "r.yaml", "type: [unclosed\n")
    with pytest.raises(ValueError, match="is not valid YAML"):
        resolve(None, str(cfg))


def test_a_file_in_the_old_schema_names_its_type_in_optimizer(tmp_path):
    cfg = _write(tmp_path / "old.yaml", "optimizer:\n  type: adaptive_analytical\n")
    assert resolve(None, str(cfg)) == ("adaptive_analytical", cfg)


def test_a_bad_type_or_base_value_is_a_clear_error(tmp_path):
    cfg = _write(tmp_path / "t.yaml", "type: [adaptive_analytical]\n")
    with pytest.raises(ValueError, match="must name its retargeter"):
        resolve(None, str(cfg))
    cfg = _write(tmp_path / "b.yaml", "type: adaptive_analytical\nbase: 5\n")
    with pytest.raises(ValueError, match="must be a file name"):
        load_config(cfg)


def test_a_copy_elsewhere_still_finds_a_packaged_base(tmp_path):
    copy = _write(
        tmp_path / "manus.yaml", default_config_path("adaptive_analytical", "manus").read_text()
    )
    assert load_config(copy) == load_config(default_config_path("adaptive_analytical", "manus"))


def test_the_cli_picks_the_config_of_its_source():
    from orca_teleop import cli

    parser = cli._common_parser("test")
    args = parser.parse_args(["--source", "manus"])
    cli._resolve_retargeter(parser, args, args.source)
    assert args.retargeter_config.name == "manus.yaml"


def test_run_manus_local_uses_the_glove_config_by_default(monkeypatch):
    from orca_teleop import pipeline

    seen = {}
    monkeypatch.setattr(pipeline, "run_with_publisher", lambda *a, **kw: seen.update(kw))
    pipeline.run_manus_local(handedness="right")
    assert seen["retargeter_config_path"] == str(
        default_config_path("adaptive_analytical", "manus")
    )
    pipeline.run_manus_local(handedness="right", retargeter_config_path="x.yaml")
    assert seen["retargeter_config_path"] == "x.yaml"


def test_a_file_named_like_its_packaged_base_starts_from_the_packaged_one(tmp_path):
    over = _write(tmp_path / "manus.yaml", "base: manus.yaml\nretarget:\n  maxeval: 7\n")
    cfg = load_config(over)
    assert cfg["retarget"]["maxeval"] == 7 and cfg["retarget"]["palm_offset_cm"] == 0.0


def test_a_base_with_a_folder_is_never_swapped_for_a_packaged_file(tmp_path):
    (tmp_path / "hand").mkdir()
    over = _write(tmp_path / "hand" / "s.yaml", "base: ../configs/manus.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_config(over)
    for bad in ("''", "."):
        cfg = _write(tmp_path / "e.yaml", f"type: adaptive_analytical\nbase: {bad}\n")
        with pytest.raises(ValueError, match="must be a file name"):
            load_config(cfg)


@pytest.mark.parametrize("entry", ["teleop_hand", "teleop_urdf"])
def test_each_entry_point_picks_the_config_of_its_source(monkeypatch, entry):
    from orca_teleop import cli

    class Sink:
        handedness, joint_ids = "right", []

        def __init__(self, **kwargs):
            pass

    got = {}
    monkeypatch.setattr(cli, "OrcaHandSink", Sink)
    monkeypatch.setattr("orca_teleop.urdf_viz.OrcaHandUrdfVizSink", Sink)
    monkeypatch.setattr(cli, "_launch", lambda args, sink, hand: got.update(vars(args)))
    getattr(cli, entry)(
        ["--source", "manus", "--dry-run"] if entry == "teleop_hand" else ["--source", "manus"]
    )
    assert got["retargeter_config"].name == "manus.yaml"
