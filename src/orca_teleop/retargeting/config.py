"""Retargeter config files: one per retargeter and input, picked by name, with an override.

``--retargeter NAME`` picks the algorithm and its config file for the input
(``DEFAULT_CONFIGS``); ``--retargeter-config FILE`` replaces that file. Every file names
its algorithm (``type: NAME``), so a file never runs under another one: given alone, a
file picks its algorithm; given with a ``--retargeter`` it disagrees with, it stops
with an error.

A file can start from another one (``base: OTHER.yaml``, a path relative to the file,
else a packaged config's name) and set only what differs: mappings merge key by key, any
other value replaces. Files from before 19 Nov 2026 name their retargeter in
``optimizer: {type: NAME}``; that still counts.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RETARGETER = "adaptive_analytical"
# Each retargeter's config file per input source (the publisher it reads from);
# "default" serves every source without a file of its own.
DEFAULT_CONFIGS: dict[str, dict[str, str]] = {
    "adaptive_analytical": {
        "default": "adaptive_analytical_orca_tuned.yaml",  # tuned on the webcam
        "manus": "manus.yaml",
    },
    "rmsprop": {"default": "rmsprop.yaml"},
}


def default_config_path(retargeter: str, source: str = "default") -> Path:
    """The packaged config file of ``retargeter`` for input ``source``."""
    files = DEFAULT_CONFIGS.get(retargeter)
    if files is None:
        raise ValueError(
            f"Unknown retargeter backend {retargeter!r}; expected one of {sorted(DEFAULT_CONFIGS)}"
        )
    return _packaged(files.get(source, files["default"]))


def _packaged(name: str) -> Path:
    return Path(str(resources.files("orca_teleop.retargeting") / "configs" / name))


def load_config(path: str | Path) -> dict[str, Any]:
    """The YAML file at ``path``, laid over its ``base:`` file (and that one's, in turn)."""
    return _load(Path(path), ())


def _load(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    if path.resolve() in seen:
        raise ValueError(f"{path}: its base: files form a loop")
    with open(path) as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as err:
            raise ValueError(f"{path} is not valid YAML: {err}") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a retargeter config must be a mapping")
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    if not isinstance(base, str):
        raise ValueError(f"{path}: `base:` must be a file name, got {base!r}")
    # Next to the file, else a packaged config (so a copy elsewhere still finds its base).
    base_path = path.parent / base
    if not base_path.exists() and _packaged(base).exists():
        base_path = _packaged(base)
    return _merge(_load(base_path, (*seen, path.resolve())), cfg)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        both = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = _merge(out[key], value) if both else value
    return out


def config_type(path: str | Path) -> str:
    """The retargeter a config file is for: its ``type:``."""
    cfg = load_config(path)
    kind = cfg.get("type")
    if kind is None and isinstance(cfg.get("optimizer"), dict):
        kind = cfg["optimizer"].get("type")  # the schema before 19 Nov 2026
    if not isinstance(kind, str) or kind not in DEFAULT_CONFIGS:
        raise ValueError(
            f"{path} must name its retargeter with `type:`, one of {sorted(DEFAULT_CONFIGS)}; "
            f"got {kind!r}"
        )
    return kind


def resolve(
    retargeter: str | None, config_path: str | Path | None, source: str = "default"
) -> tuple[str, Path]:
    """The retargeter and config file the flags ask for.

    No file: ``retargeter`` (else ``DEFAULT_RETARGETER``) with its file for ``source``.
    A file: its ``type``, which ``retargeter``, if given, must match (``ValueError``).
    """
    if config_path is None:
        name = retargeter or DEFAULT_RETARGETER
        return name, default_config_path(name, source)
    path = Path(config_path)
    kind = config_type(path)
    if retargeter is not None and retargeter != kind:
        raise ValueError(
            f"--retargeter {retargeter} disagrees with {path}, which is `type: {kind}`; "
            "drop one of them or make them match"
        )
    return kind, path
