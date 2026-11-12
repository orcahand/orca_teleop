import os
import time
from pathlib import Path

ORCAHAND_DESCRIPTION_DIR_ENV = "ORCAHAND_DESCRIPTION_DIR"
ORCAHAND_DESCRIPTION_DIR_DEFAULT = os.path.join(
    os.path.expanduser("~"), "Documents", "orcahand_description"
)


def infer_model_version(path: "str | os.PathLike | None") -> str:
    """Read the embodiment version out of an orca_core model path."""
    if path is None:
        return "v2"
    for part in Path(path).parts:
        if part.startswith("v") and part[1:].isdigit():
            return part
    return "v2"


def default_urdf_path(hand_type: str, version: str = "v2") -> str:
    """Resolve a hand URDF from the orcahand_description checkout.

    Only a model with ``{side}_{joint_id}`` joint names is accepted: an older
    checkout names v2 joints after CAD parts, which nothing here can resolve, so
    it fails here with a clear message instead of later in the solver.
    """
    base = os.environ.get(ORCAHAND_DESCRIPTION_DIR_ENV, ORCAHAND_DESCRIPTION_DIR_DEFAULT)
    candidate = Path(base) / version / "models" / "urdf" / f"orcahand_{hand_type}.urdf"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No {hand_type} {version} URDF at {candidate}. Clone "
            "https://github.com/orcahand/orcahand_description there, set "
            f"{ORCAHAND_DESCRIPTION_DIR_ENV}, or pass a URDF path explicitly."
        )
    if f'name="{hand_type}_wrist"' not in candidate.read_text():
        raise ValueError(
            f"{candidate} has no {hand_type}_wrist joint: this orcahand_description "
            "checkout predates the v2 joint names. Use its main branch or newer."
        )
    return str(candidate)


def description_root(urdf_path: "str | os.PathLike") -> Path:
    """The orcahand_description folder holding ``urdf_path`` (any folder name)."""
    path = Path(urdf_path).resolve()
    for parent in path.parents:
        if (parent / "v2" / "models").is_dir() or (parent / "v1" / "models").is_dir():
            return parent
    return path.parent


def urdf_with_local_meshes(urdf_path: "str | os.PathLike") -> str:
    """The URDF's text with ``package://orcahand_description/`` mesh paths made absolute.

    Viewers then find the meshes whatever the description folder is called.
    """
    root = description_root(urdf_path)
    return Path(urdf_path).read_text().replace("package://orcahand_description/", f"{root}/")


class RateTicker:
    """Sleep to maintain a fixed loop rate.

    Usage::

        RENDER_FPS = 30
        limiter = RateLimiter(dt=1 / RENDER_FPS)
        while running:  # loops at RENDER_FPS
            # ... do work here
            limiter.tick()
    """

    def __init__(self, dt: float) -> None:
        self._dt = dt
        self._next_tick = time.perf_counter()

    def tick(self) -> None:
        """Sleep until the next scheduled tick, then advance the schedule.

        If the deadline was missed the schedule resets to now, preventing
        a spiral of back-to-back zero-sleep ticks.
        """
        self._next_tick += self._dt
        now = time.perf_counter()
        sleep_s = self._next_tick - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            self._next_tick = now
