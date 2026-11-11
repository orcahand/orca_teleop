"""URDF-backed kinematic teleop sink.

This is intentionally lighter than ``orca_teleop.sim``: it does not simulate
contacts, actuator dynamics, or MuJoCo physics. It only displays the latest
retargeted joint command on the ORCA URDF.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import orca_core
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.pipeline import _SHUTDOWN, RobotSink
from orca_teleop.utils import RateTicker, default_urdf_path, urdf_with_local_meshes

logger = logging.getLogger(__name__)


def _default_model_path(hand_type: str, version: str) -> str:
    models = Path(orca_core.__file__).resolve().parent / "models" / version
    config_path = models / f"orcahand-{hand_type}" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find OrcaCore model config: {config_path}")
    return str(config_path)


# The real hand's look: black prints, tower and palm skin; white silicone finger skins.
HAND_RGBA = np.array([0.07, 0.07, 0.08, 1.0])
SKIN_RGBA = np.array([0.94, 0.94, 0.92, 1.0])
# The viewer always shows the wrist straight (wrist joint -> middle MCP in line with the
# forearm), whatever is commanded; the webcam never sets the wrist anyway. A Quest's
# wrist angle is not shown.
STRAIGHT_WRIST_DEG = 3.0


def _style_hand(visual_model: Any) -> None:
    """Colour the hand like the real one: the URDF's white skins white, the rest black."""
    for obj in visual_model.geometryObjects:
        obj.meshColor = SKIN_RGBA if float(np.mean(obj.meshColor[:3])) > 0.5 else HAND_RGBA


def _studio_scene(viewer: Any) -> None:
    """Light grey-white backdrop, bright soft light, no grid or axes."""
    viewer["/Background"].set_property("top_color", [0.97, 0.97, 0.97])
    viewer["/Background"].set_property("bottom_color", [0.82, 0.82, 0.83])
    viewer["/Grid"].set_property("visible", False)
    viewer["/Axes"].set_property("visible", False)
    for light, intensity in (
        ("AmbientLight", 0.75),
        ("FillLight", 0.5),
        ("SpotLight", 0.5),
        ("PointLightNegativeX", 0.25),
        ("PointLightPositiveX", 0.25),
    ):
        viewer[f"/Lights/{light}/<object>"].set_property("intensity", intensity)


class OrcaHandUrdfVizSink(RobotSink):
    """Kinematic Meshcat viewer sink for ``OrcaJointPositions`` commands."""

    def __init__(
        self,
        hand_type: str = "right",
        version: str = "v2",
        model_path: str | None = None,
        urdf_path: str | None = None,
        rate_hz: float = 90.0,
        open_browser: bool = True,
    ) -> None:
        self._hand_type = hand_type
        self._version = version
        self._model_path = model_path
        self._urdf_path = urdf_path
        self._rate_hz = float(rate_hz)
        self._open_browser = open_browser

        self._hand: OrcaHand | None = None
        self._pin: Any | None = None
        self._viz: Any | None = None
        self._model: Any | None = None
        self._qpos: np.ndarray | None = None
        self._last_action: OrcaJointPositions | None = None
        self._joint_q_indices: dict[str, int] = {}

    @property
    def retarget_model_path(self) -> str:
        if self._model_path is None:
            return _default_model_path(self._hand_type, self._version)
        return self._model_path

    def connect(self) -> None:
        try:
            import pinocchio as pin
            from pinocchio.visualize import MeshcatVisualizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "URDF visualization requires optional dependencies. Run with "
                "`uv run --extra adaptive --extra urdf-viz python scripts/teleop_urdf.py ...`."
            ) from exc

        self._pin = pin
        model_path = self.retarget_model_path
        urdf_path = self._urdf_path or default_urdf_path(self._hand_type, self._version)

        hand = OrcaHand(model_path)
        if hand.config.type != self._hand_type:
            raise ValueError(
                f"Model hand type {hand.config.type!r} does not match sink hand "
                f"{self._hand_type!r}"
            )
        self._hand = hand
        self._urdf_path = urdf_path

        urdf_xml = urdf_with_local_meshes(urdf_path)
        model = pin.buildModelFromXML(urdf_xml)
        visual_model = pin.buildGeomFromUrdfString(model, urdf_xml, pin.GeometryType.VISUAL)
        _style_hand(visual_model)
        self._model = model
        self._qpos = np.zeros(model.nq, dtype=np.float64)
        self._build_joint_mapping()

        self._last_action = OrcaJointPositions(hand.config.neutral_position)
        self._qpos[:] = self._to_qpos(self._last_action)

        self._viz = MeshcatVisualizer(model, pin.GeometryModel(), visual_model)
        self._viz.initViewer(open=self._open_browser)
        _studio_scene(self._viz.viewer)
        self._viz.loadViewerModel(rootNodeName=f"orca_{self._hand_type}_urdf")
        self._viz.display(self._qpos)

        logger.info(
            "URDF Viz connected: hand=%s version=%s joints=%d rate=%.1fHz url=%s",
            self._hand_type,
            self._version,
            len(self._joint_q_indices),
            self._rate_hz,
            self._viewer_url(),
        )

    def current_positions(self) -> OrcaJointPositions | None:
        return self._last_action

    def run_loop(
        self,
        actions_q: queue.Queue[OrcaJointPositions | object],
        stop_event: threading.Event,
    ) -> None:
        assert self._last_action is not None, "connect() must be called before run_loop()"
        ticker = RateTicker(dt=1.0 / self._rate_hz)

        while not stop_event.is_set():
            shutdown_received = False
            latest = self._last_action

            while True:
                try:
                    item = actions_q.get_nowait()
                except queue.Empty:
                    break

                if item is _SHUTDOWN:
                    shutdown_received = True
                    break
                if isinstance(item, OrcaJointPositions):
                    latest = item

            if shutdown_received:
                break

            if latest is not self._last_action:
                self._last_action = latest
                self._display(latest)

            ticker.tick()

    def close(self) -> None:
        # Meshcat has no hard process to close here; leaving the browser tab open
        # is useful after Ctrl-C for visual inspection.
        self._viz = None

    def _build_joint_mapping(self) -> None:
        assert self._hand is not None
        assert self._model is not None

        missing = []
        for joint_id in self._hand.config.joint_ids:
            urdf_name = f"{self._hand_type}_{joint_id}"
            pin_joint_id = self._model.getJointId(urdf_name)
            if pin_joint_id >= self._model.njoints:
                missing.append(urdf_name)
                continue
            self._joint_q_indices[joint_id] = int(self._model.idx_qs[pin_joint_id])

        if missing:
            raise ValueError(f"URDF is missing joints required by Orca config: {missing}")

    def _to_qpos(self, positions: OrcaJointPositions) -> np.ndarray:
        assert self._hand is not None
        assert self._model is not None

        qpos = np.zeros(self._model.nq, dtype=np.float64)
        position_data = positions.as_dict()
        for joint_id in self._hand.config.joint_ids:
            q_idx = self._joint_q_indices[joint_id]
            value_deg = float(
                position_data.get(joint_id, self._hand.config.neutral_position[joint_id])
            )
            if joint_id == "wrist":
                value_deg = STRAIGHT_WRIST_DEG
            qpos[q_idx] = np.deg2rad(value_deg)
        return qpos

    def _display(self, action: OrcaJointPositions) -> None:
        assert self._viz is not None
        self._qpos = self._to_qpos(action)
        self._viz.display(self._qpos)

    def _viewer_url(self) -> str:
        if self._viz is None:
            return "?"
        viewer = getattr(self._viz, "viewer", None)
        if viewer is None or not hasattr(viewer, "url"):
            return "?"
        try:
            return str(viewer.url())
        except Exception:
            return "?"
