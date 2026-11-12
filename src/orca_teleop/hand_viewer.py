"""Browser view of the hand model next to the physical hand (viser).

``HandViewer`` shows the model at the pose teleop commands and, see-through, at the
pose the motors report, with a per-joint table of both and their difference. Where
the two hands part, the physical hand isn't doing what it was told (slack, a stall,
an off calibration). ``HandMeshes`` draws and poses model hands in a viser scene.

viser is not in an extra yet: run with ``uv run --with viser==1.1.1 …``.
"""

from __future__ import annotations

import logging
import pathlib
import struct
import threading
import time

import numpy as np
from orca_core import OrcaJointPositions

logger = logging.getLogger(__name__)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DIFF_DEG = 5.0  # the table marks joints whose measured pose is further off than this


def read_stl(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of a binary STL."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    tri = np.frombuffer(
        data,
        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        count=n,
        offset=84,
    )["v"].reshape(-1, 3)
    verts, faces = np.unique(tri, axis=0, return_inverse=True)
    return verts, faces.reshape(-1, 3).astype(np.uint32)


class HandMeshes:
    """Model hands in a viser scene, posed by orca_core joint ids in degrees.

    Each added hand gets its own copy of the moving meshes (the forearm and tower,
    fixed to the world, are left out), its own colours and an offset in the scene.
    """

    def __init__(self, server, urdf_path: str, hand_type: str, joint_ids: list[str]):
        import pinocchio as pin

        from orca_teleop.utils import urdf_with_local_meshes

        self._pin, self._server, self.hand_type = pin, server, hand_type
        xml = urdf_with_local_meshes(urdf_path)
        self.model = pin.buildModelFromXML(xml)
        self._vmodel = pin.buildGeomFromUrdfString(self.model, xml, pin.GeometryType.VISUAL)
        self.data, self._vdata = self.model.createData(), pin.GeometryData(self._vmodel)
        self.joint_ids = list(joint_ids)
        self._q_index = {}
        for j in self.joint_ids:
            joint = self.model.getJointId(f"{hand_type}_{j}")
            if joint < self.model.njoints:
                self._q_index[j] = int(self.model.idx_qs[joint])
        missing = [j for j in self.joint_ids if j not in self._q_index]
        if missing:  # e.g. the wrong hand's model: those joints would sit at 0 unnoticed
            logger.warning("Hand model %s has no joints %s; drawn at 0°", urdf_path, missing)
        self._meshes = {
            g.name: read_stl(pathlib.Path(g.meshPath))
            for g in self._vmodel.geometryObjects
            if g.parentJoint != 0
        }
        self.hands: dict[str, np.ndarray] = {}
        self._handles: dict[str, list] = {}
        self._colors: dict[str, list] = {}  # each part's own colour, for set_style

    def add(self, name: str, light, dark, offset=(0.0, 0.0, 0.0), opacity: float | None = None):
        """A new hand in the scene; ``light``/``dark`` colour the white/black parts."""
        self.hands[name] = np.asarray(offset, dtype=np.float64)
        handles, colors = [], []
        for g in self._vmodel.geometryObjects:
            if g.parentJoint == 0:
                handles.append(None)
                colors.append(None)
                continue
            v, f = self._meshes[g.name]
            color = light if float(np.mean(g.meshColor[:3])) > 0.5 else dark
            handles.append(
                self._server.scene.add_mesh_simple(
                    f"/{name}/{g.name}",
                    v * np.asarray(g.meshScale),
                    f,
                    color=color,
                    opacity=opacity,
                )
            )
            colors.append(color)
        self._handles[name], self._colors[name] = handles, colors

    def q(self, positions: dict[str, float]) -> np.ndarray:
        """The model's configuration for joint positions in degrees (missing joints at 0)."""
        q = np.zeros(self.model.nq)
        for j, i in self._q_index.items():
            value = positions.get(j)
            if value is not None and np.isfinite(value):
                q[i] = np.radians(value)
        return q

    def pose(self, name: str, positions: dict[str, float]) -> None:
        """Move one hand; afterwards ``self.data`` holds its frames (without the offset)."""
        pin, q = self._pin, self.q(positions)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self._vmodel, self._vdata, q)
        pin.updateFramePlacements(self.model, self.data)
        offset = self.hands[name]
        for handle, placement in zip(self._handles[name], self._vdata.oMg, strict=True):
            if handle is not None:
                handle.position = placement.translation + offset
                quat = pin.Quaternion(placement.rotation)
                handle.wxyz = np.array([quat.w, quat.x, quat.y, quat.z])

    def set_visible(self, name: str, visible: bool) -> None:
        for handle in self._handles[name]:
            if handle is not None:
                handle.visible = visible

    def set_style(self, name: str, opacity: float | None, color=None) -> None:
        """See-through (``opacity`` < 1) and/or one ``color`` for every part; None
        restores solid parts and their own colours."""
        for handle, own in zip(self._handles[name], self._colors[name], strict=True):
            if handle is not None:
                handle.opacity = opacity
                handle.color = own if color is None else color

    def palm_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """The palm centre and axes (columns: across the knuckles, to the fingers, out of
        the palm) at the zero pose, as the retargeter defines them."""
        from orca_teleop.retargeting.utils import get_hand_center_and_rotation

        self._pin.framesForwardKinematics(self.model, self.data, np.zeros(self.model.nq))
        side = self.hand_type

        def at(frame: str) -> np.ndarray:
            return self.data.oMf[self.model.getFrameId(f"{side}_{frame}")].translation.copy()

        center, axes = get_hand_center_and_rotation(
            thumb_base=at("thumb_tp"),
            index_base=at("index_ap"),
            middle_base=at("middle_ap"),
            ring_base=at("ring_ap"),
            pinky_base=at("pinky_ap"),
            wrist=at("carpals"),
        )
        return np.asarray(center).reshape(3), np.asarray(axes)


class HandViewer:
    """The commanded and the measured pose of the physical hand, on the model.

    ``commanded()`` and ``measured()`` may be called from any thread; they only keep
    the newest value. A thread of its own draws at ``rate_hz``.
    """

    def __init__(
        self,
        urdf_path: str,
        hand_type: str,
        joint_ids: list[str],
        port: int = 8098,
        rate_hz: float = 30.0,
    ):
        try:
            import viser
        except ImportError:
            raise SystemExit(
                "The viewer needs viser: run with `uv run --with viser==1.1.1 …`"
            ) from None
        self.port = port
        self._server = viser.ViserServer(host="127.0.0.1", port=port, label="ORCA hand viewer")
        self._server.gui.configure_theme(control_width="large", show_share_button=False)
        self._hands = HandMeshes(self._server, urdf_path, hand_type, joint_ids)
        self._center, self._axes = self._hands.palm_axes()
        self._half_apart = 0.1 * self._axes[:, 0]  # side by side: commanded on the thumb side
        self._hands.add("commanded", (238, 238, 234), (24, 24, 28))
        self._hands.add("measured", (255, 178, 102), (232, 118, 36))
        gui = self._server.gui
        gui.add_markdown(
            "**White: commanded** (what teleop sends). **Orange: measured** (what the "
            "motors report). Overlaid, the commanded hand is a pale ghost: wherever it "
            "shows outside the orange one, the physical hand isn't following."
        )
        self._show_commanded = gui.add_checkbox("Commanded", True)
        self._show_measured = gui.add_checkbox("Measured", True)
        self._layout = gui.add_dropdown(
            "Layout", ("overlaid", "side by side"), initial_value="overlaid"
        )
        self._status = gui.add_markdown("Waiting for the first command…")
        self._table = gui.add_markdown("")
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, float] | None] = {"commanded": None, "measured": None}
        self._counts = {"commanded": 0, "measured": 0}
        self._read = {"failed": False, "ms": None}  # the last motor read; under _lock
        self._dirty = False
        self._server.on_client_connect(self._place_camera)
        self._layout.on_update(lambda _e: self._relayout())
        for toggle in (self._show_commanded, self._show_measured):
            toggle.on_update(lambda _e: self._redraw())
        self._ghost = False  # the commanded hand drawn pale and see-through
        self._relayout()
        self._stop = threading.Event()
        self._interval = 1.0 / rate_hz
        self._thread = threading.Thread(target=self._draw_loop, name="hand-viewer", daemon=True)
        self._thread.start()
        logger.info("Hand viewer at %s", self.url)

    @property
    def url(self) -> str:
        """Where it really serves: viser takes the next free port if ``port`` is busy."""
        return f"http://localhost:{self._server.get_port()}"

    def _side_by_side(self) -> bool:
        return self._layout.value == "side by side"

    def _place_camera(self, client) -> None:
        """Palm side, fingers up, a bit right of the hands (the panel covers the right)."""
        axes = self._axes
        aim = self._center + 0.06 * axes[:, 1] + 0.04 * axes[:, 0]
        distance = 0.55 if self._side_by_side() else 0.34
        client.camera.near = 0.001
        client.camera.up_direction = tuple(axes[:, 1])
        client.camera.position = tuple(aim + distance * axes[:, 2])
        client.camera.look_at = tuple(aim)

    def _relayout(self) -> None:
        side_by_side = self._side_by_side()
        self._hands.hands["commanded"] = self._half_apart if side_by_side else np.zeros(3)
        self._hands.hands["measured"] = -self._half_apart if side_by_side else np.zeros(3)
        for client in self._server.get_clients().values():
            self._place_camera(client)
        self._redraw()

    def _redraw(self) -> None:
        with self._lock:
            self._dirty = True

    def commanded(self, positions: OrcaJointPositions) -> None:
        self._put("commanded", positions)

    def measured(self, positions: OrcaJointPositions | None, read_ms: float | None = None) -> None:
        """The motors' pose; None after a failed read (the measured hand is hidden)."""
        self._put("measured", positions, read_ms)

    def _put(
        self, which: str, positions: OrcaJointPositions | None, read_ms: float | None = None
    ) -> None:
        values = None if positions is None else positions.as_dict()
        with self._lock:
            self._latest[which] = values
            self._counts[which] += 1
            self._dirty = True
            if which == "measured":
                self._read = {"failed": positions is None, "ms": read_ms}

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._server.stop()

    def _draw_loop(self) -> None:
        window_t, window = time.monotonic(), {"commanded": 0, "measured": 0}
        rates = {"commanded": 0.0, "measured": 0.0}
        last_draw = 0.0
        while not self._stop.wait(self._interval):
            with self._lock:
                latest, dirty, counts = dict(self._latest), self._dirty, dict(self._counts)
                read = dict(self._read)
                self._dirty = False
            now = time.monotonic()
            if now - window_t >= 1.0:
                rates = {k: (counts[k] - window[k]) / (now - window_t) for k in counts}
                window_t, window = now, counts
            if not dirty and now - last_draw < 1.0:  # nothing new: refresh the rates only
                continue
            last_draw = now
            try:
                self._draw(latest, rates, read)
            except Exception:  # a drawing hiccup must never stop teleop
                logger.exception("Hand viewer: drawing failed")

    def _draw(self, latest: dict, rates: dict, read: dict) -> None:
        shown = {}
        for which, toggle in (
            ("commanded", self._show_commanded),
            ("measured", self._show_measured),
        ):
            values = latest[which]
            shown[which] = toggle.value and values is not None
            self._hands.set_visible(which, shown[which])
            if values is not None:
                self._hands.pose(which, values)
        # Overlaid on a measured hand, the commanded one is a ghost; alone, it is solid.
        ghost = shown["measured"] and not self._side_by_side()
        if ghost != self._ghost:
            self._ghost = ghost
            self._hands.set_style(
                "commanded", 0.35 if ghost else None, (235, 240, 255) if ghost else None
            )
        cmd, meas = latest["commanded"], latest["measured"]
        if read["failed"]:
            reads = "**motor read failed**: measured hidden until a read succeeds"
        elif meas is None:
            reads = "no motor reads yet (none in a dry run)"
        else:
            reads = f"motor reads {rates['measured']:.0f}/s" + (
                "" if read["ms"] is None else f" ({read['ms']:.0f} ms each)"
            )
        if cmd is None and meas is None and not read["failed"]:
            return
        diffs = {j: d for j in self._hands.joint_ids if (d := _diff(cmd, meas, j)) is not None}
        worst = max(diffs, key=lambda j: abs(diffs[j]), default=None)
        self._status.content = f"commands {rates['commanded']:.0f}/s · {reads}" + (
            "" if worst is None else f" · largest difference: **{worst} {diffs[worst]:+.0f}°**"
        )
        self._table.content = _joint_table(self._hands.joint_ids, cmd, meas)


def _joint_table(joint_ids: list[str], cmd: dict | None, meas: dict | None) -> str:
    def cell(values: dict | None, j: str) -> str:
        v = None if values is None else values.get(j)
        return "-" if v is None or not np.isfinite(v) else f"{round(v)}°"  # round: no "-0°"

    order = [j for f in ("wrist",) + FINGERS for j in joint_ids if j.split("_")[0] == f]
    order += [j for j in joint_ids if j not in order]
    rows = ["| joint | commanded | measured | difference |", "|---|---|---|---|"]
    for j in order:
        diff, d = _diff(cmd, meas, j), "-"
        if diff is not None:
            d = f"**{round(diff):+d}°**" if abs(diff) > DIFF_DEG else f"{round(diff):+d}°"
        rows.append(f"| {j} | {cell(cmd, j)} | {cell(meas, j)} | {d} |")
    return "\n".join(rows)


def _diff(cmd: dict | None, meas: dict | None, j: str) -> float | None:
    """Measured minus commanded, if both are there and finite."""
    if cmd is None or meas is None or cmd.get(j) is None or meas.get(j) is None:
        return None
    diff = meas[j] - cmd[j]
    return diff if np.isfinite(diff) else None
