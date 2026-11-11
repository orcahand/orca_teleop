"""Sim-backed ``RobotSink`` driving an ``orca_sim`` Gymnasium env.

Mirrors the physical-robot path of the teleop pipeline: the ingress and
retargeter stages are unchanged, streaming``OrcaJointPositions``to a
``OrcaHandSimSink``, stepping a MuJoCo environment from ``orca_sim``.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orca_core
from orca_core import OrcaHand, OrcaJointPositions
from orca_sim.envs import BaseOrcaHandEnv

from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
from orca_teleop.pipeline import _SHUTDOWN, RecordableSink, SinkObservation
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)
DEFAULT_RENDER_FPS = 30

# Maps the single-letter finger prefix used in the orca_sim v2 MuJoCo joint
# names (e.g. ``right_t-mcp``) to the orca_core finger name.
_SIM_FINGER_LETTERS = {"t": "thumb", "i": "index", "m": "middle", "r": "ring", "p": "pinky"}


def _sim_joint_to_config_id(core_name: str, valid_ids: set[str]) -> str | None:
    """Translate a (hand-prefix-stripped) MuJoCo joint name to an orca_core
    ``config.joint_ids`` entry, or ``None`` if it has no counterpart.

    Two naming conventions are bridged. orca_sim ``v1`` names already match the
    config ids verbatim (``thumb_mcp``, ``wrist``). orca_sim ``v2`` abbreviates
    finger+segment as ``<letter>-<segment>`` (``t-mcp``, ``p-abd``); the thumb's
    distal joint is ``t-pip`` in the model but ``thumb_dip`` in the config.

    The result is validated against ``valid_ids`` by the caller, so any future
    naming drift fails loudly rather than silently driving the wrong joint.
    """
    if core_name in valid_ids:  # v1: bare id matches the config verbatim
        return core_name
    if "-" in core_name:  # v2: '<finger-letter>-<segment>'
        letter, _, segment = core_name.partition("-")
        finger = _SIM_FINGER_LETTERS.get(letter)
        if finger is not None:
            candidate = f"{finger}_{segment}"
            if candidate in valid_ids:
                return candidate
            # The thumb's distal joint is named 'pip' in the v2 model but 'dip'
            # in the orca_core config.
            if finger == "thumb" and segment == "pip" and "thumb_dip" in valid_ids:
                return "thumb_dip"
    return None


@dataclass(frozen=True)
class SimCameraConfig:
    name: str = "frontal"
    width: int = 320
    height: int = 240


class OrcaHandSimSink(RecordableSink):
    """``RobotSink`` that steps an ``orca_sim`` env from the actions queue.

    ``OrcaJointPositions`` values arrive in physical degrees (retargeter
    convention); ``_to_action_array`` converts them to radians before writing
    to the MuJoCo ctrl vector, which uses radians throughout.
    """

    def __init__(
        self,
        env_name: str = "right",
        version: str | None = "v2",
        render_mode: str = "human",
        camera_config: SimCameraConfig | None = None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
    ) -> None:
        self._env_name = env_name
        self._version = version
        self._render_mode = render_mode
        self._camera_config = SimCameraConfig() if camera_config is None else camera_config
        # OpenCV cameras composed into the observation alongside MuJoCo env renders.
        self._cameras = CameraManager(camera_configs or [])
        self._env: BaseOrcaHandEnv = None
        self._actuator_joint_names: list[str] = []
        self._joint_qpos_adr: np.ndarray = np.empty(0, dtype=np.int64)
        self._last_action: OrcaJointPositions | None = None
        self._dt: float = 1.0 / DEFAULT_RENDER_FPS
        self._renderer: Any | None = None
        self._record_camera: Any | None = None
        # Known before connect(): run() builds the retargeter before the sink connects.
        self._retarget_model_path: str | None = self._resolve_retarget_model_path(version)
        self._hand_config: Any | None = None

    def connect(self) -> None:
        import mujoco
        from orca_sim import OrcaHandLeft, OrcaHandRight

        builders = {"left": OrcaHandLeft, "right": OrcaHandRight}
        if self._env_name not in builders:
            raise ValueError(
                f"Unknown orca_sim env '{self._env_name}'. " f"Choices: {sorted(builders)}"
            )

        kwargs: dict[str, Any] = {"render_mode": self._render_mode}
        if self._version is not None:
            kwargs["version"] = self._version
        env = builders[self._env_name](**kwargs)
        env.reset()

        self._env = env
        self._retarget_model_path = self._resolve_retarget_model_path(getattr(env, "version", None))

        # orca_sim's env exposes only the raw MuJoCo model/data, so derive the
        # joint ids and neutral pose from the matching orca_core config and map
        # the MuJoCo actuators onto those ids (the keys the retargeter emits).
        hand_config = self._load_hand_config()
        self._hand_config = hand_config
        self._actuator_joint_names, self._joint_qpos_adr = self._map_actuators(
            mujoco, env.model, hand_config
        )

        # Hold neutral pose until the first retargeted command arrives.
        self._last_action = OrcaJointPositions(hand_config.neutral_position)

        self._dt = 1.0 / float(getattr(env, "metadata", {}).get("render_fps", DEFAULT_RENDER_FPS))

        self._renderer = mujoco.Renderer(
            env.model,
            height=self._camera_config.height,
            width=self._camera_config.width,
        )
        self._record_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self._record_camera)

        if self._camera_config.name in self._cameras.names:
            raise ValueError(
                f"Real camera name {self._camera_config.name!r} collides with the sim "
                "render camera name; rename it (e.g. --camera side:1) so both "
                "observations get distinct keys."
            )
        self._cameras.open()
        self._cameras.ensure_live()

        logger.info(
            "SimSink connected: env=%s version=%s actuators=%d dt=%.3fs cameras=%s",
            self._env_name,
            getattr(env, "version", "?"),
            len(self._actuator_joint_names),
            self._dt,
            self.camera_shapes,
        )

    @property
    def joint_ids(self) -> list[str]:
        return list(self._actuator_joint_names)

    @property
    def retarget_model_path(self) -> str | None:
        return self._retarget_model_path

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        # The intrinsic MuJoCo render plus any real cameras this sink owns.
        return {
            self._camera_config.name: (
                self._camera_config.height,
                self._camera_config.width,
                3,
            ),
            **self._cameras.shapes,
        }

    def get_observation(self) -> SinkObservation:
        return SinkObservation(
            joint_state=self._read_joint_state(),
            images={self._camera_config.name: self._render(), **self._cameras.capture()},
        )

    def _read_joint_state(self) -> np.ndarray:
        assert self._env is not None, "connect() must be called before get_observation()"
        # qpos addresses are stored in actuator order, so the returned array
        # lines up with ``self._actuator_joint_names`` (i.e. ``joint_ids``).
        state_rad = self._env.data.qpos[self._joint_qpos_adr]
        return np.rad2deg(state_rad).astype(np.float32)

    def _render(self) -> np.ndarray:
        assert self._env is not None, "connect() must be called before get_observation()"
        assert self._renderer is not None
        assert self._record_camera is not None
        self._renderer.update_scene(self._env.data, camera=self._record_camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def dispatch_action(self, action: OrcaJointPositions) -> None:
        assert self._env is not None, "connect() must be called before dispatch_action()"
        self._env.step(self._to_action_array(action))

    def go_home(self) -> None:
        assert self._env is not None, "connect() must be called before go_home()"
        self._last_action = self.home_position()
        self.dispatch_action(self._last_action)

    def home_position(self) -> OrcaJointPositions:
        assert self._hand_config is not None, "connect() must be called before home_position()"
        positions = dict(self._hand_config.neutral_position)
        positions["wrist"] = 0.0
        return OrcaJointPositions(positions)

    def run_loop(
        self,
        actions_q: queue.Queue[OrcaJointPositions | object],
        stop_event: threading.Event,
    ) -> None:
        assert self._env is not None, "connect() must be called before run_loop()"
        assert self._last_action is not None

        ticker = RateTicker(dt=self._dt)

        while not stop_event.is_set():
            shutdown_received = False
            try:
                item = actions_q.get_nowait()
                if item is _SHUTDOWN:
                    shutdown_received = True
                elif isinstance(item, OrcaJointPositions):
                    try:
                        self._last_action = item
                    except Exception:
                        logger.exception(
                            "SimSink failed to convert OrcaJointPositions; holding last action"
                        )
            except queue.Empty:
                pass

            if shutdown_received:
                break

            try:
                self.dispatch_action(self._last_action)

            except Exception as e:
                logger.exception("orca_sim env.step() failed: %s", e)
                break

            ticker.tick()  # sleeps to control frequency

    def close(self) -> None:
        if self._env is None:
            return
        try:
            self._cameras.close()
            if self._renderer is not None:
                self._renderer.close()
            self._env.close()
        except Exception:
            logger.exception(".close() encountered an error")
        finally:
            self._env = None
            self._renderer = None
            self._record_camera = None

    def _to_action_array(self, positions: OrcaJointPositions) -> np.ndarray:
        # Retargeter outputs degrees; MuJoCo accepts radians
        return np.deg2rad(positions.as_array(self._actuator_joint_names))

    def _load_hand_config(self) -> Any:
        """Load the orca_core hand config matching this sim env (joint ids +
        neutral pose). orca_sim no longer ships an ``OrcaHand``, so the config
        is resolved from orca_core's packaged models for this env/version."""
        if self._retarget_model_path is None:
            raise RuntimeError(
                f"Could not resolve an orca_core model config for env "
                f"'{self._env_name}' version '{self._version}'. The sim sink "
                "needs it for joint ids and the neutral pose."
            )
        return OrcaHand(self._retarget_model_path).config

    def _map_actuators(
        self, mujoco: Any, model: Any, hand_config: Any
    ) -> tuple[list[str], np.ndarray]:
        """Map MuJoCo actuators (in ctrl order) onto orca_core ``joint_ids``.

        Returns the per-actuator config joint id and the qpos address of each
        actuator's transmission joint, both in MuJoCo actuator order so the
        ctrl vector and the observed joint state stay aligned with the action keys.
        """
        valid_ids = set(hand_config.joint_ids)
        prefix = f"{self._env_name}_"
        joint_names: list[str] = []
        qpos_adr: list[int] = []
        unmapped: list[str] = []
        for actuator in range(model.nu):
            jnt_id = int(model.actuator_trnid[actuator, 0])
            mj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            core = mj_name.removeprefix(prefix) if mj_name is not None else ""
            config_id = _sim_joint_to_config_id(core, valid_ids)
            if config_id is None:
                unmapped.append(mj_name)
                continue
            joint_names.append(config_id)
            qpos_adr.append(int(model.jnt_qposadr[jnt_id]))

        if unmapped:
            raise ValueError(
                f"orca_sim actuators have no orca_core joint_id counterpart: "
                f"{unmapped}. The sim model naming may have drifted from the config."
            )
        if set(joint_names) != valid_ids or len(set(joint_names)) != len(joint_names):
            raise ValueError(
                "Actuator->joint_id mapping is not a 1:1 cover of the orca_core "
                f"config. Mapped {sorted(joint_names)} vs config {sorted(valid_ids)}."
            )
        return joint_names, np.array(qpos_adr, dtype=np.int64)

    def _resolve_retarget_model_path(self, version: str | None) -> str | None:
        if not version:
            return None
        model_path = (
            Path(orca_core.__file__).resolve().parent
            / "models"
            / version
            / f"orcahand-{self._env_name}"
            / "config.yaml"
        )
        return str(model_path) if model_path.exists() else None
