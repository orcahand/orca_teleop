"""How faithfully the robot hand follows a recorded human hand.

Used by the regression test (``tests/test_fidelity.py``). Everything is measured on
the commanded joint angles, the way the robot would execute them:

- pinch success: at moments the human's thumb touches a fingertip (seen in two
  camera views), does the robot's thumb touch that finger? Measured as the
  distance between the fingertips' collision shapes from the hand model
  (convex pieces within about 1 mm of the skin);
- false touches: the robot's fingertips touch while the human's thumb is well away
  from that finger's last two phalanges (which the robot's last segment stands for);
- curl agreement: correlation of each robot joint with the human joint it copies
  (finger MCP and spread, the finger's PIP+DIP bend, thumb MCP and IP);
- lag: the delay at which that correlation peaks;
- jitter: frame-to-frame noise in the commands (second difference);
- limit saturation: how often a joint sits at either end of its range;
- solve time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_LANDMARKS = (4, 8, 12, 16, 20)
_FINGER_LANDMARKS = {  # MCP, PIP, DIP, TIP
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TOUCH_MM = 2.0  # robot fingertips this close count as touching (shapes are within ~1 mm)
APART_CM = 3.0  # human last phalanges this far apart (landmarks): certainly not touching
FRAME_S = 1.0 / 30.0


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle between row vectors a and b, degrees."""
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def human_joint_angles(world: np.ndarray) -> dict[str, np.ndarray]:
    """The human angle each robot joint copies, per frame, degrees.

    ``world``: (n, 21, 3) MediaPipe landmarks. The robot's last finger segment spans
    the human's middle and distal phalanges, so ``{finger}_pip`` is compared with the
    bend between the human's proximal phalanx (MCP->PIP) and PIP->TIP.
    Spread (``{finger}_abd``) is the proximal phalanx's angle to the metacarpal in
    the palm plane, positive towards the thumb, as the robot's.
    """
    w = np.asarray(world, dtype=np.float64)
    out: dict[str, np.ndarray] = {
        "thumb_mcp": _angle_deg(w[:, 2] - w[:, 1], w[:, 3] - w[:, 2]),
        "thumb_dip": _angle_deg(w[:, 3] - w[:, 2], w[:, 4] - w[:, 3]),
    }
    normal = np.cross(w[:, 5] - w[:, 0], w[:, 17] - w[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    towards_thumb = w[:, 5] - w[:, 17]
    for finger, (mcp, pip, _dip, tip) in _FINGER_LANDMARKS.items():
        metacarpal = w[:, mcp] - w[:, 0]
        proximal = w[:, pip] - w[:, mcp]
        out[f"{finger}_mcp"] = _angle_deg(metacarpal, proximal)
        out[f"{finger}_pip"] = _angle_deg(proximal, w[:, tip] - w[:, pip])
        in_plane = lambda v: v - np.sum(v * normal, axis=1, keepdims=True) * normal  # noqa: E731
        m, p = in_plane(metacarpal), in_plane(proximal)
        sign = np.sign(np.sum(np.cross(m, p) * np.cross(m, towards_thumb), axis=1))
        out[f"{finger}_abd"] = sign * _angle_deg(m, p)
    return out


def human_tip_gaps_cm(world: np.ndarray) -> np.ndarray:
    """(n, 4) human thumb-tip to index/middle/ring/pinky-tip landmark distance, cm."""
    w = np.asarray(world, dtype=np.float64)
    return np.linalg.norm(w[:, list(TIP_LANDMARKS[1:])] - w[:, [4]], axis=2) * 100.0


def human_distal_gaps_cm(world: np.ndarray, samples: int = 11) -> np.ndarray:
    """(n, 4) closest distance between the human thumb's last phalanx (IP->TIP landmarks)
    and each finger's PIP->TIP landmarks, the part the robot's last segment stands for, cm.

    Closest points are found on ``samples`` points along each segment (about 4 mm apart).
    """
    w = np.asarray(world, dtype=np.float64)
    s = np.linspace(0.0, 1.0, samples)[None, :, None]
    thumb = w[:, None, 3] + s * (w[:, None, 4] - w[:, None, 3])  # (n, k, 3)
    out = np.empty((len(w), 4))
    for i, (_mcp, pip, _dip, tip) in enumerate(_FINGER_LANDMARKS.values()):
        finger = w[:, None, pip] + s * (w[:, None, tip] - w[:, None, pip])
        d = np.linalg.norm(thumb[:, :, None] - finger[:, None, :], axis=3)
        out[:, i] = d.min(axis=(1, 2)) * 100.0
    return out


class FingertipGaps:
    """Distance between the robot thumb's and each finger's last segment, from the model's
    collision shapes (pinocchio + coal)."""

    def __init__(self, urdf_path: str, hand_type: str, joint_ids: list[str]) -> None:
        import pinocchio as pin

        from orca_teleop.utils import urdf_with_local_meshes

        self._pin = pin
        xml = urdf_with_local_meshes(urdf_path)
        self.model = pin.buildModelFromXML(xml)
        self.geom = pin.buildGeomFromUrdfString(self.model, xml, pin.GeometryType.COLLISION)
        self._data = self.model.createData()

        def pieces(link: str) -> list[int]:
            frame = self.model.getFrameId(link)
            joint = self.model.frames[frame].parentJoint
            ids = [
                i
                for i, g in enumerate(self.geom.geometryObjects)
                if g.parentJoint == joint and g.name.startswith(link)
            ]
            if not ids:
                raise ValueError(f"{urdf_path} has no collision shapes for {link}")
            return ids

        thumb = pieces(f"{hand_type}_thumb_dp")
        self._pairs: list[list[int]] = []
        for finger in FINGERS[1:]:
            ids = []
            for a in thumb:
                for b in pieces(f"{hand_type}_{finger}_dp"):
                    ids.append(len(self.geom.collisionPairs))
                    self.geom.addCollisionPair(pin.CollisionPair(a, b))
            self._pairs.append(ids)
        self._gdata = self.geom.createData()
        self._q_index = []
        for jid in joint_ids:
            jid_model = self.model.getJointId(f"{hand_type}_{jid}")
            if jid_model >= self.model.njoints:
                raise ValueError(f"{urdf_path} has no joint {hand_type}_{jid}")
            self._q_index.append(int(self.model.idx_qs[jid_model]))

    def __call__(self, q_deg: np.ndarray) -> np.ndarray:
        """(4,) mm for one pose given in ``joint_ids`` order, degrees."""
        q = np.zeros(self.model.nq)
        q[self._q_index] = np.radians(q_deg)
        self._pin.computeDistances(self.model, self._data, self.geom, self._gdata, q)
        results = self._gdata.distanceResults
        return np.array([min(results[i].min_distance for i in ids) * 1000.0 for ids in self._pairs])


@dataclass
class SessionResult:
    """Per-frame outputs of replaying one recording through a retargeter."""

    name: str
    joint_ids: list[str]
    lower_deg: np.ndarray
    upper_deg: np.ndarray
    q_deg: np.ndarray  # (n, J); NaN while calibrating
    solve_ms: np.ndarray  # (n,)
    human: dict[str, np.ndarray]  # human_joint_angles
    human_gap_cm: np.ndarray  # (n, 4) human_distal_gaps_cm: apart beyond APART_CM = no touch
    contact: np.ndarray | None  # (n, 4) bool: both camera views show the touch
    robot_gap_mm: np.ndarray = field(default=None)  # (n, 4)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) < 1e-6 or np.std(b[ok]) < 1e-6:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _best_lag(human: np.ndarray, robot: np.ndarray, max_lag: int = 10) -> float | None:
    """Delay in frames at which the robot correlates best with the human (parabolic peak)."""
    corrs = [_corr(human[: len(human) - k], robot[k:]) for k in range(max_lag + 1)]
    if any(c is None for c in corrs):
        return None
    k = int(np.argmax(corrs))
    if 0 < k < max_lag:
        a, b, c = corrs[k - 1], corrs[k], corrs[k + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return k + 0.5 * (a - c) / denom
    return float(k)


def score_session(r: SessionResult) -> dict:
    ok = np.all(np.isfinite(r.q_deg), axis=1)
    q = r.q_deg[ok]
    out: dict = {"session": r.name, "frames": int(len(ok)), "scored": int(ok.sum())}

    pinch = {}
    for i, finger in enumerate(FINGERS[1:]):
        row: dict = {}
        gap = r.robot_gap_mm[ok, i]
        apart = r.human_gap_cm[ok, i] > APART_CM
        row["false_touch_pct"] = (
            float(100 * np.mean(gap[apart] <= TOUCH_MM)) if apart.any() else None
        )
        if r.contact is not None:
            at = r.contact[ok, i]
            row["contact_frames"] = int(at.sum())
            if at.sum() >= 5:
                row["success_pct"] = float(100 * np.mean(gap[at] <= TOUCH_MM))
                row["gap_mm_median"] = float(np.median(gap[at]))
        pinch[finger] = row
    out["pinch"] = pinch

    curl = {}
    for jid, human in r.human.items():
        if jid not in r.joint_ids:
            continue
        # Gaps stay in as NaN (both skip them pair by pair), so a shift of k rows is
        # always k frames of time. The human side keeps only the frames the robot was
        # given: elsewhere it may come from another source (the front camera where the
        # two views disagreed), which would favour the one shift that never pairs them.
        robot = np.where(ok, r.q_deg[:, r.joint_ids.index(jid)], np.nan)
        human = np.where(ok, human, np.nan)
        curl[jid] = {"corr": _corr(human, robot), "lag_frames": _best_lag(human, robot)}
    out["curl"] = curl

    out["at_lower_pct"] = {
        jid: float(100 * np.mean(q[:, j] <= r.lower_deg[j] + 1.0))
        for j, jid in enumerate(r.joint_ids)
    }
    out["at_upper_pct"] = {
        jid: float(100 * np.mean(q[:, j] >= r.upper_deg[j] - 1.0))
        for j, jid in enumerate(r.joint_ids)
    }
    # Over three consecutive scored frames only: across a gap it would be a jump, not jitter.
    second = np.abs(r.q_deg[2:] - 2 * r.q_deg[1:-1] + r.q_deg[:-2])
    second = second[np.isfinite(second)]
    out["jitter_deg"] = float(np.median(second)) if len(second) else None
    out["solve_ms"] = {
        "median": float(np.median(r.solve_ms[ok])) if ok.any() else None,
        "p99": float(np.percentile(r.solve_ms[ok], 99)) if ok.any() else None,
    }
    return out


def summarize(scores: list[dict]) -> dict:
    """Corpus-level numbers: pooled per finger, averaged over sessions otherwise."""
    pinch = {}
    for finger in FINGERS[1:]:
        rows = [s["pinch"][finger] for s in scores]
        n = sum(row.get("contact_frames", 0) for row in rows if "success_pct" in row)
        hit = sum(
            row["success_pct"] / 100 * row["contact_frames"] for row in rows if "success_pct" in row
        )
        false = [row["false_touch_pct"] for row in rows if row.get("false_touch_pct") is not None]
        pinch[finger] = {
            "contact_frames": int(n),
            "success_pct": float(100 * hit / n) if n else None,
            "false_touch_pct": float(np.mean(false)) if false else None,
        }
    curl_ids = sorted({jid for s in scores for jid in s["curl"]})
    curl = {}
    for jid in curl_ids:
        cs = [
            s["curl"][jid]["corr"] for s in scores if s["curl"].get(jid, {}).get("corr") is not None
        ]
        lags = [
            s["curl"][jid]["lag_frames"]
            for s in scores
            if s["curl"].get(jid, {}).get("lag_frames") is not None
        ]
        curl[jid] = {
            "corr": float(np.mean(cs)) if cs else None,
            "lag_ms": float(np.median(lags) * FRAME_S * 1000) if lags else None,
        }
    flex = [v["corr"] for k, v in curl.items() if v["corr"] is not None and not k.endswith("_abd")]
    lags = [v["lag_ms"] for v in curl.values() if v["lag_ms"] is not None]
    at_lower = {
        j: float(np.mean([s["at_lower_pct"][j] for s in scores])) for j in scores[0]["at_lower_pct"]
    }
    at_upper = {
        j: float(np.mean([s["at_upper_pct"][j] for s in scores])) for j in scores[0]["at_upper_pct"]
    }
    return {
        "pinch": pinch,
        "curl": curl,
        "curl_corr_mean": float(np.mean(flex)) if flex else None,
        "lag_ms_median": float(np.median(lags)) if lags else None,
        "jitter_deg": float(
            np.median([s["jitter_deg"] for s in scores if s["jitter_deg"] is not None])
        ),
        "at_lower_pct": at_lower,
        "at_upper_pct": at_upper,
        "solve_ms_median": float(
            np.median(
                [s["solve_ms"]["median"] for s in scores if s["solve_ms"]["median"] is not None]
            )
        ),
        "solve_ms_p99": float(
            np.max([s["solve_ms"]["p99"] for s in scores if s["solve_ms"]["p99"] is not None])
        ),
    }
