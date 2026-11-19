"""The glove bridge's messages: ``GloveState`` protobufs on a ZMQ PUB socket.

One message per glove frame, holding each streaming hand as 25 node poses in
``conversion.MANUS_NODE_NAMES`` order (see ``conversion`` for the units and frame).
``bridge.py`` and ``fake_bridge.py`` send them, ``publisher.py`` reads them, and so
does the legacy C++ client (orcahand/manus-client), port 2044 by default.

Needs only numpy and protobuf, so ``bridge.py`` can load it on the glove computer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOURCE_DIR = str(Path(__file__).resolve().with_name("source"))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import manus_stream_pb2  # noqa: E402  (generated, lives in source/)

DEFAULT_PORT = 2044
SIDES = ("left", "right")


def zmq_address(address: str) -> str:
    """A bridge address as ZMQ wants it: ``HOST:PORT`` gets ``tcp://`` in front."""
    return address if "://" in address else f"tcp://{address}"


def encode(hands: dict[str, np.ndarray], t_ns: int, frame_id: int = 0) -> bytes:
    """``{side: (N, 3) positions or (N, 7) position + quaternion wxyz}`` -> message bytes."""
    msg = manus_stream_pb2.GloveState()
    msg.header.stamp.sec = int(t_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(t_ns % 1_000_000_000)
    msg.header.frame_id = str(frame_id)
    for side, nodes in hands.items():
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        poses = msg.left_mocap_pose if side == "left" else msg.right_mocap_pose
        for node in np.asarray(nodes, dtype=np.float64):
            pose = poses.add()
            pose.position.x, pose.position.y, pose.position.z = node[:3]
            if node.shape[0] >= 7:
                q = pose.orientation
                q.w, q.x, q.y, q.z = node[3:7]
            else:
                pose.orientation.w = 1.0
    return msg.SerializeToString()


def decode(raw: bytes) -> tuple[int, dict[str, np.ndarray]]:
    """Message bytes -> (stamp in ns, ``{side: (N, 3) positions}`` for the hands present)."""
    msg = manus_stream_pb2.GloveState()
    msg.ParseFromString(raw)
    t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    hands = {}
    for side, poses in (("left", msg.left_mocap_pose), ("right", msg.right_mocap_pose)):
        if len(poses):
            hands[side] = np.array(
                [[p.position.x, p.position.y, p.position.z] for p in poses], dtype=np.float32
            )
    return t_ns, hands
