"""MANUS glove skeleton -> the retargeter's 21 hand landmarks.

The wire contract between a glove bridge and ``publisher.py`` (``GloveState`` over ZMQ,
see ``source/manus_stream.proto``): each hand is a list of 25 node poses in
``MANUS_NODE_NAMES`` order, positions in metres, in the MANUS SDK's world coordinates
set up as Z up, X from the viewer, right-handed (``CoordinateSystemVUH`` with
``useWorldCoordinates``). ``bridge.py`` guarantees that order by sorting the SDK's
nodes by their ``NodeInfo`` (chain and joint type); the SDK's own array order is not
documented and differs between SDK versions and clients, so nothing here relies on it.

The retargeter derives its palm frame from the points themselves, so any right-handed
frame works and no axis remapping is needed. Only a mirror would break it (a right
hand would turn into a left one).

This module needs only numpy, so ``bridge.py`` can load it on the glove computer
without the rest of orca_teleop.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# The retargeter's 21 landmarks, MediaPipe's order.
MANO_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

_FINGERS = ("index", "middle", "ring", "pinky")
_FINGER_JOINTS = ("cmc", "mcp", "pip", "dip", "tip")

# The wire order: the hand (wrist), each long finger from its CMC (the metacarpal's
# base, which the 21 landmarks skip) to its tip, then the thumb. It is the order the
# legacy C++ client (orcahand/manus-client) was aligned to on hardware in May 2026.
MANUS_NODE_NAMES: tuple[str, ...] = (
    ("wrist",)
    + tuple(f"{f}_{j}" for f in _FINGERS for j in _FINGER_JOINTS)
    + ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip")
)
NUM_NODES = len(MANUS_NODE_NAMES)  # 25

# Wire index of each of the 21 landmarks.
MANUS_TO_MANO: list[int] = [MANUS_NODE_NAMES.index(name) for name in MANO_LANDMARK_NAMES]

# MANUS SDK enum values (ManusSDKTypes.h) by name, for when the SDK's own enum classes
# can't name them.
CHAIN_TYPES = {
    5: "FingerThumb",
    6: "FingerIndex",
    7: "FingerMiddle",
    8: "FingerRing",
    9: "FingerPinky",
    13: "Hand",
}
FINGER_JOINT_TYPES = {1: "Metacarpal", 2: "Proximal", 3: "Intermediate", 4: "Distal", 5: "Tip"}

_CHAIN_FINGER = {
    "FingerThumb": "thumb",
    "FingerIndex": "index",
    "FingerMiddle": "middle",
    "FingerRing": "ring",
    "FingerPinky": "pinky",
}
# A MANUS node sits at the joint at the base of its bone: the Metacarpal node at the
# CMC, Proximal at the MCP, Intermediate at the PIP, Distal at the DIP. The thumb has
# no Distal node (ManusSDKTypes.h: "thumb doesn't have it"), so its Intermediate node
# is the IP joint.
_FINGER_JOINT = {
    "Metacarpal": "cmc",
    "Proximal": "mcp",
    "Intermediate": "pip",
    "Distal": "dip",
    "Tip": "tip",
}
_THUMB_JOINT = {
    "Metacarpal": "cmc",
    "Proximal": "mcp",
    "Intermediate": "ip",
    "Distal": "ip",
    "Tip": "tip",
}


def node_name(chain_type: str, finger_joint_type: str) -> str | None:
    """Our name for a MANUS node, from its ``NodeInfo`` enum names; None if unused."""
    if chain_type == "Hand":
        return "wrist"
    finger = _CHAIN_FINGER.get(chain_type)
    if finger is None:
        return None
    joint = (_THUMB_JOINT if finger == "thumb" else _FINGER_JOINT).get(finger_joint_type)
    return None if joint is None else f"{finger}_{joint}"


def wire_order(names: Sequence[str | None]) -> list[int]:
    """For each of ``MANUS_NODE_NAMES``, its index in a skeleton whose nodes are ``names``.

    Raises ``ValueError`` naming what is missing or doubled: a wrong guess would move
    the robot's fingers wrongly instead of failing.
    """
    index: dict[str, int] = {}
    doubled = set()
    for i, name in enumerate(names):
        if name is None:
            continue
        if name in index:
            doubled.add(name)
        index[name] = i
    missing = [name for name in MANUS_NODE_NAMES if name not in index]
    if missing or doubled:
        raise ValueError(
            f"MANUS skeleton nodes don't match a hand: missing {missing}, doubled "
            f"{sorted(doubled)} (nodes: {list(names)})"
        )
    return [index[name] for name in MANUS_NODE_NAMES]


def wire_to_keypoints(nodes: np.ndarray) -> np.ndarray:
    """``(25, 3)`` wire-order node positions -> the retargeter's ``(21, 3)`` landmarks.

    Wrist-relative float32 metres in MediaPipe's landmark order, ready for
    ``HandFrame.keypoints``. No axis change: see the module docstring.
    """
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.shape != (NUM_NODES, 3):
        raise ValueError(f"expected ({NUM_NODES}, 3) node positions, got {nodes.shape}")
    keypoints = nodes[MANUS_TO_MANO]
    return keypoints - keypoints[0:1]
