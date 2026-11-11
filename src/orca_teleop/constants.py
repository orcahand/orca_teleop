JOIN_TIMEOUT = 3.0
HEARTBEAT_INTERVAL = 0.1
QUEUES_MAXSIZE = 8
# Tracking loss (DD3): no frame for LOSS_TIMEOUT_S -> hold; the return eases in over RAMP_S.
LOSS_TIMEOUT_S = 0.15
# A slow steady stream raises the loss timeout to 3 frame intervals, never above this.
LOSS_TIMEOUT_MAX_S = 0.5
RAMP_S = 0.25
# After this many failed motor reads in a row (viewer, joint log), wait READ_RETRY_S
# before the next; a silent motor makes each read slow, and the commands come first.
READ_FAILS_IN_ROW = 3
READ_RETRY_S = 1.0
INGRESS_FPS = 30
MOTION_NUM_STEPS = 1

WRIST_MOTOR_IDX = 16

DEFAULT_PORT = 50051
DEFAULT_HAND = "right"
DEFAULT_CONFIDENCE = 0.7
_NUM_KEYPOINTS = 21
_COORDS_PER_POINT = 3
_EXPECTED_LEN = _NUM_KEYPOINTS * _COORDS_PER_POINT  # 63

SIM_RENDER_MODE = "human"

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30.0

MACOS_PLATFORM = "Darwin"
STALE_FRAME_TIMEOUT_S = 2.0
DEFAULT_MAX_PROBE_INDEX = 8
