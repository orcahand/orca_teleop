"""MediaPipe hand-tracking gRPC publisher.

Run this on the operator's machine (any OS with a webcam). It captures hand
landmarks via MediaPipe and streams them to the robot-side ``IngressServer``
over gRPC.

Usage::

    # Stream to robot on the same machine
    python -m orca_teleop.ingress.mediapipe.publisher

    # Stream to a remote robot
    python -m orca_teleop.ingress.mediapipe.publisher --server 192.168.1.42:50051

    # Left hand, high confidence
    python -m orca_teleop.ingress.mediapipe.publisher --hand left --confidence 0.9

    # Also record the session (raw frames + landmarks) for offline replay
    python -m orca_teleop.ingress.mediapipe.publisher --record recordings/thumb-01
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import cv2
import grpc
import mediapipe as mp
import numpy as np

from orca_teleop.constants import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
)
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.landmark_recording import LandmarkRecorder
from orca_teleop.utils import RateTicker

_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _draw_hand_landmarks(frame: np.ndarray, landmarks, color: tuple = (0, 255, 0)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1)


logger = logging.getLogger(__name__)

_LOG_EVERY = 30
CONNECT_TIMEOUT_S = 10.0


class MediaPipePublisher:
    """Captures hand landmarks from a webcam and streams them over gRPC."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        handedness: str = "right",
        confidence: float = 0.7,
        show_video: bool = False,
        camera_index: int = 0,
        camera_size: tuple[int, int] | None = None,
        capture_fps: float = DEFAULT_CAMERA_FPS,
        record_dir: str | None = None,
    ) -> None:
        self._server_address = server_address
        self._handedness = handedness.lower()
        self._confidence = confidence
        self._show_video = show_video
        self._camera_index = int(camera_index)
        self._camera_size = camera_size
        self._capture_fps = float(capture_fps)
        self._record_dir = record_dir
        self._recorder: LandmarkRecorder | None = None
        self._frame_by_timestamp: dict[int, int] = {}

        # MediaPipe setup
        mediapipe_task_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(mediapipe_task_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=self._confidence,
            min_hand_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # Latest frame data (written by callback, read by stream generator)
        self._lock = threading.Lock()
        # The stream generator sleeps on this until a result arrives, instead of polling.
        self._new_result = threading.Condition(self._lock)
        self._latest_keypoints: np.ndarray | None = None
        self._latest_capture_ns = 0
        self._fresh = False

        # Visualization state
        self._latest_frame: np.ndarray | None = None
        self._latest_image_landmarks = None

    def _on_result(self, result, _output_image, timestamp_ms: int) -> None:
        """MediaPipe async callback — fires on each detection."""
        with self._lock:
            frame = self._frame_by_timestamp.pop(timestamp_ms, -1)
        if not result.hand_landmarks:
            return

        # Only accept the hand we care about
        detected_hand = result.handedness[0][0].category_name.lower()
        if detected_hand != self._handedness:
            return

        world_landmarks = result.hand_world_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)

        if self._recorder is not None:
            image = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
            score = result.handedness[0][0].score
            self._recorder.add_landmarks(frame, keypoints, image, detected_hand, score)

        with self._lock:
            self._latest_keypoints = keypoints
            self._latest_capture_ns = timestamp_ms * 1_000_000  # MediaPipe's stamp = capture time
            self._fresh = True
            if self._show_video:
                self._latest_image_landmarks = result.hand_landmarks[0]
            self._new_result.notify()

    def _frame_generator(self):
        """Yield HandFrame protos as fast as new data arrives."""
        while True:
            with self._new_result:
                if not self._new_result.wait_for(lambda: self._fresh, timeout=0.1):
                    continue
                kp = self._latest_keypoints.copy()
                capture_ns = self._latest_capture_ns
                self._fresh = False

            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness=self._handedness,
                timestamp_ns=capture_ns,
            )

    def run(self) -> None:
        """Open the webcam, connect to the server, and stream until interrupted."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self._camera_index}. On Linux a UVC "
                "camera often registers more than one /dev/video node and only the "
                "first carries frames; try another index with --camera-index."
            )

        if self._camera_size is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_size[1])
        cap.set(cv2.CAP_PROP_FPS, self._capture_fps)
        logger.info(
            "Camera: %.0fx%.0f, pacing to %.0f fps",
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self._capture_fps,
        )

        logger.info(
            "Connecting to %s (hand=%s, confidence=%.2f)",
            self._server_address,
            self._handedness,
            self._confidence,
        )
        channel = grpc.insecure_channel(self._server_address)
        try:
            grpc.channel_ready_future(channel).result(timeout=CONNECT_TIMEOUT_S)
        except grpc.FutureTimeoutError as err:
            channel.close()
            cap.release()
            raise RuntimeError(
                f"No ingress server answered at {self._server_address} within "
                f"{CONNECT_TIMEOUT_S:.0f}s. Start it before the publisher."
            ) from err
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)

        # Streams in a background thread; the future carries any transport failure.
        stream_future = stub.StreamHandFrames.future(self._frame_generator())

        ticker = RateTicker(dt=1.0 / self._capture_fps)
        read_ms: list[float] = []
        window_start = time.perf_counter()

        try:
            if self._record_dir is not None:
                self._recorder = LandmarkRecorder(
                    self._record_dir,
                    meta={
                        "handedness": self._handedness,
                        "confidence": self._confidence,
                        "camera_index": self._camera_index,
                        "camera_size": [
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ],
                        "capture_fps": self._capture_fps,
                        "landmarker": "hand_landmarker.task (LIVE_STREAM)",
                        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    fps=self._capture_fps,
                )
                logger.info("Recording to %s", self._record_dir)

            while True:
                t_read_start = time.perf_counter()
                ret, frame = cap.read()
                read_ms.append((time.perf_counter() - t_read_start) * 1e3)
                if not ret:
                    continue

                if self._show_video:
                    with self._lock:
                        self._latest_frame = frame.copy()

                capture_time_ns = time.time_ns()
                timestamp_ms = capture_time_ns // 1_000_000
                if self._recorder is not None:
                    index = self._recorder.add_frame(frame, capture_time_ns)
                    with self._lock:
                        self._frame_by_timestamp[timestamp_ms] = index
                        while len(self._frame_by_timestamp) > 300:  # frames MediaPipe dropped
                            self._frame_by_timestamp.pop(next(iter(self._frame_by_timestamp)))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self._landmarker.detect_async(mp_image, timestamp_ms)

                if self._show_video:
                    self._display_frame()
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if stream_future.done():
                    raise RuntimeError(
                        f"Ingress stream to {self._server_address} ended early: "
                        f"{stream_future.exception() or 'server closed the stream'}"
                    )

                if len(read_ms) >= _LOG_EVERY:
                    elapsed_s = time.perf_counter() - window_start
                    logger.info(
                        "Capture | %.1f fps | read %.2f ms",
                        len(read_ms) / elapsed_s,
                        sum(read_ms) / len(read_ms),
                    )
                    read_ms.clear()
                    window_start = time.perf_counter()

                ticker.tick()

        except KeyboardInterrupt:
            pass
        finally:
            stream_future.cancel()
            channel.close()
            cap.release()
            self._landmarker.close()
            if self._recorder is not None:
                frames, results = self._recorder.close()
                logger.info(
                    "Recorded %d frames, %d with a hand, to %s", frames, results, self._record_dir
                )
            if self._show_video:
                cv2.destroyAllWindows()
            logger.info("Publisher shut down.")

    def _display_frame(self) -> None:
        """Show the webcam feed with landmarks overlaid."""
        with self._lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
            image_landmarks = self._latest_image_landmarks

        if image_landmarks:
            _draw_hand_landmarks(frame, image_landmarks)

        cv2.imshow("MediaPipe Publisher", frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream hand landmarks from a webcam to the orca_teleop server via gRPC.",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Which hand to track (default: right)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="MediaPipe detection confidence (default: 0.7)",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show webcam feed with landmarks overlay",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)"
    )
    parser.add_argument(
        "--camera-size",
        default=None,
        help=(
            "Override the capture resolution, as WIDTHxHEIGHT "
            f"(e.g. {DEFAULT_CAMERA_WIDTH}x{DEFAULT_CAMERA_HEIGHT}). "
            "Omitted, the camera keeps whatever format it opens in. Smaller "
            "frames cost less per-frame work but do not raise the capture rate, "
            "which the camera paces."
        ),
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help=f"Target capture rate in frames per second (default: {DEFAULT_CAMERA_FPS:g})",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="DIR",
        help="Also record raw frames and landmarks into DIR (must be new or empty)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    camera_size = None
    if args.camera_size is not None:
        try:
            width_str, _, height_str = args.camera_size.partition("x")
            camera_size = (int(width_str), int(height_str))
        except ValueError:
            parser.error(f"--camera-size must be WIDTHxHEIGHT, got {args.camera_size!r}")

    publisher = MediaPipePublisher(
        server_address=args.server,
        handedness=args.hand,
        confidence=args.confidence,
        show_video=args.show_video,
        camera_index=args.camera_index,
        camera_size=camera_size,
        capture_fps=args.capture_fps,
        record_dir=args.record,
    )
    publisher.run()


if __name__ == "__main__":
    main()
