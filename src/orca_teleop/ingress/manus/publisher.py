"""MANUS glove publisher: glove bridge (ZMQ) -> gRPC ``HandFrame`` messages.

Subscribes to a glove bridge's ``GloveState`` stream (``bridge.py`` on the computer
with the gloves, the legacy ``manus-client run``, or ``fake_bridge.py``), takes one
hand's 25 skeleton nodes, turns them into the retargeter's 21 landmarks
(``conversion``) and streams them to the ``IngressServer`` like any other publisher.
The newest glove frame wins. The bridge's warnings (licence, an uncalibrated or lost
glove) are logged once when they appear and once when they clear; ``warnings`` has them.

Usage::

    python -m orca_teleop.ingress.manus.publisher                                # bridge here
    python -m orca_teleop.ingress.manus.publisher --glove-address 192.168.64.5:2044
    python -m orca_teleop.ingress.manus.publisher --hand left --server robot:50051
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time

import grpc
import numpy as np

from orca_teleop.constants import DEFAULT_MANUS_ADDRESS
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.manus import conversion, wire

logger = logging.getLogger(__name__)

WAIT_LOG_S = 5.0  # how often to say what is (not) arriving
CONNECT_TIMEOUT_S = 10.0  # for the ingress server, which starts with the publisher
STOP_TIMEOUT_S = 2.0  # for the stream to close cleanly once stopped


NO_IMAGE = np.zeros((21, 3))  # a glove has no image landmarks to record


class ManusPublisher:
    """Receives MANUS glove frames over ZMQ and streams one hand to IngressServer via gRPC."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        handedness: str = "right",
        zmq_address: str = DEFAULT_MANUS_ADDRESS,
        record_dir: str | None = None,
        hand_ready: threading.Event | None = None,
    ) -> None:
        """``hand_ready``: set by the robot side once the hand is ready; the recording waits
        for it, so a start that fails leaves no folder (``None``: record from the first hand).
        """
        if handedness.lower() not in wire.SIDES:
            raise ValueError(f"handedness must be one of {wire.SIDES}, got {handedness!r}")
        self._server_address = server_address
        self._handedness = handedness.lower()
        self._zmq_address = wire.zmq_address(zmq_address)  # HOST:PORT works too

        self._lock = threading.Lock()
        self._latest_keypoints: np.ndarray | None = None
        self._fresh = threading.Event()
        self._stop = threading.Event()
        self._warnings: dict[str, str] = {}
        self._record_dir = record_dir
        self._hand_ready = hand_ready
        self._recording_pending = record_dir is not None
        self._recorder = None  # a LandmarkRecorder while recording

    def stop(self) -> None:
        self._stop.set()

    @property
    def warnings(self) -> dict[str, str]:
        """The glove bridge's active warnings (code -> text), as of its last message."""
        with self._lock:
            return dict(self._warnings)

    def update_warnings(self, raw: bytes) -> None:
        """Take a bridge message's warnings: log each new one, and each cleared one, once."""
        current = wire.bridge_warnings(raw)
        if current == self._warnings:
            return
        for code, text in current.items():
            if self._warnings.get(code) != text:
                logger.warning("Glove bridge: %s", text)
        for code in self._warnings.keys() - current.keys():
            logger.info("Glove bridge, cleared: %s", self._warnings[code])
        with self._lock:
            self._warnings = current

    def keypoints_from_message(self, raw: bytes) -> tuple[np.ndarray | None, list[str]]:
        """This hand's ``(21, 3)`` landmarks from one bridge message (None if it has none),
        and the hands the message has."""
        _, hands = wire.decode(raw)
        nodes = hands.get(self._handedness)
        keypoints = None if nodes is None else conversion.wire_to_keypoints(nodes)
        return keypoints, sorted(hands)

    def _zmq_receiver(self) -> None:
        """Background thread: subscribe to the bridge and keep the newest keypoints.

        If it fails, it stops the publisher, which ends the teleop session."""
        import zmq

        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        received, used, other_sides, problems = 0, 0, set(), set()
        t_log = time.monotonic()
        try:
            # Every message in turn, the newest hand winning (no CONFLATE): the bridge sends
            # its other SDK streams on this socket too, and conflating could keep one of
            # those in place of a hand.
            sub.setsockopt_string(zmq.SUBSCRIBE, "")
            if "[" in self._zmq_address:  # an IPv6 address
                sub.setsockopt(zmq.IPV6, 1)
            sub.connect(self._zmq_address)
            logger.info("Listening for MANUS glove frames on %s", self._zmq_address)
            while not self._stop.is_set():
                if sub.poll(timeout=100):
                    raw = sub.recv()
                    received += 1
                    try:  # the warnings first: a hand that fails must not hide them
                        self.update_warnings(raw)
                        keypoints, sides = self.keypoints_from_message(raw)
                    except Exception as err:  # a bad message must not end the stream
                        problems.add(str(err))
                        keypoints, sides = None, []
                    if keypoints is None:
                        other_sides.update(sides)
                    else:
                        used += 1
                        if self._recording_pending:
                            self._start_recording_when_ready()
                        if self._recorder is not None:
                            self._record(keypoints)
                        with self._lock:  # set inside: else a frame can go out twice
                            self._latest_keypoints = keypoints
                            self._fresh.set()
                now = time.monotonic()
                if now - t_log >= WAIT_LOG_S:
                    if received == 0:
                        logger.warning(
                            "No glove frames from %s yet. Is the bridge running there, and "
                            "can this computer reach it?",
                            self._zmq_address,
                        )
                    elif used == 0:
                        logger.warning(
                            "Bridge messages arrive, but none for the %s hand (hands: %s%s%s)",
                            self._handedness,
                            sorted(other_sides) or "none",
                            f"; problems: {sorted(problems)}" if problems else "",
                            f"; bridge warnings: {list(self._warnings.values())}"
                            if self._warnings
                            else "",
                        )
                    received, used, other_sides, problems = 0, 0, set(), set()
                    t_log = now
        except Exception:
            logger.exception("The glove stream from %s failed", self._zmq_address)
        finally:
            self._stop.set()
            sub.close(linger=0)
            ctx.term()

    def _start_recording_when_ready(self) -> None:
        if self._hand_ready is not None and not self._hand_ready.is_set():
            return
        from orca_teleop.ingress.landmark_recording import LandmarkRecorder

        self._recording_pending = False
        meta = {"input": "glove", "glove": "manus", "handedness": self._handedness, "video": False}
        try:
            self._recorder = LandmarkRecorder(self._record_dir, meta, 0.0)
        except OSError as err:  # a disk problem must not stop the hand
            logger.error("Recording did not start, teleop goes on: %s", err)
            self._recording_stopped(err)
            return
        logger.info("Recording to %s", self._record_dir)

    def _record(self, keypoints: np.ndarray) -> None:
        try:
            frame = self._recorder.add_time(time.time_ns())
            self._recorder.add_landmarks(frame, keypoints, NO_IMAGE, self._handedness, 1.0)
        except OSError as err:  # a full disk must not stop the hand
            logger.error("Recording stopped, teleop goes on: %s", err)
            recorder, self._recorder = self._recorder, None
            try:
                recorder.close()
            except OSError:
                pass
            self._recording_stopped(err)

    def _recording_stopped(self, err: OSError) -> None:
        """Called once when a disk error stops or prevents the recording (teleop goes on);
        a subclass may show ``err``."""

    def _frame_generator(self):
        """Yield a HandFrame for each new glove frame."""
        while not self._stop.is_set():
            if not self._fresh.wait(timeout=0.1):
                continue
            with self._lock:
                kp = self._latest_keypoints
                self._fresh.clear()
            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness=self._handedness,
                # Stamped here, not by the bridge (another computer's clock), so teleop's
                # capture-to-command time leaves out the glove-to-bridge-to-here leg.
                timestamp_ns=time.time_ns(),
            )

    def run(self) -> None:
        """Start the ZMQ receiver thread and the gRPC stream; block until stopped."""
        logger.info(
            "ManusPublisher starting (server=%s, hand=%s, bridge=%s)",
            self._server_address,
            self._handedness,
            self._zmq_address,
        )
        # A plain TCP probe every 0.1 s, as the webcam publisher does: a refused gRPC connect
        # backs off for seconds, which can pass the deadline after the server is up.
        host, _, port = self._server_address.rpartition(":")
        host = host.strip("[]")  # an IPv6 address comes as [::1]:PORT
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while True:
            try:
                with socket.create_connection((host, int(port)), timeout=0.5):
                    break
            except OSError as err:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"No ingress server answered at {self._server_address} within "
                        f"{CONNECT_TIMEOUT_S:.0f}s. Start it before the publisher."
                    ) from err
                time.sleep(0.1)
        channel = grpc.insecure_channel(self._server_address)
        zmq_thread = threading.Thread(target=self._zmq_receiver, name="zmq-receiver", daemon=True)
        zmq_thread.start()
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        stream_future = stub.StreamHandFrames.future(self._frame_generator())
        try:
            stream_future.result()
        except KeyboardInterrupt:
            pass
        except grpc.RpcError as e:
            logger.error("gRPC stream ended: %s", e)
        finally:
            # Stopped, the frames end within 0.1 s and the stream closes cleanly (a
            # cancelled one is an error in the ingress server's log); cancel only if not.
            self._stop.set()
            try:
                stream_future.result(timeout=STOP_TIMEOUT_S)
            except Exception:
                stream_future.cancel()
            channel.close()
            zmq_thread.join(timeout=2.0)
            if self._recorder is not None:
                _, hands = self._recorder.close()
                logger.info("Recorded %d hands to %s", hands, self._record_dir)
            elif self._recording_pending:
                logger.info("Nothing recorded: the hand was not ready, or no hand came")
            logger.info("ManusPublisher shut down.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream MANUS glove data from a glove bridge to the orca_teleop server.",
    )
    parser.add_argument(
        "--server", default="localhost:50051", help="gRPC server address (default: localhost:50051)"
    )
    parser.add_argument(
        "--hand", default="right", choices=["left", "right"], help="Hand to stream (default: right)"
    )
    parser.add_argument(
        "--glove-address",
        default=DEFAULT_MANUS_ADDRESS,
        help=f"The MANUS glove bridge, [tcp://]HOST:PORT (default: {DEFAULT_MANUS_ADDRESS})",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ManusPublisher(
        server_address=args.server, handedness=args.hand, zmq_address=args.glove_address
    ).run()


if __name__ == "__main__":
    main()
