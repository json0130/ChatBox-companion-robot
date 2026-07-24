"""
face_tracking_output.py — state-driven head tracking as an OutputModule.

Ports the standalone CHATBOX_CLIENT/face_tracker_esp32.py into the client's
module framework, with one deliberate change: instead of a free-running thread
that fights with speech and gestures, tracking is an explicit STATE machine
advanced by tick() from the client's main loop.

    TRACKING  — idle. Detect the person and stream the horizontal pixel error
                to the ESP32 so it keeps the face centred.
    HOLDING   — a reply is being spoken or a gesture is playing. Do not move.
                A '0' keep-alive is still sent: the firmware treats 0 as
                "inside the deadband" (holds its angle) while the keep-alive
                stops its LOST_TIMEOUT from drifting the head back to centre.

Handshake used by robot.py when a response arrives:

    request_hold()             # ask the tracker to stop
    wait_until_held(timeout)   # block until it has actually stopped
    ... speak + gesture ...
    release_hold()             # tracking resumes

All coordination lives here on the Jetson — the ESP32 firmware is unchanged and
stays a dumb proportional servo loop. Holds are re-entrant, so overlapping
speech and gestures cannot resume tracking early.
"""

import glob
import logging
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

from client import OutputModule

logger = logging.getLogger(__name__)

# Optional deps — the module degrades instead of crashing when any is missing.
try:
    import numpy as np
except ImportError:
    np = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import serial  # pyserial
except ImportError:
    serial = None
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class TrackingState(Enum):
    TRACKING = "tracking"   # free to move the head
    HOLDING = "holding"     # frozen while speech / gestures play


class FaceTrackingOutputModule(OutputModule):
    """Head tracking driven by tick(), gated by an explicit hold state."""

    def __init__(self, name: str = "face_tracking_output",
                 config: Dict[str, Any] = None):
        super().__init__(name, config)
        c = self.config

        # Frame / control tuning (defaults match face_tracker_esp32.py)
        self.width = c.get("width", 640)
        self.height = c.get("height", 480)
        self.baud = c.get("baud", 115200)
        self.deadband = c.get("deadband", 30)
        self.send_interval = c.get("send_interval", 0.05)   # 20 Hz max
        self.keepalive_interval = c.get("keepalive_interval", 1.0)

        # Sources
        self.serial_port = c.get("serial_port")             # None -> auto-detect
        self.use_realsense = c.get("use_realsense", True)
        self.camera_index = c.get("camera_index", 0)
        self.model_path = c.get("model_path", "yolov8n.pt")
        self.verbose_commands = c.get("verbose_commands", False)

        # State machine
        self._state = TrackingState.TRACKING
        self._hold_depth = 0
        self._hold_requested = False
        self._held_event = threading.Event()
        self._lock = threading.RLock()

        # Hardware / model handles
        self._ser = None
        self._pipeline = None
        self._cap = None
        self._model = None
        self._cascade = None

        # Bookkeeping
        self._last_send = 0.0
        self._last_keepalive = 0.0
        self.last_error: Optional[int] = None
        self.frames_seen = 0
        self.commands_sent = 0

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> TrackingState:
        with self._lock:
            return self._state

    @property
    def is_tracking(self) -> bool:
        """True when the head is free to move (what the spec calls 'tracking')."""
        return self.state is TrackingState.TRACKING

    def request_hold(self) -> None:
        """Ask the tracker to stop. Re-entrant; pair every call with release_hold().

        The transition is applied by the next tick() so the caller can be sure
        the tracking loop has observed it — wait_until_held() blocks for that.
        """
        with self._lock:
            self._hold_depth += 1
            self._hold_requested = True
            if not self.enabled:
                # Nothing is ticking, so it is already effectively held.
                self._state = TrackingState.HOLDING
                self._held_event.set()

    def release_hold(self) -> None:
        """Release one hold. Tracking resumes when the last hold is released."""
        with self._lock:
            self._hold_depth = max(0, self._hold_depth - 1)
            if self._hold_depth == 0:
                self._hold_requested = False
                self._state = TrackingState.TRACKING
                self._held_event.clear()
                logger.debug("[FaceTracking] resumed")

    def wait_until_held(self, timeout: float = 2.0) -> bool:
        """Block until tracking has actually stopped. True if it did."""
        if not self.enabled:
            return True
        return self._held_event.wait(timeout=timeout)

    # ── Module lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> bool:
        if cv2 is None or np is None:
            logger.error("[FaceTracking] needs opencv + numpy — disabled")
            return False

        self._open_serial()
        if not self._open_camera():
            return False
        self._load_detector()

        if self._model is None and self._cascade is None:
            logger.error("[FaceTracking] no detector available — disabled")
            self._close_camera()
            return False
        return True

    def start(self) -> bool:
        self.enabled = True
        with self._lock:
            if self._hold_depth == 0:
                self._state = TrackingState.TRACKING
                self._held_event.clear()
        logger.info("[FaceTracking] started — state machine active (tick-driven)")
        return True

    def stop(self):
        self.enabled = False
        self._close_camera()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        logger.info(f"[FaceTracking] stopped — {self.frames_seen} frames, "
                    f"{self.commands_sent} commands")

    def process_output(self, data: Any) -> bool:
        """OutputModule contract. Accepts explicit control dicts:
        {"face_tracking": "hold"} / {"face_tracking": "resume"}.
        Ordinary chat text is ignored — robot.py drives the hold explicitly so
        the head freezes *before* audio starts."""
        if isinstance(data, dict):
            cmd = data.get("face_tracking")
            if cmd == "hold":
                self.request_hold()
            elif cmd == "resume":
                self.release_hold()
        return True

    # ── The state machine — one step per call ────────────────────────────────

    def tick(self) -> None:
        """Advance one step. Safe to call at any rate; cheap when holding."""
        if not self.enabled:
            return

        # Apply a pending hold here so the caller knows the loop has seen it.
        with self._lock:
            if self._hold_requested and self._state is not TrackingState.HOLDING:
                self._state = TrackingState.HOLDING
                self._held_event.set()
                logger.debug("[FaceTracking] holding (speech/gesture active)")
            state = self._state

        now = time.time()

        if state is TrackingState.HOLDING:
            # Keep the head where it is; suppress the firmware's recentre timer.
            if now - self._last_keepalive >= self.keepalive_interval:
                self._send_error(0, quiet=True)
                self._last_keepalive = now
            return

        frame = self._read_frame()
        if frame is None:
            return
        self.frames_seen += 1

        error = self._detect_error(frame)
        self.last_error = error
        if error is None:
            return

        if abs(error) > self.deadband and (now - self._last_send) >= self.send_interval:
            self._send_error(error)
            self._last_send = now

    # ── Serial ───────────────────────────────────────────────────────────────

    def _open_serial(self) -> None:
        if serial is None:
            logger.warning("[FaceTracking] pyserial missing — print-only mode")
            return
        port = self.serial_port
        if not port:
            ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
            port = ports[0] if ports else None
        if not port:
            logger.warning("[FaceTracking] no ESP32 serial port — print-only mode")
            return
        try:
            self._ser = serial.Serial(port, self.baud, timeout=0.05)
            time.sleep(2)   # ESP32 resets when the port opens
            logger.info(f"[FaceTracking] serial {port} @ {self.baud}")
        except Exception as e:
            logger.warning(f"[FaceTracking] serial {port} failed: {e} — print-only")
            self._ser = None

    def _send_error(self, error: int, quiet: bool = False) -> None:
        if self._ser is not None:
            try:
                self._ser.write(f"{int(error)}\n".encode())
            except Exception as e:
                logger.error(f"[FaceTracking] serial write failed: {e}")
        self.commands_sent += 1
        if not quiet and self.verbose_commands:
            direction = "RIGHT" if error > 0 else "LEFT"
            logger.info(f"[FaceTracking] error={error:+d}px -> turn {direction}")

    # ── Camera ───────────────────────────────────────────────────────────────

    def _open_camera(self) -> bool:
        if self.use_realsense and rs is not None:
            try:
                self._pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height,
                                  rs.format.bgr8, 30)
                self._pipeline.start(cfg)
                logger.info("[FaceTracking] RealSense colour stream started")
                return True
            except Exception as e:
                logger.warning(f"[FaceTracking] RealSense unavailable ({e}) — "
                               "falling back to OpenCV")
                self._pipeline = None

        try:
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                logger.error(f"[FaceTracking] cannot open camera {self.camera_index}")
                self._cap = None
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            logger.info(f"[FaceTracking] OpenCV camera {self.camera_index} opened")
            return True
        except Exception as e:
            logger.error(f"[FaceTracking] camera error: {e}")
            self._cap = None
            return False

    def _close_camera(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _read_frame(self):
        """Non-blocking frame grab — must never stall the client's main loop."""
        if self._pipeline is not None:
            try:
                frames = self._pipeline.poll_for_frames()   # non-blocking
                if not frames:
                    return None
                color = frames.get_color_frame()
                if not color:
                    return None
                return np.asanyarray(color.get_data())
            except Exception as e:
                logger.debug(f"[FaceTracking] realsense read error: {e}")
                return None
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        return None

    # ── Detection ────────────────────────────────────────────────────────────

    def _load_detector(self) -> None:
        if YOLO is not None:
            try:
                self._model = YOLO(self.model_path)
                logger.info(f"[FaceTracking] YOLO detector: {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"[FaceTracking] YOLO load failed ({e}) — "
                               "falling back to Haar face cascade")
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                logger.error("[FaceTracking] Haar cascade failed to load")
                return
            self._cascade = cascade
            logger.info("[FaceTracking] Haar face cascade detector")
        except Exception as e:
            logger.error(f"[FaceTracking] no detector: {e}")

    def _detect_error(self, frame) -> Optional[int]:
        """Horizontal pixel error of the largest subject, or None if none seen."""
        center_x = frame.shape[1] // 2

        if self._model is not None:
            try:
                results = self._model(frame, classes=[0], verbose=False)
                boxes = results[0].boxes
                if len(boxes) == 0:
                    return None
                xyxy = boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                x1, _y1, x2, _y2 = xyxy[areas.argmax()]
                return int((x1 + x2) / 2) - center_x
            except Exception as e:
                logger.debug(f"[FaceTracking] YOLO inference error: {e}")
                return None

        if self._cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15,
                                                   minNeighbors=4, minSize=(40, 40))
            if len(faces) == 0:
                return None
            x, _y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x + w / 2) - center_x

        return None

    # ── Introspection ────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_tracking": self.is_tracking,
            "hold_depth": self._hold_depth,
            "frames_seen": self.frames_seen,
            "commands_sent": self.commands_sent,
            "last_error": self.last_error,
            "serial": self._ser is not None,
            "detector": "yolo" if self._model is not None else
                        ("haar" if self._cascade is not None else "none"),
        }
