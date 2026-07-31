"""
face_tracking_output.py — state-driven head tracking as an OutputModule.

The Jetson half of face tracking. The ESP32 half lives in the pan block of
CHATBOX_ARDUINO/ChatBoxPlus_ESP32.ino (pan servo on D19); the firmware stays a
dumb proportional loop and all the coordination happens here.

Tracking is an explicit STATE machine advanced by tick() from the client's main
loop, rather than a free-running thread that fights with speech and gestures:

    TRACKING   — idle. Detect the person and stream the horizontal pixel error
                 so the firmware keeps the face centred.
    CENTERING  — a reply is about to be spoken. Same motion as TRACKING, but we
                 also measure when the subject has *settled* inside the deadband
                 and signal that to the waiting caller.
    HOLDING    — the reply is being spoken or a gesture is playing. Do not move.

Handshake used by robot.py when a response arrives:

    center_and_hold()          # centre on the person, then freeze
    ... speak + gesture ...
    release_hold()             # tracking resumes

Holds are re-entrant, so overlapping speech and gestures cannot resume tracking
early. A hold always outranks a pending centring request.

The '0' keep-alive
------------------
Whenever a subject is visible but inside the deadband we send '0' rather than
going silent. The firmware ignores it (abs(0) > PAN_DEADBAND is false, so no
motion — it only refreshes lastPanMsg), but it gives silence on the wire exactly
one meaning:

    "0"      -> I can see them and they are centred; hold your angle.
    silence  -> nobody is in frame.

Without that, a person standing perfectly centred produces no traffic at all,
which is indistinguishable from an empty room. That ambiguity is why the
recentre-when-idle homing in ChatBoxPlus_ESP32.ino is currently commented out:
it could not tell success from failure and would drag the head back to 90°
while someone was standing right in front of it. Homing stays disabled — but
with these keep-alives it is now safe to re-enable if a head ever gets stranded
facing a wall after someone walks off.
"""

import glob
import logging
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

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
    TRACKING = "tracking"     # free to move the head
    CENTERING = "centering"   # moving, and watching for the subject to settle
    HOLDING = "holding"       # frozen while speech / gestures play


class FaceTrackingOutputModule(OutputModule):
    """Head tracking driven by tick(), gated by an explicit hold state."""

    def __init__(self, name: str = "face_tracking_output",
                 config: Dict[str, Any] = None):
        super().__init__(name, config)
        c = self.config

        # Frame / control tuning
        self.width = c.get("width", 640)
        self.height = c.get("height", 480)
        self.baud = c.get("baud", 115200)
        self.deadband = c.get("deadband", 30)
        self.send_interval = c.get("send_interval", 0.05)   # 20 Hz max
        self.keepalive_interval = c.get("keepalive_interval", 1.0)

        # Centring — how "centred enough to start talking" is decided.
        # center_tolerance defaults to the deadband so that 'centred' means
        # exactly 'the firmware would not move', which is what PAN_DEADBAND does.
        self.center_tolerance = c.get("center_tolerance", self.deadband)
        self.center_settle = c.get("center_settle", 0.3)
        self.center_timeout = c.get("center_timeout", 3.0)
        self.center_lost_grace = c.get("center_lost_grace", 1.5)

        # Sources
        self.serial_port = c.get("serial_port")             # None -> auto-detect
        self.use_realsense = c.get("use_realsense", True)
        self.camera_index = c.get("camera_index", 0)
        self.model_path = c.get("model_path", "yolov8n.pt")
        # Inference resolution. YOLO runs synchronously inside tick(), so this
        # sets the real tracking frame rate — 640 costs ~4x the compute of 320
        # and pins the loop to a few fps on the Jetson. We only need a horizontal
        # centroid, which survives the lower resolution fine.
        self.yolo_imgsz = c.get("yolo_imgsz", 320)
        self.yolo_half = c.get("yolo_half", False)   # True only helps on GPU
        self.verbose_commands = c.get("verbose_commands", False)
        self.show_preview = c.get("show_preview", False)

        # Terminal meter — a one-line live readout of where the subject is and
        # what the firmware will do about it. Unlike show_preview it needs no
        # display, so it is the way to watch tracking over SSH or in Docker.
        # pan_kp / pan_max_step only exist to *predict* the servo response for
        # the display; the firmware owns the real values and ignores whatever
        # is set here. Keep them matching PAN_KP and PAN_MAX_STEP in
        # ServoControl.ino or the degrees column will quietly lie.
        self.show_meter = c.get("show_meter", False)
        self.meter_interval = c.get("meter_interval", 0.15)
        self.meter_width = c.get("meter_width", 31)
        self.pan_kp = c.get("pan_kp", 0.03)
        self.pan_max_step = c.get("pan_max_step", 4.0)

        # State machine
        self._state = TrackingState.TRACKING
        self._hold_depth = 0
        self._hold_requested = False
        self._held_event = threading.Event()
        self._lock = threading.RLock()

        # Centring bookkeeping
        self._center_requested = False
        self._centered_event = threading.Event()
        self._center_result = False
        self._center_started: Optional[float] = None
        self._in_band_since: Optional[float] = None
        self._last_seen: Optional[float] = None

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

        # Meter bookkeeping. The tick rate is measured over a short rolling
        # window rather than since start-up, so the fps column reacts when YOLO
        # slows down instead of averaging the problem away.
        self._last_meter = 0.0
        self._tick_times: list = []

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> TrackingState:
        with self._lock:
            return self._state

    @property
    def is_tracking(self) -> bool:
        """True when the head is free to move (what the spec calls 'tracking')."""
        return self.state is not TrackingState.HOLDING

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

    # ── Centring ─────────────────────────────────────────────────────────────

    def request_center(self) -> None:
        """Ask the tracker to centre on the subject before the robot speaks.

        Resets the settle timers so a stale 'was centred a minute ago' can never
        satisfy the wait — we always measure fresh.
        """
        with self._lock:
            self._center_result = False
            self._centered_event.clear()
            self._center_started = time.time()
            self._in_band_since = None
            self._last_seen = None

            if (not self.enabled or self._hold_requested
                    or self._state is TrackingState.HOLDING):
                # Nothing is ticking, or an earlier reply already froze the head
                # (possibly a hold that tick() has not applied yet). Either way
                # there is nothing to centre and nothing may move — say so now
                # rather than leaving the caller to time out.
                self._center_result = True
                self._centered_event.set()
                return

            self._center_requested = True

    def wait_until_centered(self, timeout: float = None) -> bool:
        """Block until the subject has settled inside the deadband.

        Returns True only if it actually centred — False if nobody was there to
        centre on, or `timeout` elapsed first. Callers are expected to carry on
        regardless: talking late is worse than talking slightly off-centre.
        """
        if not self.enabled:
            return True
        if timeout is None:
            timeout = self.center_timeout

        if not self._centered_event.wait(timeout=timeout):
            logger.warning(f"[FaceTracking] not centred after {timeout:.1f}s")
            self._abandon_centering()
            return False
        return self._center_result

    def center_and_hold(self, timeout: float = None) -> bool:
        """Centre on the subject, then freeze the head for speech / gestures.

        The one call robot.py makes. Returns whether it managed to centre; the
        hold is applied either way, so the head never moves mid-sentence.
        """
        self.request_center()
        centered = self.wait_until_centered(timeout)
        self.request_hold()
        if not self.wait_until_held(timeout=2.0):
            logger.warning("[FaceTracking] hold not confirmed — proceeding anyway")
        return centered

    def _abandon_centering(self) -> None:
        with self._lock:
            self._center_requested = False
            if self._state is TrackingState.CENTERING:
                self._state = TrackingState.TRACKING

    def _finish_centering(self, ok: bool, detail: str) -> None:
        with self._lock:
            if self._centered_event.is_set():
                return
            self._center_result = ok
            self._center_requested = False
            if self._state is TrackingState.CENTERING:
                self._state = TrackingState.TRACKING
            self._centered_event.set()
        if ok:
            logger.info(f"[FaceTracking] centred ({detail})")
        else:
            logger.info(f"[FaceTracking] centring gave up — {detail}")

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
        if self.show_preview and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info(f"[FaceTracking] stopped — {self.frames_seen} frames, "
                    f"{self.commands_sent} commands")

    def process_output(self, data: Any) -> bool:
        """OutputModule contract. Accepts explicit control dicts:
        {"face_tracking": "hold"} / {"face_tracking": "resume"}.
        Ordinary chat text is ignored — robot.py drives the hold explicitly so
        the head centres and freezes *before* audio starts."""
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

        # Apply pending transitions here so the caller knows the loop has seen
        # them. A hold outranks a centring request: drop the request so that
        # release_hold() returns to TRACKING rather than resuming a stale centre.
        with self._lock:
            if self._hold_requested and self._state is not TrackingState.HOLDING:
                self._state = TrackingState.HOLDING
                self._center_requested = False
                self._held_event.set()
                logger.debug("[FaceTracking] holding (speech/gesture active)")
            elif self._center_requested and self._state is TrackingState.TRACKING:
                self._state = TrackingState.CENTERING
                logger.info("[FaceTracking] centring on subject")
            state = self._state

        now = time.time()

        if state is TrackingState.HOLDING:
            # Keep the head where it is. Detection is skipped entirely — YOLO
            # inference during speech is wasted CPU on the Jetson.
            if now - self._last_keepalive >= self.keepalive_interval:
                self._send_error(0, keepalive=True)
                self._last_keepalive = now
            if self.show_preview:
                self._preview(self._read_frame(), None, None, state)
            if self.show_meter:
                # No detection runs while held, so there is no position to show —
                # but keep printing so the line does not freeze and look crashed.
                self._meter(now, None, state)
            return

        frame = self._read_frame()
        if frame is None:
            # A dropped frame is not evidence of an empty room, but waiting
            # forever for one is how the reply gets stuck — let the grace run.
            if state is TrackingState.CENTERING:
                self._update_centering(now, None)
            return
        self.frames_seen += 1
        if self.show_meter:
            # Timestamp before inference, so the interval we measure is one
            # whole detection cycle — that is what the head actually sees.
            self._tick_times.append(now)
            if len(self._tick_times) > 20:
                self._tick_times.pop(0)

        error, box = self._detect(frame)
        self.last_error = error

        if error is not None:
            if abs(error) > self.deadband:
                if now - self._last_send >= self.send_interval:
                    self._send_error(error)
                    self._last_send = now
            elif now - self._last_keepalive >= self.keepalive_interval:
                # Inside the deadband: say "still here, still centred" so that
                # silence on the wire keeps its one meaning — nobody in frame.
                self._send_error(0, keepalive=True)
                self._last_keepalive = now

        if state is TrackingState.CENTERING:
            self._update_centering(now, error)

        if self.show_preview:
            self._preview(frame, box, error, state)
        if self.show_meter:
            self._meter(now, error, state)

    def _update_centering(self, now: float, error: Optional[int]) -> None:
        """Decide whether the head has settled on the subject. CENTERING only."""
        if error is None:
            # Nobody detected. Give them a grace period to reappear, then stop
            # blocking — a reply must never be held hostage by an empty room.
            since = self._last_seen or self._center_started or now
            if now - since > self.center_lost_grace:
                self._finish_centering(False, "no subject in frame")
            return

        self._last_seen = now

        if abs(error) > self.center_tolerance:
            self._in_band_since = None
            return

        # In band — but require it to stay there, so one lucky frame mid-swing
        # cannot pass for a centred head.
        if self._in_band_since is None:
            self._in_band_since = now
        elif now - self._in_band_since >= self.center_settle:
            took = now - (self._center_started or now)
            self._finish_centering(True, f"err={error:+d}px in {took:.1f}s")

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

    def _send_error(self, error: int, keepalive: bool = False) -> None:
        if self._ser is not None:
            try:
                self._ser.write(f"{int(error)}\n".encode())
            except Exception as e:
                logger.error(f"[FaceTracking] serial write failed: {e}")
        self.commands_sent += 1
        if self.verbose_commands:
            if keepalive:
                logger.info("[FaceTracking] keepalive 0 (centred/held)")
            else:
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

    def _detect(self, frame) -> Tuple[Optional[int], Optional[tuple]]:
        """Horizontal pixel error of the largest subject and its box.

        Returns (None, None) when nobody is visible. The box is (x1, y1, x2, y2)
        and is only used to draw the preview.
        """
        center_x = frame.shape[1] // 2

        if self._model is not None:
            try:
                results = self._model(frame, classes=[0], verbose=False,
                                      imgsz=self.yolo_imgsz, half=self.yolo_half)
                boxes = results[0].boxes
                if len(boxes) == 0:
                    return None, None
                xyxy = boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                x1, y1, x2, y2 = xyxy[areas.argmax()]
                return int((x1 + x2) / 2) - center_x, (int(x1), int(y1),
                                                       int(x2), int(y2))
            except Exception as e:
                logger.debug(f"[FaceTracking] YOLO inference error: {e}")
                return None, None

        if self._cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15,
                                                   minNeighbors=4, minSize=(40, 40))
            if len(faces) == 0:
                return None, None
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x + w / 2) - center_x, (int(x), int(y),
                                               int(x + w), int(y + h))

        return None, None

    # ── Terminal meter ───────────────────────────────────────────────────────

    def _predicted_step(self, error: int) -> float:
        """Degrees the firmware will move for this error — display only.

        Mirrors updatePanTracking() in ServoControl.ino: inside PAN_DEADBAND
        nothing happens at all, and outside it the correction is
        'panAngle -= error * PAN_KP' clamped to +/-PAN_MAX_STEP. The sign is
        negative for a positive error because the servo angle decreases as the
        head turns toward a subject on the right.
        """
        if abs(error) <= self.deadband:
            return 0.0
        step = -error * self.pan_kp
        return max(-self.pan_max_step, min(self.pan_max_step, step))

    def _meter(self, now: float, error: Optional[int],
               state: TrackingState) -> None:
        """One line of live tracking state, throttled to meter_interval.

        Needs no display, unlike _preview(), so this is what you watch when the
        robot is running over SSH or inside Docker — which is the normal case.
        """
        if now - self._last_meter < self.meter_interval:
            return
        self._last_meter = now

        # Force an odd width so there is a single true centre cell to aim at.
        w = max(11, self.meter_width | 1)
        mid = w // 2
        half_px = max(1, self.width // 2)

        cells = ["."] * w
        cells[mid] = "|"
        # Deadband edges, so you can see at a glance whether the firmware would
        # even act on the current position.
        db = int(round(self.deadband / half_px * mid))
        for idx in (mid - db, mid + db):
            if 0 <= idx < w and idx != mid:
                cells[idx] = ":"

        if error is None:
            bar = "".join(cells)
            if state is TrackingState.HOLDING:
                detail = "  held — not detecting"
            else:
                detail = "  no subject in frame"
        else:
            pos = mid + int(round(error / half_px * mid))
            cells[max(0, min(w - 1, pos))] = "O"
            bar = "".join(cells)

            step = self._predicted_step(error)
            if step == 0.0:
                # Same width as the moving case, so the columns to the right of
                # this one stay put instead of jittering every other line.
                move = "centred     "
            else:
                move = f"{'RIGHT' if error > 0 else 'LEFT':<5} {step:+5.1f}d"
            detail = f"  err={error:+5d}px  {move}"

        # Measured over a rolling window — this is the true tracking rate, since
        # YOLO runs synchronously inside tick().
        fps = 0.0
        if len(self._tick_times) >= 2:
            span = self._tick_times[-1] - self._tick_times[0]
            if span > 0:
                fps = (len(self._tick_times) - 1) / span

        logger.info(f"[FT] |{bar}|{detail}  {state.value.upper():<9} "
                    f"{fps:4.1f}fps")

    # ── Preview (debug window) ───────────────────────────────────────────────

    def _preview(self, frame, box, error: Optional[int],
                 state: TrackingState) -> None:
        """Draw the debug window. Disables itself when there is no display."""
        if frame is None:
            return

        img = frame.copy()
        h, w = img.shape[:2]
        center_x = w // 2

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 4, (0, 0, 255), -1)

        cv2.line(img, (center_x, 0), (center_x, h), (255, 0, 0), 1)
        if error is not None:
            cv2.putText(img, f"err={error:+d}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(img, state.value.upper(), (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        link = "SERIAL" if self._ser is not None else "PRINT-ONLY"
        cv2.putText(img, link, (w - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        try:
            cv2.imshow("ChatBox Face Tracker", img)
            cv2.waitKey(1)
        except cv2.error:
            logger.info("[FaceTracking] no display available — preview off")
            self.show_preview = False

    # ── Introspection ────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_tracking": self.is_tracking,
            "hold_depth": self._hold_depth,
            "centered": self._center_result and self._centered_event.is_set(),
            "frames_seen": self.frames_seen,
            "commands_sent": self.commands_sent,
            "last_error": self.last_error,
            "serial": self._ser is not None,
            "preview": self.show_preview,
            "meter": self.show_meter,
            "detector": "yolo" if self._model is not None else
                        ("haar" if self._cascade is not None else "none"),
        }
