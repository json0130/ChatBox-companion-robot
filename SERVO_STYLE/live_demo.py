#!/usr/bin/env python3
"""
live_demo.py — the whole chain, live: your face to the robot's servos.

    camera -> valence/arousal -> PAD -> gesture style -> STYLE + tag -> ESP32

Click a gesture and the robot performs it in whatever mood it currently reads on
your face, at whichever persona is selected. Look sad and press GREETING and it
greets you sadly — small, slow, ears down.

    python live_demo.py                  # auto-detect the ESP32
    python live_demo.py --port COM10     # or name it, if detection picks wrong
    python live_demo.py --dry-run        # no serial, just show what would be sent
    python live_demo.py --list-ports

Click the persona buttons to switch robot, the gesture buttons to perform one.
Keys: C/E persona, [ ] empathy, S screenshot, Q quit.

Reuses AFFECT_LAB for the camera and the affect maths, and servo_style for the
angle model, so there is one source of truth for each and this file is only the
glue plus the serial link.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import servo_style as S

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "AFFECT_LAB"))
import affect as A          # noqa: E402
import webcam_demo as W     # noqa: E402

GESTURES = ["greeting", "wave", "shrug", "confused", "angry", "sad", "default"]

GESTURE_W = 268
INK, MUTED, LINE = W.INK, W.MUTED, W.LINE
ACCENT, BG = W.ACCENT, W.BG
OK_GREEN = (110, 200, 130)
WARN = (90, 160, 235)


# ── The link to the robot ───────────────────────────────────────────────────

class RobotLink:
    """Serial link to the ESP32. Degrades to a no-op when there is nothing there.

    Opening a CH340 port normally yanks DTR/RTS and reboots the board — which
    then spends up to 30s waiting on WiFi before it will listen. Holding both
    lines low before opening avoids the reset, so the sketch keeps running.
    """

    def __init__(self, port=None, baud=115200, dry_run=False):
        self.port = port
        self.dry_run = dry_run
        self.ser = None
        self.status = "dry run — nothing sent" if dry_run else "not connected"
        self.last_sent = ""
        self.last_reply = ""
        self._rx = b""
        if dry_run:
            return

        if port is None:
            port = self.autodetect()
            if port is None:
                self.status = "no ESP32 found — running as preview only"
                return
            self.port = port

        try:
            s = __import__("serial").Serial()
            s.port = port
            s.baudrate = baud
            s.timeout = 0
            s.dtr = False        # do not reset the board on open
            s.rts = False
            s.open()
            self.ser = s
            self.status = f"connected {port}"
        except Exception as e:
            self.status = f"{port}: {e}"

    @staticmethod
    def autodetect():
        try:
            from serial.tools import list_ports
        except ImportError:
            return None
        for p in list_ports.comports():
            blob = f"{p.description} {p.manufacturer or ''}".lower()
            if any(k in blob for k in ("ch340", "cp210", "ft232", "esp",
                                       "silicon labs", "usb-serial")):
                return p.device
        ports = list(list_ports.comports())
        return ports[0].device if len(ports) == 1 else None

    def send(self, style, tag):
        """One style line then the tag, exactly as the firmware expects."""
        line = S.wire_message(style)
        self.last_sent = f"{line}  ->  {tag}"
        if self.ser is None:
            return False
        try:
            self.ser.write((line + "\n").encode())
            # The firmware parses line by line; a beat keeps the two apart in the
            # RX buffer rather than relying on it to split a single burst.
            time.sleep(0.05)
            self.ser.write((tag + "\n").encode())
            return True
        except Exception as e:
            self.status = f"write failed: {e}"
            self.ser = None
            return False

    def poll(self):
        """Pick up whatever the ESP32 said, keeping only the most useful line."""
        if self.ser is None:
            return
        try:
            waiting = self.ser.in_waiting
            if waiting:
                self._rx += self.ser.read(waiting)
        except Exception:
            return
        while b"\n" in self._rx:
            raw, self._rx = self._rx.split(b"\n", 1)
            text = raw.decode(errors="replace").strip()
            # Skip the noisy per-gesture heap prints; keep style acks and errors.
            if text and not text.startswith(("Free heap", "//=====")):
                self.last_reply = text[:44]

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


# ── Style for the current moment ────────────────────────────────────────────

def current_style(robot, valence, arousal, empathy):
    """Persona + what the camera sees -> the five values the firmware wants.

    All five come from `A.gesture_style` via the pipeline — `amplitude` and
    `tempo` from Hagane & Venture (2022), the other three from the deck.
    Nothing is scaled afterwards: the per-robot `travel` fraction that used to
    be applied here has been removed, so these are the deck's published values.
    """
    out = A.pipeline(A.ROBOTS[robot]["ocean"], valence, arousal, robot, empathy)
    return S.clamp_style(dict(out["style"])), out


# ── The gesture column ──────────────────────────────────────────────────────

def draw_gestures(height, link, busy_until, x_offset, hit_boxes, style):
    panel = np.full((height, GESTURE_W, 3), BG, np.uint8)
    cv2.putText(panel, "PERFORM", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, MUTED, 1)

    hit_boxes.clear()
    busy = time.time() < busy_until
    y, bh, gap = 32, 32, 6
    for tag in GESTURES:
        x0, x1 = 12, GESTURE_W - 12
        y1 = y + bh
        # Greyed while a gesture is still playing: the firmware blocks inside
        # EXECUTE, so a second tag sent now would just sit in the buffer.
        colour = LINE if busy else ACCENT
        cv2.rectangle(panel, (x0, y), (x1, y1), colour, 1)
        cv2.putText(panel, tag, (x0 + 12, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, MUTED if busy else INK, 1)
        hit_boxes[tag] = (x0 + x_offset, y, x1 + x_offset, y1)
        y = y1 + gap

    y += 6
    cv2.line(panel, (12, y), (GESTURE_W - 12, y), LINE, 1)
    y += 18

    connected = link.ser is not None
    cv2.putText(panel, link.status[:34], (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, OK_GREEN if connected else WARN, 1)
    y += 20
    if busy:
        cv2.putText(panel, "performing...", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, ACCENT, 1)
    y += 24

    cv2.putText(panel, "last sent", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, MUTED, 1)
    y += 15
    for chunk in (link.last_sent[:30], link.last_sent[30:60]):
        if chunk:
            cv2.putText(panel, chunk, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, INK, 1)
            y += 14
    y += 10
    if link.last_reply:
        cv2.putText(panel, "esp32", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, MUTED, 1)
        y += 15
        cv2.putText(panel, link.last_reply, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, OK_GREEN, 1)

    # A couple of the angles, so it is obvious the mood reaches the servos.
    y = height - 46
    resolved = S.resolve_gesture("greeting", style)
    cv2.putText(panel, "greeting would move", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, MUTED, 1)
    y += 15
    ears = resolved["angles"]["Ears"][0]
    shoulder = resolved["angles"]["RShoulder"][0]
    cv2.putText(panel, f"ears {ears}   arm {shoulder}   "
                       f"{resolved['step_ms']}ms/step",
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, INK, 1)
    return panel


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM8")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and display, but send nothing")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--robot", default="CHATBOX", choices=list(A.ROBOTS))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--empathy", type=float, default=A.EMPATHY)
    ap.add_argument("--smoothing", type=float, default=0.35)
    args = ap.parse_args()

    if args.list_ports:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"  {p.device:<8} {p.description}")
        return 0

    link = RobotLink(args.port, args.baud, args.dry_run)
    print(f"robot link: {link.status}")

    print("loading emotion model ...")
    try:
        model = W.load_va_model()
    except Exception as e:
        print(f"could not load the emotion model: {e}")
        return 1
    finder = W.FaceFinder()
    print(f"face detection: {finder.mode}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera} — try --camera 1")
        link.close()
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ui = {"robot": args.robot, "buttons": {}, "gestures": {}, "click": None}

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in ui["buttons"].items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                ui["robot"] = name
                return
        for tag, (x0, y0, x1, y1) in ui["gestures"].items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                ui["click"] = tag
                return

    window = "Affect -> servos   (click a gesture)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    smooth, label, times = None, "-", []
    empathy = args.empathy
    busy_until = 0.0
    print("running — click a gesture, Q to quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            box = finder.find(frame)
            if box is not None:
                x0, y0, x1, y1 = box
                crop = frame[y0:y1, x0:x1]
                if crop.size:
                    v, a, label = W.read_va(model, crop)
                    k = args.smoothing
                    smooth = (v, a) if (smooth is None or k <= 0) else (
                        k * v + (1 - k) * smooth[0], k * a + (1 - k) * smooth[1])
                cv2.rectangle(frame, (x0, y0), (x1, y1), (110, 220, 110), 2)
            else:
                cv2.putText(frame, "no face", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARN, 2)

            robot = ui["robot"]
            va = smooth or (0.0, 0.0)
            style, out = current_style(robot, va[0], va[1], empathy)

            # A click was queued by the mouse callback; act on it here so the
            # send happens on the main thread with the current style.
            if ui["click"] and time.time() >= busy_until:
                tag = ui["click"]
                ui["click"] = None
                sent = link.send(style, tag)
                # Block further clicks for as long as the gesture will take —
                # five steps at the styled tempo, plus a beat for the firmware's
                # return-to-default. A tag sent during EXECUTE just sits in the
                # buffer, so the button would appear to do nothing.
                gesture_secs = 5 * S.BASE_STEP_MS / style["tempo"] / 1000.0
                busy_until = time.time() + gesture_secs + 1.0
                print(f"{'sent' if sent else 'preview'}: "
                      f"{S.wire_message(style)} -> {tag}")
            elif ui["click"]:
                ui["click"] = None      # discard clicks while it is performing

            link.poll()

            now = time.time()
            times.append(now)
            if len(times) > 20:
                times.pop(0)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            affect_panel = W.draw_circumplex(
                h, out["shown"], out["name"], robot, empathy, fps, va, label,
                "VA model", style=out["style"], x_offset=w,
                hit_boxes=ui["buttons"])
            gesture_panel = draw_gestures(
                h, link, busy_until, w + affect_panel.shape[1], ui["gestures"],
                style)
            cv2.imshow(window, np.hstack([frame, affect_panel, gesture_panel]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                ui["robot"] = "CHATBOX"
            elif key == ord("e"):
                ui["robot"] = "ELLEBOT"
            elif key == ord("["):
                empathy = max(0.0, round(empathy - 0.05, 2))
            elif key == ord("]"):
                empathy = min(1.0, round(empathy + 0.05, 2))
            elif key == ord("s"):
                name = f"live_{int(now)}.png"
                cv2.imwrite(name, np.hstack([frame, affect_panel, gesture_panel]))
                print(f"saved {os.path.abspath(name)}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
