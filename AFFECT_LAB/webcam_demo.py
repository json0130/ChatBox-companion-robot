#!/usr/bin/env python3
"""
webcam_demo.py — the affect pipeline running live on a laptop webcam.

Reads your face, estimates valence and arousal, fuses that with a robot's OCEAN
persona through PAD, and draws the result on Russell's circumplex beside the
video. Nothing here touches the robot code — this is a bench rig for checking
the model behaves before any of it goes near a Jetson.

    python webcam_demo.py                 # CHATBOX persona, VA model
    python webcam_demo.py --robot ELLEBOT
    python webcam_demo.py --categorical   # force the classifier + lookup path
    python webcam_demo.py --camera 1      # a different webcam

Keys:  C / E  switch robot        [ ]  empathy down / up
       S      save a screenshot   Q or Esc  quit

Why valence-arousal rather than seven categories: the categories are a
quantisation of this space, and the fusion needs a position in it. Asking the
model for the position directly removes a lookup table full of estimates.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import affect as A

# ── Model loading ───────────────────────────────────────────────────────────


def load_va_model(model_name="enet_b0_8_va_mtl"):
    """HSEmotion multi-task model: 8 class logits plus valence and arousal.

    Two library quirks worth knowing about, both handled here:
      * facial_emotions.py calls urllib.request.urlretrieve but only imports
        urllib, so a fresh install crashes on the first download. Importing the
        submodule ourselves populates it and the call resolves.
      * the weights are fetched from GitHub on first use (~16 MB), so the very
        first run needs a network connection. Later runs are offline.
    """
    import urllib.request  # noqa: F401  — see above, this is the fix
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    return HSEmotionRecognizer(model_name=model_name)


# YuNet weights. opencv_zoo stores this through git-lfs, so the ordinary
# raw.githubusercontent URL returns a ~130-byte pointer file that then fails to
# parse as ONNX. The media.githubusercontent.com path resolves LFS properly and
# gives the real 232 KB model.
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


def ensure_yunet(path=YUNET_FILE, quiet=False):
    """Return a path to the YuNet weights, downloading them once if needed."""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    import urllib.request
    if not quiet:
        print(f"  fetching YuNet face detector ({YUNET_FILE}) ...")
    try:
        urllib.request.urlretrieve(YUNET_URL, path)
        if os.path.getsize(path) < 100_000:
            # An LFS pointer rather than the model — delete it so a later run
            # retries instead of trying to parse 130 bytes as ONNX.
            os.remove(path)
            return None
        return path
    except Exception as e:
        if not quiet:
            print(f"  could not download YuNet ({e})")
        return None


class FaceFinder:
    """Locates a face, using whatever this OpenCV build actually provides.

    Preference order:

        1. YuNet         — a small DNN detector, far better than Haar at angles
                           and through glasses; needs a 232 KB download once
        2. Haar cascade  — OpenCV 4.x only; no download, but frontal-only
        3. centre crop   — no detector at all

    OpenCV 5.0 removed cv2.CascadeClassifier and the objdetect namespace, so on
    a current pip install option 2 does not exist. Option 3 is an acceptable
    bench fallback — one person at a laptop is centred by construction, and the
    emotion model only needs a rough crop — but it is useless in a room with
    several people, which is why it is last.
    """

    def __init__(self, yunet_path=None, allow_download=True):
        self.mode = "centre crop"
        self.cascade = None
        self.yunet = None

        if hasattr(cv2, "FaceDetectorYN"):
            path = yunet_path if (yunet_path and os.path.exists(yunet_path)) else None
            if path is None and allow_download:
                path = ensure_yunet(yunet_path or YUNET_FILE)
            if path:
                try:
                    self.yunet = cv2.FaceDetectorYN.create(
                        path, "", (320, 320), 0.7, 0.3, 5000)
                    self.mode = "YuNet"
                    return
                except cv2.error as e:
                    first = e.err.splitlines()[0] if e.err else str(e)
                    print(f"  YuNet would not load ({first})")

        if hasattr(cv2, "CascadeClassifier"):
            c = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not c.empty():
                self.cascade = c
                self.mode = "Haar cascade"

    def find(self, frame):
        """Largest face as (x0, y0, x1, y1), or None if nothing was detected.

        Largest = closest person, the same rule the robot's own tracker uses.
        """
        h, w = frame.shape[:2]

        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            # detect() returns (retval, faces); faces is None — not an empty
            # array — when nothing was found, so check for None explicitly.
            _, faces = self.yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            best = max(range(len(faces)), key=lambda i: faces[i][2] * faces[i][3])
            x, y, fw, fh = faces[best][:4]
            return self._pad(int(x), int(y), int(fw), int(fh), w, h)

        if self.cascade is not None:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(grey, 1.2, 5, minSize=(90, 90))
            if len(faces) == 0:
                return None
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            return self._pad(x, y, fw, fh, w, h)

        # Centre crop: a square covering the middle of the frame.
        side = int(min(w, h) * 0.62)
        return ((w - side) // 2, (h - side) // 2,
                (w + side) // 2, (h + side) // 2)

    @staticmethod
    def _pad(x, y, fw, fh, w, h):
        # Detectors cut tight to the face; the model was trained on looser
        # AffectNet boxes, so widen a little before cropping.
        pad = int(0.15 * fw)
        return (max(0, x - pad), max(0, y - pad),
                min(w, x + fw + pad), min(h, y + fh + pad))


def read_va(model, face_bgr):
    """(valence, arousal, label) from a face crop.

    predict_emotions returns class logits with valence and arousal appended, so
    the coordinate is the last two entries. The model normalises with ImageNet
    statistics in R,G,B order, so the crop has to be converted out of OpenCV's
    BGR first — feeding it BGR silently degrades the estimate rather than
    failing, which is the sort of bug that costs an afternoon.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    label, scores = model.predict_emotions(face_rgb, logits=True)
    if getattr(model, "is_mtl", False) and len(scores) >= 2:
        valence, arousal = float(scores[-2]), float(scores[-1])
    else:
        # A non-MTL model gives categories only — fall back to the lookup.
        valence, arousal = A.category_to_va(label)
    return (max(-1.0, min(1.0, valence)),
            max(-1.0, min(1.0, arousal)), label)


# ── Drawing ─────────────────────────────────────────────────────────────────

INK = (232, 238, 244)
MUTED = (150, 141, 134)
LINE = (61, 49, 44)
ACCENT = (147, 74, 222)      # BGR for the magenta used in the explorer
POS = (212, 139, 62)         # BGR blue
NEG = (46, 127, 206)         # BGR orange
BG = (26, 22, 18)

EMO_POINTS = [(n, *A.category_to_va(n)) for n in
              ("happy", "surprise", "fear", "angry", "disgust", "sad", "neutral")]


def draw_buttons(panel, active, x_offset, hit_boxes):
    """A two-way persona selector along the top of the panel.

    Records each button's rectangle in *window* coordinates — the window shows
    the camera frame and this panel side by side, so a click's x has the frame's
    width already added to it. Storing window coordinates keeps the mouse
    callback from having to know the layout.
    """
    hit_boxes.clear()
    pad, top, height = 10, 10, 30
    width = (panel.shape[1] - pad * 3) // 2
    for i, name in enumerate(A.ROBOTS):
        x0 = pad + i * (width + pad)
        x1, y1 = x0 + width, top + height
        on = (name == active)
        cv2.rectangle(panel, (x0, top), (x1, y1),
                      ACCENT if on else LINE, -1 if on else 1)
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2 if on else 1)
        cv2.putText(panel, name, (x0 + (width - tw) // 2, top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if on else MUTED, 2 if on else 1)
        hit_boxes[name] = (x0 + x_offset, top, x1 + x_offset, y1)


BAR_GAP, BAR_H, BAR_TOP_PAD = 17, 9, 6
# Total vertical space the block needs, header included. Kept as one expression
# so the caller's placement can never drift out of step with the drawing.
STYLE_BLOCK_H = 16 + BAR_TOP_PAD + (len(A.STYLE_LIMITS) - 1) * BAR_GAP + BAR_H


def draw_style_bars(panel, style, y):
    """The four gesture parameters as small bars — the servo-facing output.

    Bars rather than bare numbers because what matters when tuning is the
    relative shape of the four, not their third decimal place.
    """
    x, width, height, gap = 12, 118, BAR_H, BAR_GAP
    cv2.putText(panel, "gesture style ->  ESP32", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, MUTED, 1)
    for i, (key, (lo, hi)) in enumerate(A.STYLE_LIMITS.items()):
        yy = y + 6 + i * gap
        value = style[key]
        frac = (value - lo) / (hi - lo)
        cv2.rectangle(panel, (x + 78, yy), (x + 78 + width, yy + height), LINE, 1)
        cv2.rectangle(panel, (x + 78, yy),
                      (x + 78 + int(frac * width), yy + height), ACCENT, -1)
        cv2.putText(panel, key, (x, yy + height - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, INK, 1)
        cv2.putText(panel, f"{value:+.2f}", (x + 78 + width + 8, yy + height - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, INK, 1)


def draw_circumplex(size, coord, name, robot, empathy, fps, va, label, source,
                    style=None, x_offset=0, hit_boxes=None):
    """The right-hand panel: persona buttons, circumplex, dot, and readout."""
    panel = np.full((size, size, 3), BG, np.uint8)
    cx = cy = size // 2
    radius = int(size * 0.36)

    if hit_boxes is not None:
        draw_buttons(panel, robot, x_offset, hit_boxes)

    def to_px(v, a):
        return int(cx + v * radius), int(cy - a * radius)

    # Rings, axes, and the emotion anchors that give the space its meaning.
    for r in (radius, radius // 2):
        cv2.circle(panel, (cx, cy), r, LINE, 1)
    cv2.line(panel, (cx - radius, cy), (cx + radius, cy), LINE, 1)
    cv2.line(panel, (cx, cy - radius), (cx, cy + radius), LINE, 1)
    cv2.putText(panel, "arousal +", (cx + 6, cy - radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, MUTED, 1)
    cv2.putText(panel, "valence +", (cx + radius - 60, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, MUTED, 1)

    for nm, v, a in EMO_POINTS:
        x, y = to_px(v, a)
        cv2.circle(panel, (x, y), 3, MUTED, -1)
        cv2.putText(panel, nm, (x - 18, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, MUTED, 1)

    # Where the robot currently is.
    x, y = to_px(coord["P"], coord["Ar"])
    cv2.circle(panel, (x, y), 13, (int(ACCENT[0] * .3), int(ACCENT[1] * .3),
                                   int(ACCENT[2] * .3)), -1)
    cv2.circle(panel, (x, y), 7, ACCENT, -1)
    cv2.putText(panel, name, (x - 40, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ACCENT, 2)

    # Readout.
    rows = [
        (f"{robot}  ({A.ROBOTS[robot]['show']:.0%} shown)", INK, 0.5),
        (A.ROBOTS[robot]["body"], MUTED, 0.36),
        ("", INK, 0.4),
        (f"face      v {va[0]:+.2f}   a {va[1]:+.2f}   [{source}]", INK, 0.42),
        (f"          model says: {label}", MUTED, 0.38),
        (f"shown     P {coord['P']:+.2f}  Ar {coord['Ar']:+.2f}  "
         f"D {coord['D']:+.2f}", INK, 0.42),
        (f"prompt    {', '.join(A.descriptors(coord))}", ACCENT, 0.44),
        ("", INK, 0.4),
        (f"empathy {empathy:.2f}   {fps:.1f} fps", MUTED, 0.38),
        ("click a name above, or C/E   [ ] empathy   S save   Q quit", MUTED, 0.36),
    ]
    y0 = size - 14 * len(rows) - 8
    for i, (text, colour, scale) in enumerate(rows):
        if text:
            cv2.putText(panel, text, (12, y0 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)

    # The four servo-facing parameters, sat just above the text readout so the
    # chain reads top to bottom: face -> coordinate -> movement. Placed from the
    # block's own measured height, plus a small margin, so the two never touch.
    if style is not None:
        draw_style_bars(panel, style, y0 - STYLE_BLOCK_H - 8)
    return panel


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="CHATBOX", choices=list(A.ROBOTS))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--empathy", type=float, default=A.EMPATHY)
    ap.add_argument("--smoothing", type=float, default=0.35,
                    help="0 disables; higher follows the face faster")
    ap.add_argument("--categorical", action="store_true",
                    help="use the 8-class model plus the circumplex lookup")
    ap.add_argument("--yunet", default=YUNET_FILE,
                    help="path for the YuNet face detector weights")
    ap.add_argument("--no-download", action="store_true",
                    help="never fetch YuNet; use Haar or centre crop instead")
    args = ap.parse_args()

    robot, empathy = args.robot, args.empathy

    print(f"loading model ({'categorical' if args.categorical else 'valence-arousal'}) ...")
    try:
        model = load_va_model("enet_b0_8_best_vgaf" if args.categorical
                              else "enet_b0_8_va_mtl")
    except Exception as e:
        print(f"could not load the emotion model: {e}")
        print("try:  pip install -r requirements.txt")
        return 1
    source = "categorical" if args.categorical else "VA model"

    finder = FaceFinder(args.yunet, allow_download=not args.no_download)
    print(f"face detection: {finder.mode}  (OpenCV {cv2.__version__})")
    if finder.mode == "centre crop":
        print("  no detector available — sit centred in frame.")
        print("  YuNet needs one 232 KB download; check your network, or")
        print("  install OpenCV 4 for the bundled Haar cascade:")
        print("    pip install \"opencv-python<5\"")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera} — try --camera 1")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Shared with the mouse callback. A dict rather than a closure variable so
    # the callback can write to it without nonlocal gymnastics.
    ui = {"robot": robot, "buttons": {}}

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in ui["buttons"].items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                ui["robot"] = name
                return

    window = "Affect lab - persona x emotion x body"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("running — click a persona button, or press Q to quit\n")
    smooth = None          # smoothed (valence, arousal)
    label = "-"
    times = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped delivering frames")
                break
            frame = cv2.flip(frame, 1)          # mirror, so it reads as a mirror
            h, w = frame.shape[:2]

            box = finder.find(frame)
            if box is not None:
                x0, y0, x1, y1 = box
                crop = frame[y0:y1, x0:x1]
                if crop.size:
                    v, a, label = read_va(model, crop)
                    if args.smoothing > 0:
                        # The raw estimate jitters frame to frame even when you
                        # hold still; without this the descriptors flicker.
                        k = args.smoothing
                        smooth = (v, a) if smooth is None else (
                            k * v + (1 - k) * smooth[0],
                            k * a + (1 - k) * smooth[1])
                    else:
                        smooth = (v, a)
                colour = (110, 220, 110) if finder.mode != "centre crop" \
                    else (150, 150, 150)
                cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
            else:
                cv2.putText(frame, "no face", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 200, 255), 2)

            robot = ui["robot"]          # a click may have changed it
            va = smooth if smooth else (0.0, 0.0)
            out = A.pipeline(A.ROBOTS[robot]["ocean"], va[0], va[1], robot, empathy)

            now = time.time()
            times.append(now)
            if len(times) > 20:
                times.pop(0)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            panel = draw_circumplex(h, out["shown"], out["name"], robot,
                                    empathy, fps, va, label, source,
                                    style=out["style"],
                                    x_offset=w, hit_boxes=ui["buttons"])
            cv2.imshow(window, np.hstack([frame, panel]))

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
                name = f"affect_{int(now)}.png"
                cv2.imwrite(name, np.hstack([frame, panel]))
                print(f"saved {os.path.abspath(name)}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
