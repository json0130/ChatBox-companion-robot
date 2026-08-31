"""
Pluggable emotion detection backends for the webcam KG loop.

Usage:
    det = EmotionDetector.create('hsemotion')          # default, ONNX, 191 fps CPU
    det = EmotionDetector.create('hsemotion-b2')       # larger B2 variant, ~more accurate
    det = EmotionDetector.create('efficientnet')       # original HQRAF model via PyTorch

    emotion, confidence = det.detect(face_crop_bgr)
    emotion, confidence = det.detect(face_crop_bgr, smooth=True)  # 5-frame window

All backends return labels from STANDARD_LABELS and confidence in [0, 100].

All backends also return valence and arousal on [-1, +1]. The default backend
REGRESSES them; the others reconstruct them from a lookup table. That difference
matters more than it sounds — see HSEmotionDetector for the measurements.

Backends
--------
hsemotion      EfficientNet-B0 ONNX, `enet_b0_8_va_mtl` — AffectNet, ~5 ms/frame
               CPU. 8 classes PLUS a valence/arousal regression head, so V/A comes
               from the network rather than a table.  [default]

hsemotion-lookup
               EfficientNet-B0 ONNX, `enet_b0_8_best_vgaf` — the previous default.
               Same 8 classes, but no V/A head, so V/A is a softmax-weighted blend
               over `_VA_TABLE` and is confined to the hull of its seven points.
               Retained as the baseline for that comparison.

hsemotion-b2   EfficientNet-B2 ONNX — same dataset, higher capacity, ~12 ms/frame.
               No V/A head; uses the blended path.

efficientnet   Original HQRAF EfficientNet-B0 via PyTorch + Haar cascade face detect.
               Requires EmotionProcessor from modules/emotion_processor.py.
               Forces device='cpu' to avoid RTX 5060 sm_120 / torch 2.2.2 conflict.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

# ── Label normalisation ───────────────────────────────────────────────────────

STANDARD_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Russell (1980) circumplex, used ONLY by the non-regression fallback path (see
# HSEmotionDetector._infer). The default model regresses V/A directly and never
# touches this table.
#
# WARNING: this used to claim it "mirrors kg_bridge._EMOTION_VA". It does not, and
# no test covers it. It disagrees with BOTH kg_bridge._EMOTION_VA and
# affect.CATEGORY_VA on six of seven shared labels — e.g. angry is (-0.6, 0.7) here
# and (-0.7, 0.65) in the other two; happy is (0.8, 0.6) here and (0.8, 0.5) there.
# Only `neutral` agrees. Left as-is rather than silently re-tuned, because these
# values are what the published measurements of the blended path were taken with.
_VA_TABLE: dict[str, tuple[float, float]] = {
    "angry":    (-0.6,  0.7),
    "disgust":  (-0.6,  0.3),
    "fear":     (-0.5,  0.8),
    "happy":    ( 0.8,  0.6),
    "neutral":  ( 0.0,  0.0),
    "sad":      (-0.7, -0.4),
    "surprise": ( 0.1,  0.8),
}

_NORM = {
    # hsemotion (capitalised)
    'Anger':     'angry',
    'Contempt':  'disgust',   # contempt ≈ disgust in Russell VA space
    'Disgust':   'disgust',
    'Fear':      'fear',
    'Happiness': 'happy',
    'Neutral':   'neutral',
    'Sadness':   'sad',
    'Surprise':  'surprise',
    # lowercase / misc
    'anger':     'angry',
    'contempt':  'disgust',
    'disgust':   'disgust',
    'fear':      'fear',
    'happy':     'happy',
    'happiness': 'happy',
    'neutral':   'neutral',
    'sad':       'sad',
    'sadness':   'sad',
    'surprise':  'surprise',
}

def _norm(label: str) -> str:
    return _NORM.get(label, 'neutral')


def _clamp_unit(x: float) -> float:
    """Hold a V/A value inside the [-1, +1] interval PAD is defined on."""
    return max(-1.0, min(1.0, float(x)))


# ── Simple sliding-window smoother ────────────────────────────────────────────

class _Smoother:
    def __init__(self, window: int = 5):
        self._hist: deque = deque(maxlen=window)

    def update(self, label: str, conf: float) -> tuple[str, float]:
        self._hist.append((label, conf))
        counts: dict[str, float] = {}
        for lbl, c in self._hist:
            counts[lbl] = counts.get(lbl, 0.0) + c
        best = max(counts, key=counts.__getitem__)
        avg_conf = counts[best] / sum(1 for l, _ in self._hist if l == best)
        return best, avg_conf


# ── Abstract base ─────────────────────────────────────────────────────────────

class EmotionDetector:
    """
    Detect emotion from a BGR face crop.

    Subclass and override _infer(face_bgr) → (label, confidence_0_to_100).
    Call detect() for the public interface with optional smoothing.
    """

    name: str = "base"

    def __init__(self, smooth_window: int = 5):
        self._smoother = _Smoother(smooth_window)

    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        """Return (label, confidence_0_to_100, valence, arousal)."""
        raise NotImplementedError

    def detect(
        self,
        frame_bgr:  np.ndarray,
        box:        Optional[tuple] = None,
        smooth:     bool = True,
    ) -> tuple[str, float, float, float]:
        """
        Detect emotion.

        Args:
            frame_bgr : Full BGR webcam frame (or a face crop if box is None).
            box       : (x1, y1, x2, y2) face bounding box in frame_bgr.
                        If None, treats frame_bgr as the face crop directly.
            smooth    : Apply 5-frame sliding-window smoothing (label + conf only).

        Returns:
            (emotion_label, confidence_0_to_100, valence, arousal)
            v/a are from the latest raw frame; label/conf may be window-smoothed.
        """
        if box is not None:
            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                return 'neutral', 0.0, 0.0, 0.0
        else:
            face = frame_bgr

        try:
            label, conf, v, a = self._infer(face)
        except Exception:
            return 'neutral', 0.0, 0.0, 0.0

        label = _norm(label)
        if smooth:
            label, conf = self._smoother.update(label, conf)
        return label, conf, v, a

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def create(backend: str = 'hsemotion', **kwargs) -> 'EmotionDetector':
        """
        Factory method.

        backend options:
            'hsemotion'        — EfficientNet-B0 ONNX, V/A REGRESSED   [default]
            'hsemotion-lookup' — EfficientNet-B0 ONNX, V/A from _VA_TABLE
            'hsemotion-b2'     — EfficientNet-B2 ONNX (AffectNet, accurate)
            'efficientnet'     — original HQRAF EfficientNet via PyTorch

        'hsemotion-lookup' is the previous default, kept so the regressed and
        blended V/A paths stay comparable on the same frames; only the model
        differs between it and 'hsemotion'.
        """
        b = backend.lower()
        if b == 'hsemotion':
            return HSEmotionDetector(model='enet_b0_8_va_mtl', **kwargs)
        if b in ('hsemotion-lookup', 'hsemotion_lookup', 'hsemotion-vgaf'):
            return HSEmotionDetector(model='enet_b0_8_best_vgaf', **kwargs)
        if b in ('hsemotion-b2', 'hsemotion_b2'):
            return HSEmotionDetector(model='enet_b2_8', **kwargs)
        if b == 'efficientnet':
            return EfficientNetDetector(**kwargs)
        raise ValueError(
            f"Unknown backend {backend!r}. Choose: 'hsemotion', "
            "'hsemotion-lookup', 'hsemotion-b2', 'efficientnet'"
        )

    @staticmethod
    def available_backends() -> list[str]:
        backends = ['efficientnet']
        try:
            import hsemotion_onnx  # noqa: F401
            backends += ['hsemotion', 'hsemotion-lookup', 'hsemotion-b2']
        except ImportError:
            pass
        return backends


# ─────────────────────────────────────────────────────────────────────────────
# HSEmotion ONNX backend
# ─────────────────────────────────────────────────────────────────────────────

class HSEmotionDetector(EmotionDetector):
    """
    HSE EfficientNet ONNX emotion detector.

    Trained on AffectNet (450k images, 8 classes) — one of the largest
    publicly available emotion datasets. ONNX runtime means zero CUDA/PyTorch
    version dependency: runs at 5 ms/frame on CPU via onnxruntime.

    Models (cached to ~/.hsemotion/ on first use):
        enet_b0_8_va_mtl     — B0, AffectNet, 8 classes + V/A regression [default]
        enet_b0_8_best_vgaf  — B0, AffectNet+VGAF, 8 classes   [lookup-table V/A]
        enet_b2_8            — B2, AffectNet, 8 classes        [more accurate, ~12 ms]
        enet_b2_7            — B2, AffectNet, 7 classes        [no Contempt class]

    Two ways V/A is produced, and which one runs depends on the model:

    REGRESSED (`_mtl` models).  The network emits 8 class logits followed by two
    extra scalars, so `predict_emotions` returns a 10-vector. Valence and arousal
    are read straight off the end. Ordering is not documented by the library, but
    it is fixed by the author's own training code for this exact checkpoint
    (face-emotion-recognition, training_and_examples/affectnet/
    train_emotions-pytorch.ipynb):

        loss_valence = self.loss_valence(preds[:, num_classes],   target[1])
        loss_arousal = self.loss_arousal(preds[:, num_classes+1], target[2])

    with num_classes = 8 — so index 8 (= [-2]) is valence and index 9 (= [-1]) is
    arousal. The regression targets are raw AffectNet annotations on [-1, +1] with
    no normalisation, fitted with a Concordance-Correlation-Coefficient loss.

    BLENDED (every other model).  No V/A head exists, so it is reconstructed as a
    softmax-weighted average over `_VA_TABLE`. Kept selectable because it is the
    baseline the regression is measured against, but note what it costs: every
    output is a convex combination of seven fixed points, so it can only ever land
    inside their hull. Measured over 480 real faces (60/class, AffectNet-HQ +
    RAF-DB), the blend reached v [-0.698, +0.800] / a [-0.390, +0.799] and filled
    93 of 400 cells of a 20x20 grid on [-1,+1]^2; the regression reached
    v [-1.063, +0.985] / a [-0.649, +1.484] and filled 204. Categorical accuracy
    was 68.8% (va_mtl) against 67.5% (best_vgaf) on the same images, so the
    regression head is not bought at the price of classification.
    """

    name = "hsemotion"

    def __init__(self, model: str = 'enet_b0_8_va_mtl', smooth_window: int = 5):
        super().__init__(smooth_window)
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError:
            raise ImportError("pip install hsemotion-onnx")

        self._model_name = model
        print(f"[EmotionDetector] Loading HSEmotion ONNX model: {model} …")
        self._recognizer = HSEmotionRecognizer(model_name=model)
        self._labels     = self._recognizer.idx_to_class
        # The library sets this from the model name; it decides whether the score
        # vector carries the two extra V/A outputs.
        self._is_mtl     = bool(getattr(self._recognizer, 'is_mtl', False))
        print(f"[EmotionDetector] HSEmotion ready  labels={list(self._labels.values())}  "
              f"V/A={'regressed' if self._is_mtl else 'blended from _VA_TABLE'}")

    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        label, scores = self._recognizer.predict_emotions(face_rgb, logits=False)

        # Class probabilities are the leading `len(self._labels)` entries. On an
        # _mtl model `scores` is longer than the label set, so slicing rather than
        # zipping over the whole vector is what keeps confidence honest and stops
        # the V/A tail being read as if it were a class.
        n = len(self._labels)
        probs = scores[:n]
        conf = float(np.max(probs)) * 100.0

        if self._is_mtl:
            # Regressed directly by the network — see the class docstring for how
            # the index ordering is established. Clamped because the CCC loss does
            # not bound the head: on real faces this model returned arousal up to
            # +1.484 and valence down to -1.063, and everything downstream (the
            # empathy fusion in pad_core.affect.feel, then the published Eq. 9 /
            # Eq. A1 style mapping) assumes a PAD coordinate on [-1, +1].
            v = _clamp_unit(float(scores[-2]))
            a = _clamp_unit(float(scores[-1]))
        else:
            v = a = 0.0
            for i, score in enumerate(probs):
                lv, la = _VA_TABLE.get(_norm(self._labels[i]), (0.0, 0.0))
                v += float(score) * lv
                a += float(score) * la
            v, a = _clamp_unit(v), _clamp_unit(a)

        return label, conf, v, a

    @property
    def all_labels(self) -> list[str]:
        return [_norm(v) for v in self._labels.values()]


# ─────────────────────────────────────────────────────────────────────────────
# Original EfficientNet (HQRAF) via Modules/emotion_processor.py
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetDetector(EmotionDetector):
    """
    Wraps the existing EmotionProcessor (EfficientNet-B0, HQRAF dataset).

    Kept as a comparison baseline.  Runs on CPU to avoid RTX 5060 sm_120
    incompatibility with torch 2.2.2.
    """

    name = "efficientnet"

    def __init__(self, smooth_window: int = 5):
        super().__init__(smooth_window)

        # Add CHATBOX_SERVER root to path if needed
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            from modules.emotion_processor import EmotionProcessor
        except ImportError:
            raise ImportError("modules/emotion_processor.py not found")

        print("[EmotionDetector] Loading EfficientNet (original HQRAF) …")
        self._proc = EmotionProcessor(device="cpu")
        ok, total  = self._proc.initialize()
        print(f"[EmotionDetector] EfficientNet ready ({ok}/{total} components)")

        # Keep Haar cascade for face detection inside the crop
        self._haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        emotion, conf, _ = self._proc.process_emotion_detection_realtime(face_bgr)
        v, a = _VA_TABLE.get(_norm(emotion), (0.0, 0.0))
        return emotion, float(conf), v, a


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, time as _time

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="hsemotion",
                   choices=["hsemotion", "hsemotion-lookup",
                            "hsemotion-b2", "efficientnet"])
    p.add_argument("--camera",  type=int, default=0)
    args = p.parse_args()

    print(f"\nAvailable backends: {EmotionDetector.available_backends()}")
    det = EmotionDetector.create(args.backend)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera"); sys.exit(1)

    print(f"\nRunning {args.backend} live — press Q to quit\n")
    t0 = _time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        emotion, conf = det.detect(frame, smooth=True)

        fps = frames / max(0.001, _time.time() - t0)
        label = f"{emotion} ({conf:.0f}%)  [{fps:.1f} fps]"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imshow(f"emotion: {args.backend}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
