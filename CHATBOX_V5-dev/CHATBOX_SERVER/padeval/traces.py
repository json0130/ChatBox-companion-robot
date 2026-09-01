"""
Valence/arousal frame traces — the input side of the chattering measurement.

The camera emits V/A at frame rate. Whether that is a problem depends entirely on
which effector the signal is routed to, and measuring that needs a trace rather
than a single coordinate.

No video corpus is available here, so the trace is SYNTHESISED from two parts
that are separately defensible:

  1. a piecewise-constant emotion schedule — the person holds an expression for a
     few seconds, then changes. Expression dwell times of 2-6 s are the regime
     the 1 Hz pipeline tick and the 5-sample AffectStream window were designed
     around.
  2. additive Gaussian frame noise of scale sigma.

Sigma is NOT guessed. `measure_model_dispersion()` derives it from the actual
regression head running over the real face corpus, and the chattering result is
reported as a curve over sigma rather than at a single point, so the conclusion
does not rest on one number.

numpy only — this module must import on a machine with no analysis stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modules.affect_bridge import affect

# The expressions a child-facing camera actually sees, and roughly how often.
# Weighted toward neutral because that is what a face does most of the time.
DEFAULT_SCHEDULE = ("neutral", "happy", "neutral", "sad",
                    "neutral", "happy", "neutral", "surprise")


@dataclass(frozen=True)
class Trace:
    """A V/A frame trace plus the ground-truth emotion label per frame."""
    va: np.ndarray          # (n_frames, 2)
    labels: list            # n_frames emotion names
    fps: float

    @property
    def duration_s(self) -> float:
        return len(self.labels) / self.fps


def synth_trace(schedule: Sequence[str] = DEFAULT_SCHEDULE,
                dwell_s: float = 4.0,
                fps: float = 30.0,
                sigma: float = 0.0,
                seed: int = 0) -> Trace:
    """Piecewise-constant emotion schedule + Gaussian frame noise.

    Corner V/A comes from `affect.CATEGORY_VA`, i.e. the project's own table, so
    the trace's anchor points are the ones the rest of the system already uses.
    """
    rng = np.random.default_rng(seed)
    per = int(round(dwell_s * fps))
    va, labels = [], []
    for emo in schedule:
        v, a = affect.category_to_va(emo)
        block = np.tile(np.array([v, a], dtype=float), (per, 1))
        if sigma > 0:
            block = block + rng.normal(0.0, sigma, size=block.shape)
        va.append(np.clip(block, -1.0, 1.0))
        labels.extend([emo] * per)
    return Trace(va=np.vstack(va), labels=labels, fps=fps)


def measure_model_dispersion(zip_path: str, per_class: int = 40,
                             seed: int = 0) -> dict:
    """Within-class dispersion of the live V/A regression over the real corpus.

    Returns per-class and pooled standard deviations of (valence, arousal) for
    images that share a ground-truth label. This is an UPPER BOUND on
    frame-to-frame noise for a single face — it mixes identity, pose and lighting
    variation as well as sensor noise — and must be reported as such. It is used
    to bound the sigma sweep, not to claim a specific operating point.

    Requires the face corpus and hsemotion-onnx; callers should treat an
    exception here as "skip, report the sweep without an anchor".
    """
    import zipfile
    import cv2
    from modules.face_webcam.emotion_detector import EmotionDetector

    det = EmotionDetector.create("hsemotion")
    z = zipfile.ZipFile(zip_path)
    rng = np.random.default_rng(seed)

    by_class: dict = {}
    for name in z.namelist():
        if name.endswith("/") or name.endswith(".csv"):
            continue
        by_class.setdefault(name.split("/")[0], []).append(name)

    out, pooled = {}, []
    for cls, names in sorted(by_class.items()):
        names = sorted(names)
        take = rng.choice(len(names), size=min(per_class, len(names)),
                          replace=False)
        rows = []
        for i in take:
            buf = np.frombuffer(z.read(names[i]), np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            _lbl, _conf, v, a = det.detect(bgr, box=None, smooth=False)
            rows.append((v, a))
        arr = np.asarray(rows, dtype=float)
        out[cls] = {"n": len(arr),
                    "sd_valence": float(arr[:, 0].std()),
                    "sd_arousal": float(arr[:, 1].std())}
        pooled.append(arr - arr.mean(axis=0))

    stacked = np.vstack(pooled)
    out["_pooled"] = {"n": int(stacked.shape[0]),
                      "sd_valence": float(stacked[:, 0].std()),
                      "sd_arousal": float(stacked[:, 1].std())}
    return out
