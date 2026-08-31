"""
The camera's valence/arousal path — the one thing nothing used to test.

Before this file existed, `emotion_detector._VA_TABLE` and the V/A arithmetic
around it had no automated coverage whatsoever. A comment in kg_bridge claimed
`test_pad_affect` guarded the tables; that file was deleted in 324f569 and the
claim went stale without anything failing. Three tables drifted apart in the
meantime and nobody noticed.

Needs no camera. Model weights are cached to ~/.hsemotion/ after the first run;
tests that need them skip cleanly when hsemotion-onnx is not installed.

    python3 test_emotion_va.py          # or: pytest test_emotion_va.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.face_webcam.emotion_detector import (   # noqa: E402
    _VA_TABLE, EmotionDetector, _clamp_unit,
)

_HAS_ONNX = "hsemotion" in EmotionDetector.available_backends()
_needs_onnx = pytest.mark.skipif(not _HAS_ONNX, reason="hsemotion-onnx not installed")


def _images(n=6):
    """Synthetic frames — enough to exercise the arithmetic without a camera."""
    rng = np.random.default_rng(0)
    return [(rng.random((96, 96, 3)) * 255).astype(np.uint8) for _ in range(n)]


def test_clamp_unit_holds_the_pad_domain():
    """The regression head is unbounded; PAD is not."""
    assert _clamp_unit(1.484) == 1.0        # a real observed arousal value
    assert _clamp_unit(-1.063) == -1.0      # a real observed valence value
    assert _clamp_unit(0.25) == 0.25
    print("1. _clamp_unit holds V/A inside [-1, +1] ✓")


@_needs_onnx
def test_default_backend_regresses_va():
    """'hsemotion' must be the multi-task model, not the lookup one. If this
    flips back, V/A silently collapses onto seven fixed points again."""
    det = EmotionDetector.create("hsemotion")
    assert det._model_name == "enet_b0_8_va_mtl", det._model_name
    assert det._is_mtl is True
    print("2. default backend regresses V/A (enet_b0_8_va_mtl) ✓")


@_needs_onnx
def test_lookup_backend_still_reachable():
    """The blended path is the baseline the regression is measured against."""
    det = EmotionDetector.create("hsemotion-lookup")
    assert det._model_name == "enet_b0_8_best_vgaf", det._model_name
    assert det._is_mtl is False
    print("3. lookup backend still selectable for comparison ✓")


@_needs_onnx
@pytest.mark.parametrize("backend", ["hsemotion", "hsemotion-lookup"])
def test_va_stays_inside_the_pad_domain(backend):
    """Everything downstream — the empathy fusion, then the published Eq. 9 /
    Eq. A1 mapping — assumes a coordinate on [-1, +1]."""
    det = EmotionDetector.create(backend)
    for img in _images():
        _label, conf, v, a = det.detect(img, box=None, smooth=False)
        assert -1.0 <= v <= 1.0, f"{backend}: valence {v} outside PAD domain"
        assert -1.0 <= a <= 1.0, f"{backend}: arousal {a} outside PAD domain"
        assert 0.0 <= conf <= 100.0, f"{backend}: confidence {conf}"
    print(f"4. {backend}: V/A within [-1,+1], confidence within [0,100] ✓")


@_needs_onnx
def test_confidence_ignores_the_va_tail():
    """On an _mtl model `scores` is 10 long while there are 8 labels. Taking the
    max over the whole vector would let a V/A value masquerade as class
    confidence, and zipping labels against it would raise KeyError at index 8."""
    det = EmotionDetector.create("hsemotion")
    n = len(det._labels)
    assert n == 8
    for img in _images(4):
        rgb = img[:, :, ::-1]
        _lbl, scores = det._recognizer.predict_emotions(rgb, logits=False)
        assert len(scores) == n + 2, f"expected {n}+2 outputs, got {len(scores)}"
        _l, conf, _v, _a = det.detect(img, box=None, smooth=False)
        expected = float(np.max(scores[:n])) * 100.0
        assert abs(conf - expected) < 1e-6, (conf, expected)
    print("5. confidence comes from the 8 class probs, not the V/A tail ✓")


@_needs_onnx
def test_regressed_va_is_read_from_the_documented_indices():
    """Ordering is fixed by the author's training code for this checkpoint:
    preds[:, num_classes] is valence and preds[:, num_classes+1] is arousal."""
    det = EmotionDetector.create("hsemotion")
    n = len(det._labels)
    for img in _images(4):
        rgb = img[:, :, ::-1]
        _lbl, scores = det._recognizer.predict_emotions(rgb, logits=False)
        _l, _c, v, a = det.detect(img, box=None, smooth=False)
        assert abs(v - _clamp_unit(float(scores[n]))) < 1e-6      # index 8  == [-2]
        assert abs(a - _clamp_unit(float(scores[n + 1]))) < 1e-6  # index 9  == [-1]
    print("6. valence <- scores[8], arousal <- scores[9] ✓")


def test_va_table_divergence_is_pinned():
    """`_VA_TABLE` disagrees with affect.CATEGORY_VA on six of seven labels. That
    is a known, deliberate state — the blended path's published measurements were
    taken with these values — but it must not drift further unnoticed, which is
    exactly what happened when the guarding test was deleted."""
    from modules.affect_bridge import affect

    shared = sorted(set(_VA_TABLE) & set(affect.CATEGORY_VA))
    differing = {k for k in shared if _VA_TABLE[k] != affect.CATEGORY_VA[k]}
    assert differing == {"angry", "disgust", "fear", "happy", "sad", "surprise"}, (
        f"the divergence changed: {sorted(differing)}. Update this test "
        "deliberately, or align the tables deliberately — do not do either by "
        "accident."
    )
    assert _VA_TABLE["neutral"] == affect.CATEGORY_VA["neutral"] == (0.0, 0.0)
    print(f"7. _VA_TABLE vs CATEGORY_VA: {len(differing)}/{len(shared)} labels "
          "differ, as documented ✓")


if __name__ == "__main__":
    test_clamp_unit_holds_the_pad_domain()
    if _HAS_ONNX:
        test_default_backend_regresses_va()
        test_lookup_backend_still_reachable()
        for b in ("hsemotion", "hsemotion-lookup"):
            test_va_stays_inside_the_pad_domain(b)
        test_confidence_ignores_the_va_tail()
        test_regressed_va_is_read_from_the_documented_indices()
    else:
        print("   (skipped model-backed checks — hsemotion-onnx not installed)")
    test_va_table_divergence_is_pinned()
    print("\nthe camera V/A path is covered.")
