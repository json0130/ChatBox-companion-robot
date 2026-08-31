"""
Rapport and trust are two axes, not one written twice.

The accrual rule had no automated coverage, and under that cover the live path
wrote the SAME delta to both axes. Since the tier reads
`score = (rapport + trust) / 2`, that collapsed to `score = rapport` and made
trust a decorative duplicate — while also letting `close`, the top rung of a
signal the design calls slow-moving, arrive after ~48 s of smiling.

What separates the two is already stated in the extractor's own brief
(extraction.py): "rapport_delta rises with warmth and positive affect;
trust_delta rises with the child sharing personal things." Warmth is readable
from a face. Disclosure is not.

    python3 test_closeness.py        # or: pytest test_closeness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.face_webcam.webcam_loop import (          # noqa: E402
    _read_rapport_trust, _update_rapport_trust,
)
from modules.graph_relationship.kg_bridge import _tier_from_scores   # noqa: E402
from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment  # noqa: E402
from modules.graph_relationship.store import InMemoryGraphStore      # noqa: E402

PID, RID = "kid", "chatbox"


def _store():
    s = InMemoryGraphStore()
    s.upsert_node(PersonNode(id=PID))
    s.upsert_node(RobotNode(id=RID, name=RID, embodiment=Embodiment.CAT))
    return s


def test_the_axes_move_independently():
    """The whole point: one may move without the other."""
    s = _store()
    _update_rapport_trust(s, PID, RID, d_rapport=0.30)
    assert _read_rapport_trust(s, PID, RID) == (0.30, 0.0)

    _update_rapport_trust(s, PID, RID, d_trust=0.10)
    r, t = _read_rapport_trust(s, PID, RID)
    assert (round(r, 6), round(t, 6)) == (0.30, 0.10)
    print("1. rapport and trust move independently ✓")


def test_deltas_clamp_at_one():
    s = _store()
    _update_rapport_trust(s, PID, RID, d_rapport=5.0, d_trust=5.0)
    assert _read_rapport_trust(s, PID, RID) == (1.0, 1.0)
    print("2. both axes clamp at 1.0 ✓")


def test_a_smile_alone_cannot_reach_close():
    """Replays the live per-tick rule: RAPPORT only, gated on felt P > 0.05.

    Before the split this reached `close` in ~48 s. Trust has to come from
    somewhere a face cannot show it.
    """
    s = _store()
    felt_p = 0.586                      # CHATBOX, happy face, empathy 0.6
    for _ in range(3600):               # a full hour at 1 Hz
        if felt_p > 0.05:
            _update_rapport_trust(s, PID, RID, d_rapport=0.025 * felt_p)
    r, t = _read_rapport_trust(s, PID, RID)
    assert r == 1.0, r
    assert t == 0.0, f"a face must not move trust, got {t}"

    tier = _tier_from_scores(r, t, count=3600)
    assert tier == "known", f"an hour of smiling should stop at 'known', got {tier}"
    print("3. an hour of smiling saturates rapport but never reaches 'close' ✓")


def test_close_needs_trust_from_disclosure():
    """Trust is what unlocks the top rung, and it arrives per session."""
    s = _store()
    _update_rapport_trust(s, PID, RID, d_rapport=1.0)        # rapport maxed
    assert _tier_from_scores(*_read_rapport_trust(s, PID, RID), 100) == "known"

    # End-of-session extraction, at its +0.2 per-session cap.
    for session, expected in ((1, "known"), (2, "known"), (3, "close")):
        _update_rapport_trust(s, PID, RID, d_trust=0.2)
        r, t = _read_rapport_trust(s, PID, RID)
        got = _tier_from_scores(r, t, count=100)
        assert got == expected, f"session {session}: expected {expected}, got {got}"
    print("4. 'close' needs trust — 3 sessions at the +0.2 cap, not 48 seconds ✓")


def test_operator_override_still_reaches_close():
    """The B key is an explicit override, not an inference, so it moves both."""
    s = _store()
    for _ in range(5):
        _update_rapport_trust(s, PID, RID, d_rapport=0.15, d_trust=0.15)
    r, t = _read_rapport_trust(s, PID, RID)
    assert round(r, 6) == round(t, 6) == 0.75
    assert _tier_from_scores(r, t, count=1) == "close"
    print("5. five B presses still reach 'close' for testing ✓")


def test_score_no_longer_collapses_to_rapport():
    """Regression guard on the actual defect. If some future caller writes the
    same delta to both again, score degenerates to rapport and this fails."""
    s = _store()
    _update_rapport_trust(s, PID, RID, d_rapport=0.8, d_trust=0.2)
    r, t = _read_rapport_trust(s, PID, RID)
    assert r != t, "the two axes have become identical again"
    score = (r + t) / 2
    assert abs(score - r) > 1e-9, "score has collapsed back onto rapport"
    print("6. score is a genuine average of two distinct axes ✓")


if __name__ == "__main__":
    test_the_axes_move_independently()
    test_deltas_clamp_at_one()
    test_a_smile_alone_cannot_reach_close()
    test_close_needs_trust_from_disclosure()
    test_operator_override_still_reaches_close()
    test_score_no_longer_collapses_to_rapport()
    print("\nrapport and trust are two axes again.")
