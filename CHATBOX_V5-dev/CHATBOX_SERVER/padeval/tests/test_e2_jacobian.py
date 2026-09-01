"""
E2 gate (analytic half): the identifiability metric is calibrated, and the
degenerate assignment is provably degenerate.

Two claims are pinned:

  1. `collapse_D` is RANK-DEFICIENT. Two sources move the effectors along the
     same direction, so no observer can attribute a change to one of them. That
     is unidentifiability proved rather than estimated — no dataset, no noise
     model, no classifier.

  2. The leakage metric RESPONDS TO A KNOWN MANIPULATION. Reading a single
     leakage number off the shipped system tells you nothing about whether the
     metric works. Running it with the close-tier +0.10 arousal term present and
     again with that one term zeroed must recover exactly 0.10.

Also asserts the two structural facts the analysis leans on, straight off the
real `gesture_style`: droop is a pure function of pleasure and idle a pure
function of arousal. If either stops being true, the leakage attribution loses
its footing and this fails loudly.

numpy only. No LLM, no camera.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.analysis.e2_identify import (                    # noqa: E402
    LEAK_OFF, LEAK_ON, SOURCES, STYLE_KEYS, analyse, conditioning, jacobian,
    leakage_matrix,
)
from padeval.axes import ASSIGNMENTS                          # noqa: E402

ROBOTS = ("CHATBOX", "ELLEBOT")


def test_droop_is_pure_pleasure_and_idle_is_pure_arousal():
    """The attribution in E2 rests on these two, so they are asserted against the
    real gesture_style rather than trusted from a comment."""
    J = jacobian(ASSIGNMENTS["identity"], "CHATBOX", ramp=LEAK_OFF)
    droop, idle = STYLE_KEYS.index("droop"), STYLE_KEYS.index("idle")
    v, a, t = (SOURCES.index(s) for s in ("valence", "arousal", "tier"))
    assert abs(J[droop, a]) < 1e-9 and abs(J[droop, t]) < 1e-9, J[droop]
    assert abs(J[droop, v]) > 0.1, J[droop, v]
    assert abs(J[idle, v]) < 1e-9 and abs(J[idle, t]) < 1e-9, J[idle]
    assert abs(J[idle, a]) > 0.1, J[idle, a]
    print("1. droop is pure pleasure; idle is pure arousal ✓")


def test_collapse_is_rank_deficient_and_the_others_are_not():
    for robot in ROBOTS:
        for name in ("identity", "perm_emotion_D", "perm_arousal_D"):
            c = conditioning(jacobian(ASSIGNMENTS[name], robot))
            assert c["rank"] == 3, f"{robot}/{name} rank {c['rank']}"
            assert np.isfinite(c["cond"]), f"{robot}/{name} cond not finite"
        c = conditioning(jacobian(ASSIGNMENTS["collapse_D"], robot))
        assert c["rank"] < 3, f"{robot}/collapse_D rank {c['rank']} — expected < 3"
    print("2. collapse_D is rank-deficient; the others are full rank ✓")


def test_leakage_metric_recovers_the_injected_violation():
    """The calibration. Exactly 0.10 in, exactly 0.10 out."""
    for robot in ROBOTS:
        on = {r["assignment"]: r for r in analyse(robot, ramp=LEAK_ON)}
        off = {r["assignment"]: r for r in analyse(robot, ramp=LEAK_OFF)}
        got_on = on["identity"]["d_arousal_d_tier"]
        got_off = off["identity"]["d_arousal_d_tier"]
        assert abs(got_on - 0.10) < 1e-6, f"{robot}: leak-on {got_on}"
        assert abs(got_off) < 1e-9, f"{robot}: leak-off {got_off}"
    print("3. leakage metric recovers the injected 0.10 to 1e-6, and reads 0 "
          "when it is removed ✓")


def test_identity_leakage_is_confined_to_the_known_term():
    """With the one documented violation removed, the deployed assignment's
    source->axis map is exactly diagonal. That is the disjoint-axis claim as an
    equation rather than a sentence."""
    for robot in ROBOTS:
        L = leakage_matrix(ASSIGNMENTS["identity"], robot, ramp=LEAK_OFF)
        off_diag = L - np.diag(np.diag(L))
        assert np.allclose(off_diag, 0.0, atol=1e-9), f"{robot} off-diagonal:\n{L}"
        assert np.all(np.diag(L) > 1e-6), f"{robot} a source writes nothing:\n{L}"
    print("4. with the leak removed, identity's source->axis map is diagonal ✓")


def test_the_shipped_system_has_exactly_one_off_diagonal_term():
    """And with the leak present, exactly ONE off-diagonal entry is non-zero —
    the tier->arousal term. Not a diffuse smear; one named, located violation."""
    L = leakage_matrix(ASSIGNMENTS["identity"], "CHATBOX", ramp=LEAK_ON)
    off_diag = L - np.diag(np.diag(L))
    nonzero = np.argwhere(off_diag > 1e-9)
    assert len(nonzero) == 1, f"expected 1 off-diagonal term, got:\n{L}"
    row, col = nonzero[0]
    assert row == 1 and col == 2, (row, col)      # Ar <- tier
    assert abs(off_diag[row, col] - 0.10) < 1e-6
    print("5. the shipped system has exactly one off-diagonal term: "
          "Ar <- tier, 0.10 ✓")


def test_jacobian_is_stable_across_step_sizes():
    """Central differences on a clamped, piecewise function can be fragile; check
    the reported numbers are not an artefact of the step."""
    a = ASSIGNMENTS["identity"]
    ref = jacobian(a, "CHATBOX", ramp=LEAK_ON, h=1e-6)
    for h in (1e-5, 1e-7):
        alt = jacobian(a, "CHATBOX", ramp=LEAK_ON, h=h)
        assert np.allclose(ref, alt, atol=1e-4), f"h={h} diverged:\n{ref - alt}"
    print("6. Jacobian stable across step sizes 1e-5 .. 1e-7 ✓")


if __name__ == "__main__":
    test_droop_is_pure_pleasure_and_idle_is_pure_arousal()
    test_collapse_is_rank_deficient_and_the_others_are_not()
    test_leakage_metric_recovers_the_injected_violation()
    test_identity_leakage_is_confined_to_the_known_term()
    test_the_shipped_system_has_exactly_one_off_diagonal_term()
    test_jacobian_is_stable_across_step_sizes()
    print("\nthe leakage metric is calibrated; collapse_D is provably degenerate.")
