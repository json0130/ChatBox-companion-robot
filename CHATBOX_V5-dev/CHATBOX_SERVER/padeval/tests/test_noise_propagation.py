"""
7.0 gate: measured-noise propagation, cross-checked two independent ways, and
reconciled against E3's simulated switching rather than declared inconsistent.

No LLM, no camera. Deterministic modulo the Monte Carlo / trace seeds, which
are fixed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.analysis.noise_propagation import (               # noqa: E402
    RUNG_GAPS, SIGMA_A, SIGMA_V, closed_form_std, monte_carlo_std,
    switch_boundary_clustering,
)
from padeval.axes import ASSIGNMENTS                            # noqa: E402


def test_identity_has_exactly_zero_noise_on_d_both_methods():
    """The one claim that does not depend on any of the crossing-rate
    subtlety below: identity has NO noise path to D at all, exactly, by
    both an analytic and a simulated route."""
    cf = closed_form_std(ASSIGNMENTS["identity"], "CHATBOX")
    mc = monte_carlo_std(ASSIGNMENTS["identity"], "CHATBOX", n=2000)
    assert cf["D"] == 0.0, cf["D"]
    assert mc["D"] < 1e-9, mc["D"]
    print("1. identity: D std is exactly 0 (closed form) and ~0 (Monte Carlo) ✓")


def test_closed_form_matches_monte_carlo():
    """Cross-validates the two implementations. If they disagreed, one of them
    would be wrong -- this is the check that rules that out before trusting
    either for the harder question below."""
    for name in ("identity", "perm_emotion_D", "perm_arousal_D"):
        cf = closed_form_std(ASSIGNMENTS[name], "CHATBOX")
        mc = monte_carlo_std(ASSIGNMENTS[name], "CHATBOX", n=20000, seed=3)
        for axis in ("P", "Ar", "D"):
            assert abs(cf[axis] - mc[axis]) < 0.01, (name, axis, cf[axis], mc[axis])
    print("2. closed-form and Monte Carlo agree to <0.01 on every axis, "
          "3 assignments ✓")


def test_permutations_have_nonzero_noise_on_d():
    for name in ("perm_emotion_D", "perm_arousal_D"):
        cf = closed_form_std(ASSIGNMENTS[name], "CHATBOX")
        assert cf["D"] > 0.05, (name, cf["D"])
    print("3. both permutations have nonzero propagated noise on D ✓")


def test_registered_prediction_as_literally_stated_is_not_met():
    """Honesty gate. The module's registered prediction said the propagated
    std would be comparable to or exceed the smallest rung gap (0.18) at the
    DEPLOYED (5-sample-smoothed) noise level. It is not — pinned here so this
    cannot quietly become a pass by editing the module's docstring after the
    fact. The resolution is a DIFFERENT, more precise claim (see the next
    test), not a redefinition of this one."""
    smallest_gap = RUNG_GAPS[0]
    for name in ("perm_emotion_D", "perm_arousal_D"):
        cf = closed_form_std(ASSIGNMENTS[name], "CHATBOX")  # window=5, deployed
        assert cf["D"] < smallest_gap, (
            f"{name}: std {cf['D']} now exceeds the smallest rung gap "
            f"{smallest_gap} -- the registered-prediction-failure this test "
            f"pins has changed; update the writeup, don't just fix this test"
        )
    print("4. as literally registered, propagated std < smallest rung gap at "
          "deployed smoothing -- prediction NOT met, exactly as reported ✓")


def test_most_switches_are_mid_block_not_at_scripted_transitions():
    """The resolution. If switching were mostly happening at scripted emotion
    transitions (a big, deterministic swing), a small marginal std would be
    perfectly consistent with E3's switch count and there would be nothing to
    explain. It is NOT mostly transitions: most switches happen away from any
    scripted change, which is what a marginal-std comparison cannot see,
    because switch COUNT is a level-crossing-rate question (depends on the
    smoothing filter's autocorrelation time), not a marginal-variance one."""
    for name in ("perm_emotion_D", "perm_arousal_D", "collapse_D"):
        r = switch_boundary_clustering(ASSIGNMENTS[name], "CHATBOX",
                                       sigma=0.20, seed=1)
        assert r["n_switches"] > 15, (name, r)
        assert r["near_boundary_frac"] < 0.40, (
            f"{name}: {r['near_boundary_frac']:.0%} of switches near a "
            f"scripted transition -- if this rises much, transitions (not "
            f"steady noise) would be the dominant driver and the writeup's "
            f"explanation would need revisiting"
        )
    print("5. most switches (>60%) occur mid-block, away from any scripted "
          "transition -- confirms genuine noise-driven crossing, not just "
          "transition-driven ✓")


if __name__ == "__main__":
    test_identity_has_exactly_zero_noise_on_d_both_methods()
    test_closed_form_matches_monte_carlo()
    test_permutations_have_nonzero_noise_on_d()
    test_registered_prediction_as_literally_stated_is_not_met()
    test_most_switches_are_mid_block_not_at_scripted_transitions()
    print("\nnoise propagation is cross-validated and the E3 apparent "
          "inconsistency is resolved, not hidden.")
