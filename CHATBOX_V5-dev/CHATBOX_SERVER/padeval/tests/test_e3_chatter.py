"""
E3 gate: the deployed assignment has a stable directive setpoint and the
permuted ones do not.

The claim being pinned is categorical, not quantitative. Under the deployed
routing the face cannot write Dominance at all, so the commanded rung is
invariant to the face NO MATTER how noisy the camera is — zero switches at every
sigma, by construction rather than by tuning. If that ever becomes non-zero,
either the routing or `manner_directive`'s band edges have moved, and E3's
headline result has quietly changed.

No LLM, no camera, no hardware, no analysis stack — numpy only.

    python3 -m pytest padeval/tests/test_e3_chatter.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect, prompt              # noqa: E402
from padeval.analysis.e3_permute import (                     # noqa: E402
    directive_volatility, rung_of, sweep,
)
from padeval.axes import ASSIGNMENTS, IDENTITY, PERM_EMOTION_D   # noqa: E402
from padeval.traces import synth_trace                        # noqa: E402

SIGMAS = (0.0, 0.05, 0.10, 0.20, 0.28)   # 0.28 = measured pooled within-class sd


def test_rung_of_matches_manner_directive():
    """rung_of must index the same ladder manner_directive walks, or every
    switch count is measuring the wrong thing."""
    for d in [x / 100.0 for x in range(-100, 101)]:
        i = rung_of(d)
        assert prompt._DIRECTIVES[i][1] == prompt.manner_directive(d), d
    print("1. rung_of agrees with prompt.manner_directive over [-1, +1] ✓")


def test_identity_never_chatters_at_any_noise_level():
    """The headline. Zero, not small — the face has no path to D."""
    for sigma in SIGMAS:
        trace = synth_trace(sigma=sigma, seed=1)
        for robot in ("CHATBOX", "ELLEBOT"):
            v = directive_volatility(IDENTITY, robot, "known", trace)
            assert v.rung_switches == 0, (
                f"identity chattered at sigma={sigma} on {robot}: "
                f"{v.rung_switches} switches"
            )
            assert v.distinct_rungs == 1
    print(f"2. identity: 0 rung switches at every sigma in {SIGMAS} ✓")


def test_permutation_chatters_and_worsens_with_noise():
    """The contrast. Also asserts monotonicity in sigma, which is what makes it
    a noise-sensitivity result rather than a single anecdote."""
    prev = -1.0
    for sigma in (0.0, 0.10, 0.20):
        trace = synth_trace(sigma=sigma, seed=1)
        v = directive_volatility(PERM_EMOTION_D, "ELLEBOT", "known", trace)
        assert v.rung_switches > 0, f"perm_emotion_D stable at sigma={sigma}"
        assert v.switches_per_min > prev, (
            f"chattering did not increase from sigma<{sigma}: "
            f"{prev} -> {v.switches_per_min}"
        )
        prev = v.switches_per_min
    print("3. perm_emotion_D chatters, and worsens monotonically with noise ✓")


def test_the_gap_is_categorical_not_marginal():
    """At the empirically measured noise level the two regimes must not be a
    judgement call — one is exactly zero and the other is many per minute."""
    trace = synth_trace(sigma=0.20, seed=1)
    ident = directive_volatility(IDENTITY, "ELLEBOT", "known", trace)
    perm = directive_volatility(PERM_EMOTION_D, "ELLEBOT", "known", trace)
    assert ident.switches_per_min == 0.0
    assert perm.switches_per_min > 60.0, perm.switches_per_min
    print(f"4. at sigma=0.20: identity 0.0/min vs perm "
          f"{perm.switches_per_min:.0f}/min ✓")


def test_rung_is_invariant_to_emotion_under_identity():
    """The static version of the same claim, straight off the affect model —
    this is the property the trace measurement is the dynamic evidence for."""
    for robot in ("CHATBOX", "ELLEBOT"):
        base = affect.to_pad(affect.ROBOTS[robot]["ocean"])
        for tier in affect.TIERS:
            seen = {
                rung_of(affect.feel_with_relationship(
                    base, *affect.category_to_va(e), tier)["D"])
                for e in ("happy", "sad", "angry", "neutral", "surprise")
            }
            assert len(seen) == 1, f"{robot}/{tier} rung varies with emotion: {seen}"
    print("5. commanded rung is invariant to emotion at every tier ✓")


def test_sweep_shape():
    rows = sweep(assignments=("identity", "perm_emotion_D"),
                 robots=("CHATBOX",), tiers=("known",), sigmas=(0.0, 0.1))
    assert len(rows) == 4
    assert {r["assignment"] for r in rows} == {"identity", "perm_emotion_D"}
    for r in rows:
        assert set(r) >= {"switches_per_min", "clamp_rate_d", "dwell_median_s"}
    print("6. sweep() returns a well-formed row per cell ✓")


if __name__ == "__main__":
    test_rung_of_matches_manner_directive()
    test_identity_never_chatters_at_any_noise_level()
    test_permutation_chatters_and_worsens_with_noise()
    test_the_gap_is_categorical_not_marginal()
    test_rung_is_invariant_to_emotion_under_identity()
    test_sweep_shape()
    print("\nthe deployed assignment is the only one with a stable setpoint.")
