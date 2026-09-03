"""
Phase 6c gate: the sensitivity sweep's core claims, and the rank-1 correction.

No LLM, no camera, no hardware. Deterministic, numpy-only checks.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect                       # noqa: E402
from padeval.analysis.tier_sensitivity import (                # noqa: E402
    DEPLOYED_M, clamp_bound, offsets_at, sweep_conditioning, sweep_switching,
)


def test_parameterisation_reproduces_deployed_table():
    for tier, expect in affect.TIER_OFFSETS.items():
        got = offsets_at(tier, DEPLOYED_M)
        assert got == expect, (tier, got, expect)
    print("1. offsets_at(tier, 0.40) reproduces the live TIER_OFFSETS exactly ✓")


def test_identity_full_rank_across_the_swept_range():
    m_values = [0.10, 0.25, 0.40, 0.60]
    for robot in ("CHATBOX", "ELLEBOT"):
        rows = sweep_conditioning(m_values, robot=robot, assignments=("identity",))
        for r in rows:
            assert r["rank"] == 3, (robot, r["m"], r["rank"])
    print("2. identity is full rank at every swept m, both robots ✓")


def test_collapse_is_rank_1_not_rank_2():
    """The corrected number. See tier_sensitivity.sweep_conditioning's docstring
    for the full derivation of why the naive finite-difference call (h=1e-6)
    misreports this as rank 2."""
    m_values = [0.10, 0.25, 0.40, 0.60]
    for robot in ("CHATBOX", "ELLEBOT"):
        rows = sweep_conditioning(m_values, robot=robot, assignments=("collapse_D",))
        for r in rows:
            assert r["rank"] == 1, (robot, r["m"], r["rank"])
    print("3. collapse_D is rank 1 (corrected from the earlier '2/3') "
          "at every swept m, both robots ✓")


def test_permuted_assignments_stay_full_rank_and_finite():
    """The other permutations are NOT structurally degenerate like collapse —
    all three PAD axes still move, just via a relabelled source. Confirms the
    h=1e-6 -> h=1e-3 change did not just move the goalposts for these."""
    m_values = [0.10, 0.25, 0.40, 0.60]
    for name in ("perm_emotion_D", "perm_arousal_D"):
        rows = sweep_conditioning(m_values, robot="CHATBOX", assignments=(name,))
        for r in rows:
            assert r["rank"] == 3, (name, r["m"])
            assert np.isfinite(r["cond"]), (name, r["m"])
    print("4. perm_emotion_D / perm_arousal_D stay full rank, finite cond ✓")


def test_identity_never_switches_across_the_swept_range():
    m_values = [0.10, 0.25, 0.40, 0.60]
    rows = sweep_switching(m_values, robot="CHATBOX", tier="known", sigma=0.20,
                           assignments=("identity",))
    for r in rows:
        assert r["tick_switches"] == 0, r
    print("5. identity: 0 tick-level switches at every swept m ✓")


def test_permutations_switch_across_the_swept_range():
    m_values = [0.10, 0.25, 0.40, 0.60]
    for name in ("perm_emotion_D", "perm_arousal_D", "collapse_D"):
        rows = sweep_switching(m_values, robot="CHATBOX", tier="known", sigma=0.20,
                               assignments=(name,))
        for r in rows:
            assert r["tick_switches"] > 0, (name, r)
    print("6. every permutation switches at every swept m ✓")


def test_clamp_bound_matches_deployed_observation():
    """CHATBOX's own baseline already exceeds the deployed m=0.40 bound by
    0.043 — the exact figure the original audit measured as the observed
    `unknown`-tier clamp and the non-uniform ladder step (0.157 vs 0.200)."""
    b_chatbox = clamp_bound("CHATBOX")
    assert abs(b_chatbox - 0.357) < 1e-9, b_chatbox
    assert DEPLOYED_M - b_chatbox > 0, "deployed m should already exceed the bound"
    assert abs((DEPLOYED_M - b_chatbox) - 0.043) < 1e-9
    b_ellebot = clamp_bound("ELLEBOT")
    assert DEPLOYED_M < b_ellebot, "ELLEBOT should NOT be clamped at deployed m"
    print(f"7. clamp bound: CHATBOX {b_chatbox:.4f} (deployed exceeds by 0.043), "
          f"ELLEBOT {b_ellebot:.4f} (deployed is inside) ✓")


if __name__ == "__main__":
    test_parameterisation_reproduces_deployed_table()
    test_identity_full_rank_across_the_swept_range()
    test_collapse_is_rank_1_not_rank_2()
    test_permuted_assignments_stay_full_rank_and_finite()
    test_identity_never_switches_across_the_swept_range()
    test_permutations_switch_across_the_swept_range()
    test_clamp_bound_matches_deployed_observation()
    print("\nthe sensitivity sweep's claims hold across the swept range.")
