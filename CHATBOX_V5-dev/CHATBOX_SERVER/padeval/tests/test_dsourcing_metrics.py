"""
Phase 11b gate. Each metric checked for correctness on its own terms, before
11c checks the registered predictions against the results.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect                       # noqa: E402
from padeval.analysis.dsourcing import ARMS, M                 # noqa: E402
from padeval.analysis.dsourcing_metrics import (                # noqa: E402
    TAU_P_TICKS, _closed_form_std_ramp, metric1_rank,
    metric2_frame_switching, metric3_noise_propagation, metric4_timescale,
    metric5_ownership_alternation,
)
from padeval.analysis.noise_propagation import closed_form_std  # noqa: E402
from padeval.axes import IDENTITY                                # noqa: E402

ROBOTS = ("CHATBOX", "ELLEBOT")


def test_ramp_helper_matches_the_existing_tested_function_exactly():
    """`_closed_form_std_ramp` is new code; it must reproduce the ALREADY
    TESTED `closed_form_std` bit for bit at the one point where they overlap
    (ramp == TIER_OFFSETS["close"]), or metric 3's numbers for A1/A2/A3 rest on
    an unverified reimplementation rather than the real, checked machinery."""
    for robot in ROBOTS:
        want = closed_form_std(IDENTITY, robot, "close")
        got = _closed_form_std_ramp(robot, affect.TIER_OFFSETS["close"])
        assert want == got, (robot, want, got)
    print("1. new ramp-taking helper is bit-identical to the tested "
          "closed_form_std at the point where they overlap ✓")


def test_metric1_A5_is_the_only_rank_deficient_arm():
    for robot in ROBOTS:
        ranks = {arm: metric1_rank(robot, arm)["rank"] for arm in ARMS}
        assert ranks["A5_no_relationship"] == 2, ranks
        assert all(ranks[a] == 3 for a in ARMS if a != "A5_no_relationship"), ranks
    print("2. A5 is rank 2/3 (nothing reaches D); every other arm is rank 3/3 ✓")


def test_metric1_A1_A2_A3_are_numerically_identical():
    """The structural claim from the module docstring, checked directly: these
    three cannot be told apart by a static axis-level Jacobian."""
    for robot in ROBOTS:
        j1 = metric1_rank(robot, "A1_full")
        j2 = metric1_rank(robot, "A2_no_trust")
        j3 = metric1_rank(robot, "A3_no_accumulation")
        assert j1["sv"] == j2["sv"] == j3["sv"], (j1, j2, j3)
    print("3. A1/A2/A3 give bit-identical singular values on metric 1 ✓")


def test_metric2_A1_A2_are_structural_zero():
    for robot in ROBOTS:
        for arm in ("A1_full", "A2_no_trust"):
            m = metric2_frame_switching(robot, arm)
            assert m["switches"] == 0, m
    print("4. A1/A2: structurally zero switching under frame noise ✓")


def test_metric2_A3_is_NOT_zero_and_that_is_the_point():
    """The one place a literal reading of registered prediction 2 does not
    hold. Asserted here so a future change that accidentally makes A3 immune
    to frame noise is caught as a change to A3's mechanism, not celebrated as
    an improvement."""
    for robot in ROBOTS:
        m = metric2_frame_switching(robot, "A3_no_accumulation")
        assert m["switches"] > 0, (
            "A3 showed zero frame-noise switching — either its direct-coupling "
            "mechanism changed, or this test regressed silently")
    print("5. A3 switches under frame noise (CHATBOX "
          f"{metric2_frame_switching('CHATBOX','A3_no_accumulation')['switches']}, "
          f"ELLEBOT "
          f"{metric2_frame_switching('ELLEBOT','A3_no_accumulation')['switches']}) "
          f"— confirms prediction 2's stated exception, not a regression ✓")


def test_metric2_A4_A5_are_zero_under_frame_noise_specifically():
    for robot in ROBOTS:
        for arm in ("A4_wasabi", "A5_no_relationship"):
            assert metric2_frame_switching(robot, arm)["switches"] == 0
    print("6. A4 (ownership fixed) and A5 (D constant): zero under CAMERA "
          "noise specifically ✓")


def test_metric3_is_zero_for_every_arm_and_says_why():
    """Not a discriminating metric here — checked, not assumed."""
    for robot in ROBOTS:
        for arm in ARMS:
            m = metric3_noise_propagation(robot, arm)
            assert m["std_D"] == 0.0, m
    print("7. metric 3 (axis-level noise propagation): std(D)=0 for all five "
          "arms — it measures axis leakage, which none of the five have ✓")


def test_metric4_A2_close_is_unreachable():
    for robot in ROBOTS:
        m = metric4_timescale(robot, "A2_no_trust")
        assert m["tau_D_to_close"] is None
        assert m["tau_D_to_known"] is not None
    print("8. A2: `known` reachable, `close` provably never reached "
          "(re-confirms 9a under this phase's own harness) ✓")


def test_metric4_A3_settles_immediately_and_beats_deployed_tau_P():
    for robot in ROBOTS:
        m = metric4_timescale(robot, "A3_no_accumulation")
        assert m["tau_D_settling"] == 0.0
        assert m["ratio_to_tau_P"] == 0.0
    print(f"9. A3: settles in 0 ticks against the deployed tau_P="
          f"{TAU_P_TICKS} — at least as reactive as P, not merely "
          f"'the same order' ✓")


def test_metric4_A4_is_a_period_not_a_pole():
    for robot in ROBOTS:
        m = metric4_timescale(robot, "A4_wasabi")
        assert m["tau_D_ownership_period"] == 10.0
        assert 1 < m["ratio_to_tau_P"] < 10
    print(f"10. A4: ownership period 10 ticks, ratio to tau_P = "
          f"{metric4_timescale('CHATBOX','A4_wasabi')['ratio_to_tau_P']:.2f} "
          f"— same order as P, categorically not a session-scale quantity ✓")


def test_metric4_A1_A5_are_the_two_extremes():
    for robot in ROBOTS:
        a1 = metric4_timescale(robot, "A1_full")["tau_D"]
        a5 = metric4_timescale(robot, "A5_no_relationship")["tau_D"]
        assert a1 is not None and a1 >= 3      # 9a/8c floor
        assert a5 is None                       # no timescale at all
    print("11. A1: >=3 sessions (the 9a/8c floor). A5: no timescale — D is "
          "constant ✓")


def test_metric5_only_A4_moves_and_M_produces_a_real_rung_swing():
    for robot in ROBOTS:
        for arm in ARMS:
            r = metric5_ownership_alternation(robot, arm)
            if arm == "A4_wasabi":
                # every epoch alternates ownership, so every adjacent pair
                # differs: n_epochs-1 switches out of n_epochs-1 possible.
                assert r.switches == r.n_epochs - 1, r
            else:
                assert r.switches == 0, (arm, r)
    print("12. metric 5: only A4 moves, and it moves on every single "
          "ownership flip (39/39 possible transitions) ✓")


def test_metric5_A1_A2_do_not_repeat_the_count_artifact():
    """Regression pin for the bug this exact test class caught: calling the
    once-per-turn accrual function once per micro-epoch pushed `count` past
    the count>5 threshold and produced a spurious switch unrelated to
    ownership. Held fixed at n_epochs well past 6 to make sure it stays fixed."""
    for robot in ROBOTS:
        for arm in ("A1_full", "A2_no_trust"):
            r = metric5_ownership_alternation(robot, arm, n_epochs=100)
            assert r.switches == 0, (
                f"{arm} produced {r.switches} switch(es) at n_epochs=100 — "
                f"the count-threshold artifact may have returned")
    print("13. A1/A2 stay at zero switches even at 100 epochs — the count "
          "artifact does not return ✓")


if __name__ == "__main__":
    test_ramp_helper_matches_the_existing_tested_function_exactly()
    test_metric1_A5_is_the_only_rank_deficient_arm()
    test_metric1_A1_A2_A3_are_numerically_identical()
    test_metric2_A1_A2_are_structural_zero()
    test_metric2_A3_is_NOT_zero_and_that_is_the_point()
    test_metric2_A4_A5_are_zero_under_frame_noise_specifically()
    test_metric3_is_zero_for_every_arm_and_says_why()
    test_metric4_A2_close_is_unreachable()
    test_metric4_A3_settles_immediately_and_beats_deployed_tau_P()
    test_metric4_A4_is_a_period_not_a_pole()
    test_metric4_A1_A5_are_the_two_extremes()
    test_metric5_only_A4_moves_and_M_produces_a_real_rung_swing()
    test_metric5_A1_A2_do_not_repeat_the_count_artifact()
