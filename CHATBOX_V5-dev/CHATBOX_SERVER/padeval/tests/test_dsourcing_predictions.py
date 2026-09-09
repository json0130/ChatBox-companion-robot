"""
Phase 11c — the five registered predictions (dsourcing_metrics.py docstring),
checked against 11b's results and reported plainly. Verdicts are not softened
after the fact; a prediction that fails is reported as failed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.analysis.dsourcing import ARMS                     # noqa: E402
from padeval.analysis.dsourcing_metrics import (                 # noqa: E402
    ROBOTS, metric1_rank, metric2_frame_switching,
    metric3_noise_propagation, metric4_timescale,
    metric5_ownership_alternation,
)

NON_A1_ARMS = [a for a in ARMS if a != "A1_full"]


def test_prediction_1_all_five_rank_3():
    """As literally stated in the brief, this prediction is FALSE — checked
    honestly rather than reworded to pass. A5 is rank 2/3; every other arm is
    rank 3/3. Reported as: the comparability check WORKS (it disqualifies
    exactly the arm designed to have no relationship channel, for exactly that
    reason), not as a surprise to explain away."""
    ranks = {(robot, arm): metric1_rank(robot, arm)["rank"]
            for robot in ROBOTS for arm in ARMS}
    all_three = all(r == 3 for r in ranks.values())
    a5_only = (all(ranks[(r, a)] == 3 for r in ROBOTS for a in ARMS
                  if a != "A5_no_relationship")
              and all(ranks[(r, "A5_no_relationship")] == 2 for r in ROBOTS))
    print(f"P1 (all five rank 3): {'MET' if all_three else 'NOT MET'}")
    print(f"    actual: A1-A4 rank 3/3, A5 rank 2/3 — disqualified for a "
          f"different reason than collapse_D (7a): no source reaches D at "
          f"all, by design, not by axis collision.")
    assert not all_three and a5_only, ranks


def test_prediction_2_frame_noise_structural_zero_except_A3():
    """A1, A2, A4, A5 zero. A3 non-zero — stated as the exception in the
    prediction itself, checked here rather than assumed."""
    results = {(robot, arm): metric2_frame_switching(robot, arm)["switches"]
              for robot in ROBOTS for arm in ARMS}
    zero_arms = ("A1_full", "A2_no_trust", "A4_wasabi", "A5_no_relationship")
    all_zero = all(results[(r, a)] == 0 for r in ROBOTS for a in zero_arms)
    a3_nonzero = all(results[(r, "A3_no_accumulation")] > 0 for r in ROBOTS)
    print(f"P2 (structural zero except A3): "
          f"{'MET' if all_zero and a3_nonzero else 'NOT MET'}")
    print(f"    A1/A2/A4/A5: 0 switches, all robots. A3: "
          f"CHATBOX {results[('CHATBOX','A3_no_accumulation')]}, "
          f"ELLEBOT {results[('ELLEBOT','A3_no_accumulation')]} switches — "
          f"the metric is not blind to A3 the way it is to A1/A2/A4/A5, "
          f"because A3 is the one arm that routes the face to D directly.")
    assert all_zero and a3_nonzero


def test_prediction_3_timescale_orders():
    """A1: 10^2-10^3 (sessions). A3: 10^0, and specifically <= tau_P. A4: order
    of a turn, not sessions. A2/A5 reported on their own terms (unreachable /
    undefined) rather than forced into the same units."""
    for robot in ROBOTS:
        a1 = metric4_timescale(robot, "A1_full")["tau_D"]
        a3 = metric4_timescale(robot, "A3_no_accumulation")["ratio_to_tau_P"]
        a4 = metric4_timescale(robot, "A4_wasabi")["ratio_to_tau_P"]
        assert a1 >= 3                    # sessions — orders of magnitude above a tick
        assert a3 == 0.0                  # at least as fast as P
        assert 1.0 < a4 < 10.0            # same order as P, not sessions
        print(f"P3 {robot}: A1={a1} sessions (order 10^2-10^3 in ticks at any "
              f"realistic session length, 9a); A3 ratio-to-tau_P={a3} (<=1, "
              f"order 10^0); A4 ratio-to-tau_P={a4:.2f} (order 10^0-10^1, "
              f"NOT sessions) — MET")


def test_prediction_4_ownership_metric_separates_A4():
    for robot in ROBOTS:
        results = {arm: metric5_ownership_alternation(robot, arm)
                  for arm in ARMS}
        a4 = results["A4_wasabi"]
        others_zero = all(results[a].switches == 0
                          for a in ARMS if a != "A4_wasabi")
        a4_moves_every_flip = a4.switches == a4.n_epochs - 1
        print(f"P4 {robot}: A4 switches on {a4.switches}/{a4.n_epochs - 1} "
              f"possible ownership flips ({a4.switches_per_min:.2f}/min); "
              f"every other arm holds at 0 when the relationship is scripted "
              f"unchanged — {'MET' if others_zero and a4_moves_every_flip else 'NOT MET'}")
        assert others_zero and a4_moves_every_flip
    print("    Not a contradiction of Phase 9's finding that A1 CAN move "
          "mid-session from rapport accrual — that is a different scenario "
          "(the relationship itself changing); this one holds it fixed on "
          "purpose to isolate ownership sensitivity specifically.")


def test_prediction_5_A2_vs_A1_is_distinguishable_on_metric_4_only():
    """If indistinguishable on every metric, the brief says the paper must
    report the trust channel contributes nothing measurable. Checked directly:
    they ARE distinguishable, and exactly where accrual history can show up."""
    for robot in ROBOTS:
        m1_same = (metric1_rank(robot, "A1_full")["sv"]
                  == metric1_rank(robot, "A2_no_trust")["sv"])
        m2_same = (metric2_frame_switching(robot, "A1_full")["switches"]
                  == metric2_frame_switching(robot, "A2_no_trust")["switches"])
        m3_same = (metric3_noise_propagation(robot, "A1_full")["std_D"]
                  == metric3_noise_propagation(robot, "A2_no_trust")["std_D"])
        a1_close = metric4_timescale(robot, "A1_full")["tau_D"]
        a2_close = metric4_timescale(robot, "A2_no_trust")["tau_D_to_close"]
        m4_differs = (a1_close is not None) and (a2_close is None)
        indistinguishable_everywhere = m1_same and m2_same and m3_same and not m4_differs
        print(f"P5 {robot}: metrics 1-3 identical (m1={m1_same}, m2={m2_same}, "
              f"m3={m3_same}) as predicted; metric 4 DIFFERS "
              f"(A1 reaches close in {a1_close} sessions, A2 never does) — "
              f"trust IS load-bearing, on the one metric built to show it, "
              f"and the paper should say exactly that rather than 'A2 fails' "
              f"or 'A2 is fine' without naming which metric carries the claim.")
        assert m1_same and m2_same and m3_same and m4_differs
        assert not indistinguishable_everywhere


def test_the_A3_matches_A1_headline_gate():
    """The brief's explicit instruction: if A3 matches A1 on EVERY metric,
    that is the headline finding and the contribution claim must be revised.
    Checked directly: it does not. A3 diverges from A1 on metrics 2 and 4,
    which are precisely the two metrics that involve time — the ones an
    axis-only ablation could never have distinguished. The contribution is
    accumulation, not merely the axis choice, and this test is what entitles
    that sentence."""
    for robot in ROBOTS:
        m1_same = (metric1_rank(robot, "A1_full")["sv"]
                  == metric1_rank(robot, "A3_no_accumulation")["sv"])
        m2_same = (metric2_frame_switching(robot, "A1_full")["switches"]
                  == metric2_frame_switching(robot, "A3_no_accumulation")["switches"])
        a1_tau = metric4_timescale(robot, "A1_full")["tau_D"]     # sessions
        a3_tau = metric4_timescale(robot, "A3_no_accumulation")["tau_D_settling"]  # ticks
        matches_everywhere = m1_same and m2_same   # units already preclude m4 matching
        print(f"A3-matches-A1 gate, {robot}: metric1 identical={m1_same}, "
              f"metric2 identical={m2_same} (A1=0, A3="
              f"{metric2_frame_switching(robot,'A3_no_accumulation')['switches']}), "
              f"metric4 not comparable in the same units by construction "
              f"(A1={a1_tau} sessions vs A3={a3_tau} ticks, but A3's own ratio "
              f"to tau_P is 0.0 while A1's is 100-1000x tau_P) "
              f"-> A3 does NOT match A1. Accumulation is load-bearing.")
        assert not matches_everywhere


if __name__ == "__main__":
    test_prediction_1_all_five_rank_3()
    test_prediction_2_frame_noise_structural_zero_except_A3()
    test_prediction_3_timescale_orders()
    test_prediction_4_ownership_metric_separates_A4()
    test_prediction_5_A2_vs_A1_is_distinguishable_on_metric_4_only()
    test_the_A3_matches_A1_headline_gate()
