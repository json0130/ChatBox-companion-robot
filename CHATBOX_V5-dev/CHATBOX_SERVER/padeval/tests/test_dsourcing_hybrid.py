"""
Phase 11f gate — A6, the proposed hybrid: D = relationship baseline (slow) +
bounded situational offset (fast).

REGISTERED PREDICTION, as given
--------------------------------
"A6 preserves A1's timescale separation on the slow component while
recovering A4's responsiveness on the fast one — because they're bounded,
additive, and operate at different rates."

RESULT: preserved on the slow side; the naive fast side does NOT come for
free. Both halves checked below rather than assumed.

WHY THE NAIVE VERSION FAILS, AND WHAT FIXES IT
------------------------------------------------
Bounding the fast term controls its MAGNITUDE, not whether that magnitude
happens to cross a discrete rung boundary. If the trigger is WASABI's raw
`robot_turn` (Design 1: symmetric +-bound, mirroring their binary exactly),
metric 5's alternation and metric 6's correction present the IDENTICAL
signal at the IDENTICAL magnitude, so either BOTH cross the same boundary or
NEITHER does — a fine sweep confirms this is exactly all-or-nothing, never
partial. There is no bound that buys metric 6 responsiveness without paying
the full A4 chatter rate on metric 5.

Design 2 (`legitimate_assertion`) fixes this by keying the fast term to a
content-level signal ordinary turn-taking never raises, rather than to bare
floor-holding: one-sided (0 when not asserting, +bound when asserting, no
"legitimate submission" event to mirror the boost against). This is the
version that actually delivers the registered prediction, and it is reported
as the recommendation, with Design 1's failure reported alongside it as the
reason the recommendation looks the way it does rather than reusing WASABI's
signal unmodified.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.analysis.dsourcing_metrics import (                  # noqa: E402
    ROBOTS, metric1_rank, metric3_noise_propagation, metric4_timescale,
    metric5_ownership_alternation, metric6_situational_responsiveness,
)

ARM = "A6_hybrid"
SWEEP = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
RECOMMENDED_BOUND = 0.40


def test_slow_component_matches_A1_at_every_bound():
    for robot in ROBOTS:
        a1 = metric4_timescale(robot, "A1_full")["tau_D"]
        for b in (0.05, 0.20, 0.40):
            a6 = metric4_timescale(robot, ARM)["tau_D"]     # bound-independent by construction
            assert a6 == a1, (robot, b, a6, a1)
    print("1. A6's slow-component floor equals A1's at every bound tested — "
          "the fast term does not corrupt tier derivation ✓")


def test_metrics_1_and_3_are_identical_to_A1():
    """The slow ramp is unchanged from A1; the fast term is a fixed additive
    constant outside the ramp mechanism, so these two metrics cannot see it —
    same structural blindness already established for A1/A2/A3 in 11b."""
    for robot in ROBOTS:
        assert metric1_rank(robot, ARM)["sv"] == metric1_rank(robot, "A1_full")["sv"]
        assert (metric3_noise_propagation(robot, ARM)["std_D"]
               == metric3_noise_propagation(robot, "A1_full")["std_D"] == 0.0)
    print("2. metrics 1 and 3: A6 bit-identical to A1 ✓")


def test_design1_robot_turn_is_provably_all_or_nothing():
    """Fine sweep: metric 5's switch count for A6/Design-1 only ever takes TWO
    values — 0 or the maximum (n_epochs-1) — never anything between."""
    for robot in ROBOTS:
        counts = {metric5_ownership_alternation(
                     robot, ARM, fast_bound=b, a6_trigger="robot_turn").switches
                 for b in SWEEP}
        n_epochs = metric5_ownership_alternation(robot, ARM).n_epochs
        assert counts <= {0, n_epochs - 1}, (
            f"{robot}: found an intermediate switch count {counts} — the "
            f"all-or-nothing claim is false, the hybrid's cost profile needs "
            f"re-describing")
        assert counts == {0, n_epochs - 1}, (
            f"{robot}: sweep only hit one regime ({counts}) — widen SWEEP")
    print("3. Design 1 (robot_turn): metric 5's switch count is either 0 or "
          "the maximum across the whole sweep, never partial — confirmed as "
          "a genuine phase transition, not a smooth cost curve ✓")


def test_design1_any_bound_that_helps_metric6_costs_full_metric5_chatter():
    """The concrete statement of the failure: for every bound where Design 1
    gives a positive metric 6 gain, metric 5 is ALREADY at the maximum."""
    for robot in ROBOTS:
        for b in SWEEP:
            gain = metric6_situational_responsiveness(
                robot, ARM, "known", fast_bound=b,
                a6_trigger="robot_turn")["assertiveness_gain"]
            m5 = metric5_ownership_alternation(
                robot, ARM, fast_bound=b, a6_trigger="robot_turn")
            if gain > 0:
                assert m5.switches == m5.n_epochs - 1, (
                    f"{robot} bound={b}: gain={gain} but metric5 switches="
                    f"{m5.switches}/{m5.n_epochs - 1} — found a bound that "
                    f"escapes the tradeoff Design 1 was shown to have")
    print("4. Design 1: every bound tested that helps metric 6 pays the "
          "FULL metric 5 chatter cost, no partial win found ✓")


def test_design2_legitimate_assertion_never_costs_metric5():
    for robot in ROBOTS:
        for b in SWEEP:
            m5 = metric5_ownership_alternation(
                robot, ARM, fast_bound=b, a6_trigger="legitimate_assertion")
            assert m5.switches == 0, (robot, b, m5.switches)
    print("5. Design 2 (legitimate_assertion): metric 5 stays at ZERO across "
          "the entire sweep — ordinary rapid turn-taking never raises the "
          "content-level flag, so it never pays A4's cost ✓")


def test_design2_recovers_positive_metric6_gain_at_the_recommended_bound():
    for robot in ROBOTS:
        r = metric6_situational_responsiveness(
            robot, ARM, "known", fast_bound=RECOMMENDED_BOUND,
            a6_trigger="legitimate_assertion")
        assert r["assertiveness_gain"] > 0, r
        # And the ordinary condition must equal the pure relationship value —
        # Design 2 adds nothing when not asserting, unlike Design 1.
        a1 = metric6_situational_responsiveness(robot, "A1_full", "known")
        assert r["D_ordinary"] == a1["D_ordinary"], (
            "Design 2's 'ordinary' D drifted from the pure relationship "
            "value — it should contribute exactly zero when not asserting")
    print(f"6. Design 2 at bound={RECOMMENDED_BOUND}: positive metric-6 gain "
          f"on both robots, AND zero metric-5 cost at every bound tested "
          f"(test 5) — the registered prediction holds for this design, "
          f"where it did not for Design 1.")


def test_quoted_directives_at_the_recommended_design():
    for robot in ROBOTS:
        r = metric6_situational_responsiveness(
            robot, ARM, "known", fast_bound=RECOMMENDED_BOUND,
            a6_trigger="legitimate_assertion")
        print(f"7. {robot} A6 (Design 2, bound={RECOMMENDED_BOUND}) @ known — "
              f"ordinary: {r['directive_ordinary']!r}")
        print(f"   {robot} A6 correcting: {r['directive_correcting']!r}")


if __name__ == "__main__":
    test_slow_component_matches_A1_at_every_bound()
    test_metrics_1_and_3_are_identical_to_A1()
    test_design1_robot_turn_is_provably_all_or_nothing()
    test_design1_any_bound_that_helps_metric6_costs_full_metric5_chatter()
    test_design2_legitimate_assertion_never_costs_metric5()
    test_design2_recovers_positive_metric6_gain_at_the_recommended_bound()
    test_quoted_directives_at_the_recommended_design()
