"""
Phase 11e gate — metric 6, situational responsiveness.

REGISTERED BEFORE COMPUTING
---------------------------
The scenario: the robot is mid-explanation, correcting a factual error, and
legitimately holds the floor for that one turn.

R1  A1, A2, A3, A5: assertiveness_gain == 0 at every tier, both robots — none
    of their D formulas read anything about who currently holds the floor, so
    the directive for a legitimate correction is BIT-IDENTICAL to the
    directive for an ordinary reply in the same relationship state.
R2  A4: assertiveness_gain > 0 at every tier, both robots — this is the one
    scenario WASABI's rule is actually built for.

This is the flip side of metric 5's finding, not a contradiction of it: metric
5 showed A4 chattering on ownership changes that carry no relational meaning;
this shows the SAME mechanism correctly serving a moment that does carry
meaning. Both are true, and the paper says both.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from padeval.analysis.dsourcing import ARMS                      # noqa: E402
from padeval.analysis.dsourcing_metrics import (                  # noqa: E402
    ROBOTS, metric6_situational_responsiveness,
)

TIERS = ("unknown", "visitor", "known", "close")
BLIND_ARMS = ("A1_full", "A2_no_trust", "A3_no_accumulation", "A5_no_relationship")


def test_R1_the_four_relationship_only_arms_cannot_feel_the_correction():
    for robot in ROBOTS:
        for arm in BLIND_ARMS:
            for tier in TIERS:
                r = metric6_situational_responsiveness(robot, arm, tier)
                assert r["assertiveness_gain"] == 0, r
                assert r["D_ordinary"] == r["D_correcting"], r
                assert r["directive_ordinary"] == r["directive_correcting"], r
    print(f"1. {len(BLIND_ARMS)} arms x {len(TIERS)} tiers x {len(ROBOTS)} "
          f"robots: bit-identical D and directive text, ordinary vs "
          f"correcting — zero situational responsiveness, confirmed not "
          f"assumed ✓")


def test_R2_A4_asserts_at_every_tier_on_both_robots():
    for robot in ROBOTS:
        for tier in TIERS:
            r = metric6_situational_responsiveness(robot, "A4_wasabi", tier)
            assert r["assertiveness_gain"] > 0, r
            assert r["rung_correcting"] < r["rung_ordinary"], (
                "correcting must command a MORE assertive (lower-index) rung")
    print("2. A4: positive assertiveness gain at all 4 tiers, both robots — "
          "the one scenario its rule is built for ✓")


def test_the_gain_is_tier_invariant_for_every_arm():
    """A1/A2/A3/A5's gain is zero regardless of tier (expected: ownership
    enters none of their formulas). A4's gain is CONSTANT across tier too —
    less obvious, and worth confirming rather than assuming: the +-M swing is
    large enough to cross the same NUMBER of rung boundaries regardless of
    where the baseline+tier offset happens to sit."""
    for robot in ROBOTS:
        for arm in ARMS:
            gains = {metric6_situational_responsiveness(robot, arm, t)
                    ["assertiveness_gain"] for t in TIERS}
            assert len(gains) == 1, (
                f"{robot}/{arm}: gain varies by tier: "
                f"{[(t, metric6_situational_responsiveness(robot, arm, t)['assertiveness_gain']) for t in TIERS]}")
    print("3. every arm's gain is CONSTANT across all four tiers — the "
          "finding does not depend on which relationship state was picked "
          "as the demonstration point ✓")


def test_quoted_directives_for_the_headline_scenario():
    """The concrete pair the paper quotes: tier='known', both robots."""
    for robot in ROBOTS:
        a1 = metric6_situational_responsiveness(robot, "A1_full", "known")
        a4 = metric6_situational_responsiveness(robot, "A4_wasabi", "known")
        print(f"4. {robot} @ known — A1 (ordinary): {a1['directive_ordinary']!r}")
        print(f"   {robot} @ known — A1 (correcting): {a1['directive_correcting']!r}  "
              f"[UNCHANGED]")
        print(f"   {robot} @ known — A4 (ordinary): {a4['directive_ordinary']!r}")
        print(f"   {robot} @ known — A4 (correcting): {a4['directive_correcting']!r}")
        assert a1["directive_ordinary"] == a1["directive_correcting"]
        assert a4["directive_ordinary"] != a4["directive_correcting"]


if __name__ == "__main__":
    test_R1_the_four_relationship_only_arms_cannot_feel_the_correction()
    test_R2_A4_asserts_at_every_tier_on_both_robots()
    test_the_gain_is_tier_invariant_for_every_arm()
    test_quoted_directives_for_the_headline_scenario()
