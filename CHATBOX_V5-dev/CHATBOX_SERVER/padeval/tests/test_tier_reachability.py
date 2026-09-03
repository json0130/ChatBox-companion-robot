"""
9a gate — every closed form confirmed against an independent simulation.

REGISTERED BEFORE COMPUTING
---------------------------
R1  `visitor` is reachable in exactly 1 turn, unconditionally — the `count > 0`
    branch requires nothing of affect or disclosure.
R2  `known` has two independent paths: `count > 5` (6 turns of anything) and
    `score > 0.45` (rapport alone, since rapport > 0.90 with trust = 0 clears
    it). Which is faster depends on turn pacing and the crossover must be named.
R3  `close` requires `rapport + trust > 1.40` JOINTLY. Because rapport is
    hard-clamped at 1.0 the optimum is a CORNER, not an interior tradeoff, and
    the constraint collapses to `trust > 0.40` exactly — strictly stronger than
    8b's "trust must be nonzero".

The confirmations are not decoration. The `close` closed form initially returned
2 sessions against the simulation's 3, from `2*0.70 - 1.0` evaluating to
0.3999999999999999; see `_strict_ticks`. Without the cross-check that off-by-one
would have shipped as the phase's headline number.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pytest                                                    # noqa: E402

from modules.graph_relationship.kg_bridge import _tier_from_scores  # noqa: E402
from padeval.analysis.tier_reachability import (                 # noqa: E402
    CLOSE_SCORE, KNOWN_SCORE, RAPPORT_CEILING, close_closed_form,
    crossover_seconds_per_turn, format_report, known_via_count_closed_form,
    known_via_rapport_closed_form, p_max, rho, seconds_to_known_at_valence,
    simulate, visitor_closed_form, zero_rapport_valence,
)
from padeval.coding.disclosure import SESSION_TRUST_CAP          # noqa: E402

ROBOTS = ("CHATBOX", "ELLEBOT")


def test_R1_visitor_needs_one_turn_and_nothing_else():
    assert visitor_closed_form()["value"] == 1
    # Confirmation across the least favourable inputs it could face: a cold,
    # affectless, silent turn must still land visitor.
    for robot in ROBOTS:
        sim = simulate(robot, valence=-1.0, depth=0, turns_per_session=1)
        v = sim["first"]["visitor"]
        assert v["turn"] == 1, sim["first"]
        assert v["rapport"] == 0.0 and v["trust"] == 0.0
    print("1. R1 confirmed: visitor at turn 1 even with v=-1 and no disclosure ✓")


def test_R2_known_has_two_paths_and_the_crossover_is_named():
    # Path A — pure volume. Confirmed with a HOSTILE face so rapport cannot help.
    assert known_via_count_closed_form()["value"] == 6
    for robot in ROBOTS:
        sim = simulate(robot, valence=-1.0, depth=0, turns_per_session=1)
        k = sim["first"]["known"]
        assert k["turn"] == 6, sim["first"]
        assert k["rapport"] == 0.0, "rapport contributed; not a pure-count test"

    # Path B — pure warmth. Confirmed with 1-second turns so `count > 5` fires
    # long before the rapport threshold and cannot be the cause... which it
    # would, so instead drive rapport directly against the live tier function.
    for robot in ROBOTS:
        cf = known_via_rapport_closed_form(robot)["value"]
        r = 0.0
        for tick in range(1, 400):
            r = min(RAPPORT_CEILING, r + rho(robot))
            if _tier_from_scores(r, 0.0, 1) == "known":
                assert tick == cf, f"{robot}: closed form {cf}, simulated {tick}"
                break
        else:
            pytest.fail(f"{robot}: never reached known on rapport alone")
        assert 2 * KNOWN_SCORE == 0.90
        print(f"2. R2 {robot}: count path 6 turns | rapport path {cf}s "
              f"| crossover {crossover_seconds_per_turn(robot):.2f} s/turn")

    # The crossover must sit between the two regimes, or it is not a crossover.
    for robot in ROBOTS:
        x = crossover_seconds_per_turn(robot)
        fast = simulate(robot, valence=1.0, depth=0,
                        turn_seconds=max(1, int(x // 2)), turns_per_session=1)
        slow = simulate(robot, valence=1.0, depth=0,
                        turn_seconds=int(x * 4), turns_per_session=1)
        assert fast["first"]["known"]["turn"] == 6, (
            "below the crossover the count path should win")
        assert slow["first"]["known"]["turn"] < 6, (
            "above the crossover the rapport path should win")
    print("   crossover verified: count path wins below it, rapport path above ✓")


def test_R3_close_is_a_corner_solution_not_a_tradeoff():
    """R3. The optimum sits at rapport = 1.0 because rapport is clamped there."""
    trust_needed = 2 * CLOSE_SCORE - RAPPORT_CEILING
    assert abs(trust_needed - 0.40) < 1e-9

    # The corner claim, checked directly: no feasible (rapport, trust) with
    # trust <= 0.40 reaches close, however large rapport is allowed to be.
    for t in (0.0, 0.1, 0.2, 0.3, 0.39, 0.40):
        assert _tier_from_scores(RAPPORT_CEILING, t, 100) != "close", (
            f"close reached with trust={t} — the trust > 0.40 bound is wrong")
    assert _tier_from_scores(RAPPORT_CEILING, 0.41, 100) == "close"
    print("3. R3: trust > 0.40 is exactly necessary and sufficient at "
          "rapport = 1.0 (checked at the boundary, both sides) ✓")

    for robot in ROBOTS:
        cf = close_closed_form(robot)["value"]
        sim = simulate(robot, valence=1.0, depth=3, turns_per_session=4)
        assert "close" in sim["first"], "close never reached in simulation"
        got = sim["first"]["close"]["session"]
        assert got == cf, f"{robot}: closed form {cf} sessions, simulated {got}"
        print(f"   {robot}: closed form {cf} sessions == simulated {got} ✓")


def test_the_close_bound_is_strict_not_loose():
    """The exact-boundary case the float bug hid. Two capped sessions give trust
    exactly 0.40 and score exactly 0.70, which does NOT satisfy `> 0.70`."""
    two_sessions = SESSION_TRUST_CAP * 2
    assert two_sessions == 0.40
    assert (RAPPORT_CEILING + two_sessions) / 2 == 0.70
    assert _tier_from_scores(RAPPORT_CEILING, two_sessions, 100) == "known", (
        "score exactly 0.70 was treated as close — the comparison is `>`")
    print("4. two capped sessions land score EXACTLY 0.70 -> still `known`; "
          "the third session is what crosses ✓")


def test_known_is_reachable_with_no_engagement_at_all():
    """Named as a design property, not a defect — see the 9a report.

    Six turns unlock the same tier as a minute of sustained warmth, and they can
    be actively hostile turns: v=-1 puts felt pleasure below the accrual gate, so
    rapport contributes exactly nothing and `count > 5` does all the work.
    """
    for robot in ROBOTS:
        sim = simulate(robot, valence=-1.0, depth=0, turns_per_session=1)
        k = sim["first"]["known"]
        assert k["turn"] == 6 and k["rapport"] == 0.0 and k["trust"] == 0.0
    print("5. `known` reachable in 6 HOSTILE turns, zero rapport, zero "
          "disclosure — volume alone unlocks half the ladder ✓")


def test_a_neutral_face_is_not_affectively_neutral_to_the_system():
    """Found by a test that assumed the opposite and failed.

    `P = 0.4*P_base + 0.6*v` and the gate is `P > 0.05`, so with both robots
    carrying a positive baseline P, a blank face still builds rapport. The
    no-rapport regime needs a mildly NEGATIVE face, not an absent one.
    """
    for robot in ROBOTS:
        v_star = zero_rapport_valence(robot)
        assert v_star < 0.0, f"{robot}: rapport already stops at v={v_star}"
        assert seconds_to_known_at_valence(robot, 0.0) is not None
        assert seconds_to_known_at_valence(robot, v_star - 0.01) is None
        neutral = seconds_to_known_at_valence(robot, 0.0)
        warm = seconds_to_known_at_valence(robot, 1.0)
        assert neutral > warm
        print(f"6. {robot}: neutral face still reaches `known` in {neutral}s "
              f"(vs {warm}s warm); rapport stops only below v={v_star:+.4f}")


def test_no_path_reaches_close_without_disclosure():
    """The 8b result, re-confirmed as a general statement over the whole space."""
    for robot in ROBOTS:
        for secs in (1, 5, 20, 60):
            sim = simulate(robot, valence=1.0, depth=0, turn_seconds=secs,
                           turns_per_session=50, max_sessions=20)
            assert "close" not in sim["first"], (
                f"{robot}: close reached with zero disclosure at {secs}s/turn")
            assert sim["final_trust"] == 0.0
    print("6. across every pacing tried, close is unreachable without "
          "disclosure — rapport alone caps score at 0.50 ✓")


def test_report_renders():
    print()
    print(format_report())


if __name__ == "__main__":
    test_R1_visitor_needs_one_turn_and_nothing_else()
    test_R2_known_has_two_paths_and_the_crossover_is_named()
    test_R3_close_is_a_corner_solution_not_a_tradeoff()
    test_the_close_bound_is_strict_not_loose()
    test_known_is_reachable_with_no_engagement_at_all()
    test_a_neutral_face_is_not_affectively_neutral_to_the_system()
    test_no_path_reaches_close_without_disclosure()
    test_report_renders()
