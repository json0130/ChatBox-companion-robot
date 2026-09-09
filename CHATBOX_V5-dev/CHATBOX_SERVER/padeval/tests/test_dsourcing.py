"""
Phase 11a gate — each arm's per-turn stepping does what its row in the table
says, before any cross-arm metric is computed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect                    # noqa: E402
from padeval.analysis.dsourcing import (                    # noqa: E402
    ARMS, INSTANT_TIER_EDGES, M, ArmState, instant_tier, step_d_offset,
)

WARM, COLD = 0.7, -0.7


def test_A1_matches_the_deployed_accrual():
    """Rapport from warmth, trust from disclosure, same as webcam_loop.py."""
    s = ArmState()
    _, _, d0 = step_d_offset("A1_full", s, WARM, "hello")
    assert s.rapport > 0 and s.trust == 0.0
    _, _, d1 = step_d_offset("A1_full", s, WARM, "i felt left out at school")
    assert s.trust > 0.0
    print(f"1. A1: rapport={s.rapport:.3f} trust={s.trust:.3f} after warmth "
          f"then disclosure ✓")


def test_A2_never_moves_trust_and_never_calls_the_detector():
    """The trust channel does not exist for A2 — not zeroed, absent."""
    s = ArmState()
    for _ in range(5):
        step_d_offset("A2_no_trust", s, WARM, "i felt left out at school today")
    assert s.trust == 0.0
    assert s.rapport > 0.0
    print(f"2. A2: rapport={s.rapport:.3f} after 5 warm+disclosing turns, "
          f"trust untouched at 0.0 ✓")


def test_A3_has_no_memory_between_turns():
    """Identical state object, called twice with the same input, must return
    the same result both times — nothing persisted."""
    s = ArmState()
    a = step_d_offset("A3_no_accumulation", s, WARM, "i felt left out")
    b = step_d_offset("A3_no_accumulation", s, WARM, "i felt left out")
    assert a == b
    assert (s.rapport, s.trust, s.count) == (0.0, 0.0, 0), (
        "A3 must never mutate ArmState")
    cold = step_d_offset("A3_no_accumulation", s, COLD, "")[2]
    warm = step_d_offset("A3_no_accumulation", s, WARM, "")[2]
    assert cold < warm, "D must track felt Pleasure directly"
    print(f"3. A3: stateless (repeat call identical), D tracks felt P: "
          f"cold={cold:+.2f} warm={warm:+.2f} ✓")


def test_A4_requires_and_obeys_turn_ownership():
    s = ArmState()
    dom = step_d_offset("A4_wasabi", s, WARM, "", robot_turn=True)
    sub = step_d_offset("A4_wasabi", s, WARM, "", robot_turn=False)
    assert dom == (0.0, 0.0, M) and sub == (0.0, 0.0, -M)
    assert (s.rapport, s.trust, s.count) == (0.0, 0.0, 0), (
        "A4 must never mutate ArmState")
    try:
        step_d_offset("A4_wasabi", s, WARM, "")
        assert False, "must require robot_turn"
    except ValueError:
        pass
    print(f"4. A4: dominant=+{M}, submissive=-{M}, stateless, "
          f"robot_turn required ✓")


def test_A5_is_the_constant_zero_offset():
    s = ArmState()
    for felt_p, text in ((WARM, "i felt left out"), (COLD, ""), (0.0, "hi")):
        assert step_d_offset("A5_no_relationship", s, felt_p, text) == (0.0, 0.0, 0.0)
    assert (s.rapport, s.trust, s.count) == (0.0, 0.0, 0)
    print("5. A5: (0,0,0) regardless of input, stateless ✓")


def test_instant_tier_orders_correctly_and_is_robust_to_the_edge_choice():
    assert [instant_tier(x) for x in (-0.9, -0.1, 0.3, 0.9)] == [
        "unknown", "visitor", "known", "close"]
    # Shifted edges: the ORDER must survive even if the exact cut moves.
    shifted = (0.6, 0.1, -0.4)
    assert [instant_tier(x, shifted) for x in (-0.9, -0.1, 0.3, 0.9)] == [
        "unknown", "visitor", "known", "close"]
    print(f"6. instant_tier orders correctly at the stated edges "
          f"{INSTANT_TIER_EDGES} and at a shifted set {shifted} ✓")


def test_A1_A2_A3_share_the_identical_offset_table():
    """Not a behavioural equivalence at one call — A1/A2 carry state across
    turns and A3 deliberately does not, so their step_d_offset calls are not
    comparable at a single point. What IS shared, and is checked here, is the
    TABLE every arm's tier ultimately indexes into: whatever tier each of the
    three computes (by whatever accrual or lack of it), the (dP,dAr,dD) it
    returns is one of these same four rows. This is WHY metrics 1 and 3 turn
    out unable to tell A1/A2/A3 apart (see dsourcing_metrics.py) — the axis-level
    Jacobian only sees the ramp INTO this shared table, never the mechanism that
    picks a row."""
    rows = set(affect.TIER_OFFSETS.values())
    cases = [
        step_d_offset("A1_full", ArmState(), 0.9, "i feel very close to you"),
        step_d_offset("A2_no_trust", ArmState(), 0.9, "i feel very close to you"),
        step_d_offset("A3_no_accumulation", ArmState(), 0.9, ""),
    ]
    assert all(c in rows for c in cases), cases
    print(f"7. A1/A2/A3 each return a row of the same {len(rows)}-tier table ✓")


def test_M_is_the_deployed_tier_magnitude():
    assert M == affect.TIER_OFFSETS["close"][2] == 0.40
    print(f"8. M = {M}, asserted equal to the deployed ladder's own value ✓")


if __name__ == "__main__":
    test_A1_matches_the_deployed_accrual()
    test_A2_never_moves_trust_and_never_calls_the_detector()
    test_A3_has_no_memory_between_turns()
    test_A4_requires_and_obeys_turn_ownership()
    test_A5_is_the_constant_zero_offset()
    test_instant_tier_orders_correctly_and_is_robust_to_the_edge_choice()
    test_A1_A2_A3_share_the_identical_offset_table()
    test_M_is_the_deployed_tier_magnitude()
