"""
Phase 0 gate (a): the permutation machinery is a strict SUPERSET of the shipped
pipeline, not a reimplementation of it.

If `compose(IDENTITY, ...)` ever drifts from `affect.feel_with_relationship`,
every E3 number becomes a comparison between the real system and a slightly
different one, and nothing in the paper would be safe. So it is asserted bit for
bit over the whole reachable grid plus 1000 random points.

This mirrors the guarantee affect.pipeline already makes about tier="known"
(affect.py:365-369) — same idea, same style, one layer out.

No camera, no LLM, no hardware.

    python3 -m pytest padeval/tests/test_axes.py -q
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect                      # noqa: E402
from padeval.axes import (                                    # noqa: E402
    ASSIGNMENTS, AXES, COLLAPSE_D, IDENTITY, PERM_AROUSAL_D, PERM_EMOTION_D,
    AxisAssignment, compose, tier_deltas,
)

ROBOTS = ("CHATBOX", "ELLEBOT")
EMOTIONS = ("happy", "sad", "angry", "neutral", "surprise", "fear", "disgust")


def _baseline(robot):
    return affect.to_pad(affect.ROBOTS[robot]["ocean"])


def test_identity_is_bit_identical_on_the_reachable_grid():
    """Every (robot, tier, emotion) cell the deployed system can actually reach."""
    n = 0
    for robot in ROBOTS:
        base = _baseline(robot)
        for tier in affect.TIERS:
            for emo in EMOTIONS:
                v, a = affect.category_to_va(emo)
                want = affect.feel_with_relationship(base, v, a, tier)
                got, _flags = compose(IDENTITY, base, v, a, tier)
                for axis in AXES:
                    assert got[axis] == want[axis], (
                        f"{robot}/{tier}/{emo} axis {axis}: "
                        f"{got[axis]!r} != {want[axis]!r}"
                    )
                n += 1
    print(f"1. compose(IDENTITY) bit-identical over {n} reachable cells ✓")


def test_identity_is_bit_identical_on_random_points():
    """Fuzz the continuous inputs — the grid only covers 7 corner emotions, and
    the live camera now regresses V/A continuously (commit e102d75), so the
    reachable input space is the whole square."""
    rng = random.Random(20260901)
    for _ in range(1000):
        robot = rng.choice(ROBOTS)
        base = _baseline(robot)
        v = rng.uniform(-1.0, 1.0)
        a = rng.uniform(-1.0, 1.0)
        tier = rng.choice(affect.TIERS)
        emp = rng.choice([affect.EMPATHY, 0.0, 0.25, 1.0])
        want = affect.feel_with_relationship(base, v, a, tier, emp)
        got, _flags = compose(IDENTITY, base, v, a, tier, emp)
        for axis in AXES:
            assert got[axis] == want[axis], (
                f"random point robot={robot} v={v!r} a={a!r} tier={tier} "
                f"empathy={emp!r} axis={axis}: {got[axis]!r} != {want[axis]!r}"
            )
    print("2. compose(IDENTITY) bit-identical over 1000 random points ✓")


def test_identity_tier_routing_is_the_identity_map():
    for tier in affect.TIERS:
        d_p, d_ar, d_d = affect.tier_offset(tier)
        assert tier_deltas(IDENTITY, tier) == {"P": d_p, "Ar": d_ar, "D": d_d}
    print("3. tier_deltas(IDENTITY) is the identity map ✓")


def test_permutation_actually_moves_the_signal():
    """A permutation that changed nothing would make E3 vacuous."""
    base = _baseline("CHATBOX")
    v, a = affect.category_to_va("happy")
    ident, _ = compose(IDENTITY, base, v, a, "known")
    perm, _ = compose(PERM_EMOTION_D, base, v, a, "known")
    assert ident != perm, "perm_emotion_D produced the identity coordinate"
    # Under perm_emotion_D the face owns D, so D must respond to the face.
    d_happy, _ = compose(PERM_EMOTION_D, base, *affect.category_to_va("happy"), "known")
    d_sad, _ = compose(PERM_EMOTION_D, base, *affect.category_to_va("sad"), "known")
    assert d_happy["D"] != d_sad["D"], "face does not move D under perm_emotion_D"
    # Under IDENTITY it must NOT — that invariance is what E3 measures against.
    i_happy, _ = compose(IDENTITY, base, *affect.category_to_va("happy"), "known")
    i_sad, _ = compose(IDENTITY, base, *affect.category_to_va("sad"), "known")
    assert i_happy["D"] == i_sad["D"], "face moved D under IDENTITY"
    print("4. permutation moves D with the face; identity does not ✓")


def test_collapse_puts_every_source_on_one_axis():
    base = _baseline("ELLEBOT")
    for tier in affect.TIERS:
        for emo in EMOTIONS:
            v, a = affect.category_to_va(emo)
            felt, _ = compose(COLLAPSE_D, base, v, a, tier)
            # The two axes nothing writes must stay at their baseline.
            assert felt["P"] == base["P"], (tier, emo, felt["P"], base["P"])
            assert felt["Ar"] == base["Ar"], (tier, emo)
    print("5. collapse_D leaves the unwritten axes at baseline ✓")


def test_clamp_flags_fire_where_the_audit_said_they_would():
    """CHATBOX at `unknown` wants D = -0.643 + -0.40 = -1.043."""
    base = _baseline("CHATBOX")
    felt, flags = compose(IDENTITY, base, 0.0, 0.0, "unknown")
    assert flags["D"] is True, "expected a D clamp for CHATBOX/unknown"
    assert felt["D"] == -1.0
    felt, flags = compose(IDENTITY, base, 0.0, 0.0, "known")
    assert not any(flags.values()), f"unexpected clamp at CHATBOX/known: {flags}"
    print("6. clamp flags fire at CHATBOX/unknown and nowhere at known ✓")


def test_the_close_tier_arousal_leak_is_still_present_and_routed():
    """R11 / the measured defect. This test is a TRIPWIRE, not an endorsement:
    the leak is kept deliberately so E2 can show its leakage metric resolves it.
    If someone zeroes the offset, this fails loudly and the E2 sensitivity
    result has to be regenerated rather than silently becoming a null."""
    assert affect.TIER_OFFSETS["close"][1] == 0.10, (
        "the close-tier arousal leak has changed; E2's sensitivity pair and the "
        "plan's decision 1 both assume +0.10"
    )
    # ...and it must land on whatever axis arousal owns.
    assert tier_deltas(IDENTITY, "close")["Ar"] == 0.10
    assert tier_deltas(PERM_AROUSAL_D, "close")["D"] == 0.10
    print("7. the close-tier +0.10 arousal leak is present and follows arousal ✓")


def test_assignment_rejects_a_bad_axis():
    try:
        AxisAssignment("bogus", valence_axis="X")
    except ValueError:
        print("8. AxisAssignment rejects an unknown axis name ✓")
    else:
        raise AssertionError("expected ValueError for axis 'X'")


def test_registry_names_match_their_keys():
    for name, a in ASSIGNMENTS.items():
        assert a.name == name
    print(f"9. ASSIGNMENTS registry is consistent ({len(ASSIGNMENTS)} entries) ✓")


if __name__ == "__main__":
    test_identity_is_bit_identical_on_the_reachable_grid()
    test_identity_is_bit_identical_on_random_points()
    test_identity_tier_routing_is_the_identity_map()
    test_permutation_actually_moves_the_signal()
    test_collapse_puts_every_source_on_one_axis()
    test_clamp_flags_fire_where_the_audit_said_they_would()
    test_the_close_tier_arousal_leak_is_still_present_and_routed()
    test_assignment_rejects_a_bad_axis()
    test_registry_names_match_their_keys()
    print("\npadeval.axes is a strict superset of the shipped pipeline.")
