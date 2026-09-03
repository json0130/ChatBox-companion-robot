"""
Phase 8b gate — the four registered predictions, checked against the trace.

REGISTERED BEFORE RUNNING (`docs/paper/8b_system_trace.md` carries the results)
------------------------------------------------------------------------------
P1  D moves in discrete steps at SESSION BOUNDARIES ONLY, never within a session.
P2  Mood visibly decays between sessions: a session-old mood value is not carried
    into the next session's first turn at full weight.
P3  Rapport and trust diverge at least once, exactly at the scripted disclosure
    event.
P4  CHATBOX's D clamps at the tier 7b predicts, ELLEBOT's does not clamp anywhere.

P1 is written as the architecture's own claim about itself ("relationship is the
slow, per-session axis"). It is tested as stated, and it FAILS — see
test_P1. The failure is left in place and reported rather than the prediction
being softened after the fact.

No LLM, no camera, no network anywhere in this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pytest                                              # noqa: E402

from modules.affect_bridge import affect                   # noqa: E402
from padeval.analysis.system_trace import (                # noqa: E402
    SCRIPT, format_trace, run_trace, tau_d_sessions,
)

SEED = 7


@pytest.fixture(scope="module")
def traces():
    return {r: run_trace(r, SCRIPT, seed=SEED) for r in ("CHATBOX", "ELLEBOT")}


def test_trace_is_deterministic():
    """Everything downstream is worthless if the seed does not pin the run."""
    a = run_trace("CHATBOX", SCRIPT, seed=SEED)
    b = run_trace("CHATBOX", SCRIPT, seed=SEED)
    assert [r["P"] for r in a] == [r["P"] for r in b]
    assert [r["D"] for r in a] == [r["D"] for r in b]
    c = run_trace("CHATBOX", SCRIPT, seed=SEED + 1)
    assert [r["P"] for r in a] != [r["P"] for r in c], "seed is inert"
    print("1. trace is seed-reproducible and seed-sensitive ✓")


def test_P1_does_D_only_step_at_session_boundaries(traces):
    """P1, as registered. Reported as it comes out."""
    verdict = {}
    for robot, rows in traces.items():
        within = []
        for prev, cur in zip(rows, rows[1:]):
            if cur["turn_in_session"] == 0:
                continue                       # a boundary step is allowed
            if cur["tier"] != prev["tier"]:
                within.append((cur["session"], cur["turn_in_session"],
                               prev["tier"], cur["tier"]))
        verdict[robot] = within
    for robot, within in verdict.items():
        if within:
            print(f"2. P1 FAILS for {robot}: D stepped WITHIN a session at "
                  f"{[(s, t, a, b) for s, t, a, b in within]}")
        else:
            print(f"2. P1 holds for {robot}: no within-session tier change")

    # The registered prediction is FALSE, and this asserts the falsification so a
    # later change that made it true would show up as a failure to re-examine.
    #
    # Cause, structural rather than incidental: `pre_turn` re-derives the tier
    # EVERY TURN (kg_bridge.py:296), and NEITHER kind of threshold is
    # session-scoped, so both can be crossed mid-session:
    #
    #   * count-gated — `count > 0 -> visitor` (kg_bridge.py:112-131) counts
    #     TURNS across all sessions, so it fires on turn 2 of the first session.
    #     This is CHATBOX's within-session step.
    #   * score-gated — `score > 0.45 -> known`. Rapport accrues once per TICK
    #     (~20 per turn here), so a warm run crosses the threshold mid-session.
    #     This is ELLEBOT's step at s1t1, where rapport hit 0.96 with trust
    #     still 0.
    #
    # It would have been easy to report only the first: CHATBOX alone shows just
    # the count-gated case, and that reads like a tidy bootstrap-only exception.
    # ELLEBOT shows the score-gated rung doing it too, which is the finding that
    # actually matters — nothing in the tier derivation makes D per-session.
    assert verdict["CHATBOX"] and verdict["ELLEBOT"], (
        "P1 now holds where it previously failed — re-derive tau_D before "
        "trusting any timescale claim"
    )
    assert any(a == "unknown" and b == "visitor"
               for _s, _t, a, b in verdict["CHATBOX"]), "count-gated case gone"
    assert any(a == "visitor" and b == "known"
               for _s, _t, a, b in verdict["ELLEBOT"]), "score-gated case gone"


def test_P2_mood_does_not_cross_a_session_boundary(traces):
    """P2. Checked as a difference against the pre-f5bfac9 behaviour."""
    for robot, rows in traces.items():
        gated = [r for r in rows if r["mood_gated_out"]]
        assert gated, "no mood was ever gated — the fix is not in this path"
        assert all(r["first_turn_of_session"] for r in gated), (
            "mood was gated mid-session, which the session-scoped rule forbids")
        assert all(r["session"] > 0 for r in gated)
        # Every later session's first turn must be gated, not just some.
        firsts = [r for r in rows if r["first_turn_of_session"] and r["session"] > 0]
        assert len(gated) == len(firsts)
        biggest = max(abs(r["gate_effect"]) for r in gated)
        print(f"3. P2 {robot}: {len(gated)}/{len(firsts)} session-opening turns "
              f"gated; largest shift vs ungated = {biggest:.4f} valence")
        assert biggest > 0.0, "the gate changed nothing — no stale mood to block"


def test_P2b_without_the_gate_a_stale_mood_reaches_turn_one(traces):
    """The counterfactual. Without it, P2 shows a property, not a fix."""
    gated = run_trace("CHATBOX", SCRIPT, seed=SEED, mood_gate=True)
    ungated = run_trace("CHATBOX", SCRIPT, seed=SEED, mood_gate=False)
    diffs = [(g["session"], g["blended_v"] - u["blended_v"])
             for g, u in zip(gated, ungated)
             if g["first_turn_of_session"] and g["session"] > 0]
    print("4. P2b gated vs ungated, at each session-opening turn: "
          + ", ".join(f"s{s}: {d:+.4f}" for s, d in diffs))
    assert all(abs(d) > 1e-9 for _s, d in diffs), (
        "gated and ungated runs are identical — the counterfactual is not wired")


def test_P3_rapport_and_trust_diverge_at_the_scripted_disclosure(traces):
    for robot, rows in traces.items():
        assert rows[-1]["rapport"] != rows[-1]["trust"], "no divergence at all"
        first_trust = next(r for r in rows if r["d_trust"] > 0)
        first_disc = next(r for r in rows if r["disclosed"])
        assert first_trust["global_turn"] == first_disc["global_turn"]
        assert first_disc["session"] == 2 and first_disc["turn_in_session"] == 0, (
            f"first disclosure moved off the scripted turn: "
            f"s{first_disc['session']}t{first_disc['turn_in_session']}")
        # Everything before it must have moved rapport and left trust flat.
        before = rows[:first_disc["global_turn"]]
        assert all(r["d_trust"] == 0.0 for r in before)
        assert any(r["d_rapport"] > 0.0 for r in before)
        print(f"5. P3 {robot}: trust first moves at global turn "
              f"{first_trust['global_turn']} (s2t0, the scripted disclosure); "
              f"end rapport {rows[-1]['rapport']:.4f} vs trust "
              f"{rows[-1]['trust']:.4f}")


def test_P4_chatbox_clamps_at_unknown_and_ellebot_never_does(traces):
    """P4, with the tier named in advance from 7b's bound, not read off the run.

    7b: admissible iff |D_baseline| <= 1 - m. CHATBOX D_baseline = -0.643 and the
    `unknown` offset is -0.40, so -1.043 saturates; every other rung is inside.
    ELLEBOT's +0.421 leaves headroom at all four.
    """
    predicted = {}
    for robot in ("CHATBOX", "ELLEBOT"):
        b = affect.to_pad(affect.ROBOTS[robot]["ocean"])
        predicted[robot] = {t for t, off in affect.TIER_OFFSETS.items()
                            if abs(b["D"] + off[2]) > 1.0}
    assert predicted["CHATBOX"] == {"unknown"}, predicted["CHATBOX"]
    assert predicted["ELLEBOT"] == set(), predicted["ELLEBOT"]

    for robot, rows in traces.items():
        observed = {r["tier"] for r in rows if r["clamped"]}
        assert observed == predicted[robot], (
            f"{robot}: clamped at {observed}, 7b predicted {predicted[robot]}")
        print(f"6. P4 {robot}: clamped at {observed or '{}'} — matches the 7b "
              f"bound ({predicted[robot] or 'no clamping'})")

    # And the clamp must actually cost something observable, or it is bookkeeping.
    cb = [r for r in traces["CHATBOX"] if r["clamped"]]
    assert cb and any(abs(r["raw_D"] - r["D"]) > 1e-9 for r in cb), (
        "clamp flagged but D was never actually truncated")
    lost = max(abs(r["raw_D"] - r["D"]) for r in cb)
    print(f"      CHATBOX loses {lost:.3f} of commanded D at `unknown`")


def test_tau_D_recomputed_under_the_8a_mechanism():
    """8a changed the accrual, so 6a's tau_D cannot be carried over unchanged."""
    res = tau_d_sessions("CHATBOX", turns_per_session=2, depth=3, seed=SEED)
    print(f"7. tau_D under 8a: close reached at session "
          f"{res['sessions_to_close']} "
          f"(2 turns/session, deepest disclosure every turn); "
          f"first seen at each tier {res['first_session_at_tier']}")
    assert res["sessions_to_close"] is not None, "close never reached"
    assert res["sessions_to_close"] >= 3, (
        f"close now reachable in {res['sessions_to_close']} sessions — under 3, "
        f"the relationship is no longer the slow axis and 6a's separation claim "
        f"needs restating, not just recomputing")

    # The median is a FAVOURABLE choice, and reporting only it would hide a real
    # regression. 6a's 3-session floor was hard: trust moved once per session,
    # capped at +/-0.2 by the extractor, so the floor held "independent of
    # session length or how warm the interaction is". 8a made trust accrue PER
    # TURN with no per-session cap, and that independence is gone — at the IQR
    # upper bound the floor breaks.
    across = {n: tau_d_sessions("CHATBOX", turns_per_session=n, depth=3,
                                seed=SEED)["sessions_to_close"]
              for n in (1, 2, 4)}
    print(f"      sessions-to-close by session length (6a IQR [1,4]): {across}")
    assert across[4] < 3, (
        "the floor appears restored at 4 turns/session — if a per-session trust "
        "cap was added, update this test and 6a's floor claim together")
    print("      -> tau_D is now session-length DEPENDENT; 6a's hard floor is "
          "gone. Restoring it needs a per-session trust cap (see 8b report).")


def test_the_style_vector_moves_with_the_tier(traces):
    """The trace has to reach the EFFECTORS or it is not end-to-end."""
    rows = traces["CHATBOX"]
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], set()).add(round(r["style"]["amplitude"], 6))
    assert len(by_tier) >= 3, f"only {len(by_tier)} tiers visited"
    for tier, amps in by_tier.items():
        assert len(amps) == 1, f"amplitude varied within tier {tier}: {amps}"
    print("8. amplitude is constant within a tier and differs across tiers: "
          + ", ".join(f"{t}={next(iter(a)):.3f}" for t, a in by_tier.items()))


def test_print_both_traces(traces):
    for robot, rows in traces.items():
        print()
        print(format_trace(rows, robot))


if __name__ == "__main__":
    t = {r: run_trace(r, SCRIPT, seed=SEED) for r in ("CHATBOX", "ELLEBOT")}
    test_trace_is_deterministic()
    test_P1_does_D_only_step_at_session_boundaries(t)
    test_P2_mood_does_not_cross_a_session_boundary(t)
    test_P2b_without_the_gate_a_stale_mood_reaches_turn_one(t)
    test_P3_rapport_and_trust_diverge_at_the_scripted_disclosure(t)
    test_P4_chatbox_clamps_at_unknown_and_ellebot_never_does(t)
    test_tau_D_recomputed_under_the_8a_mechanism()
    test_the_style_vector_moves_with_the_tier(t)
    test_print_both_traces(t)
