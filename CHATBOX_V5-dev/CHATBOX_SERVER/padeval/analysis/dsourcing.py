"""
Phase 11 — five ways to source Dominance, compared under one shared axis
assignment.

WHY THIS PHASE, AND WHY IT IS NOT AN AXIS-PERMUTATION SWEEP
-------------------------------------------------------------
E3/7a compared axis ROUTINGS: which source writes which axis. Every arm here
uses the SAME routing — face writes P and Ar, "the relationship" writes D — so
this phase is orthogonal to that one. What varies is what generates the D
VALUE the routing carries.

WASABI (Becker-Asano & Wachsmuth, AAMAS 2010) supplies the external comparison
ICRA reviewers will ask for. Verified from the primary source: WASABI routes at
the axis level exactly as we do — two valences into P/Ar continuously from
embodied dynamics, a level of Dominance "derived from the situational context".
**The difference is timescale, not assignment.** In their Skip-Bo scenario,
"MAX feels dominant whenever it is his turn and non-dominant, i.e. submissive,
otherwise" — turn-taking state with no memory across episodes. Ours accumulates
across sessions with a provable floor (9a: tau_D = 3 sessions, gated at
trust > 0.40).

This module implements ONE RULE from WASABI — its D-sourcing rule — as a
comparison arm inside our own pipeline, holding everything else fixed. It does
not reimplement WASABI's architecture, and no claim here should read as if it
does.

THE FIVE ARMS
-------------
    A1  full (deployed)     D = tier(accumulated rapport + trust)
    A2  no trust channel    D = tier(accumulated rapport only)
    A3  no accumulation     D = tier(THIS turn's warmth only, no memory)
    A4  WASABI-style        D = +-M by turn ownership, no memory
    A5  no relationship     D = persona baseline, constant

All five keep `padeval.axes.IDENTITY`: face -> P, face -> Ar, "the relationship
signal, however it's made" -> D. `affect.py`, `WEIGHTS`, `TIER_OFFSETS` and the
existing E2/E3 code paths are not modified anywhere in this module; every arm is
passed to them as an explicit `ramp` or `offsets` argument, the same seam E2's
leak/no-leak pair and 7a's enumeration already use.

M = 0.40 is `affect.TIER_OFFSETS["close"][2]`, the deployed tier ladder's
principal displacement — asserted below rather than restated, so a future change
to the ladder cannot silently desynchronise this module's normalisation from the
one it claims to match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from modules.affect_bridge import affect
from modules.graph_relationship.kg_bridge import _tier_from_scores
from padeval.coding.disclosure import SessionAccrual
from padeval.analysis.tier_reachability import (
    PLEASURE_FLOOR, RAPPORT_CEILING, RAPPORT_GAIN,
)

ARMS: Tuple[str, ...] = (
    "A1_full", "A2_no_trust", "A3_no_accumulation", "A4_wasabi",
    "A5_no_relationship",
)

M = affect.TIER_OFFSETS["close"][2]
assert abs(M - 0.40) < 1e-9, "the deployed tier magnitude moved; re-derive M"

TICKS_PER_TURN = 20      # matches system_trace.py's convention

# The ramp triple each arm's D reaches at full displacement, used ONLY by
# metrics 1 and 3 (see dsourcing_metrics.py for why those two metrics are
# structurally blind to A1 vs A2 vs A3).
RAMP: Dict[str, Tuple[float, float, float]] = {
    "A1_full": affect.TIER_OFFSETS["close"],
    "A2_no_trust": affect.TIER_OFFSETS["close"],
    "A3_no_accumulation": affect.TIER_OFFSETS["close"],
    # WASABI has no arousal-leak analogue — the +0.10 term is a documented
    # deviation of OUR tier ladder (E2's calibration target), not part of the
    # rule being borrowed. Carrying it into A4 would attribute our own known
    # violation to their design. Normalised to the same PRINCIPAL magnitude
    # only, as the brief requires, stated here rather than left implicit.
    "A4_wasabi": (0.0, 0.0, M),
    "A5_no_relationship": (0.0, 0.0, 0.0),
}

# A3's instantaneous re-banding of felt Pleasure into the tier ladder's order.
# Equal-width bands over felt-P's full possible range [-1, 1], chosen for being
# the least arbitrary split available (no tuned edge) rather than for matching
# any particular robot's baseline. Robustness to this choice is checked in
# test_dsourcing.py by shifting the edges and confirming the qualitative
# findings (frame-rate switching, tau_D ~ tau_P) do not depend on the exact cut.
INSTANT_TIER_EDGES: Tuple[float, float, float] = (0.5, 0.0, -0.5)


def instant_tier(felt_p: float,
                 edges: Tuple[float, float, float] = INSTANT_TIER_EDGES) -> str:
    hi, mid, lo = edges
    if felt_p > hi:
        return "close"
    if felt_p > mid:
        return "known"
    if felt_p > lo:
        return "visitor"
    return "unknown"


@dataclass
class ArmState:
    """Per-(person, robot) memory a D-source needs across turns.

    A5 and A4 use none of this — they are stateless by construction, which is
    itself part of what this phase measures. A3 uses none of it either, on
    purpose: `step_d_offset` never reads `rapport`/`trust`/`count` for A3, since
    reading them would silently reintroduce the accumulation being isolated
    against.
    """
    rapport: float = 0.0
    trust: float = 0.0
    count: int = 0
    accrual: SessionAccrual = field(default_factory=SessionAccrual)

    def new_session(self, trust_cap: Optional[float] = None) -> None:
        """Start a fresh session's trust budget. Rapport and count persist —
        only trust is session-scoped, exactly as `SessionAccrual` documents."""
        self.accrual = SessionAccrual(trust_cap=trust_cap)


def step_d_offset(arm: str, state: ArmState, felt_p: float, child_text: str,
                  ticks: int = TICKS_PER_TURN,
                  robot_turn: Optional[bool] = None,
                  ) -> Tuple[float, float, float]:
    """Advance `state` by one turn (mutated in place) and return this turn's
    (dP, dAr, dD) offset triple, ready for `compose_offsets`.

    `child_text` is coded for disclosure by A1 only. `robot_turn` is consulted
    by A4 only and ignored elsewhere — passing it for A1/A2/A3/A5 is harmless
    and lets one call site drive every arm from the same script.
    """
    if arm == "A1_full":
        d_rapport, d_trust = state.accrual.turn(child_text, felt_p, ticks=ticks)
        state.rapport = min(RAPPORT_CEILING, state.rapport + d_rapport)
        state.trust = min(1.0, state.trust + d_trust)
        state.count += 1
        tier = _tier_from_scores(state.rapport, state.trust, state.count)
        return affect.TIER_OFFSETS[tier]

    if arm == "A2_no_trust":
        # The trust channel does not exist here — `detect` is never called, not
        # merely discarded, so the disclosure detector's own defects (Phase 10)
        # cannot leak into this arm's result even in principle.
        if felt_p > PLEASURE_FLOOR:
            state.rapport = min(RAPPORT_CEILING,
                                state.rapport + RAPPORT_GAIN * felt_p * ticks)
        state.count += 1
        tier = _tier_from_scores(state.rapport, 0.0, state.count)
        return affect.TIER_OFFSETS[tier]

    if arm == "A3_no_accumulation":
        # No `state` read or write at all: the tier is re-derived from THIS
        # turn's felt Pleasure alone, with no memory of any prior turn. Uses
        # the same felt-P signal the pipeline already computes for the P axis
        # (post empathy, post AffectStream where the caller applies it) —
        # deliberately excluding the graph mood blend (kg_bridge.py:302-314),
        # which is itself a FAST-tier memory mechanism and would reintroduce a
        # form of accumulation this arm exists to remove.
        tier = instant_tier(felt_p)
        return affect.TIER_OFFSETS[tier]

    if arm == "A4_wasabi":
        if robot_turn is None:
            raise ValueError("A4_wasabi requires robot_turn (True=dominant)")
        return (0.0, 0.0, M if robot_turn else -M)

    if arm == "A5_no_relationship":
        return (0.0, 0.0, 0.0)

    raise ValueError(f"unknown arm {arm!r}")
