"""
9a — the complete reachability map of the relationship ladder.

8b established ONE path precisely: maximal warmth, zero disclosure, one session
— `close` provably unreachable, because `score = (rapport+trust)/2` tops out at
0.50 against a 0.70 threshold when trust stays 0. Correct, but one path through a
larger space. This characterises the rest of it with the discipline 7a applied to
the axis assignments: a closed form for every cell, and an independent numerical
confirmation of every closed form.

THE THREE PARAMETERS EVERY DERIVATION RESTS ON
-----------------------------------------------
1. `_tier_from_scores` (kg_bridge.py:112-131), quoted exactly:

       score = (rapport + trust) / 2
       score > 0.70 -> close ; score > 0.45 -> known
       count > 5    -> known ; count > 0    -> visitor ; else unknown

   All four comparisons are STRICT. That matters at the boundary and is the
   difference between 2 sessions and 3 for `close`.

2. Rapport accrues per TICK at `0.025 * P` when `P > 0.05`
   (webcam_loop.py:1583-1585), one tick per second (`_DEFAULT_TICK`,
   webcam_loop.py:99), and is hard-clamped to 1.0 (webcam_loop.py:742).

3. Trust accrues per TURN by disclosure depth, capped at 0.20 per session
   (disclosure.py, `SESSION_TRUST_CAP`) — the extractor's own clamp, restored
   in 8c.

Felt pleasure is `P = P_base + 0.6*(v - P_base)` (affect.feel, EMPATHY=0.6), so
the fastest possible rapport rate is at v=+1:

       P_max = 0.4 * P_base + 0.6        rho = 0.025 * P_max  per second

numpy-free; every number here is arithmetic on the three rules above.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from modules.affect_bridge import affect
from modules.graph_relationship.kg_bridge import _tier_from_scores
from padeval.coding.disclosure import DEPTH_TRUST_DELTA, SESSION_TRUST_CAP

RAPPORT_GAIN = 0.025          # webcam_loop.py:1585
PLEASURE_FLOOR = 0.05         # webcam_loop.py:1583
RAPPORT_CEILING = 1.0         # webcam_loop.py:742
CLOSE_SCORE, KNOWN_SCORE = 0.70, 0.45
COUNT_KNOWN, COUNT_VISITOR = 5, 0


def p_max(robot: str) -> float:
    """Fastest achievable felt Pleasure: the face at v=+1."""
    return affect.feel(affect.to_pad(affect.ROBOTS[robot]["ocean"]), 1.0, 0.0)["P"]


def rho(robot: str) -> float:
    """Maximum rapport accrued per tick (= per second)."""
    return RAPPORT_GAIN * p_max(robot)


def _strict_ticks(target: float, rate: float) -> int:
    """Steps for a strictly-increasing accrual to EXCEED `target`.

    `n + 1` rather than `ceil` when target/rate is a whole number, because the
    threshold is `>` and not `>=`: that step lands the value exactly ON the
    threshold and does not cross it.

    The near-integer test is deliberately loose (1e-9, not 1e-12). Written tight,
    it got the `close` row WRONG: `2*0.70 - 1.0` evaluates to
    0.3999999999999999, not 0.40, so the exact-multiple branch never fired and
    the closed form returned 2 sessions where the simulation returned 3. The
    simulation was right — trust reaches exactly 0.40 after two capped sessions,
    giving score exactly 0.70, which does not satisfy `> 0.70`.

    This is the entire reason every cell here carries a derivation AND an
    independent confirmation. A closed form alone would have shipped an
    off-by-one on the single most quoted number in the phase.
    """
    q = target / rate
    n = round(q)
    if abs(q - n) < 1e-9:
        return int(n) + 1
    return math.ceil(q)


# ── closed forms, one per transition x path ─────────────────────────────────

def visitor_closed_form() -> Dict:
    """`count > 0`. Requires nothing of affect or disclosure — one turn, always."""
    return {"transition": "unknown -> visitor", "path": "count > 0",
            "unit": "turns", "value": 1,
            "derivation": "count increments once per turn and the branch tests "
                          "count > 0, so turn 1 always lands visitor",
            "requires": "nothing — any affect, any content"}


def known_via_count_closed_form() -> Dict:
    """`count > 5`. Six turns of anything at all, including silence."""
    return {"transition": "visitor -> known", "path": "count > 5",
            "unit": "turns", "value": COUNT_KNOWN + 1,
            "derivation": f"strict `count > {COUNT_KNOWN}` is first true at "
                          f"count = {COUNT_KNOWN + 1}",
            "requires": "nothing — neutral, disengaged turns qualify"}


def known_via_rapport_closed_form(robot: str) -> Dict:
    """`score > 0.45` on rapport alone: rapport > 0.90, at rho per second."""
    need = 2 * KNOWN_SCORE                       # trust = 0
    ticks = _strict_ticks(need, rho(robot))
    return {"transition": "visitor -> known", "path": "score > 0.45 (rapport only)",
            "unit": "seconds", "value": ticks, "robot": robot,
            "derivation": f"rapport must exceed 2*{KNOWN_SCORE} = {need:.2f}; "
                          f"rho = {rho(robot):.6f}/s, so {need:.2f}/rho = "
                          f"{need / rho(robot):.2f} -> {ticks} ticks",
            "requires": "continuous v=+1 face; no disclosure needed"}


def close_closed_form(robot: str, depth: int = 3) -> Dict:
    """`score > 0.70` — the joint constraint, and why its optimum is a CORNER.

    The boundary is `rapport + trust > 1.40`, so in principle the two channels
    trade off and one should look for the cheapest point on that line rather
    than fixing one and solving for the other.

    They do not trade off here, and the reason is worth stating rather than
    assuming: rapport is HARD-CLAMPED at 1.0 (webcam_loop.py:742). The feasible
    set is therefore `rapport in [0,1]`, and since `trust` is the expensive
    channel (0.20 per session against rapport's ~0.9 per minute), the cheapest
    feasible point is always the corner `rapport = 1.0`. There is no interior
    optimum to find. That collapses the constraint to an exact statement:

        close  <=>  trust > 0.40

    which is strictly stronger than 8b's "trust must be nonzero" — necessary,
    but it understated the bar by a factor of the whole trust budget.
    """
    trust_needed = 2 * CLOSE_SCORE - RAPPORT_CEILING          # 0.40
    sessions = _strict_ticks(trust_needed, SESSION_TRUST_CAP)
    per_turn = DEPTH_TRUST_DELTA[depth]
    turns_per_session = math.ceil(SESSION_TRUST_CAP / per_turn)
    rapport_ticks = _strict_ticks(RAPPORT_CEILING - 1e-9, rho(robot))
    return {"transition": "known -> close", "path": "score > 0.70 (joint)",
            "unit": "sessions", "value": sessions, "robot": robot,
            "derivation": f"rapport clamps at {RAPPORT_CEILING}, so the corner "
                          f"rapport=1.0 is optimal and trust must exceed "
                          f"2*{CLOSE_SCORE} - 1.0 = {trust_needed:.2f}; at "
                          f"{SESSION_TRUST_CAP}/session that is {sessions} sessions",
            "requires": f"{turns_per_session} depth-{depth} disclosures per "
                        f"session, plus ~{rapport_ticks}s of warmth once"}


# ── numerical confirmation: simulate the rules, do not reuse the algebra ────

def simulate(robot: str, *, valence: float = 1.0, depth: int = 0,
             turn_seconds: int = 20, turns_per_session: int = 4,
             max_sessions: int = 40, trust_cap: Optional[float] = SESSION_TRUST_CAP,
             ) -> Dict:
    """Step the deployed rules forward and record when each tier first appears.

    Independent of the closed forms above: it accrues tick by tick and asks
    `_tier_from_scores` — the live function — what tier that is.
    """
    base = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    felt_p = affect.feel(base, valence, 0.0)["P"]
    per_turn_trust = DEPTH_TRUST_DELTA[depth]

    rapport = trust = 0.0
    count = 0
    elapsed = 0
    first: Dict[str, Dict] = {}
    for s in range(max_sessions):
        session_trust = 0.0
        for _ in range(turns_per_session):
            tier = _tier_from_scores(rapport, trust, count)
            first.setdefault(tier, {"session": s + 1, "turn": count,
                                    "seconds": elapsed, "rapport": rapport,
                                    "trust": trust})
            if felt_p > PLEASURE_FLOOR:
                rapport = min(RAPPORT_CEILING,
                              rapport + RAPPORT_GAIN * felt_p * turn_seconds)
            gain = per_turn_trust
            if trust_cap is not None:
                gain = max(0.0, min(gain, trust_cap - session_trust))
            trust = min(1.0, trust + gain)
            session_trust += gain
            count += 1
            elapsed += turn_seconds
        tier = _tier_from_scores(rapport, trust, count)
        first.setdefault(tier, {"session": s + 1, "turn": count,
                                "seconds": elapsed, "rapport": rapport,
                                "trust": trust})
        if tier == "close":
            break
    return {"robot": robot, "first": first, "final_rapport": rapport,
            "final_trust": trust}


def zero_rapport_valence(robot: str) -> float:
    """The face valence BELOW which rapport stops accruing entirely.

    Not zero, and that is the point. Felt pleasure is
    `P = 0.4*P_base + 0.6*v` (affect.feel with EMPATHY=0.6), and the accrual
    gate is `P > 0.05` (webcam_loop.py:1583). Both robots have a positive
    baseline P, so a NEUTRAL face still clears the gate and still builds
    rapport — CHATBOX at 0.00266/s, ELLEBOT at 0.00425/s.

    Found by a test that assumed the opposite and failed. "Disengaged" in the
    affective sense is not the same as neutral here: the system reads a blank
    face as mildly positive, because the robot's own temperament supplies most
    of the felt value and the face only moves it 60% of the way.
    """
    p_base = affect.to_pad(affect.ROBOTS[robot]["ocean"])["P"]
    return (PLEASURE_FLOOR - 0.4 * p_base) / 0.6


def seconds_to_known_at_valence(robot: str, valence: float) -> Optional[int]:
    """Time to cross `score > 0.45` on rapport alone at a fixed face valence."""
    felt = affect.feel(affect.to_pad(affect.ROBOTS[robot]["ocean"]), valence, 0.0)["P"]
    if felt <= PLEASURE_FLOOR:
        return None
    return _strict_ticks(2 * KNOWN_SCORE, RAPPORT_GAIN * felt)


def crossover_seconds_per_turn(robot: str) -> float:
    """Turn pacing at which the two `known` paths take equally long.

    Below it the pure-volume path (6 turns of anything) wins; above it the
    warmth path does. Reported because "which is faster" has no answer that is
    independent of how fast the conversation moves.
    """
    return (2 * KNOWN_SCORE / rho(robot)) / (COUNT_KNOWN + 1)


def full_table(robots: Sequence[str] = ("CHATBOX", "ELLEBOT")) -> List[Dict]:
    rows: List[Dict] = [visitor_closed_form(), known_via_count_closed_form()]
    for r in robots:
        rows.append(known_via_rapport_closed_form(r))
    for r in robots:
        rows.append(close_closed_form(r))
    return rows


def format_report(robots: Sequence[str] = ("CHATBOX", "ELLEBOT")) -> str:
    out: List[str] = []
    out.append("| transition | channel path | minimum | unit | gated on |")
    out.append("|---|---|---|---|---|")
    for row in full_table(robots):
        who = f" ({row['robot']})" if "robot" in row else ""
        out.append(f"| `{row['transition']}` | {row['path']}{who} "
                   f"| **{row['value']}** | {row['unit']} | {row['requires']} |")
    out.append("")
    for r in robots:
        out.append(f"- **{r}**: P_max = {p_max(r):.4f}, rho = {rho(r):.6f}/s, "
                   f"two-path crossover at **{crossover_seconds_per_turn(r):.2f} s/turn**")
    out.append("")
    out.append("| robot | to `known` at v=+1 | at v=0 (neutral face) | rapport stops below v |")
    out.append("|---|---|---|---|")
    for r in robots:
        out.append(f"| {r} | {seconds_to_known_at_valence(r, 1.0)}s "
                   f"| {seconds_to_known_at_valence(r, 0.0)}s "
                   f"| {zero_rapport_valence(r):+.4f} |")
    return "\n".join(out)
