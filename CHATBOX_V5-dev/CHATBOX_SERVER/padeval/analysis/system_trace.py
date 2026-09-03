"""
8b — a whole system, traced end to end, with nothing stochastic left unseeded.

WHAT THIS IS FOR
----------------
Phases 6 and 7 established properties one at a time: timescale separation (6a),
tier sensitivity (6c), noise propagation (7.0), identifiability (7a),
admissibility (7b). Each is a statement about a component. None of them shows the
components running TOGETHER, and a reviewer is entitled to ask whether the
separation that holds in the algebra survives contact with a real schedule of
sessions, faces and utterances.

So this runs the deployed mechanism forward over multiple sessions and logs every
established quantity per turn: felt P/Ar/D, tier, rapport and trust SEPARATELY,
whether disclosure fired, the style vector, and the mood value on both sides of a
session boundary.

NO CAMERA, NO LLM, NO NETWORK. The face signal is synthetic, drawn with the same
per-frame dispersion measured on the real corpus in 7.0 (SIGMA_V=0.279,
SIGMA_A=0.292, `noise_propagation.py:71-72`); the utterances are authored; the
tier, the mood gate and both closeness deltas are the deployed code paths. Every
number below is reproducible from a seed.

WHAT IS SIMULATED VS. WHAT IS THE REAL PATH
--------------------------------------------
Real, imported, not reimplemented:
  * `affect.feel_with_relationship` and `affect.gesture_style` — the coordinate
    and the effectors.
  * `_tier_from_scores` — the tier thresholds, imported from kg_bridge.
  * `disclosure.turn_deltas` — which itself reproduces the deployed rapport rule
    (`webcam_loop.py:1583-1585`) verbatim.

Simulated, because they need a process and a graph on disk:
  * The mood blend and its session gate. Reproduced from `kg_bridge.py:302-314`:
    0.7 camera / 0.3 graph mood, and the graph mood counts ONLY if this session
    wrote it. Modelled rather than driven so the counterfactual — what turn 1
    would have been WITHOUT the gate — can be logged beside it. That difference
    is the entire content of the f5bfac9 fix, and it is invisible unless both
    are computed.

A NOTE ON INTER-SESSION GAPS
-----------------------------
6a characterised `sessions.db` as 31 sessions / 97 turns / 4 persons, median 2
turns per session, IQR [1, 4]. It did not characterise gap DURATION, and this
trace does not invent one — because it cannot matter. The mood gate is scoped to
a SESSION, not to a decay curve with a time constant (`kg_bridge.py:242-256`
argues that choice explicitly). A five-minute gap and a five-day gap are
therefore the same event to this mechanism. That is a property worth stating
rather than a limitation worth apologising for: there is no tuned time constant
here to defend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.affect_bridge import affect
from modules.graph_relationship.kg_bridge import _tier_from_scores
from padeval.analysis.noise_propagation import SIGMA_A, SIGMA_V, WINDOW
from padeval.coding.disclosure import detect, turn_deltas

STYLE_KEYS = ("amplitude", "tempo", "posture", "droop", "idle")

# Ticks of live pipeline per conversational turn. The rapport path runs at
# controller rate (1 tick/s, `webcam_loop.py:99`) while a turn is an utterance,
# so a turn is worth many rapport increments and exactly one disclosure check.
# 20 s/turn sits inside 6a's quoted 10-60 s range.
TICKS_PER_TURN = 20

MOOD_CAMERA_W = 0.7      # kg_bridge.py:314
MOOD_GRAPH_W = 0.3


@dataclass(frozen=True)
class Turn:
    emotion: str              # drives the synthetic face signal
    text: str                 # the child's utterance, coded for disclosure
    note: str = ""


@dataclass(frozen=True)
class Session:
    label: str
    turns: Sequence[Turn]


def _camera_va(emotion: str, rng: np.random.Generator) -> Tuple[float, float]:
    """One smoothed face reading: corpus-dispersion noise through AffectStream.

    The WINDOW-sample mean is applied analytically — averaging w iid draws is a
    draw from a distribution WINDOW times narrower — rather than by simulating
    frames, so the trace stays exactly reproducible from the seed and does not
    depend on a frame count nobody would be able to check.
    """
    v0, a0 = affect.CATEGORY_VA.get(emotion.lower(), (0.0, 0.0))
    v = v0 + rng.normal(0.0, SIGMA_V / np.sqrt(WINDOW))
    a = a0 + rng.normal(0.0, SIGMA_A / np.sqrt(WINDOW))
    return float(np.clip(v, -1.0, 1.0)), float(np.clip(a, -1.0, 1.0))


def run_trace(robot: str, sessions: Sequence[Session], seed: int = 0,
              mood_gate: bool = True) -> List[Dict]:
    """Walk the deployed mechanism forward. One row per turn.

    `mood_gate=False` reproduces the PRE-f5bfac9 behaviour — a persisted mood
    edge blended into turn 1 of a fresh session regardless of its age — so the
    fix can be shown as a difference rather than asserted as a property.
    """
    rng = np.random.default_rng(seed)
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])

    rapport = trust = 0.0
    count = 0
    graph_mood: Optional[float] = None        # last felt P written to the graph
    graph_mood_session: Optional[int] = None  # which session wrote it
    rows: List[Dict] = []

    for s_idx, session in enumerate(sessions):
        for t_idx, turn in enumerate(session.turns):
            camera_v, camera_a = _camera_va(turn.emotion, rng)

            # The gate. A mood edge counts only if THIS session wrote it
            # (kg_bridge.py:309-314). Both branches are computed so the
            # counterfactual is on the record.
            same_session = graph_mood_session == s_idx
            usable = graph_mood if (graph_mood is not None
                                    and (same_session or not mood_gate)) else None
            blended_v = (MOOD_CAMERA_W * camera_v + MOOD_GRAPH_W * usable
                         if usable is not None else camera_v)
            ungated_v = (MOOD_CAMERA_W * camera_v + MOOD_GRAPH_W * graph_mood
                         if graph_mood is not None else camera_v)

            # Tier is read fresh every turn, exactly as pre_turn does.
            tier = _tier_from_scores(rapport, trust, count)
            felt = affect.feel_with_relationship(baseline, blended_v, camera_a, tier)
            style = affect.gesture_style(felt)

            raw_d = baseline["D"] + affect.TIER_OFFSETS[tier][2]
            clamped = abs(raw_d) > 1.0

            # Closeness. Rapport accrues per TICK across the turn; disclosure is
            # one event per utterance. The two are read from the same function
            # so neither can silently drift from the deployed rule.
            d_rap_tick, d_trust_turn = turn_deltas(turn.text, felt["P"])
            d_rapport = d_rap_tick * TICKS_PER_TURN
            disc = detect(turn.text)

            r_before, t_before = rapport, trust
            rapport = min(1.0, rapport + d_rapport)
            trust = min(1.0, trust + d_trust_turn)
            count += 1

            rows.append({
                "session": s_idx, "session_label": session.label,
                "turn_in_session": t_idx, "global_turn": len(rows),
                "first_turn_of_session": t_idx == 0,
                "emotion": turn.emotion, "text": turn.text,
                "camera_v": camera_v, "camera_a": camera_a,
                "graph_mood_in": graph_mood,
                "mood_used": usable, "mood_gated_out":
                    graph_mood is not None and usable is None,
                "blended_v": blended_v, "ungated_blended_v": ungated_v,
                "gate_effect": blended_v - ungated_v,
                "tier": tier, "raw_D": raw_d, "clamped": clamped,
                "P": felt["P"], "Ar": felt["Ar"], "D": felt["D"],
                "style": {k: style[k] for k in STYLE_KEYS},
                "disclosed": disc.disclosed, "disclosure_depth": disc.depth,
                "d_rapport": d_rapport, "d_trust": d_trust_turn,
                "rapport_before": r_before, "trust_before": t_before,
                "rapport": rapport, "trust": trust,
                "score": (rapport + trust) / 2, "interaction_count": count,
            })

            # post_turn writes the new mood, stamped with this session.
            graph_mood, graph_mood_session = felt["P"], s_idx

    return rows


# ── the scripted schedule ───────────────────────────────────────────────────
# 4 sessions. Turn counts 3/2/4/2 sit on 6a's empirical median of 2 with an IQR
# of [1,4]. Disclosure is scheduled AWAY from the warmest turns on purpose: if
# the two co-occurred, a divergence between rapport and trust could not be
# attributed to either.

SCRIPT: Tuple[Session, ...] = (
    Session("s0 first meeting", (
        Turn("neutral", "hello", "neither"),
        Turn("happy", "you're funny", "warmth only"),
        Turn("happy", "haha that's a good one", "warmth only"),
    )),
    Session("s1 warm, still guarded", (
        Turn("happy", "that's brilliant", "warmth only"),
        Turn("neutral", "mm", "neither"),
    )),
    Session("s2 opens up", (
        Turn("neutral", "i felt left out at school today", "DISCLOSURE d3"),
        Turn("sad", "my dad works nights so i don't see him much", "DISCLOSURE d2"),
        Turn("neutral", "i get nervous before tests", "DISCLOSURE d3"),
        Turn("happy", "thanks for listening", "neither"),
    )),
    Session("s3 both", (
        Turn("happy", "haha you're the best", "warmth only"),
        Turn("neutral", "i live with my grandma", "DISCLOSURE d2"),
    )),
)


def tau_d_sessions(robot: str = "CHATBOX", max_sessions: int = 40,
                   turns_per_session: int = 2, depth: int = 3,
                   seed: int = 0) -> Dict:
    """Recompute 6a's tau_D under the 8a accrual mechanism, same method.

    6a measured tau_D as SESSIONS TO TRAVERSE ONE TIER STEP, under the old rule
    where trust moved only at end-of-session and was capped at +/-0.2 by the
    extractor. 8a replaced that with a per-turn rule, so the figure has to be
    recomputed rather than carried over.

    Deliberately uses the empirical median session length (2 turns) and the
    warmest realistic profile — an upper bound on speed, i.e. a LOWER bound on
    tau_D, which is the conservative direction for a separation claim.
    """
    rng = np.random.default_rng(seed)
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    text = {1: "i like jazz", 2: "my dad works nights",
            3: "i felt left out at school"}[depth]

    rapport = trust = 0.0
    count = 0
    first: Dict[str, int] = {}
    for s in range(max_sessions):
        for _ in range(turns_per_session):
            tier = _tier_from_scores(rapport, trust, count)
            first.setdefault(tier, s + 1)
            v, a = _camera_va("happy", rng)
            felt = affect.feel_with_relationship(baseline, v, a, tier)
            dr, dt = turn_deltas(text, felt["P"])
            rapport = min(1.0, rapport + dr * TICKS_PER_TURN)
            trust = min(1.0, trust + dt)
            count += 1
        if _tier_from_scores(rapport, trust, count) == "close":
            first.setdefault("close", s + 1)
            break
    return {"robot": robot, "first_session_at_tier": first,
            "sessions_to_close": first.get("close"),
            "final_rapport": rapport, "final_trust": trust,
            "turns_per_session": turns_per_session, "depth": depth}


def format_trace(rows: Sequence[Dict], robot: str) -> str:
    out: List[str] = [f"### {robot}\n"]
    out.append("| s | t | emotion | tier | D | clamp | mood in | used | "
               "P | rapport | trust | disc | ampl |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    last_session = None
    for r in rows:
        if last_session is not None and r["session"] != last_session:
            out.append("| | | | | | | | | | | | | |")
        last_session = r["session"]
        mi = "—" if r["graph_mood_in"] is None else f"{r['graph_mood_in']:+.3f}"
        mu = "**gated**" if r["mood_gated_out"] else (
            "—" if r["mood_used"] is None else f"{r['mood_used']:+.3f}")
        out.append(
            f"| {r['session']} | {r['turn_in_session']} | {r['emotion']} "
            f"| `{r['tier']}` | {r['D']:+.3f} | {'**Y**' if r['clamped'] else '·'} "
            f"| {mi} | {mu} | {r['P']:+.3f} | {r['rapport']:.4f} | {r['trust']:.4f} "
            f"| {'d' + str(r['disclosure_depth']) if r['disclosed'] else '·'} "
            f"| {r['style']['amplitude']:.3f} |")
    return "\n".join(out)
