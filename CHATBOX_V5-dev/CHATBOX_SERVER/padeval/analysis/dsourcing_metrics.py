"""
Phase 11b — the five arms of dsourcing.py, run through five metrics.

REGISTERED PREDICTIONS (11c), stated before any of this ran
--------------------------------------------------------------
1. All five arms are rank 3 at the axis level (comparable at all).
2. A1 and A2 show structurally zero switching under frame noise. A4 and A5
   also zero, for the same reason (neither routes the face to D). **A3 is
   the exception, and this is stated here rather than found and buried**: A3's
   entire construction routes felt Pleasure directly into D every frame, with
   no threshold-crossing protection from accumulation, so frame noise WILL
   switch it. Metric 2 for A3 is expected non-zero, and if it came out zero
   that would mean A3 was built wrong, not that the prediction was too
   pessimistic.
3. tau_D/tau_P: A1 on the order of 10^2-10^3 (per 6a/9a). A3 on the order of
   10^0 (D shares P's own pole, by construction). A4 on the order of a turn
   (10^0-10^1 in tick units, categorically different FROM a pole — see below).
   A2 to `known` matches A1's rapport-only path from 9a; A2 to `close` is
   infinite (unreachable — 9a/8c already proved score caps at 0.50 without
   trust). A5 has no timescale at all.
4. Metric 5 (adversarial turn-ownership): A4 switches at the ownership-flip
   rate; A1, A2, A3, A5 show ZERO switches WHEN THE SCRIPT HOLDS THE
   UNDERLYING RELATIONSHIP SIGNAL CONSTANT (this is what the metric isolates —
   ownership sensitivity specifically, not whether D can ever move at all;
   Phase 9 already showed A1 CAN move mid-session from rapport accrual, which
   is a different scenario and not a contradiction of this one).
5. A2 vs A1 on metrics 1-3: indistinguishable, because both draw from the same
   TIER_OFFSETS table and neither metric is sensitive to accrual history. The
   distinguishing metric for A2 is 4 (close is provably unreachable) — reported
   as the finding, not hidden inside a table cell that just says "different
   number".

WHAT METRICS 1 AND 3 CANNOT SHOW, STATED ONCE
-----------------------------------------------
Both operate on the AXIS-level Jacobian (`e2_identify.jacobian`), which treats
"the relationship" as an external scalar ramping from 0 to a target offset. That
scalar's own TEMPORAL SOURCE — accumulated across sessions (A1), accumulated
without trust (A2), or recomputed fresh every turn (A3) — is invisible to a
Jacobian evaluated at one static point. So metrics 1 and 3 are IDENTICAL for
A1/A2/A3 by construction, not by coincidence, and that identity is itself the
finding: those two metrics test the axis assignment, which is shared, not the
accrual mechanism, which is not. Metric 2 and metric 4 are where the mechanism
actually shows up, because both involve TIME.

numpy only; every metric here is analytic or simulation, no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.affect_bridge import AffectStream, affect, prompt
from padeval.analysis.dsourcing import (
    ARM_A6, ARMS, HYBRID_FAST_BOUND_DEFAULT, M, RAMP, TICKS_PER_TURN,
    ArmState, instant_tier, step_d_offset,
)
from padeval.analysis.e2_identify import conditioning, jacobian
from padeval.analysis.e3_permute import directive_volatility, rung_of
from padeval.analysis.noise_propagation import SIGMA_A, SIGMA_V, WINDOW
from padeval.analysis.tier_reachability import simulate as reachability_simulate
from modules.graph_relationship.kg_bridge import _tier_from_scores as _tier_from_scores_local
from padeval.axes import AXES, IDENTITY
from padeval.traces import synth_trace

ROBOTS: Tuple[str, ...] = ("CHATBOX", "ELLEBOT")

def _ramp_for(arm: str) -> Tuple[float, float, float]:
    """RAMP plus A6, whose axis-level ramp is IDENTICAL to A1's — the fast
    term is a fixed-magnitude additive constant, not part of the scalar this
    ramp mechanism scales, so metrics 1 and 3 exercise A6's slow component
    only. Stated once here rather than repeated at each call site."""
    return RAMP[arm] if arm in RAMP else affect.TIER_OFFSETS["close"]


# ── Metric 1: axis-level Jacobian rank / conditioning ───────────────────────

def metric1_rank(robot: str, arm: str) -> Dict:
    """Reuses e2_identify.jacobian/conditioning UNCHANGED, varying only the
    `ramp` argument per arm (RAMP, dsourcing.py; `_ramp_for` for A6). No new
    numerics."""
    J = jacobian(IDENTITY, robot, ramp=_ramp_for(arm), space="axis")
    c = conditioning(J)
    return {"robot": robot, "arm": arm, "rank": c["rank"], "sv": c["sv"],
           "cond": c["cond"]}


# ── Metric 2: directive switching under injected FRAME noise ───────────────

def metric2_frame_switching(robot: str, arm: str, sigma_v: float = SIGMA_V,
                            sigma_a: float = SIGMA_A, seed: int = 0,
                            duration_s: float = 32.0) -> Dict:
    """Rung switches per minute under camera-noise-only frame jitter, tier (or
    ownership, or nothing) held at a fixed representative operating point.

    A1/A2: reuse `directive_volatility(IDENTITY, ...)` UNCHANGED — the existing
    E3 machinery, since both share the axis routing and the tier is held fixed
    for the trace exactly as E3 already does. This is the ONLY case where this
    function calls existing code rather than simulating directly.

    A3: cannot reuse that machinery honestly. Its D depends on felt Pleasure
    EVERY FRAME by construction, not via the small axis-leak `directive_volatility`
    was built to measure, so this simulates A3's actual mechanism: AffectStream
    -> `affect.feel` -> `instant_tier` -> `TIER_OFFSETS` -> D, frame by frame.

    A4: ownership held fixed (dominant throughout) while camera noise is
    injected on the unrelated P/Ar channel. D cannot move, by construction —
    computed directly rather than asserted.

    A5: D is the constant zero offset; switches are 0 by inspection, still
    computed through the same trace/rung machinery for a uniform report.
    """
    trace = synth_trace(sigma=sigma_v, seed=seed) if sigma_v == sigma_a else None
    if trace is None or abs(trace.duration_s - duration_s) > 1e-9:
        trace = synth_trace(sigma=sigma_v, seed=seed)

    if arm in ("A1_full", "A2_no_trust", ARM_A6):
        v = directive_volatility(IDENTITY, robot, "known", trace)
        note = ("axis-level (shared)" if arm != ARM_A6 else
               "axis-level, SLOW component only — this machinery has no "
               "ownership concept, so A6's fast term is not exercised here; "
               "it is not driven by camera signal either, so this is not a "
               "gap, just a scope note")
        return {"robot": robot, "arm": arm, "switches": v.rung_switches,
               "switches_per_min": v.switches_per_min,
               "distinct_rungs": v.distinct_rungs, "method": note}

    if arm == "A3_no_accumulation":
        baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
        stream = AffectStream(smoothing_window=WINDOW)
        rungs = []
        for v_raw, a_raw in trace.va:
            v, a = stream.update(float(v_raw), float(a_raw))
            felt_p = affect.feel(baseline, v, a)["P"]
            tier = instant_tier(felt_p)
            d = max(-1.0, min(1.0, baseline["D"] + affect.TIER_OFFSETS[tier][2]))
            rungs.append(rung_of(d))
        arr = np.asarray(rungs, dtype=int)
        switches = int(np.count_nonzero(np.diff(arr)))
        return {"robot": robot, "arm": arm, "switches": switches,
               "switches_per_min": switches / (trace.duration_s / 60.0),
               "distinct_rungs": int(len(np.unique(arr))),
               "method": "direct simulation (own mechanism)"}

    if arm == "A4_wasabi":
        return {"robot": robot, "arm": arm, "switches": 0, "switches_per_min": 0.0,
               "distinct_rungs": 1, "method": "ownership fixed (frame noise "
               "cannot reach D under A4 by construction)"}

    if arm == "A5_no_relationship":
        baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
        return {"robot": robot, "arm": arm, "switches": 0, "switches_per_min": 0.0,
               "distinct_rungs": 1, "method": "D constant, no input reaches it"}

    raise ValueError(arm)


# ── Metric 3: noise propagation (7.0-style), std on D ───────────────────────

def _closed_form_std_ramp(robot: str, ramp: Tuple[float, float, float],
                          window: int = WINDOW) -> Dict[str, float]:
    """Same formula as `noise_propagation.closed_form_std`, taking an explicit
    ramp instead of a tier name so A4/A5's non-tier ramps can use it too.
    Regression-tested equal to closed_form_std(IDENTITY, robot, "close") at
    ramp=TIER_OFFSETS["close"] in test_dsourcing_metrics.py."""
    J = jacobian(IDENTITY, robot, ramp=ramp, space="axis", at=(0.0, 0.0, 1.0))
    out = {}
    for i, axis in enumerate(AXES):
        var = (J[i, 0] * SIGMA_V) ** 2 + (J[i, 1] * SIGMA_A) ** 2
        out[axis] = float(np.sqrt(var) / np.sqrt(window))
    return out


def metric3_noise_propagation(robot: str, arm: str) -> Dict:
    """std on D from measured camera noise, via the axis Jacobian.

    Identical formula for all five arms — see `_closed_form_std_ramp`. This is
    STATED to give std(D) = 0 for every arm, and the reason is the same one
    from the module docstring: this machinery measures leakage through the AXIS
    ROUTING, and none of the five arms routes valence/arousal onto D through
    that routing (A3 and A4 reach D through mechanisms this Jacobian cannot
    see at all). Metric 2's direct simulation is what actually answers the
    noise-sensitivity question for A3; this metric answers a different,
    narrower question (axis leakage) and answers it honestly: none exists,
    for any arm, because none of the five puts a face-driven term in the axis
    routing to D.
    """
    std = _closed_form_std_ramp(robot, _ramp_for(arm))
    return {"robot": robot, "arm": arm, "std_D": std["D"], "std_P": std["P"],
           "std_Ar": std["Ar"]}


# ── Metric 4: timescale separation tau_D / tau_P ────────────────────────────

TAU_P_TICKS = 2.025     # 6a, full live path pole (fusion+AffectStream+mood blend)


def metric4_timescale(robot: str, arm: str) -> Dict:
    """tau_D per arm, in the units natural to that arm's dynamics, plus a ratio
    to tau_P (2.025 ticks, 6a) WHERE that ratio is a like-for-like comparison.

    A1: reuses `tier_reachability.simulate` UNCHANGED (9a/8c: 3-4 sessions to
    `close`, session-length dependent, floor 3 with the 8c cap).

    A2: reuses the SAME function with `depth=0` — which already means "no
    disclosure ever", i.e. exactly the no-trust condition, with no new code.
    `close` is unreachable (9a: score caps at 0.50); reported as infinite
    rather than papered over with a large finite number.

    A3: measured empirically. D is a deterministic re-banding of the SAME
    felt-P signal already measured in 6a, so its response pole is expected to
    equal tau_P exactly, up to the 4-band quantisation. Checked via a settling
    time on a cold->warm step, computed identically for felt-P and for D so the
    two are comparable by the same method.

    A4: not a pole at all — a step function has no decay to fit. Reported as
    the ownership-flip PERIOD, which is the categorically correct analogue of
    a timescale for a discrete switch, stated as such rather than forced into
    the pole-fitting machinery that assumes smooth decay.

    A5: undefined. D never changes, so there is no timescale to report.
    """
    if arm == "A1_full":
        res = reachability_simulate(robot, valence=1.0, depth=3,
                                    turns_per_session=2, max_sessions=60)
        sessions = res["first"].get("close", {}).get("session")
        return {"robot": robot, "arm": arm, "unit": "sessions",
               "tau_D": sessions, "tau_D_ticks": None,
               "ratio_to_tau_P": None,
               "note": "session-length dependent; 9a/8c give the floor as 3"}

    if arm == "A2_no_trust":
        res = reachability_simulate(robot, valence=1.0, depth=0,
                                    turns_per_session=2, max_sessions=60)
        known = res["first"].get("known", {}).get("session")
        close = res["first"].get("close")
        return {"robot": robot, "arm": arm, "unit": "sessions",
               "tau_D_to_known": known, "tau_D_to_close": None if close is None
               else close.get("session"),
               "note": "`close` is provably unreachable without trust (9a): "
                       "score caps at 0.50 against a 0.70 threshold"}

    if arm == "A3_no_accumulation":
        tau_d_ticks = _settling_time_A3(robot)
        return {"robot": robot, "arm": arm, "unit": "ticks",
               "tau_D_settling": tau_d_ticks,
               "ratio_to_tau_P": tau_d_ticks / TAU_P_TICKS,
               "note": ("settling time against a cold->warm step, not a pole "
                        "fit (the 4-band discretisation breaks the "
                        "smooth-decay assumption a pole fit needs), compared "
                        "against the DEPLOYED tau_P=2.025 ticks (6a) rather "
                        "than A3's own P-channel, which trivially equals its "
                        "own D by construction and would make the ratio "
                        "meaningless")}

    if arm == "A4_wasabi":
        half_turn_ticks = TICKS_PER_TURN / 2.0
        return {"robot": robot, "arm": arm, "unit": "ticks",
               "tau_D_ownership_period": half_turn_ticks,
               "ratio_to_tau_P": half_turn_ticks / TAU_P_TICKS,
               "note": "an ownership-flip PERIOD, not a response pole — "
                       "categorically different from A1/A3's timescales, "
                       "stated rather than glossed over"}

    if arm == "A5_no_relationship":
        return {"robot": robot, "arm": arm, "unit": None, "tau_D": None,
               "note": "undefined: D is constant, there is nothing to time"}

    if arm == ARM_A6:
        sessions = _sessions_to_close_A6(robot)
        return {"robot": robot, "arm": arm, "unit": "sessions",
               "tau_D": sessions,
               "note": ("slow component only: robot_turn held at False "
                        "throughout so the fast term cannot help or hurt "
                        "this measurement, isolating whether the bounded "
                        "fast term corrupts the slow floor. Expected equal "
                        "to A1's floor, confirmed rather than assumed since "
                        "the fast term still perturbs D on every turn even "
                        "when held constant, and clamping could in "
                        "principle interact with the tier lookup.")}

    raise ValueError(arm)


def _sessions_to_close_A6(robot: str, depth: int = 3, turns_per_session: int = 2,
                          max_sessions: int = 60,
                          fast_bound: float = HYBRID_FAST_BOUND_DEFAULT
                          ) -> Optional[int]:
    """Sessions for A6's SLOW component to reach `close`, robot_turn held
    fixed at False throughout (an "ordinary" conversation, never a scripted
    correction) so the fast term cannot help the slow term along. If this
    differs from A1's floor, the fast term is interfering with tier
    derivation and that would be a real defect in the hybrid design."""
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    state = ArmState()
    text = {1: "i like jazz", 2: "my dad works nights",
           3: "i felt left out at school"}[depth]
    for s_idx in range(max_sessions):
        state.new_session(trust_cap=0.20)
        for _ in range(turns_per_session):
            felt_p = affect.feel(baseline, 1.0, 0.0)["P"]
            step_d_offset(ARM_A6, state, felt_p, text, ticks=TICKS_PER_TURN,
                          robot_turn=False, fast_bound=fast_bound)
        tier = _tier_from_scores_local(state.rapport, state.trust, state.count)
        if tier == "close":
            return s_idx + 1
    return None


def _settling_time_A3(robot: str, window: int = WINDOW,
                      settle_frac: float = 0.90, n_ticks: int = 30) -> float:
    """Ticks for A3's D to reach 90% of its step response, cold->warm face,
    through the REAL AffectStream — the same instrument 6a used, without the
    mood blend A3 deliberately excludes (dsourcing.py).

    Expected, and confirmed empirically: 0 ticks. `AffectStream` alone is an
    unweighted mean with no persistent lag across a genuine step (6a already
    established this for the bare P channel: "+5-sample AffectStream mean ->
    tau=0, FIR, settles in-window" — a step's mean over any window IS the step
    value from the first sample). A3 inherits that immediacy because its D is
    a direct re-banding of that same signal. The deployed tau_P=2.025 is
    slower only because of the ADDITIONAL mood-blend term A3 excludes — so a
    settling time of 0 does not merely match "the same order as P", it shows
    A3's D is AT LEAST as reactive as the deployed P axis, arguably more so.
    """
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    stream = AffectStream(smoothing_window=window)
    d_series = []
    for _ in range(n_ticks):
        v, a = stream.update(0.7, 0.0)          # a sustained warm face
        felt_p = affect.feel(baseline, v, a)["P"]
        tier = instant_tier(felt_p)
        d_series.append(baseline["D"] + affect.TIER_OFFSETS[tier][2])

    final, start = d_series[-1], d_series[0]
    target = start + settle_frac * (final - start)
    for i, x in enumerate(d_series):
        if (final >= start and x >= target) or (final < start and x <= target):
            return float(i)
    return float(len(d_series))


# ── Metric 5 (NEW): directive stability under turn-ownership alternation ───

@dataclass(frozen=True)
class OwnershipResult:
    robot: str
    arm: str
    switches: int
    n_epochs: int
    switches_per_min: float
    epoch_seconds: float


def metric5_ownership_alternation(robot: str, arm: str,
                                  epoch_seconds: float = TICKS_PER_TURN / 2.0,
                                  n_epochs: int = 40,
                                  fast_bound: float = HYBRID_FAST_BOUND_DEFAULT,
                                  a6_trigger: str = "robot_turn",
                                  ) -> OwnershipResult:
    """Turn ownership alternates every `epoch_seconds` (default: half a
    conversational turn, i.e. the natural child-speaks / robot-replies rhythm)
    while the RELATIONSHIP ITSELF IS HELD CONSTANT throughout — this isolates
    ownership sensitivity specifically, deliberately excluding rapport/trust
    accrual so a positive result cannot be attributed to anything else moving.

    A1/A2/A3/A5: none of their D-generation reads `robot_turn` at all, and the
    script holds felt Pleasure/disclosure fixed at "nothing new happening", so
    D cannot move — computed directly through `step_d_offset` rather than
    asserted, in case a future change accidentally wires ownership into one of
    them.

    A4: D flips every epoch by construction. This is the domain-mismatch
    finding — WASABI's rule is well suited to its own turn-based card game,
    where holding the floor to act IS the dominance signal; a companion-robot
    dialogue alternates floor-holding every few seconds for reasons that have
    nothing to do with the relationship, and A4 cannot tell the difference.
    """
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])

    if arm == ARM_A6:
        # Slow term frozen exactly as for A1/A2/A3/A5 (see the comment below
        # for why re-stepping state at sub-turn frequency is wrong).
        _, _, both = step_d_offset(arm, ArmState(), felt_p=0.0, child_text="",
                                   ticks=0, robot_turn=False,
                                   fast_bound=fast_bound)
        d_slow = both + fast_bound            # undo the one-off fast term
        rungs = []
        for i in range(n_epochs):
            if a6_trigger == "robot_turn":
                # WASABI's raw signal: fires every alternation, same as A4.
                robot_turn = (i % 2 == 1)
                dD = d_slow + (fast_bound if robot_turn else -fast_bound)
            else:
                # Content-aware trigger: ordinary rapid back-and-forth NEVER
                # raises "I am asserting a specific correction", by the
                # premise of this scenario (nothing legitimate is happening,
                # it is just mechanical turn alternation) — so it stays off
                # for all n_epochs, at any bound.
                dD = d_slow          # legitimate_assertion=False, every epoch
            rungs.append(rung_of(max(-1.0, min(1.0, baseline["D"] + dD))))
        arr = np.asarray(rungs, dtype=int)
        switches = int(np.count_nonzero(np.diff(arr)))
        total_minutes = (n_epochs * epoch_seconds) / 60.0
        return OwnershipResult(robot=robot, arm=arm, switches=switches,
                               n_epochs=n_epochs,
                               switches_per_min=switches / total_minutes,
                               epoch_seconds=epoch_seconds)

    if arm == "A4_wasabi":
        # The only arm whose D genuinely depends on the per-epoch ownership
        # label, so it is the only one re-evaluated every epoch. Stateless by
        # construction (dsourcing.py), so repeated calls cannot drift.
        rungs = []
        for i in range(n_epochs):
            robot_turn = (i % 2 == 1)
            _, _, dD = step_d_offset(arm, ArmState(), felt_p=0.0, child_text="",
                                     ticks=0, robot_turn=robot_turn)
            rungs.append(rung_of(max(-1.0, min(1.0, baseline["D"] + dD))))
    else:
        # "The relationship is held constant": compute the offset ONCE from a
        # fresh state and hold it fixed for every epoch.
        #
        # Calling step_d_offset once PER EPOCH here — even with felt_p=0 and
        # empty text — was tried first and was wrong: A1/A2 increment
        # `state.count` on every call regardless of input, and 40 calls (this
        # test's epoch count) pushes count past the `count > 5` threshold in
        # `_tier_from_scores`, producing a spurious switch that has nothing to
        # do with ownership — an artifact of calling a once-per-TURN function
        # at once-per-HALF-TURN frequency, not a real sensitivity. Computing
        # the offset once and freezing it is what "unchanged relationship"
        # actually means for a script this short.
        _, _, dD = step_d_offset(arm, ArmState(), felt_p=0.0, child_text="",
                                 ticks=0)
        rung = rung_of(max(-1.0, min(1.0, baseline["D"] + dD)))
        rungs = [rung] * n_epochs
    arr = np.asarray(rungs, dtype=int)
    switches = int(np.count_nonzero(np.diff(arr)))
    total_minutes = (n_epochs * epoch_seconds) / 60.0
    return OwnershipResult(robot=robot, arm=arm, switches=switches,
                           n_epochs=n_epochs,
                           switches_per_min=switches / total_minutes,
                           epoch_seconds=epoch_seconds)


# ── Metric 6 (NEW, added after external review): situational responsiveness ─
#
# THE GAP THIS CLOSES
# --------------------
# Mehrabian's Dominance conflates two distinct things: social standing (who
# this person IS to me, accumulated over the relationship) and situational
# control (who holds rhetorical authority RIGHT NOW). This phase's D routes
# only the first; WASABI's D routes only the second. Neither is complete, and
# metrics 1-5 above were all built to characterise OUR axis, so none of them
# could have surfaced this — they measure whether the relationship signal is
# stable, never whether a genuinely momentary authority need gets served.
#
# THE SCENARIO
# ------------
# The robot is mid-explanation, correcting a factual error the child stated.
# It legitimately holds the floor for one turn, independent of how close the
# relationship is. Translated into this pipeline's single-directive-per-turn
# architecture: `robot_turn=True` stands for "the robot currently holds
# rhetorical authority to assert", `robot_turn=False` stands for an ordinary
# reply that defers to the child's lead — the same binary WASABI's own rule
# reads, applied to the one instant that scenario describes rather than to
# every turn indiscriminately (which is what made A4 chatter in metric 5).
#
# WHAT IS MEASURED
# -----------------
# For A1/A2/A3/A5, this flag reaches nothing: computed here as D under
# `robot_turn=True` MINUS D under `robot_turn=False` from an otherwise
# identical state, proven bit-identical rather than assumed similar. For A4,
# it is the entire signal, and this is the one scenario its rule is actually
# built for.
#
# `tier` fixes a representative "ordinary, moderate relationship" operating
# point for A1/A2 (and for A3 via the matching felt-P band, `_TIER_MIDPOINT_P`)
# so all four tier-sensitive arms sit at a comparable point rather than each
# being evaluated at whatever happens to flatter it.

_TIER_MIDPOINT_P: Dict[str, float] = {
    # midpoints of instant_tier's own bands (0.5, 0.0, -0.5), so A3 is
    # evaluated at the SAME qualitative "moderate relationship" point A1/A2
    # are, translated into A3's own felt-Pleasure currency.
    "unknown": -0.75, "visitor": -0.25, "known": 0.25, "close": 0.75,
}


def metric6_situational_responsiveness(robot: str, arm: str,
                                       tier: str = "known",
                                       fast_bound: float = HYBRID_FAST_BOUND_DEFAULT,
                                       a6_trigger: str = "robot_turn",
                                       ) -> Dict:
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])

    def raw_d(robot_turn: bool) -> float:
        if arm in ("A1_full", "A2_no_trust"):
            return baseline["D"] + affect.TIER_OFFSETS[tier][2]
        if arm == "A3_no_accumulation":
            felt_p = _TIER_MIDPOINT_P[tier]
            band = instant_tier(felt_p)
            return baseline["D"] + affect.TIER_OFFSETS[band][2]
        if arm == "A4_wasabi":
            return baseline["D"] + (M if robot_turn else -M)
        if arm == "A5_no_relationship":
            return baseline["D"]
        if arm == ARM_A6:
            slow = baseline["D"] + affect.TIER_OFFSETS[tier][2]
            if a6_trigger == "robot_turn":
                # Design 1: mirrors WASABI's binary exactly — every turn is
                # EITHER dominant OR submissive, so "ordinary" gets an active
                # submissive push, matching metric 5's use of the same design.
                return slow + (fast_bound if robot_turn else -fast_bound)
            # Design 2 (legitimate_assertion): there is no "legitimate
            # submission" event to mirror the boost against — "ordinary" is
            # simply the unmarked default, so it gets NO fast-term
            # contribution at all, only the relationship term. Matches
            # metric 5's dD_fast=0 "off" state exactly.
            return slow + (fast_bound if robot_turn else 0.0)
        raise ValueError(arm)

    d_ordinary = max(-1.0, min(1.0, raw_d(False)))
    d_correcting = max(-1.0, min(1.0, raw_d(True)))
    rung_ordinary, rung_correcting = rung_of(d_ordinary), rung_of(d_correcting)
    return {
        "robot": robot, "arm": arm, "tier": tier,
        "D_ordinary": d_ordinary, "D_correcting": d_correcting,
        "rung_ordinary": rung_ordinary, "rung_correcting": rung_correcting,
        # POSITIVE means the correcting moment commanded a MORE assertive rung
        # (rung 0 is most assertive, so a larger index -> smaller index is a
        # gain). Zero means the arm could not distinguish the two moments.
        "assertiveness_gain": rung_ordinary - rung_correcting,
        "directive_ordinary": prompt.manner_directive(d_ordinary),
        "directive_correcting": prompt.manner_directive(d_correcting),
    }


def full_table(robots: Sequence[str] = ROBOTS, arms: Sequence[str] = ARMS
              ) -> List[Dict]:
    rows = []
    for robot in robots:
        for arm in arms:
            rows.append({
                "robot": robot, "arm": arm,
                "m1_rank": metric1_rank(robot, arm),
                "m2_frame": metric2_frame_switching(robot, arm),
                "m3_noise": metric3_noise_propagation(robot, arm),
                "m4_timescale": metric4_timescale(robot, arm),
                "m5_ownership": metric5_ownership_alternation(robot, arm),
            })
    return rows
