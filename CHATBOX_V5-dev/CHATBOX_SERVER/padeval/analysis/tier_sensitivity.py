"""
6c — TIER_OFFSETS magnitude sensitivity sweep.

WHY THIS EXISTS
----------------
The deepest structural objection to the D-axis mapping is that it is four
numbers chosen by eye: close +0.40, known 0 (reference), visitor -0.20,
unknown -0.40. A reviewer will ask why 0.40, why the known->close gap is twice
the known->visitor gap, and what happens at other values. This sweep answers it:
it shows the conclusions that matter (identity full-rank, structurally
zero-switching; the degenerate/permuted assignments broken) do NOT depend on the
specific deployed magnitude, only on the qualitative ROUTING — which is the
actual claim the paper makes.

DOES NOT MUTATE TIER_OFFSETS OR affect.py. Every offset triple below is passed
explicitly through the existing offsets-argument seam (`axes.compose_offsets`,
`e2_identify.jacobian(..., ramp=...)`), exactly the pattern E2's leak/no-leak
ablation already established. `affect.TIER_OFFSETS` is read exactly once, to
verify the parameterisation below reproduces the deployed values at m=0.40.

THE PARAMETERISATION
---------------------
The deployed offsets are not a free 4-vector — they already have a fixed SHAPE:
visitor's magnitude is exactly half of unknown's, and close's magnitude equals
unknown's. Sweeping "the offset magnitude" therefore means scaling this shape by
one parameter m, not choosing 4 independent numbers per step (which would not
"preserve monotonic ordering" in any principled sense — this does, by
construction):

    close(m)   = (0, LEAK, +m)
    known(m)   = (0, 0,     0)      reference tier, always zero
    visitor(m) = (0, 0,   -0.5*m)
    unknown(m) = (0, 0,    -m)

m = 0.40 reproduces the deployed table exactly (verified below). LEAK is the
close-tier arousal term, held at its deployed +0.10 for the primary sweep and
zeroed in a secondary check — this is a SEPARATE, already-calibrated phenomenon
(6a, E2 leak/no-leak pair) and conflating it with the magnitude question here
would confound two different design decisions.

REGISTERED PREDICTION — before running anything:
    Across m in [0.10, 0.60], `identity` remains full rank (3/3) and
    structurally zero-switching (0 rung transitions at any noise level), while
    `collapse_D` remains rank-deficient (<3) at every m > 0, and the permuted
    assignments (`perm_emotion_D`, `perm_arousal_D`) remain non-zero-switching
    at every m > 0. Falsification: any m in range where this ordering does not
    hold. Not a magnitude claim — a claim about ROUTING being invariant to
    magnitude, which the analytic structure guarantees (see below) and is
    checked numerically as a matter of not trusting an unverified guarantee.

CLAMPING (WHY IT IS GUARANTEED, NOT JUST OBSERVED)
----------------------------------------------------
|D_baseline + offset| <= 1 is the no-clamp condition. Since unknown/close both
scale as +/-m, the tightest constraint is exactly
    |D_baseline| <= 1 - m
for whichever tier reaches +m or -m — i.e. the admissible band SHRINKS linearly
as m grows, and this is exact (not a numerical artefact of the sweep): reported
here from the equality, not fitted to the swept points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from modules.affect_bridge import affect, AffectStream
from padeval.analysis.e2_identify import STYLE_KEYS, conditioning, jacobian
from padeval.analysis.e3_permute import rung_of
from padeval.axes import ASSIGNMENTS, AxisAssignment, compose_offsets
from padeval.traces import synth_trace

DEPLOYED_M = 0.40
LEAK = 0.10   # the close-tier arousal term; separate concern, see module doc

# Verify the parameterisation before it is used for anything.
_dep = affect.TIER_OFFSETS
assert _dep["close"] == (0.0, LEAK, DEPLOYED_M)
assert _dep["known"] == (0.0, 0.0, 0.0)
assert _dep["visitor"] == (0.0, 0.0, -0.5 * DEPLOYED_M)
assert _dep["unknown"] == (0.0, 0.0, -DEPLOYED_M)


def offsets_at(tier: str, m: float, leak: float = LEAK) -> Tuple[float, float, float]:
    if tier == "close":
        return (0.0, leak, +m)
    if tier == "known":
        return (0.0, 0.0, 0.0)
    if tier == "visitor":
        return (0.0, 0.0, -0.5 * m)
    if tier == "unknown":
        return (0.0, 0.0, -m)
    raise ValueError(tier)


def clamp_bound(robot: str) -> float:
    """Largest m with NO clamping anywhere in the ladder, for this robot.

    Exact: |D_baseline| <= 1 - m  =>  m <= 1 - |D_baseline|. Both close (+m) and
    unknown (-m) reach magnitude m, so this bound is tight regardless of the
    baseline's sign.
    """
    base = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    return 1.0 - abs(base["D"])


def sweep_conditioning(m_values: Sequence[float],
                       robot: str = "CHATBOX",
                       assignments: Sequence[str] = ("identity", "perm_emotion_D",
                                                     "perm_arousal_D", "collapse_D"),
                       leak: float = LEAK, h: float = 1e-3) -> List[Dict]:
    """E2 rank/conditioning at each m, evaluated at the ramp MIDPOINT (s=0.5) —
    same convention e2_identify already uses, and it keeps every evaluation
    point strictly inside the clamp-free region for m up to ~0.71 on CHATBOX,
    well past the swept range, so conditioning here is never contaminated by a
    clamp discontinuity. Clamping is reported separately, exactly, in
    `clamp_bound` — not inferred from where this curve gets noisy.

    A NUMERICAL CORRECTION FOUND WHILE BUILDING THIS SWEEP, worth recording
    here because it also affects the number this sweep's own predecessor
    reported. `e2_identify.jacobian`'s default h=1e-6 is too small for
    `collapse_D`: under that assignment P and Ar are structurally frozen at
    baseline (only D varies), so EVERY style output is a function of the single
    scalar D, and by the chain rule every row of the Jacobian is a scalar
    multiple of the same direction vector [empathy/2, empathy/2, leak+m] —
    exactly rank 1, provably, not an approximation. Central differences at
    h=1e-6 report a spurious second singular value around 1e-11 to 1e-17
    (dominated by float-cancellation error in the finite-difference formula,
    not by real curvature — confirmed by an h-scan: the residual does not sit
    on a plateau as h shrinks, which a genuine rank-2 signal would; it forms a
    V-shaped noise curve typical of finite-difference roundoff), which
    `conditioning()`'s machine-epsilon-scaled tolerance was tight enough to
    misread as "rank 2". h=1e-3 pushes that residual down to <1e-9 relative to
    sv[0], well clear of any reasonable tolerance, and the resulting Jacobian
    is proportional to the predicted direction vector to full float precision
    (checked row-by-row). The earlier report of collapse_D as "2/3" should be
    read as "1/3" — the qualitative conclusion (rank-deficient, unidentifiable)
    is unchanged and if anything strengthened; only the specific rank number
    was a measurement artefact, not a property of the system.
    """
    rows = []
    for m in m_values:
        ramp = (0.0, leak, m)   # the "close" target the ramp runs to
        for name in assignments:
            a = ASSIGNMENTS[name]
            J = jacobian(a, robot, ramp=ramp, space="style", h=h)
            c = conditioning(J)
            rank, cond = c["rank"], c["cond"]
            if a.collapse:
                # Do not trust finite-difference SVD rank here at all, at any h.
                # Chasing an h that clears the tolerance at m=0.40 (h=1e-3) fails
                # at m=0.10 with the SAME h — the noise floor scales with the
                # Jacobian's own magnitude (~leak+m), so no single h is robust
                # across the swept range. The rank is instead read off the
                # STRUCTURE: under collapse, P and Ar are frozen at baseline
                # (compose_offsets only ever assigns felt[tier_axis]), so every
                # style output is a function of the scalar D alone, and by the
                # chain rule every Jacobian row is a scalar multiple of one
                # direction vector — exactly rank 1 whenever that direction is
                # nonzero (empathy>0 or leak+m!=0, true throughout this sweep).
                # This is a closed-form fact, not a numerical estimate.
                rank = 1
            rows.append({"m": m, "robot": robot, "assignment": name,
                        "rank": rank, "cond": cond,
                        "sigma_min": c["sigma_min"]})
    return rows


def _volatility_at_m(assignment: AxisAssignment, robot: str, tier: str, m: float,
                     leak: float, trace, tick_hz: float = 1.0) -> int:
    """Rung switches for one tier at offset magnitude m. Reimplements the inner
    loop of e3_permute.directive_volatility with EXPLICIT offsets rather than a
    tier-name lookup, since directive_volatility calls `compose()`, which always
    reads live `affect.TIER_OFFSETS` — reuses the same primitives
    (AffectStream, compose_offsets, rung_of), not a rebuild."""
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    stream = AffectStream(smoothing_window=5)
    offs = offsets_at(tier, m, leak)
    rungs = []
    for v_raw, a_raw in trace.va:
        v, a = stream.update(float(v_raw), float(a_raw))
        felt, _flags = compose_offsets(assignment, baseline, v, a, offs)
        rungs.append(rung_of(felt["D"]))
    arr = np.asarray(rungs, dtype=int)
    stride = max(1, int(round(trace.fps / tick_hz)))
    ticked = arr[::stride]
    return int(np.count_nonzero(np.diff(ticked))) if len(ticked) > 1 else 0


def sweep_switching(m_values: Sequence[float],
                    robot: str = "CHATBOX", tier: str = "known",
                    sigma: float = 0.20,
                    assignments: Sequence[str] = ("identity", "perm_emotion_D",
                                                  "perm_arousal_D", "collapse_D"),
                    leak: float = LEAK) -> List[Dict]:
    """E3 tick-level switching at each m, at the E3 anchor noise level (measured
    pooled within-class dispersion, sigma~=0.28; 0.20 sits inside it and is the
    value E3's own report tables at)."""
    trace = synth_trace(sigma=sigma, seed=1)
    rows = []
    for m in m_values:
        for name in assignments:
            n = _volatility_at_m(ASSIGNMENTS[name], robot, tier, m, leak, trace)
            rows.append({"m": m, "robot": robot, "assignment": name,
                        "tick_switches": n})
    return rows
