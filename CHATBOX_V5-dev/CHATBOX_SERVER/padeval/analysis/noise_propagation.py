"""
7.0 (finishing 6d) — propagate MEASURED camera noise through the identity
assignment and the permutations, and cross-check against E3's simulated
switching result.

WHY THIS EXISTS
----------------
E2's rank-deficiency result is exact arithmetic on noise-free inputs. The
obvious question is whether the qualitative separation between `identity` and
the permutations survives real sensor noise, propagated at its MEASURED
magnitude rather than an assumed one.

sigma_v = 0.279, sigma_a = 0.292 — pooled within-class dispersion of the
deployed regression head over the 320-face corpus (docs/PROGRESS.md:192-193,
`traces.measure_model_dispersion`). CAVEAT, stated once and meant to be read
every time these numbers appear: this mixes identity, pose and lighting
variation together with sensor noise, since the corpus is not a repeated-frame
video of one face. It therefore BOUNDS the operating point rather than naming
it exactly — an upper bound on frame-to-frame noise for a single face's video
stream, not a measurement of that specific quantity.

TWO INDEPENDENT ROUTES TO THE SAME NUMBER, DELIBERATELY
----------------------------------------------------------
(1) CLOSED FORM. The pipeline from raw camera (v,a) to felt PAD is linear up to
    the clamp (compose_offsets is affine; AffectStream is an unweighted mean).
    So variance propagates by the ordinary sum-of-squares rule through the E2
    axis Jacobian's valence/arousal columns, then scales down by the
    5-sample window (independent noise -> std divides by sqrt(5)).

(2) MONTE CARLO. Draw many noisy (v,a) pairs, run them through the REAL
    AffectStream + compose_offsets — the actual code, not the linearisation —
    and measure the empirical std of the resulting axis values.

If these disagree by more than sampling noise, that is a bug in the closed
form (or a clamp interfering), not a footnote — checked explicitly below.

CROSS-CHECK AGAINST E3
------------------------
E3 (e3_permute.py) already showed non-zero switching for the permutations at
sigma up to 0.28 via literal trace simulation. This module's Monte Carlo route
reuses the SAME primitives (AffectStream, compose_offsets) as E3's
`directive_volatility`, so the two are not independent implementations that
might quietly disagree — they are two different measurements built on the same
mechanism. The propagated std on D under a permutation is compared against the
gap between adjacent directive rungs (prompt._DIRECTIVES); if that std is
comparable to or larger than a typical rung gap, frequent switching is exactly
what should follow, which is what E3 measured directly.

REGISTERED PREDICTION -- before running anything:
    Under `identity`, propagated std on D is exactly 0 (no noise path exists:
    D receives no contribution from v or a in the routing, only from the
    deterministic tier ramp). Under `perm_emotion_D` and `perm_arousal_D`,
    propagated std on D is strictly positive, and its magnitude is comparable
    to or exceeds the SMALLEST gap between adjacent directive rungs (0.20,
    the 0.32->0.12 step) -- which would make frequent rung-switching the
    expected consequence, consistent with E3's direct measurement, not an
    unexplained coincidence.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from modules.affect_bridge import affect, AffectStream
from pad_core import prompt as P
from padeval.analysis.e2_identify import jacobian
from padeval.axes import AXES, ASSIGNMENTS, AxisAssignment, compose_offsets

SIGMA_V = 0.279
SIGMA_A = 0.292
WINDOW = 5          # AffectStream's default smoothing_window (stream.py:21)

# The gaps between adjacent directive rungs, smallest first -- the scale a
# propagated D-std is compared against.
_RUNG_EDGES = [e for e, _t in P._DIRECTIVES]
RUNG_GAPS = sorted(abs(a - b) for a, b in zip(_RUNG_EDGES, _RUNG_EDGES[1:]))


def closed_form_std(assignment: AxisAssignment, robot: str, tier: str = "known",
                    window: int = WINDOW) -> Dict[str, float]:
    """Propagated std per PAD axis from (sigma_v, sigma_a), via the E2 axis
    Jacobian's valence/arousal columns, scaled for the AffectStream mean.

    Uses the DEPLOYED tier offset for `tier` (not the 6c sweep parameter) --
    this module is about noise propagation, not offset magnitude, so it reads
    affect.tier_offset directly rather than reusing the offsets-argument seam;
    that seam is for when the offset itself is what varies, which it is not
    here.
    """
    dP, dAr, dD = affect.tier_offset(tier)
    ramp = (dP, dAr, dD)
    J = jacobian(assignment, robot, ramp=ramp, space="axis", at=(0.0, 0.0, 1.0))
    # columns: 0=valence, 1=arousal, 2=tier(ramp scalar s) -- only 0,1 carry
    # camera noise.
    out = {}
    for i, axis in enumerate(AXES):
        var = (J[i, 0] * SIGMA_V) ** 2 + (J[i, 1] * SIGMA_A) ** 2
        out[axis] = float(np.sqrt(var) / np.sqrt(window))
    return out


def monte_carlo_std(assignment: AxisAssignment, robot: str, tier: str = "known",
                    n: int = 20000, window: int = WINDOW, seed: int = 0
                    ) -> Dict[str, float]:
    """Same question, answered by running the REAL AffectStream + the REAL
    compose_offsets over many independent noisy camera draws -- not a
    linearisation of them."""
    rng = np.random.default_rng(seed)
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    offsets = affect.tier_offset(tier)
    samples = {axis: [] for axis in AXES}
    for _ in range(n):
        stream = AffectStream(smoothing_window=window)
        # fill the window with independent noisy draws around a fixed true
        # signal (0,0) -- a steady-state face, camera noise only.
        for _ in range(window):
            v_raw = rng.normal(0.0, SIGMA_V)
            a_raw = rng.normal(0.0, SIGMA_A)
            v, a = stream.update(v_raw, a_raw)
        felt, _flags = compose_offsets(assignment, baseline, v, a, offsets)
        for axis in AXES:
            samples[axis].append(felt[axis])
    return {axis: float(np.std(samples[axis])) for axis in AXES}


def report(robot: str = "CHATBOX", tier: str = "known",
          assignments: Sequence[str] = ("identity", "perm_emotion_D",
                                        "perm_arousal_D")) -> List[Dict]:
    rows = []
    for name in assignments:
        a = ASSIGNMENTS[name]
        cf = closed_form_std(a, robot, tier)
        mc = monte_carlo_std(a, robot, tier)
        for axis in AXES:
            rows.append({"assignment": name, "robot": robot, "tier": tier,
                        "axis": axis, "closed_form_std": cf[axis],
                        "monte_carlo_std": mc[axis],
                        "agree": abs(cf[axis] - mc[axis]) < 0.01})
    return rows

def switch_boundary_clustering(assignment: AxisAssignment, robot: str,
                               sigma: float = 0.20, dwell_s: float = 4.0,
                               fps: float = 30.0, near_s: float = 0.5,
                               seed: int = 1) -> Dict[str, float]:
    """What fraction of E3-measured rung switches occur near a SCRIPTED
    schedule transition, vs mid-block (i.e. genuinely noise-driven)?

    This is the check that resolves an apparent tension in this module: the
    MARGINAL std propagated above (a static "how far does D typically sit from
    its mean" quantity) can be smaller than a rung gap while E3 still measures
    frequent switching, because switching COUNT is a level-CROSSING-RATE
    question, which depends on the smoothing filter's autocorrelation
    timescale, not on the marginal variance alone. A low-marginal-variance
    process can still cross a nearby boundary often if it moves slowly and
    lingers near it (clustered crossings) rather than jumping cleanly across
    once. If most switches cluster at the scripted transitions, the marginal
    std correctly says "not much is happening between transitions"; if most
    switches are mid-block, real ongoing noise-driven crossing is happening at
    a rate the marginal std under-signals on its own.
    """
    from padeval.analysis.e3_permute import rung_of
    from padeval.traces import synth_trace

    trace = synth_trace(sigma=sigma, seed=seed, dwell_s=dwell_s, fps=fps)
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    stream = AffectStream(smoothing_window=WINDOW)
    rungs = []
    for v, a in trace.va:
        sv, sa = stream.update(float(v), float(a))
        felt, _flags = compose_offsets(
            assignment, baseline, sv, sa, affect.tier_offset("known"))
        rungs.append(rung_of(felt["D"]))
    arr = np.asarray(rungs, dtype=int)
    switch_idx = np.flatnonzero(np.diff(arr)) + 1
    block_frames = int(dwell_s * fps)
    boundaries = set(range(block_frames, len(arr), block_frames))
    near = int(near_s * fps)
    if len(switch_idx) == 0:
        return {"n_switches": 0, "near_boundary_frac": float("nan")}
    n_near = sum(1 for i in switch_idx
                if boundaries and min(abs(int(i) - b) for b in boundaries) <= near)
    return {"n_switches": int(len(switch_idx)),
            "near_boundary_frac": n_near / len(switch_idx)}
