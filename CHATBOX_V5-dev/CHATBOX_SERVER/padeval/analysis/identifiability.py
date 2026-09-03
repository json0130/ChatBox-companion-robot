"""
7a — the general identifiability theorem, and the complete assignment table.

WHAT THIS REPLACES
-------------------
E2 previously read as: "we checked four assignments; three were fine and one was
not." That is a spot-check. This module states and verifies the general fact of
which those four are instances, so the result reads as a characterization rather
than a coincidence.

THE TWO STATEMENTS (registered as the prediction; the check is that the code
agrees with them, not the other way round)
------------------------------------------------------------------------------
Let an assignment route each of the three input sources (valence, arousal, tier)
onto one of the three PAD axes (P, Ar, D). Let

    T = { valence_axis, arousal_axis, tier_axis }        (as a SET)
    j = 3 - |T|                                          (axes never written)

**Theorem 1 (freezing bounds rank).** If j axes are frozen at baseline — receive
zero contribution from every source — then rank(J) <= 3 - j, for J the axis-space
Jacobian d(P,Ar,D)/d(valence,arousal,tier), regardless of how the remaining
(3-j) axes are driven.

  Proof. A frozen axis is a constant function of the inputs, so its Jacobian row
  is identically zero (chain rule; not an approximation, and independent of step
  size — this is exactly why 6c's finite-difference rank detection was unreliable
  for the degenerate case and the structural argument was authoritative). J has
  at most 3-j nonzero rows, hence rank <= 3-j. QED

**Theorem 2 (permutations preserve rank).** If the assignment is a bijection —
every source to a distinct axis, no axis frozen — then rank(J) = 3.

  Proof. Under a bijection each axis is written by exactly one source, with gain
  `empathy` for the two face sources and the routed offset magnitude for tier.
  J is then a permutation matrix scaled by a nonzero diagonal, which is
  invertible by construction whenever empathy != 0 and the offset magnitude != 0.
  Hence full rank. QED

**Corollary (the whole E2 table).** `identity`, `perm_emotion_D` and
`perm_arousal_D` are bijections, so rank 3 by Theorem 2. `collapse_D` writes
only one axis (|T|=1, j=2), so rank <= 1 by Theorem 1, and exactly 1 since that
one row is nonzero. Every previously-measured value is now predicted, not
observed.

WHAT THIS DOES *NOT* ESTABLISH — stated explicitly so 7a's cleanliness cannot
overstate it
-----------------------------------------------------------------------------
Identifiability is NECESSARY but NOT SUFFICIENT for correctness. All six
bijections are equally full-rank and therefore equally identifiable by
Theorem 2. Nothing here singles out `identity` as the right one. The argument
that does is E3's: timescale separation and setpoint stability (a per-frame
source must not drive a per-session effector). The structure of the overall
claim is two-part and must be reported that way:

    collapse is ruled out by IDENTIFIABILITY alone (this module);
    among the identifiable bijections, `identity` is singled out by DYNAMICS
    (E3, and 7.0's noise propagation).

A SEMANTIC SUBTLETY IN THE NON-BIJECTIVE CASES, found by reading the code rather
than assuming
------------------------------------------------------------------------------
`axes.compose_offsets` assigns `felt[valence_axis]` and then `felt[arousal_axis]`
SEQUENTIALLY. When those are the same axis, the second assignment OVERWRITES the
first, so valence is discarded entirely rather than mixed with arousal. These
cases therefore mean "one source is silently dropped", not "two sources share an
axis". The rank arithmetic is unaffected (a dropped source contributes a zero
COLUMN, a frozen axis a zero ROW — both reduce rank), but the interpretation
differs, and only the explicit `collapse=True` branch actually averages sources.
Noted because a reader would otherwise reasonably assume duplicate-axis routing
means mixing.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from modules.affect_bridge import affect
from padeval.analysis.e2_identify import jacobian
from padeval.axes import AXES, AxisAssignment

DEPLOYED_TIER = "close"     # nonzero offset, so the tier column is live


def analytic_axis_jacobian(a: AxisAssignment,
                           offsets: Tuple[float, float, float],
                           empathy: float = None) -> np.ndarray:
    """The 3x3 axis-space Jacobian in closed form.

    Derived from `axes.compose_offsets` directly rather than estimated: the map
    is affine in (valence, arousal, s) up to the clamp, so the Jacobian is
    exact and constant. Having this lets the numerical result be checked
    against something that cannot suffer the finite-difference failure mode
    that misreported collapse_D's rank in 6c.
    """
    from padeval.axes import route_offsets
    e = affect.EMPATHY if empathy is None else empathy
    J = np.zeros((3, 3))
    idx = {ax: i for i, ax in enumerate(AXES)}

    if a.collapse:
        # felt[tier_axis] = base + e*((v+a)/2 - base): both face sources at e/2.
        r = idx[a.tier_axis]
        J[r, 0] = e / 2.0
        J[r, 1] = e / 2.0
    else:
        # Sequential assignment: arousal is written second, so if the two share
        # an axis, valence's contribution is overwritten and vanishes.
        if a.valence_axis != a.arousal_axis:
            J[idx[a.valence_axis], 0] = e
        J[idx[a.arousal_axis], 1] = e

    # tier column: the routed offset, differentiated w.r.t. the ramp scalar s.
    deltas = route_offsets(a, offsets)
    for ax in AXES:
        J[idx[ax], 2] = deltas[ax]
    return J


def predicted_rank(a: AxisAssignment) -> int:
    """Theorem 1 + Theorem 2, as a computation on the assignment's structure.

    |T| = number of DISTINCT axes written. Frozen axes contribute zero rows;
    the written rows are linearly independent in every case here because each
    carries a distinct combination of source gains (verified numerically
    against the analytic Jacobian below, not assumed).
    """
    if a.collapse:
        return 1                      # only tier_axis is ever written
    return len({a.valence_axis, a.arousal_axis, a.tier_axis})


def enumerate_assignments() -> List[Tuple[str, AxisAssignment]]:
    """Every structurally distinct assignment: 27 non-collapse + 3 collapse."""
    out = []
    for v, ar, t in itertools.product(AXES, repeat=3):
        distinct = len({v, ar, t})
        if distinct == 3:
            cls = "bijection"
        elif distinct == 2:
            cls = "one-axis-frozen"
        else:
            cls = "two-axes-frozen"
        name = f"v{v}_a{ar}_t{t}"
        out.append((cls, AxisAssignment(name, v, ar, t)))
    for t in AXES:
        out.append(("collapse", AxisAssignment(f"collapse_{t}", t, t, t,
                                               collapse=True)))
    return out


def measured_rank(a: AxisAssignment, robot: str = "CHATBOX",
                  tier: str = DEPLOYED_TIER, h: float = 1e-3) -> Dict:
    """Rank from BOTH the analytic Jacobian and finite differences.

    Reporting both is the point: 6c showed finite-difference rank detection can
    silently misreport a degenerate case. If the two disagree anywhere, that is
    the finite-difference method failing, not the theorem — and it should be
    visible rather than smoothed over.
    """
    offsets = affect.tier_offset(tier)
    Ja = analytic_axis_jacobian(a, offsets)
    Jn = jacobian(a, robot, ramp=offsets, space="axis", h=h, at=(0.0, 0.0, 1.0))

    def _rank(M):
        sv = np.linalg.svd(M, compute_uv=False)
        tol = max(M.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
        return int(np.count_nonzero(sv > max(tol, 1e-9)))

    return {"analytic_rank": _rank(Ja), "numeric_rank": _rank(Jn),
            "max_abs_diff": float(np.max(np.abs(Ja - Jn)))}


def full_table(robot: str = "CHATBOX", tier: str = DEPLOYED_TIER) -> List[Dict]:
    rows = []
    for cls, a in enumerate_assignments():
        m = measured_rank(a, robot, tier)
        pred = predicted_rank(a)
        rows.append({"class": cls, "name": a.name, "predicted": pred,
                    "analytic": m["analytic_rank"], "numeric": m["numeric_rank"],
                    "jacobian_max_diff": m["max_abs_diff"],
                    "agree": pred == m["analytic_rank"] == m["numeric_rank"]})
    return rows
