"""
E2 (analytic half) — is the source of a behaviour recoverable from what the
robot emits?

WHY THIS IS ALGEBRA AND NOT A CLASSIFIER
----------------------------------------
The style vector is a closed-form function of the felt coordinate
(`affect.gesture_style`), so a classifier trained on noiseless style data scores
100% and has demonstrated nothing. The real question is about the FORWARD MAP:
does it preserve the distinction between sources, or fold them together?

That is a question about a Jacobian, and it is answerable exactly — no dataset,
no seed, no noise assumption, no train/test split.

    J = d[amplitude, tempo, posture, droop, idle] / d[valence, arousal, tier]     (5x3)

The relationship is a discrete factor, so a derivative with respect to it is
undefined as stated. It is made continuous by ramping the tier offset from the
`known` reference (0,0,0) to a target tier's offset along a scalar s in [0,1],
which is exactly the path the system traverses as a relationship deepens.

TWO STRUCTURAL FACTS TO STATE RATHER THAN DISCOVER
--------------------------------------------------
By inspection of affect.gesture_style (affect.py:347-359):

    droop = -1.4 * P            depends on PLEASURE alone
    idle  = 0.45 + 0.4 * Ar     depends on AROUSAL alone
    amplitude = f(D)            depends on DOMINANCE alone   (Eq. 9)
    tempo, posture              mixed

So under the identity assignment the Jacobian is near block-diagonal BY
CONSTRUCTION. Presenting that as an empirical finding would be circular. What is
worth measuring is (a) how much energy sits off the block structure, (b) how the
condition number degrades under permutation, and (c) whether the metric is
sensitive enough to resolve a known, deliberately-introduced violation.

(c) is why the `close`-tier +0.10 arousal leak is kept rather than fixed. `idle`
is a pure function of arousal, so ANY tier -> idle sensitivity is arousal leakage
and is exactly quantifiable. The leak/no-leak pair is the metric's calibration.

numpy only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from modules.affect_bridge import affect
from padeval.axes import ASSIGNMENTS, AXES, AxisAssignment, compose_offsets

STYLE_KEYS = ("amplitude", "tempo", "posture", "droop", "idle")
SOURCES = ("valence", "arousal", "tier")

# The relationship ramp: `known` (the reference tier, zero displacement) -> the
# target. Deployed values come from affect.TIER_OFFSETS.
LEAK_ON = affect.TIER_OFFSETS["close"]        # (0.00, +0.10, +0.40)
LEAK_OFF = (LEAK_ON[0], 0.0, LEAK_ON[2])      # the disjoint-axis counterfactual


def _felt(assignment: AxisAssignment, baseline, v, a, s, ramp):
    offsets = tuple(s * c for c in ramp)
    felt, _flags = compose_offsets(assignment, baseline, v, a, offsets)
    return felt


def _style_vec(assignment: AxisAssignment, baseline, v, a, s, ramp) -> np.ndarray:
    style = affect.gesture_style(_felt(assignment, baseline, v, a, s, ramp))
    return np.array([style[k] for k in STYLE_KEYS], dtype=float)


def _axis_vec(assignment: AxisAssignment, baseline, v, a, s, ramp) -> np.ndarray:
    felt = _felt(assignment, baseline, v, a, s, ramp)
    return np.array([felt[k] for k in AXES], dtype=float)


def jacobian(assignment: AxisAssignment,
             robot: str,
             at: Tuple[float, float, float] = (0.0, 0.0, 0.5),
             ramp: Tuple[float, float, float] = LEAK_ON,
             space: str = "style",
             h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of the real pipeline at one operating point.

    `space="style"` gives the 5x3 effector Jacobian; `space="axis"` gives the 3x3
    source -> PAD-axis routing Jacobian, which is where leakage is defined.

    `at` is (valence, arousal, s). The default s=0.5 sits mid-ramp so the
    derivative is taken away from the clamp and away from the ramp endpoints.
    """
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    f = _style_vec if space == "style" else _axis_vec
    point = list(at)
    cols = []
    for i in range(3):
        hi, lo = list(point), list(point)
        hi[i] += h
        lo[i] -= h
        cols.append((f(assignment, baseline, *hi, ramp)
                     - f(assignment, baseline, *lo, ramp)) / (2 * h))
    return np.column_stack(cols)          # rows = outputs, cols = sources


def leakage_matrix(assignment: AxisAssignment, robot: str,
                   ramp: Tuple[float, float, float] = LEAK_ON,
                   at: Tuple[float, float, float] = (0.0, 0.0, 0.5),
                   ) -> np.ndarray:
    """|d(axis)/d(source)|, 3x3, rows = P/Ar/D, cols = valence/arousal/tier.

    Under a clean assignment this is diagonal up to the source's own gain: each
    source writes exactly one axis. Off-diagonal mass IS the leakage.
    """
    return np.abs(jacobian(assignment, robot, at=at, ramp=ramp, space="axis"))


def conditioning(J: np.ndarray) -> Dict[str, float]:
    """Singular values, condition number and numerical rank of a Jacobian.

    A rank-deficient J means two or more sources move the outputs along the SAME
    direction, so no observer — however clever, however noise-free — can tell
    which one moved. That is unidentifiability proved rather than estimated.
    """
    sv = np.linalg.svd(J, compute_uv=False)
    tol = max(J.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    rank = int(np.count_nonzero(sv > tol))
    smallest = float(sv[-1]) if sv.size else 0.0
    cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > tol else float("inf")
    return {"sv": [float(x) for x in sv],
            "sigma_max": float(sv[0]) if sv.size else 0.0,
            "sigma_min": smallest,
            "rank": rank,
            "cond": cond}


def analyse(robot: str = "CHATBOX",
            assignments: Sequence[str] = ("identity", "perm_emotion_D",
                                          "perm_arousal_D", "collapse_D"),
            ramp: Tuple[float, float, float] = LEAK_ON,
            ramp_label: str = "leak_on") -> List[Dict]:
    """One row per assignment: conditioning of the style map plus leakage."""
    rows = []
    for name in assignments:
        a = ASSIGNMENTS[name]
        J = jacobian(a, robot, ramp=ramp, space="style")
        L = leakage_matrix(a, robot, ramp=ramp)
        cond = conditioning(J)
        # Arousal leakage from the relationship, read off `idle`, which is a pure
        # function of arousal (idle = 0.45 + 0.4*Ar) — so d(idle)/d(tier) / 0.4
        # IS d(Ar)/d(tier), with no attribution ambiguity.
        idle_row = STYLE_KEYS.index("idle")
        tier_col = SOURCES.index("tier")
        rows.append({
            "robot": robot,
            "assignment": name,
            "ramp": ramp_label,
            "rank": cond["rank"],
            "cond": cond["cond"],
            "sigma_min": cond["sigma_min"],
            "sv": cond["sv"],
            "d_idle_d_tier": float(J[idle_row, tier_col]),
            "d_arousal_d_tier": float(J[idle_row, tier_col] / 0.4),
            "leakage": L.tolist(),
        })
    return rows


def format_report(rows_on: Sequence[Dict], rows_off: Sequence[Dict]) -> str:
    out: List[str] = []
    out.append("### Conditioning of the style map  d(5 style)/d(3 sources)\n")
    out.append("| assignment | rank | cond number | sigma_min | verdict |")
    out.append("|---|---|---|---|---|")
    for r in rows_on:
        verdict = ("**sources unidentifiable**" if r["rank"] < 3
                   else "sources separable")
        cond = "inf" if not np.isfinite(r["cond"]) else f"{r['cond']:.1f}"
        out.append(f"| `{r['assignment']}` | {r['rank']}/3 | {cond} "
                   f"| {r['sigma_min']:.3e} | {verdict} |")

    out.append(
        "\n**What this does and does not show.** Rank is the strong result: under "
        "`collapse_D` the map is rank-deficient, so two sources move the effectors "
        "along the same direction and NO observer can tell which one moved — "
        "unidentifiability proved, not estimated, with no dataset and no noise "
        "assumption.\n\n"
        "The condition number does **not** favour the deployed assignment, and "
        "saying otherwise would be overclaiming: `perm_emotion_D` is better "
        "conditioned here than `identity`. Conditioning separates the degenerate "
        "assignment from the non-degenerate ones; it does not rank the "
        "non-degenerate ones against each other.\n\n"
        "> **The one sentence to carry into the paper:** Jacobian rank and "
        "conditioning establish that `identity` and `perm_emotion_D` are both "
        "well-posed; the case for `identity` over `perm_emotion_D` is made "
        "empirically in E2's noise sweep and structurally in E3's setpoint "
        "stability, **not by this table**.\n\n"
        "Without that stated, a reader pattern-matches \"lower condition number "
        "is better\" out of habit, sees `perm_emotion_D` winning on a metric this "
        "table was never built to rank assignments with, and the paper ends up "
        "litigating a comparison it never made."
    )

    out.append("\n### Leakage calibration — does the metric resolve a known "
               "0.10 violation?\n")
    out.append("`idle = 0.45 + 0.4*Ar` is a pure function of arousal "
               "(affect.py:355), so any tier->idle sensitivity is arousal "
               "leakage and is exactly attributable — no inference required.\n")
    out.append("| assignment | d(Ar)/d(tier) with leak | with leak zeroed | resolved |")
    out.append("|---|---|---|---|")
    off = {r["assignment"]: r for r in rows_off}
    for r in rows_on:
        o = off[r["assignment"]]
        delta = abs(r["d_arousal_d_tier"] - o["d_arousal_d_tier"])
        if delta > 1e-9:
            note = f"**yes, {delta:.4f}**"
        elif abs(r["d_arousal_d_tier"]) > 1e-9:
            note = "n/a — tier owns arousal here"
        else:
            note = "n/a — nothing writes arousal here"
        out.append(f"| `{r['assignment']}` | {r['d_arousal_d_tier']:+.4f} "
                   f"| {o['d_arousal_d_tier']:+.4f} | {note} |")

    out.append(
        "\nThe `identity` row is the calibration: the metric reads exactly "
        "**+0.1000** with the shipped `close`-tier offset and exactly **0.0000** "
        "with that one term zeroed, recovering the injected violation to four "
        "decimal places. So the leakage number is a measurement with a known "
        "response, not a single reading taken on faith.\n\n"
        "The two `n/a` rows are correct, not failures. Under `perm_arousal_D` the "
        "relationship's PRINCIPAL displacement is routed to arousal, so "
        "d(Ar)/d(tier) = 0.40 by design and the 0.10 secondary term is not "
        "separable from it. Under `collapse_D` nothing writes arousal at all."
    )
    return "\n".join(out)
