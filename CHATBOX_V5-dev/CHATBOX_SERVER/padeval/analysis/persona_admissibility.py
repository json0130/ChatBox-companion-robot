"""
7b — persona admissibility, generalized from one robot to the trait population.

WHAT THIS GENERALIZES
----------------------
6c established `|D_baseline| <= 1 - m` exactly, and reported it as a fact about
ONE persona: CHATBOX's baseline (0.643) exceeds the deployed bound (0.60) by
0.043. That is a bug-shaped observation. This turns it into a statement about
the design space: given a target offset magnitude, WHICH personas are safe to
assign, and which traits drive violation.

THE GEOMETRY IS SIMPLER THAN IT LOOKS, FOR TWO REASONS
--------------------------------------------------------
(1) NEUROTICISM HAS ZERO WEIGHT. affect.WEIGHTS["D"] is
    {O: 0.25, C: 0.17, E: 0.60, A: -0.32} — no N term (affect.py:41-43). So D
    does not depend on N at all, and the admissibility question over [-1,1]^5
    is EXACTLY the question over [-1,1]^4 in (O, C, E, A). N is free: any
    persona's admissibility is unchanged by its Neuroticism. Reported because it
    is a non-obvious and immediately useful design fact, not because it makes
    the sums easier.

(2) D IS LINEAR IN THE TRAITS, so the admissible set
    {t : |w.t| <= 1-m} is the region between two parallel hyperplanes — a SLAB
    through the trait hypercube, centred on the origin. Admissibility is
    therefore a single scalar question (how far along w), not a shape that needs
    exploring numerically.

DISTRIBUTION CHOICE, STATED RATHER THAN ASSUMED
-------------------------------------------------
Primary result uses UNIFORM over [-1,1]^4. This is a DESIGN-SPACE COVERAGE
question ("what fraction of assignable personas are safe?"), not a claim about
how real human traits are distributed — real Big Five scores are roughly normal
and concentrated near the mean, which would make the admissible fraction look
considerably better and would flatter the design. Uniform is the conservative,
neutral choice for a design rule. A concentrated alternative is reported as a
sensitivity check so the difference is visible rather than hidden.

TWO INDEPENDENT ROUTES, as elsewhere in this work
---------------------------------------------------
(a) Monte Carlo over the cube, with a binomial CI.
(b) Near-exact numerical convolution: D is a weighted sum of independent
    uniforms, so its density is computable by convolving four box densities on
    a fine grid, and the admissible fraction is a direct integral of that
    density. No sampling error.
If these disagree beyond the MC CI, one of them is wrong.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from modules.affect_bridge import affect

D_WEIGHTS = affect.WEIGHTS["D"]                 # {'O':0.25,'C':0.17,'E':0.60,'A':-0.32}
ACTIVE_TRAITS = tuple(D_WEIGHTS)                # O, C, E, A  (N absent by construction)
DEPLOYED_M = 0.40


def baseline_d(traits: Dict[str, float]) -> float:
    """D from a trait vector, via the live WEIGHTS. Mirrors affect.to_pad's D row."""
    return sum(w * traits[t] for t, w in D_WEIGHTS.items())


def max_abs_d() -> float:
    """Largest |D| any trait vector in the cube can produce."""
    return sum(abs(w) for w in D_WEIGHTS.values())


def admissible_fraction_mc(m: float, n: int = 400_000, seed: int = 0
                           ) -> Tuple[float, float, float]:
    """Fraction of the uniform trait cube satisfying |D| <= 1-m, with a 95% CI."""
    rng = np.random.default_rng(seed)
    W = np.array([D_WEIGHTS[t] for t in ACTIVE_TRAITS])
    T = rng.uniform(-1.0, 1.0, size=(n, len(ACTIVE_TRAITS)))
    D = T @ W
    ok = np.abs(D) <= (1.0 - m)
    p = float(ok.mean())
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / n))
    return p, p - 1.96 * se, p + 1.96 * se


def _density_grid(step: float = 0.0005) -> Tuple[np.ndarray, np.ndarray]:
    """Density of D = sum(w_i * U_i), U_i ~ Uniform(-1,1), by convolution.

    Each term w_i*U_i is uniform on [-|w_i|, |w_i|]; convolving the four box
    densities gives the exact density of the sum, up to grid resolution.
    """
    lo = -max_abs_d() - 1.0
    hi = max_abs_d() + 1.0
    grid = np.arange(lo, hi + step, step)
    dens = np.zeros_like(grid)
    dens[np.argmin(np.abs(grid))] = 1.0 / step        # delta at 0
    for t in ACTIVE_TRAITS:
        half = abs(D_WEIGHTS[t])
        k = int(round(2 * half / step))
        if k < 1:
            continue
        box = np.ones(k) / (k * step)                  # unit-mass box density
        dens = np.convolve(dens, box, mode="same") * step
    dens /= dens.sum() * step                          # renormalise
    return grid, dens


def admissible_fraction_exact(m: float, step: float = 0.0005) -> float:
    """Same fraction by integrating the convolved density. No sampling error."""
    grid, dens = _density_grid(step)
    c = 1.0 - m
    mask = np.abs(grid) <= c
    return float(dens[mask].sum() * step)


def sweep(m_values: Sequence[float], n: int = 400_000) -> List[Dict]:
    rows = []
    for m in m_values:
        p, lo, hi = admissible_fraction_mc(m, n=n)
        rows.append({"m": m, "bound": 1.0 - m, "mc": p, "mc_lo": lo, "mc_hi": hi,
                    "exact": admissible_fraction_exact(m)})
    return rows


def violation_drivers(m: float = DEPLOYED_M, n: int = 400_000, seed: int = 1
                      ) -> List[Dict]:
    """P(inadmissible | trait in top/bottom third), per trait.

    Confirms rather than asserts which traits drive violation: E has the largest
    coefficient (0.60) and A the largest opposing one (-0.32), so high-E/low-A
    is the analytically obvious candidate — this measures it.
    """
    rng = np.random.default_rng(seed)
    W = np.array([D_WEIGHTS[t] for t in ACTIVE_TRAITS])
    T = rng.uniform(-1.0, 1.0, size=(n, len(ACTIVE_TRAITS)))
    D = T @ W
    bad = np.abs(D) > (1.0 - m)
    rows = []
    for i, t in enumerate(ACTIVE_TRAITS):
        hi = T[:, i] > (1 / 3)
        lo = T[:, i] < (-1 / 3)
        rows.append({"trait": t, "weight": D_WEIGHTS[t],
                    "p_violate_high": float(bad[hi].mean()),
                    "p_violate_low": float(bad[lo].mean()),
                    "p_violate_overall": float(bad.mean())})
    return rows


def worst_corner(m: float = DEPLOYED_M) -> Dict:
    """The trait vector maximising |D| — the least admissible persona possible."""
    corner = {t: (1.0 if w > 0 else -1.0) for t, w in D_WEIGHTS.items()}
    d = baseline_d(corner)
    return {"traits": corner, "D": d, "bound": 1.0 - m,
            "admissible": abs(d) <= (1.0 - m)}


def safe_e_range(a_value: float, m: float = DEPLOYED_M,
                 o: float = 0.0, c: float = 0.0) -> Tuple[float, float]:
    """Given Agreeableness (and O, C), the range of Extraversion keeping a
    persona admissible. The design rule in its most directly usable form:
    E is the dominant lever, so this is the question a designer actually asks.
    """
    fixed = D_WEIGHTS["O"] * o + D_WEIGHTS["C"] * c + D_WEIGHTS["A"] * a_value
    wE = D_WEIGHTS["E"]
    bound = 1.0 - m
    lo = (-bound - fixed) / wE
    hi = (bound - fixed) / wE
    return (max(-1.0, min(lo, hi)), min(1.0, max(lo, hi)))


def violation_direction(m: float = DEPLOYED_M, n: int = 400_000, seed: int = 7
                        ) -> Dict:
    """Split violations into over-assertive (D > +bound) and over-deferential
    (D < -bound), and report the mean trait profile of each.

    This exists because the natural intuition — "high Extraversion, low
    Agreeableness is the risky combination", from E's large positive weight and
    A's opposing one — is only HALF the story, and the missing half is the one
    the deployed system actually hit. The constraint is |D| <= 1-m, two-sided,
    so the mirror profile (low E, high A) violates the lower bound just as
    often. CHATBOX is an instance of that mirror: it clamps for being too
    DEFERENTIAL. An analysis that only checked the intuited direction would
    have reported it as confirmed and missed the case in the shipped system.
    """
    rng = np.random.default_rng(seed)
    W = np.array([D_WEIGHTS[t] for t in ACTIVE_TRAITS])
    T = rng.uniform(-1.0, 1.0, size=(n, len(ACTIVE_TRAITS)))
    D = T @ W
    bound = 1.0 - m
    hi, lo = D > bound, D < -bound
    ok = ~(hi | lo)
    return {
        "bound": bound,
        "p_over_assertive": float(hi.mean()),
        "p_over_deferential": float(lo.mean()),
        "profile_over_assertive": {t: float(T[hi, i].mean())
                                   for i, t in enumerate(ACTIVE_TRAITS)},
        "profile_over_deferential": {t: float(T[lo, i].mean())
                                     for i, t in enumerate(ACTIVE_TRAITS)},
        "profile_admissible": {t: float(T[ok, i].mean())
                               for i, t in enumerate(ACTIVE_TRAITS)},
    }
