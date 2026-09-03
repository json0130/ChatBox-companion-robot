"""
7b gate: persona admissibility generalized to the trait population.
Analytic + Monte Carlo, cross-checked. No LLM, no camera.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect                          # noqa: E402
from padeval.analysis.persona_admissibility import (              # noqa: E402
    ACTIVE_TRAITS, DEPLOYED_M, D_WEIGHTS, admissible_fraction_exact,
    admissible_fraction_mc, baseline_d, max_abs_d, safe_e_range,
    violation_direction, worst_corner,
)


def test_neuroticism_has_no_effect_on_admissibility():
    """The 5-D question is exactly the 4-D question. Pinned because it is a
    directly usable design fact, and because if a future WEIGHTS change adds an
    N term this analysis silently becomes incomplete."""
    assert "N" not in D_WEIGHTS, D_WEIGHTS
    assert set(ACTIVE_TRAITS) == {"O", "C", "E", "A"}
    base = {"O": 0.3, "C": -0.2, "E": 0.5, "A": -0.4, "N": 0.0}
    for n_val in (-1.0, 0.0, 1.0):
        t = dict(base, N=n_val)
        assert abs(baseline_d(t) - baseline_d(base)) < 1e-12
        assert abs(affect.to_pad(t)["D"] - affect.to_pad(base)["D"]) < 1e-12
    print("1. Neuroticism has zero weight on D — admissibility is a 4-D "
          "question, confirmed against live to_pad ✓")


def test_monte_carlo_matches_exact_convolution():
    """Two independent routes. Disagreement means one is wrong."""
    for m in (0.10, 0.25, 0.40, 0.60):
        p, lo, hi = admissible_fraction_mc(m, n=200_000, seed=11)
        exact = admissible_fraction_exact(m)
        assert lo - 0.005 <= exact <= hi + 0.005, (m, p, lo, hi, exact)
    print("2. Monte Carlo and exact convolution agree at every swept m ✓")


def test_admissible_fraction_decreases_with_m():
    fracs = [admissible_fraction_exact(m) for m in
             (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)]
    assert all(a > b for a, b in zip(fracs, fracs[1:])), fracs
    print("3. admissible fraction decreases monotonically as m grows ✓")


def test_deployed_personas_match_6c():
    """Consistency with the per-persona result 6c already reported."""
    bound = 1.0 - DEPLOYED_M
    d_chatbox = affect.to_pad(affect.ROBOTS["CHATBOX"]["ocean"])["D"]
    d_ellebot = affect.to_pad(affect.ROBOTS["ELLEBOT"]["ocean"])["D"]
    assert abs(d_chatbox) > bound, d_chatbox          # clamps, as 6c found
    assert abs(abs(d_chatbox) - bound - 0.043) < 1e-9  # by exactly 0.043
    assert abs(d_ellebot) < bound, d_ellebot          # does not clamp
    print("4. CHATBOX exceeds the bound by exactly 0.043, ELLEBOT is inside — "
          "matches 6c ✓")


def test_violation_is_two_sided_not_just_high_E():
    """The correction. The intuited 'high E, low A' profile is only the UPPER
    violation; the mirror (low E, high A) violates the lower bound equally
    often, and the deployed system's clamping persona is an instance of the
    MIRROR, not the intuited case."""
    v = violation_direction()
    assert abs(v["p_over_assertive"] - v["p_over_deferential"]) < 0.01, v
    hi = v["profile_over_assertive"]
    lo = v["profile_over_deferential"]
    assert hi["E"] > 0.5 and hi["A"] < -0.3, hi     # intuited profile
    assert lo["E"] < -0.5 and lo["A"] > 0.3, lo     # its mirror
    # and CHATBOX is the mirror case
    o = affect.ROBOTS["CHATBOX"]["ocean"]
    assert o["E"] < 0 and o["A"] > 0
    assert affect.to_pad(o)["D"] < 0
    print("5. violation is two-sided and symmetric; CHATBOX clamps for being "
          "too DEFERENTIAL, the mirror of the intuited case ✓")


def test_worst_corner_is_the_weight_signs():
    w = worst_corner()
    assert w["traits"] == {"O": 1.0, "C": 1.0, "E": 1.0, "A": -1.0}
    assert abs(w["D"] - max_abs_d()) < 1e-12
    assert not w["admissible"]
    print(f"6. worst corner |D| = {max_abs_d():.2f}, far outside any bound ✓")


def test_safe_e_range_is_consistent_with_the_bound():
    """Spot-check the design rule against direct evaluation."""
    for a_val in (-1.0, 0.0, 0.6):
        lo, hi = safe_e_range(a_val, m=DEPLOYED_M)
        for e in (lo + 1e-6, 0.0 if lo < 0 < hi else (lo + hi) / 2, hi - 1e-6):
            if not (lo <= e <= hi):
                continue
            d = baseline_d({"O": 0.0, "C": 0.0, "E": e, "A": a_val})
            assert abs(d) <= (1.0 - DEPLOYED_M) + 1e-9, (a_val, e, d)
    print("7. safe_e_range agrees with direct evaluation of the bound ✓")


if __name__ == "__main__":
    test_neuroticism_has_no_effect_on_admissibility()
    test_monte_carlo_matches_exact_convolution()
    test_admissible_fraction_decreases_with_m()
    test_deployed_personas_match_6c()
    test_violation_is_two_sided_not_just_high_E()
    test_worst_corner_is_the_weight_signs()
    test_safe_e_range_is_consistent_with_the_bound()
    print("\npersona admissibility is a design rule, not an anecdote.")
