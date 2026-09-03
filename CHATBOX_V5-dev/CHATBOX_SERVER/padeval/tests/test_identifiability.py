"""
7a gate: the identifiability theorem holds over the COMPLETE assignment space,
by two independent routes, and does not overclaim.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from modules.affect_bridge import affect                        # noqa: E402
from padeval.analysis.identifiability import (                  # noqa: E402
    analytic_axis_jacobian, enumerate_assignments, full_table, predicted_rank,
)
from padeval.axes import ASSIGNMENTS                            # noqa: E402


def test_space_is_completely_enumerated():
    """27 non-collapse (3 sources x 3 axes) + 3 collapse variants."""
    cases = enumerate_assignments()
    assert len(cases) == 30, len(cases)
    assert sum(1 for c, _ in cases if c == "bijection") == 6
    assert sum(1 for c, _ in cases if c == "collapse") == 3
    print("1. all 30 structurally distinct assignments enumerated "
          "(27 non-collapse + 3 collapse) ✓")


def test_theorem_predicts_every_case():
    """The headline. Every row must agree, by BOTH the analytic Jacobian and
    finite differences — if any disagreed, the theorem or the measurement would
    be wrong and that matters more than any other result in this subphase."""
    rows = full_table()
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"{len(bad)} disagreement(s): {bad}"
    print(f"2. theorem predicts all {len(rows)}/{len(rows)} cases, confirmed by "
          "analytic AND numeric rank ✓")


def test_analytic_and_numeric_jacobians_match_to_machine_precision():
    """Guards against the 6c failure mode recurring silently: if finite
    differences ever drift from the closed form, this catches it before a rank
    number is trusted."""
    rows = full_table()
    worst = max(r["jacobian_max_diff"] for r in rows)
    assert worst < 1e-9, worst
    print(f"3. analytic vs numeric Jacobian agree to {worst:.1e} "
          "(machine precision) across all 30 ✓")


def test_theorem_1_freezing_bounds_rank():
    """rank <= 3 - j, for j = number of axes never written."""
    for _cls, a in enumerate_assignments():
        if a.collapse:
            written = {a.tier_axis}
        else:
            written = {a.valence_axis, a.arousal_axis, a.tier_axis}
        j = 3 - len(written)
        assert predicted_rank(a) <= 3 - j, (a.name, j, predicted_rank(a))
    print("4. Theorem 1 holds: rank <= 3 - (frozen axes), every case ✓")


def test_theorem_2_every_bijection_is_full_rank():
    rows = [r for r in full_table() if r["class"] == "bijection"]
    assert len(rows) == 6
    for r in rows:
        assert r["predicted"] == 3 and r["analytic"] == 3, r
    print("5. Theorem 2 holds: all 6 bijections are rank 3 ✓")


def test_identifiability_does_not_single_out_identity():
    """The anti-overclaim guard. If this ever fails it means someone has made
    identifiability appear to do work it cannot do — the case for `identity`
    specifically rests on E3's dynamics, not on rank."""
    rows = [r for r in full_table() if r["class"] == "bijection"]
    ranks = {r["analytic"] for r in rows}
    assert ranks == {3}, (
        "bijections now differ in rank — identifiability would then appear to "
        "single out one routing, which would contradict the writeup's stated "
        "two-part structure. Investigate before changing the claim."
    )
    print("6. all 6 bijections equally identifiable — identity is NOT singled "
          "out by rank (E3's dynamics does that) ✓")


def test_the_four_original_assignments_are_corollaries():
    """The four E2 spot-checked now fall out of the theorem, with the corrected
    rank-1 for collapse_D."""
    expect = {"identity": 3, "perm_emotion_D": 3, "perm_arousal_D": 3,
              "collapse_D": 1}
    for name, want in expect.items():
        got = predicted_rank(ASSIGNMENTS[name])
        assert got == want, (name, got, want)
    print("7. the 4 originally-tested assignments are recovered as corollaries "
          "(collapse_D = 1, the 6c correction) ✓")


def test_sequential_overwrite_semantics_documented_in_code():
    """When valence and arousal route to the SAME axis, compose_offsets'
    sequential assignment means valence is DISCARDED, not mixed. Pinned because
    it changes what those cases mean, and a reader would reasonably assume
    mixing."""
    from padeval.axes import AxisAssignment
    a = AxisAssignment("dup", "P", "P", "D")
    J = analytic_axis_jacobian(a, affect.tier_offset("close"))
    assert np.allclose(J[:, 0], 0.0), (
        "valence column should be all-zero when valence shares an axis with "
        f"arousal (it is overwritten), got {J[:, 0]}")
    assert not np.allclose(J[:, 1], 0.0), "arousal should still be live"
    print("8. duplicate-axis routing discards valence (overwrite, not mixing) — "
          "only collapse=True actually averages sources ✓")


if __name__ == "__main__":
    test_space_is_completely_enumerated()
    test_theorem_predicts_every_case()
    test_analytic_and_numeric_jacobians_match_to_machine_precision()
    test_theorem_1_freezing_bounds_rank()
    test_theorem_2_every_bijection_is_full_rank()
    test_identifiability_does_not_single_out_identity()
    test_the_four_original_assignments_are_corollaries()
    test_sequential_overwrite_semantics_documented_in_code()
    print("\nidentifiability is characterized, not spot-checked.")
