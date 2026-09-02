"""
Two-rung screen analysis, shared by D1 (prompt position) and D2 (model capacity).

Both diagnostics ask the same question of different manipulations — does the
rung-ordering effect become detectable? — so they must be measured identically
or the comparison between them is meaningless. One code path, one CI method.

Inference is cluster-robust over STIMULI, matching E1. Resampling replies would
treat 8 draws from one prompt as 8 independent facts; measured ICC is 0.128 on
the binary outcome, so nearly half the nominal sample is not real.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from padeval.analysis.stats import (cluster_bootstrap, design_effect,
                                    effective_n, icc1, kendall_tau_stat,
                                    rate_diff_stat)
from padeval.coding.rules import rule_code
from padeval.stimuli import BY_ID

# The comparator, from the L_full ladder arm (persona + tier pinned, Dominance
# forced per rung, memory present) — the only published baseline where the
# directive is the SOLE manipulation. Tier-level tables confound rung with tier
# and persona and are NOT valid comparators here.
BASELINE_L_FULL = {3: 0.66, 6: 0.75}
BASELINE_TAU_BINARY = +0.083     # over all 7 rungs, L_full


def code_rows(rows: Sequence[Dict]) -> List[Dict]:
    for r in rows:
        c = rule_code(r["reply"], BY_ID[r["stimulus_id"]])
        r["init"] = int(c.initiated)
        r["lvl"] = c.initiative_level
    return list(rows)


def screen_report(rows: Sequence[Dict], group_key: str,
                  low_rung: int = 3, high_rung: int = 6) -> str:
    """Per-group rung rates, the low-vs-high difference, and cluster-robust CIs.

    `low_rung` is the permissive directive ("either of you may open a topic"),
    `high_rung` the suppressing one ("answer only what they ask"). The predicted
    sign is rate(low) > rate(high), i.e. a POSITIVE difference.
    """
    out: List[str] = []
    out.append(f"| {group_key} | n | rung {low_rung} init% | rung {high_rung} init% "
               f"| difference | 95% CI (cluster) | tau | ICC |")
    out.append("|---|---|---|---|---|---|---|---|")
    for g in sorted({r[group_key] for r in rows}):
        sub = [r for r in rows if r[group_key] == g]
        lo = [r["init"] for r in sub if r["rung"] == low_rung]
        hi = [r["init"] for r in sub if r["rung"] == high_rung]
        d, dlo, dhi = cluster_bootstrap(
            sub, "stimulus_id", rate_diff_stat("rung", low_rung, high_rung, "init"))
        t, _tlo, _thi = cluster_bootstrap(
            sub, "stimulus_id", kendall_tau_stat("rung", "init"))
        i = icc1([r["init"] for r in sub], [r["stimulus_id"] for r in sub])
        out.append(f"| `{g}` | {len(sub)} | {np.mean(lo):.1%} | {np.mean(hi):.1%} "
                   f"| {d:+.3f} | [{dlo:+.3f}, {dhi:+.3f}] | {t:+.3f} | {i:.3f} |")
    return "\n".join(out)


def verdict(rows: Sequence[Dict], group_key: str, treatment: str,
            control: str = "deployed", threshold: float = 0.10,
            low_rung: int = 3, high_rung: int = 6) -> str:
    """Did the registered prediction clear its pre-stated bar?

    The bar is stated as a NUMBER before the run and is not renegotiated after
    seeing the result. A CI that merely touches zero is not an effect.
    """
    def tau_of(g):
        sub = [r for r in rows if r[group_key] == g]
        return cluster_bootstrap(sub, "stimulus_id",
                                 kendall_tau_stat("rung", "init"))

    t_t, tlo, thi = tau_of(treatment)
    t_c, clo, chi = tau_of(control)
    gain = abs(t_t) - abs(t_c)
    ci_excludes_zero = (tlo > 0) or (thi < 0)
    passed = gain >= threshold and ci_excludes_zero
    lines = [
        f"- control `{control}`   tau = {t_c:+.3f}  95% CI [{clo:+.3f}, {chi:+.3f}]",
        f"- treatment `{treatment}` tau = {t_t:+.3f}  95% CI [{tlo:+.3f}, {thi:+.3f}]",
        f"- |tau| gain = {gain:+.3f}, registered bar = +{threshold:.2f}",
        f"- treatment CI excludes zero: {ci_excludes_zero}",
        "",
        f"**VERDICT: {'PASS' if passed else 'FAIL — registered prediction not met'}**",
    ]
    return "\n".join(lines)
