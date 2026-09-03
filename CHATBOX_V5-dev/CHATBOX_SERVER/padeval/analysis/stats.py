"""
Cluster-aware statistics. numpy/scipy only.

WHY CLUSTERING IS NOT OPTIONAL HERE
-----------------------------------
The unit of analysis is a reply, but replies are generated in groups that share a
stimulus. Two replies to "mm." are not two independent pieces of evidence about
how the controller behaves in general — they are two draws from the same prompt.
Measured on this system, reply diversity is itself a property of the prompt: a
low-entropy stimulus produced 2 distinct replies across 5 seeds while an open
question produced 5. So the effective sample size is somewhere between the number
of replies and the number of stimuli, and where exactly is an empirical question,
not an assumption.

`icc1` measures it. `design_effect` converts it into the factor by which naive n
overstates the truth. `cluster_bootstrap` resamples STIMULI rather than replies,
which is the inference that respects the design.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def icc1(values: Sequence[float], clusters: Sequence) -> float:
    """One-way random-effects ICC(1): the share of variance between clusters.

    0 means replies within a stimulus are as different as replies across
    stimuli (replicates are full-value evidence). 1 means a stimulus fully
    determines the outcome (replicates add nothing at all).
    """
    v = np.asarray(values, dtype=float)
    groups: Dict = {}
    for x, c in zip(v, clusters):
        groups.setdefault(c, []).append(x)
    k = len(groups)
    if k < 2:
        return float("nan")
    sizes = np.array([len(g) for g in groups.values()], dtype=float)
    means = np.array([np.mean(g) for g in groups.values()])
    grand = v.mean()
    msb = float(np.sum(sizes * (means - grand) ** 2) / (k - 1))
    within = float(sum(((np.asarray(g) - m) ** 2).sum()
                       for g, m in zip(groups.values(), means)))
    dfw = len(v) - k
    if dfw <= 0:
        return float("nan")
    msw = within / dfw
    # m0: average cluster size corrected for unequal sizes
    m0 = (sizes.sum() - (sizes ** 2).sum() / sizes.sum()) / (k - 1)
    denom = msb + (m0 - 1) * msw
    if denom == 0:
        return 0.0
    return float(max(0.0, (msb - msw) / denom))


def design_effect(icc: float, avg_cluster_size: float) -> float:
    """DE = 1 + (m - 1) * ICC. Naive n divided by this is the effective n."""
    return 1.0 + (avg_cluster_size - 1.0) * max(0.0, icc)


def effective_n(n: int, icc: float, avg_cluster_size: float) -> float:
    return n / design_effect(icc, avg_cluster_size)


def cluster_bootstrap(rows: Sequence[Dict], cluster_key: str,
                      stat: Callable[[Sequence[Dict]], float],
                      n_boot: int = 5000, seed: int = 0
                      ) -> Tuple[float, float, float]:
    """Resample CLUSTERS with replacement; return (point, lo95, hi95).

    Resampling replies instead would treat 8 draws from one prompt as 8
    independent facts and produce an interval far too narrow.
    """
    point = stat(rows)
    by: Dict = {}
    for r in rows:
        by.setdefault(r[cluster_key], []).append(r)
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for i in pick for r in by[keys[i]]]
        try:
            s = stat(sample)
            if np.isfinite(s):
                draws.append(s)
        except Exception:
            continue
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def kendall_tau_stat(x_key: str, y_key: str) -> Callable[[Sequence[Dict]], float]:
    from scipy.stats import kendalltau

    def _stat(rows: Sequence[Dict]) -> float:
        return kendalltau([r[x_key] for r in rows], [r[y_key] for r in rows]).statistic

    return _stat


def rate_diff_stat(group_key: str, a, b, outcome_key: str
                   ) -> Callable[[Sequence[Dict]], float]:
    def _stat(rows: Sequence[Dict]) -> float:
        ga = [r[outcome_key] for r in rows if r[group_key] == a]
        gb = [r[outcome_key] for r in rows if r[group_key] == b]
        if not ga or not gb:
            return float("nan")
        return float(np.mean(ga) - np.mean(gb))

    return _stat
