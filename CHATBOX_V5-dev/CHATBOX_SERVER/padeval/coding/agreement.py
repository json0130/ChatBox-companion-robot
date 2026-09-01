"""
Inter-coder agreement. numpy only — no statsmodels, no sklearn.

Cohen's kappa with a bootstrap CI, weighted kappa for the ordinal, McNemar for
directional disagreement, and a per-stratum breakdown. Each is short enough to
read in full, which is the point: these are the numbers that decide whether the
automatic coder can be trusted, so they should not sit behind a dependency.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _confusion(a: Sequence, b: Sequence, labels: Sequence) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def cohens_kappa(a: Sequence, b: Sequence,
                 labels: Sequence | None = None,
                 weights: str | None = None) -> float:
    """weights=None for nominal, "linear"/"quadratic" for an ordinal scale."""
    if labels is None:
        labels = sorted(set(list(a) + list(b)))
    if len(labels) < 2:
        return 1.0 if list(a) == list(b) else 0.0
    m = _confusion(a, b, labels)
    n = m.sum()
    if n == 0:
        return float("nan")
    po_m = m / n
    row, col = po_m.sum(axis=1), po_m.sum(axis=0)
    pe_m = np.outer(row, col)
    k = len(labels)
    if weights is None:
        w = 1.0 - np.eye(k)
    else:
        i, j = np.indices((k, k))
        d = np.abs(i - j) / (k - 1)
        w = d if weights == "linear" else d ** 2
    denom = (w * pe_m).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (w * po_m).sum() / denom)


def kappa_ci(a: Sequence, b: Sequence, labels: Sequence | None = None,
             weights: str | None = None, n_boot: int = 5000,
             seed: int = 0) -> Tuple[float, float, float]:
    """Point estimate plus a percentile bootstrap 95% CI."""
    a, b = list(a), list(b)
    point = cohens_kappa(a, b, labels, weights)
    rng = np.random.default_rng(seed)
    n = len(a)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            draws.append(cohens_kappa([a[i] for i in idx], [b[i] for i in idx],
                                      labels, weights))
        except Exception:
            continue
    lo, hi = np.percentile([d for d in draws if np.isfinite(d)], [2.5, 97.5])
    return point, float(lo), float(hi)


def percent_agreement(a: Sequence, b: Sequence) -> float:
    return float(np.mean([x == y for x, y in zip(a, b)]))


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict:
    """Is the disagreement DIRECTIONAL — does one coder systematically say yes
    where the other says no? A high kappa with a lopsided McNemar means the
    coders agree often but disagree in a biased way, which matters for a rate."""
    from math import comb
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(b_only, c_only)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b": b_only, "c": c_only, "p": float(p)}


def agreement_report(pairs: List[Dict], a_key: str, b_key: str,
                     by: str | None = None) -> str:
    """Markdown block for one coder pair, optionally split by a stratum field."""
    out: List[str] = []

    def block(rows: List[Dict], label: str) -> None:
        if not rows:
            return
        ab = [r[a_key]["initiated"] for r in rows]
        bb = [r[b_key]["initiated"] for r in rows]
        al = [r[a_key]["initiative_level"] for r in rows]
        bl = [r[b_key]["initiative_level"] for r in rows]
        kb, lo, hi = kappa_ci(ab, bb, labels=[False, True])
        kl, llo, lhi = kappa_ci(al, bl, labels=list(range(6)), weights="quadratic")
        mc = mcnemar_exact(ab, bb)
        out.append(
            f"| {label} | {len(rows)} | {percent_agreement(ab, bb):.0%} | "
            f"{kb:.2f} [{lo:.2f}, {hi:.2f}] | {percent_agreement(al, bl):.0%} | "
            f"{kl:.2f} [{llo:.2f}, {lhi:.2f}] | {mc['b']}/{mc['c']} | {mc['p']:.3f} |"
        )

    out.append("| subset | n | binary agree | kappa (binary) | ordinal agree "
               "| kappa_w (ordinal) | McNemar b/c | p |")
    out.append("|---|---|---|---|---|---|---|---|")
    block(pairs, "ALL")
    if by:
        for value in sorted({r[by] for r in pairs}):
            block([r for r in pairs if r[by] == value], value)
    return "\n".join(out)
