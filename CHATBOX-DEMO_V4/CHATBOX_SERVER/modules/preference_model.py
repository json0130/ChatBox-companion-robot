"""
Bayesian preference overlay (Command B) — compiled ON READ, NEVER persisted.

`rank_suggestions(store, person_id)` ranks topics the robot could tentatively bring
up with a person, blending:
  * the person's CULTURE priors  (base rates — the robot's prior knowledge)
  * OBSERVED interests            (topics they actually engage with → strong evidence)
  * one-hop `related_topic` links (semantic neighbours → soft propagation)

This mirrors the computed-on-read `_tier_from_edges` pattern:
  * READ-ONLY over the store — zero writes, zero node/edge creation.
  * No live embeddings — consumes only the STORED `related_topic` edge weights.
  * No LLM, no PAD. Does not modify graph_relationship/ (imports its pure reads only).

Namespace join (post-redesign): culture priors live on `culture_topic` nodes
(`ck:<culture>:<slug>`) which are DISTINCT from a person's shared `topic:<slug>`
nodes. So the model works in a unified SLUG space — a culture prior for "kimchi"
and a person topic "kimchi" are the same concept, joined by normalized label.

Scope (per Command B): positive/observed evidence only. Negative/"dislike" evidence
is deliberately OUT OF SCOPE — a future extension would add a polarity signal that
clamps a disliked concept low and blocks its propagation.
"""

from __future__ import annotations

from typing import List, Tuple

from modules.graph_relationship.cultures import person_culture, culture_priors
from modules.graph_relationship.topics import (
    person_interests, topic_related, normalize_label,
)

# Inference constants (deliberately simple — no external BN library).
_DEFAULT_PRIOR = 0.30   # unobserved concept with no culture prior
_OBSERVED_P    = 0.90   # observed-positive clamp (interest/about edge)
_DAMPING       = 0.80   # noisy-OR edge damping w
_ROUNDS        = 2      # propagation rounds over related_topic edges


def rank_suggestions(
    store, person_id: str, k: int = 3, floor: float = 0.35,
) -> List[Tuple[str, float]]:
    """Top-k UNOBSERVED topics to tentatively suggest, as [(node_id, posterior)].

    Deterministic. Degrades gracefully to prior-only ranking when there are no
    `related_topic` links. Returns [] when there is nothing to suggest.
    """
    # ── base rates: culture priors, keyed by slug ─────────────────────────────
    prior_by_slug: dict = {}
    ck_id_by_slug: dict = {}
    cid = person_culture(store, person_id)
    if cid:
        for ck_id, label, prior in culture_priors(store, cid):
            s = normalize_label(label)
            prior_by_slug[s] = prior
            ck_id_by_slug.setdefault(s, ck_id)

    # ── observed evidence: the person's own interest topics (clamped) ─────────
    observed: set = set()
    topic_id_by_slug: dict = {}
    for _interest, topics in person_interests(store, person_id):
        for t in topics:
            s = normalize_label(t.label)
            observed.add(s)
            topic_id_by_slug.setdefault(s, t.id)

    # ── candidate expansion: one-hop related neighbours of the person's topics ─
    for s, tid in list(topic_id_by_slug.items()):
        for rt in topic_related(store, tid):
            topic_id_by_slug.setdefault(normalize_label(rt.label), rt.id)

    concepts: set = set(prior_by_slug) | set(topic_id_by_slug)
    if not concepts:
        return []

    # ── initialise posteriors ─────────────────────────────────────────────────
    p: dict = {
        s: (_OBSERVED_P if s in observed else prior_by_slug.get(s, _DEFAULT_PRIOR))
        for s in concepts
    }

    # ── gather related_topic edges (with STORED weights) in slug space ────────
    slug_by_id = {tid: s for s, tid in topic_id_by_slug.items()}
    edges: List[Tuple[str, str, float]] = []
    seen: set = set()
    for s, tid in list(topic_id_by_slug.items()):
        for edge, nb in store.query_neighbors(tid, "related_topic"):
            if nb.node_type != "topic":
                continue
            bs = slug_by_id.get(nb.id) or normalize_label(nb.label)
            if bs not in p:                    # neighbour-of-neighbour: add candidate
                p[bs] = prior_by_slug.get(bs, _DEFAULT_PRIOR)
                concepts.add(bs)
                topic_id_by_slug.setdefault(bs, nb.id)
                slug_by_id.setdefault(nb.id, bs)
            key = tuple(sorted((s, bs)))
            if s == bs or key in seen:
                continue
            seen.add(key)
            edges.append((s, bs, float(edge.weight)))

    # ── noisy-OR propagation (observed stay clamped; unobserved may rise) ─────
    for _ in range(_ROUNDS):
        for a, b, w in edges:
            if b not in observed:
                p[b] = max(p[b], p[a] * _DAMPING * w)
            if a not in observed:
                p[a] = max(p[a], p[b] * _DAMPING * w)

    # ── rank UNOBSERVED concepts with posterior ≥ floor ───────────────────────
    ranked: List[Tuple[str, float, float]] = []
    for s in concepts:
        if s in observed:
            continue
        post = p[s]
        if post + 1e-12 < floor:
            continue
        node_id = topic_id_by_slug.get(s) or ck_id_by_slug.get(s)
        if node_id is None:
            continue
        ranked.append((node_id, post, prior_by_slug.get(s, _DEFAULT_PRIOR)))
    # posterior desc, then higher prior, then lexicographic id
    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return [(node_id, round(post, 4)) for node_id, post, _prior in ranked[:k]]
