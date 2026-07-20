"""
Culture layer helpers: a person may (manually) BELONG to one CultureNode, and a
culture carries soft PRIORS over Topics.

  person --belongs_to_culture--> Culture("Korean") --culture_prior[0.8]--> Topic("kimchi")

`culture_id()` gives every culture a deterministic id ("culture:" + slug, the SAME
slug as TopicNode), so re-seeding "Korean" resolves to ONE node. Priors are authored
starting guesses about a background — never a fact about an individual.

Design contract: imports ONLY schema.py + store.py (+ the pure topics slug) — no LLM,
no PAD, no embeddings, nothing in modules/. Stateless functions only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .schema import (
    BelongsToCultureEdge, CultureNode, CulturePriorEdge, Provenance,
)
from .store import GraphStore
from .topics import normalize_label


def _prov(source: Optional[str], confidence: float = 1.0) -> Provenance:
    return Provenance(source=source or "cultures",
                      confidence=max(0.0, min(1.0, float(confidence))),
                      timestamp=datetime.now(timezone.utc))


def culture_id(label: str) -> str:
    """Deterministic id `culture:<slug>` (same normalization as topic_id)."""
    return f"culture:{normalize_label(label)}"


def ensure_culture(store: GraphStore, label: str) -> CultureNode:
    """Get-or-create the CultureNode for `label` (deterministic id). Idempotent —
    re-calling returns the SAME node without duplicating it."""
    cid = culture_id(label)
    existing = store.get_node(cid)
    if existing is not None and existing.node_type == "culture":
        return existing
    node = CultureNode(id=cid, label=str(label).strip())
    store.upsert_node(node)
    return node


def assign_culture(store: GraphStore, person_id: str, culture_id: str,
                   *, source: Optional[str] = None) -> None:
    """Link a person to a culture (belongs_to_culture). Idempotent — one edge per
    person-culture pair (upsert replaces provenance, never duplicates)."""
    store.upsert_edge(BelongsToCultureEdge(
        source_id=person_id, target_id=culture_id, provenance=_prov(source)))


def set_culture_prior(store: GraphStore, culture_id: str, topic_id: str,
                      prior: float, *, source: Optional[str] = None) -> None:
    """Upsert a culture→topic prior, clamped to [0,1]. Re-setting replaces the
    stored value (one edge per culture-topic pair)."""
    p = max(0.0, min(1.0, float(prior)))
    store.upsert_edge(CulturePriorEdge(
        source_id=culture_id, target_id=topic_id, prior=p, provenance=_prov(source)))


def culture_priors(store: GraphStore, culture_id: str) -> List[Tuple[str, float]]:
    """[(topic_id, prior), ...] for one culture, sorted by prior DESC (stable
    lexicographic tie-break on topic_id). Index-based read."""
    out: List[Tuple[str, float]] = []
    for edge, neighbor in store.query_neighbors(culture_id, "culture_prior"):
        if neighbor.node_type == "topic":
            out.append((neighbor.id, float(edge.prior)))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def person_culture(store: GraphStore, person_id: str) -> Optional[str]:
    """The culture_id this person belongs to, or None. If (unexpectedly) more than
    one is assigned, returns the lexicographically-first for determinism."""
    cids = [n.id for _e, n in store.query_neighbors(person_id, "belongs_to_culture")
            if n.node_type == "culture"]
    return sorted(cids)[0] if cids else None
