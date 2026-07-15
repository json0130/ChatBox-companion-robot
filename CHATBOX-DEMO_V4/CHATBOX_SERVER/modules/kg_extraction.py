"""
App-layer graph-aware topic extraction (Feature: fine-grained typing + reuse).

Lives OUTSIDE `graph_relationship/` on purpose: all LLM prompt/parse/guard logic
stays in the APP layer. It is handed an `llm_fn(system, user) -> str` (the harness
`LLMClient.respond`) and imports ONLY pure schema/store helpers from
`graph_relationship`. `graph_relationship/` never imports this module, so it stays
copy-pasteable with zero LLM/PAD dependencies.

Flow per session:
  1. read the person's EXISTING topics (label, category) from the graph (pure read)
  2. prompt the LLM, conditioned on that list, for two buckets:
        existing_topics_discussed | new_topics     (+ optional per-topic summary)
  3. deterministic guards (JSON parse, enum, hallucination, confidence, normalize)
  4. apply via pure store helpers (reinforce existing path / add new typed topic)

Nothing is written on a JSON parse failure (never partially apply). Per-item
failures drop only that item. Re-running on an already-extracted session is a
no-op (deterministic ids + upsert; caller marks the session extracted).
"""
from __future__ import annotations

import json
from typing import Callable, List, Optional, Tuple

import math

from modules.graph_relationship.schema import TOPIC_CATEGORIES
from modules.graph_relationship.topics import (
    add_person_topic,
    merge_topics,
    normalize_label,
    person_topics,
    reinforce_person_topic,
    topic_degree,
)
from modules.graph_relationship.extraction import format_transcript  # pure transcript renderer

LLMFn = Callable[[str, str], str]

# Drop any extracted item below this confidence (constant — easy to tune).
CONFIDENCE_MIN = 0.6


def build_system_prompt(existing: List[Tuple[str, str]]) -> str:
    """Graph-aware extraction prompt, conditioned on the person's known topics."""
    cats = ", ".join(sorted(TOPIC_CATEGORIES))
    listing = "\n".join(f'  - "{label}" [{cat}]' for label, cat in existing) or "  (none yet)"
    return (
        "You extract the TOPICS the CHILD talked about, from a conversation between "
        "a CHILD and a companion ROBOT. Output ONLY one JSON object — no prose, no "
        "code fences.\n\n"
        "The child ALREADY has these known topics (reuse them; do NOT invent "
        "near-duplicates):\n"
        f"{listing}\n\n"
        "Return exactly this JSON:\n"
        '{"existing_topics_discussed": ['
        '{"label": "<COPY VERBATIM from the list above>", '
        '"confidence": <0.0-1.0>, "summary": "<optional one short sentence>"}], '
        '"new_topics": ['
        '{"label": "<short canonical noun phrase, lowercase>", '
        f'"category": "<one of: {cats}>", '
        '"confidence": <0.0-1.0>, "summary": "<optional one short sentence>"}]}\n\n'
        "Rules:\n"
        "- Prefer reusing an existing label over a near-duplicate: if \"jazz\" is "
        "known and the child says \"jazz music\", REUSE \"jazz\".\n"
        "- Put a topic in new_topics ONLY if it is genuinely distinct from EVERY "
        "existing topic listed above.\n"
        f"- category MUST be one of: {cats}. Use \"other\" if unsure — never invent one.\n"
        "- Include ONLY topics the CHILD actually expressed; use empty arrays if none.\n"
        "- Return ONLY the JSON object."
    )


def _parse_json_object(raw: str) -> Optional[dict]:
    """First JSON object in a possibly-noisy response, or None."""
    if not raw:
        return None
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _conf(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _summary(v) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip()[:280]
    return s or None


def extract_and_apply_topics(
    store, person_id: str, robot_id: str, turns: list, llm_fn: LLMFn,
    *, session_id: Optional[str] = None,
) -> dict:
    """Graph-aware, typed topic extraction for one session. Returns a summary dict:
       {applied, reinforced:[(label,conf)], added:[(label,cat,conf)], dropped:[...]}"""
    source = f"extraction:{session_id}" if session_id else "extraction"

    existing = person_topics(store, person_id)                       # [(label, category)]
    existing_norm = {normalize_label(l): (l, c) for l, c in existing}

    raw = llm_fn(build_system_prompt(existing), format_transcript(turns))
    obj = _parse_json_object(raw)
    if obj is None:
        # Parse failure → write NOTHING (never partially apply).
        return {"applied": False, "reason": "json_parse_failed",
                "reinforced": [], "added": [], "dropped": []}

    reinforced: list = []
    added: list = []
    dropped: list = []

    def _reinforce(canon_label, conf, summary):
        node = reinforce_person_topic(store, person_id, canon_label,
                                      source=source, confidence=conf, summary=summary)
        if node:
            reinforced.append((canon_label, conf))
        else:
            dropped.append(("existing", canon_label, "no_path"))

    # ── existing_topics_discussed ─────────────────────────────────────────────
    for item in (obj.get("existing_topics_discussed") or []):
        if not isinstance(item, dict):
            dropped.append(("existing", item, "not_object")); continue
        label = str(item.get("label", "")).strip()
        conf = _conf(item.get("confidence"))
        summary = _summary(item.get("summary"))
        norm = normalize_label(label)
        if not norm:
            dropped.append(("existing", label, "empty")); continue
        if conf < CONFIDENCE_MIN:
            dropped.append(("existing", label, f"low_conf<{CONFIDENCE_MIN}")); continue
        if norm not in existing_norm:
            # Hallucinated 'existing' topic (not in the list we provided). It has no
            # category, so it cannot become a valid new topic → drop.
            dropped.append(("existing", label, "not_in_provided_list")); continue
        _reinforce(existing_norm[norm][0], conf, summary)   # reuse canonical label

    # ── new_topics ────────────────────────────────────────────────────────────
    for item in (obj.get("new_topics") or []):
        if not isinstance(item, dict):
            dropped.append(("new", item, "not_object")); continue
        label = str(item.get("label", "")).strip()
        conf = _conf(item.get("confidence"))
        summary = _summary(item.get("summary"))
        cat = str(item.get("category", "")).strip().lower()
        norm = normalize_label(label)
        if not norm:
            dropped.append(("new", label, "empty")); continue
        if conf < CONFIDENCE_MIN:
            dropped.append(("new", label, f"low_conf<{CONFIDENCE_MIN}")); continue
        if cat not in TOPIC_CATEGORIES:
            # Category outside the closed taxonomy → DROP the item (write nothing).
            # The prompt tells the LLM to use "other" when unsure, so a value outside
            # the enum is a malformed response, not a real topic.
            dropped.append(("new", label, f"bad_category:{cat}")); continue
        if norm in existing_norm:
            # LLM put a known topic under new_topics → reinforce, do not duplicate.
            _reinforce(existing_norm[norm][0], conf, summary); continue
        node = add_person_topic(store, person_id, label, cat,
                                source=source, confidence=conf, summary=summary)
        if node:
            added.append((label, cat, conf))

    return {"applied": True, "reinforced": reinforced, "added": added, "dropped": dropped}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — semantic topic consolidation (app layer; injected embed_fn)
# ─────────────────────────────────────────────────────────────────────────────
# Merges near-duplicate topic nodes (e.g. "hiphop" / "hip hop") that Feature-1's
# exact-label reuse cannot catch. Embedding + pairing decisions live HERE; the
# graph surgery is the pure topics.merge_topics(). NOT run during live extraction
# — invoked on demand (dry-run first), so merges are deterministic and reviewable.

EmbedFn = Callable[[str], List[float]]

CONSOLIDATE_FLOOR = 0.86        # cosine >= this → merge candidate (conservative)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _topic_nodes(store) -> List[tuple]:
    """[(id, label, category_str), ...] for every TopicNode (app-layer read)."""
    out = []
    for n in getattr(store, "_nodes", {}).values():
        if getattr(n, "node_type", None) == "topic":
            cat = n.category.value if hasattr(n.category, "value") else str(n.category)
            out.append((n.id, n.label, cat))
    return out


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def consolidate_topics(
    store, embed_fn: EmbedFn, *,
    floor: float = CONSOLIDATE_FLOOR, same_category_only: bool = True,
    dry_run: bool = False, source: str = "consolidate",
) -> dict:
    """Find near-duplicate topics by embedding cosine and merge each group into a
    single canonical node (highest-degree; ties → shortest, then lexicographic).

    same_category_only: never merge across categories (jazz[music] vs jaguar[animals]).
    dry_run: compute + return the proposed merges but write NOTHING.

    Returns {"pairs":[(a_label,b_label,sim)], "merges":[(canon_label,dup_label)],
             "groups": n, "dry_run": bool}.
    """
    topics = _topic_nodes(store)
    report = {"pairs": [], "merges": [], "groups": 0, "dry_run": dry_run}
    if len(topics) < 2:
        return report

    # Embed each label once (cache); skip topics whose embedding fails.
    vecs: dict = {}
    for tid, label, _cat in topics:
        try:
            v = embed_fn(label)
        except Exception:  # noqa: BLE001 — embedding backend down → skip
            v = None
        if v:
            vecs[tid] = v

    # Candidate pairs by cosine (optionally same-category only).
    cat_of = {tid: cat for tid, _l, cat in topics}
    label_of = {tid: lbl for tid, lbl, _c in topics}
    uf = _UnionFind([t[0] for t in topics])
    n = len(topics)
    for i in range(n):
        ai = topics[i][0]
        if ai not in vecs:
            continue
        for j in range(i + 1, n):
            bj = topics[j][0]
            if bj not in vecs:
                continue
            if same_category_only and cat_of[ai] != cat_of[bj]:
                continue
            sim = _cosine(vecs[ai], vecs[bj])
            if sim >= floor:
                report["pairs"].append((label_of[ai], label_of[bj], round(sim, 3)))
                uf.union(ai, bj)

    # Group members; canonical = highest degree, tie → shortest then lexicographic.
    groups: dict = {}
    for tid, _l, _c in topics:
        groups.setdefault(uf.find(tid), []).append(tid)
    merge_ops = []  # (canonical_id, duplicate_id)
    for members in groups.values():
        if len(members) < 2:
            continue
        report["groups"] += 1
        canonical = min(
            members,
            key=lambda t: (-topic_degree(store, t), len(label_of[t]), label_of[t]))
        for tid in members:
            if tid != canonical:
                merge_ops.append((canonical, tid))
                report["merges"].append((label_of[canonical], label_of[tid]))

    if not dry_run:
        for canonical, dup in merge_ops:
            merge_topics(store, canonical, dup, source=source)
    return report
