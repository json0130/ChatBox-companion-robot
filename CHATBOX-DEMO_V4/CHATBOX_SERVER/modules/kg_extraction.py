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

from modules.graph_relationship.schema import TOPIC_CATEGORIES
from modules.graph_relationship.topics import (
    add_person_topic,
    normalize_label,
    person_topics,
    reinforce_person_topic,
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
