"""
Spec-driven subgraph seeder for the dual-cluster relational KG.

Reads authored spec files and populates the store with a Robot subgraph and a
Human (Person) subgraph: an anchor node (Robot/Person) plus authored-attribute
subnodes (persona / role / style / capability) linked by Has* identity edges.

Design contract
---------------
* Imports ONLY schema.py and store.py — no PAD, no kg_bridge, no adapters.
  The whole graph_relationship/ folder stays copy-pasteable.
* Idempotent: attribute-node ids are derived deterministically from
  (anchor_id, kind, value), so re-seeding the same spec upserts in place and
  never duplicates nodes/edges. Identity edges follow the store's standard
  replace-on-newer merge rule (SLOW timescale) — they do NOT accumulate.
* This step seeds authored identity only. Conversation extraction, event
  nodes, embeddings, and relationship-edge accumulation are later steps.

CLI
---
    python3 -m graph_relationship.seed \
        --kg-path kg_state.json \
        --spec-dir graph_relationship/specs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_YAML = False

from .schema import (
    AboutEdge,
    CapabilityNode,
    Embodiment,
    HasCapabilityEdge,
    HasInterestEdge,
    HasPersonaEdge,
    HasRoleEdge,
    HasSkillEdge,
    InterestNode,
    PersonaNode,
    PersonNode,
    Provenance,
    RoleNode,
    RobotNode,
    SkillNode,
)
from .store import InMemoryGraphStore
from .topics import interest_id, normalize_label, resolve_topic

# kind -> (NodeClass, EdgeClass).  Order fixes the seeding order.
_SINGLE_ATTRS = {
    "persona": (PersonaNode, HasPersonaEdge),
    "role":    (RoleNode,    HasRoleEdge),
}


def _slug(value: str) -> str:
    """Stable, readable id fragment: lowercase alnum, dashes for the rest."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _attr_id(anchor_id: str, kind: str, value: str) -> str:
    """Deterministic attribute-node id — same (anchor, kind, value) => same id."""
    return f"{anchor_id}:{kind}:{_slug(value)}"


def _load_spec(spec_path: str) -> Dict[str, Any]:
    """Load a spec file. YAML when available/appropriate, else JSON."""
    ext = os.path.splitext(spec_path)[1].lower()
    with open(spec_path, "r", encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            if not _HAVE_YAML:
                raise RuntimeError(
                    f"{spec_path} is YAML but pyyaml is not installed; "
                    f"provide a .json spec instead."
                )
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    if not isinstance(data, dict) or "id" not in data:
        raise ValueError(f"{spec_path}: spec must be a mapping with an 'id' field")
    return data


def seed_from_spec(store: InMemoryGraphStore, spec_path: str) -> Dict[str, int]:
    """Upsert one spec's anchor + authored-attribute subgraph into the store.

    Returns a small summary: {"nodes": N, "edges": M} = number of nodes/edges
    this spec writes (constant across re-runs, since ids are deterministic).
    """
    spec = _load_spec(spec_path)
    anchor_id = str(spec["id"])
    provenance_source = f"spec:{os.path.basename(spec_path)}"

    # --- anchor node: Robot if an embodiment is authored, else a Person ------
    is_robot = ("embodiment" in spec) or (str(spec.get("type", "")).lower() == "robot")
    if is_robot:
        store.upsert_node(RobotNode(
            id=anchor_id,
            name=str(spec.get("name", anchor_id)),
            embodiment=Embodiment(str(spec["embodiment"]).upper()),
        ))
    else:
        store.upsert_node(PersonNode(
            id=anchor_id,
            display_name=spec.get("name"),
        ))

    n_nodes, n_edges = 1, 0

    def _prov() -> Provenance:
        # Fresh timestamp per call so a re-seed is "newer" and replaces cleanly.
        return Provenance(source=provenance_source, confidence=1.0)

    # --- single-valued attributes: persona / role / style -------------------
    for kind, (NodeCls, EdgeCls) in _SINGLE_ATTRS.items():
        value = spec.get(kind)
        if not value:
            continue
        value = str(value)
        node = NodeCls(id=_attr_id(anchor_id, kind, value), descriptor=value)
        store.upsert_node(node)
        store.upsert_edge(EdgeCls(
            source_id=anchor_id, target_id=node.id, provenance=_prov(),
        ))
        n_nodes += 1
        n_edges += 1

    # --- capabilities: a hub CapabilityNode with one SkillNode per capability,
    #     each topic-bearing skill linking to a shared Topic:
    #       robot --has_capability--> Capability(hub) --has_skill--> Skill --about--> Topic
    #     A capability item may be a plain string, or {label, topics: [...]}.
    #     Top-level `knows_topics` are attached to the skill whose label mentions
    #     them, else to a new "knows <topic>" skill. resolve_topic() gives a
    #     deterministic "topic:<slug>" id so robot + human topics collapse to ONE.
    caps = spec.get("capabilities") or []
    if isinstance(caps, (str, dict)):
        caps = [caps]
    know = [str(t) for t in (spec.get("knows_topics") or []) if str(t).strip()]
    if caps or know:
        hub = CapabilityNode(id=f"{anchor_id}:capability", label="capabilities")
        store.upsert_node(hub)
        store.upsert_edge(HasCapabilityEdge(
            source_id=anchor_id, target_id=hub.id, provenance=_prov()))
        n_nodes += 1
        n_edges += 1

        skills_by_norm: Dict[str, Any] = {}

        def _add_skill(label: str) -> Any:
            skill = SkillNode(id=f"{anchor_id}:skill:{_slug(label)}", label=label)
            store.upsert_node(skill)
            store.upsert_edge(HasSkillEdge(
                source_id=hub.id, target_id=skill.id, provenance=_prov()))
            skills_by_norm[normalize_label(label)] = skill
            return skill

        def _skill_about(skill: Any, topic_label: str) -> None:
            topic = resolve_topic(store, topic_label)  # shared node, not counted
            store.upsert_edge(AboutEdge(
                source_id=skill.id, target_id=topic.id, provenance=_prov()))

        for item in caps:
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                item_topics = [str(t) for t in (item.get("topics") or []) if str(t).strip()]
            else:
                label = str(item).strip()
                item_topics = []
            if not label:
                continue
            skill = _add_skill(label)
            n_nodes += 1
            n_edges += 1
            for tl in item_topics:
                _skill_about(skill, tl)
                n_edges += 1

        # Top-level knows_topics: attach to a matching skill, else a new one.
        for tl in know:
            ntl = normalize_label(tl)
            match = next((s for norm, s in skills_by_norm.items() if ntl in norm), None)
            if match is None:
                match = _add_skill(f"knows {tl}")
                n_nodes += 1
                n_edges += 1
            _skill_about(match, tl)
            n_edges += 1

    # --- human: interests -> Interest node --about--> shared TopicNode -------
    for interest in spec.get("interests") or []:
        if isinstance(interest, str):
            interest = {"label": interest, "topics": []}
        label = str(interest.get("label", "")).strip()
        if not label:
            continue
        inode = InterestNode(id=interest_id(anchor_id, label), label=label)
        store.upsert_node(inode)
        store.upsert_edge(HasInterestEdge(
            source_id=anchor_id, target_id=inode.id, provenance=_prov(),
        ))
        n_nodes += 1
        n_edges += 1
        for topic_label in interest.get("topics") or []:
            topic_label = str(topic_label)
            if not topic_label:
                continue
            topic = resolve_topic(store, topic_label)
            store.upsert_edge(AboutEdge(
                source_id=inode.id, target_id=topic.id, provenance=_prov(),
            ))
            n_edges += 1

    print(f"[seed] {os.path.basename(spec_path)}: anchor={anchor_id} "
          f"({'robot' if is_robot else 'person'})  +{n_nodes} nodes, +{n_edges} edges")
    return {"nodes": n_nodes, "edges": n_edges}


def _spec_files(spec_dir: str) -> List[str]:
    files: List[str] = []
    for pattern in ("*.yaml", "*.yml", "*.json"):
        files.extend(glob.glob(os.path.join(spec_dir, pattern)))
    return sorted(files)


def seed_all(store: InMemoryGraphStore, spec_dir: str) -> Dict[str, int]:
    """Seed every spec file found in spec_dir. Returns aggregate write counts."""
    files = _spec_files(spec_dir)
    if not files:
        print(f"[seed] no spec files (*.yaml/*.yml/*.json) found in {spec_dir}")
        return {"nodes": 0, "edges": 0}
    total = {"nodes": 0, "edges": 0}
    for path in files:
        summary = seed_from_spec(store, path)
        total["nodes"] += summary["nodes"]
        total["edges"] += summary["edges"]
    return total


def main(argv: Optional[List[str]] = None) -> None:
    default_specs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
    ap = argparse.ArgumentParser(
        description="Seed Robot/Human subgraphs from authored spec files."
    )
    ap.add_argument("--kg-path", default="kg_state.json",
                    help="KG JSON file to load (if present) and save back to")
    ap.add_argument("--spec-dir", default=default_specs,
                    help=f"Directory of spec files (default: {default_specs})")
    args = ap.parse_args(argv)

    store = InMemoryGraphStore()
    if os.path.exists(args.kg_path):
        store.load(args.kg_path)
    before_n, before_e = len(store._nodes), len(store._edges)

    seed_all(store, args.spec_dir)

    after_n, after_e = len(store._nodes), len(store._edges)
    store.save(args.kg_path)

    print(f"[seed] store nodes {before_n} -> {after_n}  "
          f"(+{after_n - before_n} new)")
    print(f"[seed] store edges {before_e} -> {after_e}  "
          f"(+{after_e - before_e} new)")
    print(f"[seed] wrote {args.kg_path}")


if __name__ == "__main__":
    main()
