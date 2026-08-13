"""
strip_culture_kg.py — migrate a knowledge graph off the culture schema.

This branch removes `CultureNode` / `CultureTopicNode` and the three culture edge
types from `graph_relationship.schema`. `InMemoryGraphStore.load()` deserialises
through the `AnyNode` / `AnyEdge` discriminated unions, so ANY graph still holding
those records raises a Pydantic validation error on the first run — a certain
failure, not a risk.

Rather than start from an empty graph (which would throw away the enrolled person,
their topics, interests and the InteractionNode carrying real rapport/trust — the
only relationship available to exercise the tier ladder against), this rewrites the
file in place-ish: culture records out, everything else untouched.

What it removes:
  * nodes  with node_type in {culture, culture_topic}
  * edges  with edge_type in {knows_culture, belongs_to_culture, culture_prior}
  * edges  incident to any node removed above (dangling-reference sweep)

Everything else is copied byte-for-byte, so a node's fields, ids and provenance are
untouched. Runs on plain JSON and imports no project code, so it works BEFORE or
AFTER the schema surgery.

Usage:
    python3 -m tools.strip_culture_kg kg_state.json                 # in place (.bak kept)
    python3 -m tools.strip_culture_kg kg_state.json -o kg_pad.json  # to a new file
    python3 -m tools.strip_culture_kg kg_state.json --dry-run       # report only
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

CULTURE_NODE_TYPES = {"culture", "culture_topic"}
CULTURE_EDGE_TYPES = {"knows_culture", "belongs_to_culture", "culture_prior"}


def _counts(items: list, key: str) -> dict:
    return dict(collections.Counter(i.get(key) for i in items))


def strip(graph: dict) -> tuple[dict, dict]:
    """Return (clean_graph, report). Pure — does not mutate `graph`."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    kept_nodes = [n for n in nodes if n.get("node_type") not in CULTURE_NODE_TYPES]
    dropped_ids = {n.get("id") for n in nodes
                   if n.get("node_type") in CULTURE_NODE_TYPES}

    kept_edges, dangling = [], 0
    for e in edges:
        if e.get("edge_type") in CULTURE_EDGE_TYPES:
            continue
        # Sweep edges pointing at a node we just removed — a surviving edge with a
        # dangling endpoint is worse than the culture record itself.
        if e.get("source_id") in dropped_ids or e.get("target_id") in dropped_ids:
            dangling += 1
            continue
        kept_edges.append(e)

    clean = dict(graph)
    clean["nodes"] = kept_nodes
    clean["edges"] = kept_edges

    report = {
        "nodes_before": len(nodes), "nodes_after": len(kept_nodes),
        "edges_before": len(edges), "edges_after": len(kept_edges),
        "culture_nodes_dropped": len(dropped_ids),
        "culture_edges_dropped": sum(
            1 for e in edges if e.get("edge_type") in CULTURE_EDGE_TYPES),
        "dangling_edges_swept": dangling,
        "node_types_after": _counts(kept_nodes, "node_type"),
        "edge_types_after": _counts(kept_edges, "edge_type"),
    }
    return clean, report


def _assert_survivors(report: dict) -> list:
    """Sanity-check what remains. Returns a list of warnings (never raises) — the
    point is to notice if the strip took something it shouldn't have."""
    warn = []
    nt = report["node_types_after"]
    for kind in ("person", "robot", "interaction"):
        if nt.get(kind, 0) < 1:
            warn.append(f"no '{kind}' node survived — expected at least 1")
    if any(k in nt for k in CULTURE_NODE_TYPES):
        warn.append("culture node types STILL present after strip")
    if any(k in report["edge_types_after"] for k in CULTURE_EDGE_TYPES):
        warn.append("culture edge types STILL present after strip")
    return warn


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="graph JSON to migrate (e.g. kg_state.json)")
    p.add_argument("-o", "--out", default=None,
                   help="write here instead of rewriting `path` in place")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change; write nothing")
    p.add_argument("--no-backup", action="store_true",
                   help="skip the .pre-strip.bak copy when rewriting in place")
    args = p.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"[strip] no such file: {args.path}")
        return 1
    with open(args.path) as fh:
        graph = json.load(fh)

    clean, rep = strip(graph)

    print(f"\n[strip] {args.path}")
    print(f"  nodes {rep['nodes_before']:4d} -> {rep['nodes_after']:4d}   "
          f"({rep['culture_nodes_dropped']} culture)")
    print(f"  edges {rep['edges_before']:4d} -> {rep['edges_after']:4d}   "
          f"({rep['culture_edges_dropped']} culture, "
          f"{rep['dangling_edges_swept']} dangling swept)")
    print(f"  surviving nodes: {rep['node_types_after']}")

    for w in _assert_survivors(rep):
        print(f"  WARNING: {w}")

    if args.dry_run:
        print("  (dry run — nothing written)\n")
        return 0

    dest = args.out or args.path
    if dest == args.path and not args.no_backup:
        bak = args.path + ".pre-strip.bak"
        shutil.copy2(args.path, bak)
        print(f"  backup: {bak}")
    with open(dest, "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"  wrote:  {dest}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
