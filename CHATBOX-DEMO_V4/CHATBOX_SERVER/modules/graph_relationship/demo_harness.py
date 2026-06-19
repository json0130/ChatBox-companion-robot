"""
KG × PAD integration harness — proves per-person KG adaptation.

Wires the real end-to-end loop with NO hardware and NO webcam:

    KGBridge.pre_turn  →  PADPipelineAdapter.process_turn  →  KGBridge.post_turn

A shared InMemoryGraphStore persists across the whole run.

Faked inputs (the only stubs):
  person_id     — typed by the user or scripted
  camera_emotion — typed by the user or scripted (happy/sad/neutral/angry/…)
  robot_id      — flag --robot chatbox|ellebot  (default: chatbox)

Usage:
    python3 -m modules.graph_relationship.demo_harness              # interactive
    python3 -m modules.graph_relationship.demo_harness --scripted   # automated proof
    python3 -m modules.graph_relationship.demo_harness --robot ellebot
    python3 -m modules.graph_relationship.demo_harness --scripted --obsidian
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# ── Module imports ────────────────────────────────────────────────────────────
# Relative imports work when run as: python3 -m modules.graph_relationship.demo_harness
# from the CHATBOX_SERVER directory.

from .schema import (
    Embodiment,
    PersonNode,
    Provenance,
    RapportEdge,
    RobotNode,
    TrustEdge,
)
from .store import InMemoryGraphStore
from .kg_bridge import KGBridge, derive_tier

try:
    from ..pad_persona.pipeline_adapter import PADPipelineAdapter
except ImportError:
    # Fallback for direct invocation or unusual PYTHONPATH setups.
    _here = os.path.dirname(os.path.dirname(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from pad_persona.pipeline_adapter import PADPipelineAdapter  # type: ignore


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_EMOTIONS = frozenset({
    "happy", "sad", "neutral", "angry", "calm",
    "fear", "disgust", "surprise",
})
VALID_ROBOTS = frozenset({"chatbox", "ellebot"})

# ANSI colour codes for tier labels in terminal output.
_TIER_COLOUR = {
    "unknown": "\033[90m",   # grey
    "visitor": "\033[33m",   # yellow
    "known":   "\033[36m",   # cyan
    "close":   "\033[32m",   # green
}
_RST = "\033[0m"


# ── Store helpers  (read-only queries, no writes) ─────────────────────────────

def _relationship_snapshot(
    store: InMemoryGraphStore,
    person_id: str,
    robot_id: str,
) -> dict:
    """Return rapport / trust / interaction_count for a person-robot pair."""
    rp = store.get_edge(robot_id, person_id, "rapport")
    tr = store.get_edge(robot_id, person_id, "trust")
    ic = store.get_edge(robot_id, person_id, "interaction_count")
    return {
        "rapport":           rp.weight if rp else 0.0,
        "trust":             tr.weight if tr else 0.0,
        "interaction_count": ic.count  if ic else 0,
    }


def _ensure_person(store: InMemoryGraphStore, person_id: str) -> None:
    if store.get_node(person_id) is None:
        store.upsert_node(PersonNode(id=person_id, display_name=person_id))


def _ensure_robot(store: InMemoryGraphStore, robot_id: str) -> None:
    if store.get_node(robot_id) is None:
        emb = Embodiment.CAT if robot_id.lower() == "chatbox" else Embodiment.ELEPHANT
        store.upsert_node(RobotNode(id=robot_id, name=robot_id, embodiment=emb))


# ── TEST AID ──────────────────────────────────────────────────────────────────

def seed_relationship(
    store: InMemoryGraphStore,
    person_id: str,
    robot_id: str,
    rapport: float = 0.5,
    trust: float = 0.5,
) -> None:
    """
    TEST AID — NOT production logic.

    Directly writes RapportEdge and TrustEdge into the store so the harness
    can push a person to a higher tier without running enough interaction turns.
    In the real system tier is earned through interaction; this exists only so
    the demo can show the 'close' state without requiring 70+ turns of warm-up.

    Why:  (rapport + trust) / 2 > 0.70  →  derive_tier() returns "close"
           (rapport + trust) / 2 > 0.45  →  "known"
    """
    rapport = max(0.0, min(1.0, rapport))
    trust   = max(0.0, min(1.0, trust))
    _ensure_person(store, person_id)
    _ensure_robot(store, robot_id)
    prov = Provenance(
        source="harness:seed",
        confidence=1.0,
        timestamp=datetime.now(timezone.utc),
    )
    store.apply_delta(edges=[
        RapportEdge(source_id=robot_id, target_id=person_id,
                    provenance=prov, weight=rapport),
        TrustEdge  (source_id=robot_id, target_id=person_id,
                    provenance=prov, weight=trust),
    ])
    score = (rapport + trust) / 2.0
    tier_after = "close" if score > 0.70 else ("known" if score > 0.45 else "visitor")
    tc = _TIER_COLOUR.get(tier_after, "")
    print(f"  [TEST AID] seeded {person_id}: rapport={rapport:.2f}  trust={trust:.2f}"
          f"  score={score:.2f}  → tier now {tc}{tier_after}{_RST}")


# ── Graph export ──────────────────────────────────────────────────────────────

def _edge_label(edge) -> str:
    et = edge.edge_type
    if hasattr(edge, "weight"):
        return f"{et}={edge.weight:.2f}"
    if hasattr(edge, "count"):
        return f"{et}={edge.count}"
    if hasattr(edge, "value"):
        return f"{et}={edge.value:.2f}"
    return et


def export_graph_json(store: InMemoryGraphStore, out_path: str) -> None:
    data = {
        "nodes": [n.model_dump(mode="json") for n in store._nodes.values()],
        "edges": [e.model_dump(mode="json") for e in store._edges.values()],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"  → {os.path.abspath(out_path)}")


def export_graph_html(store: InMemoryGraphStore, out_path: str) -> None:
    """Render store as an interactive pyvis HTML file (self-contained, double-click to open)."""
    try:
        from pyvis.network import Network  # type: ignore
    except ImportError:
        print("  pyvis not installed — run:  pip install pyvis")
        json_fallback = out_path.replace(".html", ".json")
        print(f"  Falling back to JSON export:")
        export_graph_json(store, json_fallback)
        return

    net = Network(
        height="780px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
    )
    net.set_options("""{
        "nodes": {
            "font": {"size": 14, "face": "monospace"},
            "borderWidth": 2
        },
        "edges": {
            "font": {"size": 10, "color": "#bbbbbb", "align": "middle"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 120}
        }
    }""")

    _NODE_COLOUR = {
        "robot":  "#4A90D9",
        "person": "#5CB85C",
        "topic":  "#F0AD4E",
        "event":  "#D9534F",
    }
    _NODE_SHAPE = {
        "robot":  "star",
        "person": "ellipse",
        "topic":  "box",
        "event":  "diamond",
    }

    for node in store._nodes.values():
        label = getattr(node, "display_name", None) or node.id
        tooltip_parts = [f"type: {node.node_type}", f"id: {node.id}"]
        if hasattr(node, "embodiment"):
            tooltip_parts.append(f"embodiment: {node.embodiment}")
        net.add_node(
            node.id,
            label=label,
            color=_NODE_COLOUR.get(node.node_type, "#AAAAAA"),
            shape=_NODE_SHAPE.get(node.node_type, "ellipse"),
            title="<br>".join(tooltip_parts),
            size=30 if node.node_type == "robot" else 20,
        )

    for edge in store._edges.values():
        lbl = _edge_label(edge)
        tip = (f"type: {edge.edge_type}<br>"
               f"source: {edge.source_id}<br>"
               f"target: {edge.target_id}")
        net.add_edge(
            edge.source_id,
            edge.target_id,
            label=lbl,
            title=tip,
        )

    net.write_html(out_path)
    print(f"  → {os.path.abspath(out_path)}")


def export_obsidian_vault(store: InMemoryGraphStore, vault_dir: str) -> None:
    """Export one Markdown file per node with [[wikilinks]] on edges."""
    os.makedirs(vault_dir, exist_ok=True)

    def _slug(node_id: str, node_type: str) -> str:
        return f"{node_type}_{node_id.replace('-', '_')}"

    for node in store._nodes.values():
        ns = _slug(node.id, node.node_type)
        outgoing = [e for e in store._edges.values() if e.source_id == node.id]
        incoming = [e for e in store._edges.values()
                    if e.target_id == node.id and e.source_id != node.id]

        lines: list[str] = [f"# {node.node_type}: {node.id}", ""]
        lines.append(f"**type**: `{node.node_type}`")
        if hasattr(node, "embodiment"):
            lines.append(f"**embodiment**: `{node.embodiment}`")
        lines.append("")

        if outgoing:
            lines.append("## Outgoing edges")
            for e in outgoing:
                tgt = store._nodes.get(e.target_id)
                if tgt:
                    lines.append(f"- `{_edge_label(e)}` → [[{_slug(tgt.id, tgt.node_type)}]]")
            lines.append("")

        if incoming:
            lines.append("## Incoming edges")
            for e in incoming:
                src = store._nodes.get(e.source_id)
                if src:
                    lines.append(f"- [[{_slug(src.id, src.node_type)}]] → `{_edge_label(e)}`")
            lines.append("")

        filepath = os.path.join(vault_dir, f"{ns}.md")
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    n_files = len(store._nodes)
    print(f"  → {os.path.abspath(vault_dir)}  ({n_files} markdown files)")


# ── Harness class ─────────────────────────────────────────────────────────────

class Harness:
    """
    Owns the shared InMemoryGraphStore and runs one turn at a time.

    The store is created once at construction and lives for the whole run —
    per-person state MUST persist across turns; this is the entire point of the demo.
    """

    def __init__(self, robot_id: str = "chatbox", obsidian: bool = False) -> None:
        self.store   = InMemoryGraphStore()
        self.bridge  = KGBridge(self.store)
        self.robot_id = robot_id
        self.obsidian = obsidian
        self.turn_n  = 0
        self._pad_adapters: dict[str, PADPipelineAdapter] = {}

    def _adapter(self, robot_id: str) -> PADPipelineAdapter:
        if robot_id not in self._pad_adapters:
            self._pad_adapters[robot_id] = PADPipelineAdapter(robot_id)
        return self._pad_adapters[robot_id]

    # ── Core turn ─────────────────────────────────────────────────────────────

    def run_turn(
        self,
        person_id: str,
        robot_id: str,
        emotion: str,
    ) -> dict:
        """
        Run one full KG → PAD → KG loop iteration and print a readable trace.

        Returns the pad_result dict (keys: system_prompt, gesture_params,
        pad_state, descriptors).
        """
        self.turn_n += 1

        # 1. Read KG state → derive tier, blend valence, fetch slow-edge memory
        bi = self.bridge.pre_turn(person_id, robot_id, emotion)

        # 2. PAD update — tier drives D-axis; V/A come from camera_emotion via bridge
        pad_result = self._adapter(robot_id).process_turn(
            valence=bi.valence,
            arousal=bi.arousal,
            relationship_tier=bi.tier,
            memory_context=bi.structured_memory,
        )

        # 3. Write PAD output back to KG (mood, attention, interaction_count)
        self.bridge.post_turn(person_id, robot_id, pad_result)

        # 4. Read updated edges AFTER post_turn so interaction_count reflects this turn
        rel = _relationship_snapshot(self.store, person_id, robot_id)

        # 5. Print trace
        p, a, d = pad_result["pad_state"]
        desc     = pad_result["descriptors"]
        mem_str  = bi.structured_memory or ""
        tc       = _TIER_COLOUR.get(bi.tier, "")

        print(
            f"turn {self.turn_n:3d} | "
            f"person={person_id:<10s} robot={robot_id:<8s} emotion={emotion}"
        )
        print(
            f"         → tier={tc}{bi.tier:<8s}{_RST} "
            f" v={bi.valence:+.2f}  a={bi.arousal:+.2f}"
        )
        print(
            f"         → PAD  P={p:+.3f}  A={a:+.3f}  D={d:+.3f}"
            f"   descriptors={desc['pleasure']}/{desc['arousal']}/{desc['dominance']}"
        )
        print(f"         → mem={mem_str!r}")
        print(
            f"         [graph]  rapport={rel['rapport']:.2f}"
            f"  trust={rel['trust']:.2f}"
            f"  interaction_count={rel['interaction_count']}"
        )

        return pad_result

    # ── Summary view ──────────────────────────────────────────────────────────

    def show_people(self) -> None:
        persons = [n for n in store._nodes.values() if n.node_type == "person"
                   ] if False else [
            n for n in self.store._nodes.values() if n.node_type == "person"
        ]
        if not persons:
            print("  (no people in store yet)")
            return
        print(f"  {'Person':<12s}  {'Tier':<10s}  {'Count':>5s}  {'Rapport':>7s}  {'Trust':>5s}")
        print("  " + "─" * 48)
        for node in sorted(persons, key=lambda n: n.id):
            ctx  = self.store.get_person_context(node.id)
            tier = derive_tier(ctx.relationship_edges)
            rel  = _relationship_snapshot(self.store, node.id, self.robot_id)
            tc   = _TIER_COLOUR.get(tier, "")
            print(
                f"  {node.id:<12s}  {tc}{tier:<10s}{_RST}"
                f"  {rel['interaction_count']:>5d}"
                f"  {rel['rapport']:>7.2f}"
                f"  {rel['trust']:>5.2f}"
            )

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, out_dir: str = ".") -> None:
        html_path = os.path.join(out_dir, "graph_snapshot.html")
        print("Exporting graph …")
        export_graph_html(self.store, html_path)
        if self.obsidian:
            vault_path = os.path.join(out_dir, "vault")
            export_obsidian_vault(self.store, vault_path)


# ── Scripted proof sequence ───────────────────────────────────────────────────

_DIVIDER = "─" * 72

def run_scripted(robot_id: str = "chatbox", obsidian: bool = False) -> None:
    """
    Fixed proof sequence — no typing required.

    Demonstrates three properties:
      1. Same person over many turns: interaction_count climbs, tier escalates,
         D-axis shifts.
      2. Two people, same emotion, different tiers → divergent PAD / descriptors.
      3. Brand-new person mid-run cold-starts at tier=unknown without crashing.
    """
    h = Harness(robot_id=robot_id, obsidian=obsidian)

    print("=" * 72)
    print("  SCRIPTED DEMO — KG × PAD per-person adaptation")
    print(f"  robot: {robot_id}")
    print("=" * 72)

    # ── PROOF 1: same person across many turns ────────────────────────────────
    print()
    print(_DIVIDER)
    print("PROOF 1 — alice × 6 turns (happy): tier escalation, D-axis shift")
    print("  expect: unknown (turn 1) → visitor (turns 2–6, count 1→5 at pre_turn)")
    print("  After 6 turns: count=6 in store; next pre_turn will see 'known'.")
    print(_DIVIDER)
    print()
    for _ in range(6):
        h.run_turn("alice", robot_id, "happy")
        print()

    # ── bob: 2 turns — builds visitor tier ───────────────────────────────────
    print(_DIVIDER)
    print("PROOF 2 setup — bob × 2 turns (neutral): builds to visitor tier")
    print(_DIVIDER)
    print()
    for _ in range(2):
        h.run_turn("bob", robot_id, "neutral")
        print()

    # ── Boost alice to 'close' ────────────────────────────────────────────────
    print(_DIVIDER)
    print("BOOST alice — rapport=0.75  trust=0.75  (score=0.75 > 0.70 → 'close')")
    print("Combined with count=6 > 5, alice is now comfortably 'close'.")
    print(_DIVIDER)
    print()
    seed_relationship(h.store, "alice", robot_id, rapport=0.75, trust=0.75)
    print()

    # ── PROOF 2: side-by-side divergence ─────────────────────────────────────
    print(_DIVIDER)
    print("PROOF 2 — alice (close) vs bob (visitor) — same emotion: happy")
    print("  Watch D-axis and 'dominance' descriptor diverge.")
    print(_DIVIDER)
    print()
    print("[alice — close tier]")
    h.run_turn("alice", robot_id, "happy")
    print()
    print("[bob — visitor tier]")
    h.run_turn("bob", robot_id, "happy")
    print()

    # ── PROOF 3: cold-start ───────────────────────────────────────────────────
    print(_DIVIDER)
    print("PROOF 3 — casey cold-start (never seen before)")
    print("  expect: tier=unknown, mem='', no crash")
    print(_DIVIDER)
    print()
    h.run_turn("casey", robot_id, "happy")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(_DIVIDER)
    print("Final people state")
    print(_DIVIDER)
    h.show_people()
    print()

    # ── Graph export ──────────────────────────────────────────────────────────
    print(_DIVIDER)
    h.export(".")
    print()

    print("=" * 72)
    print("  SCRIPTED DEMO COMPLETE")
    print("=" * 72)


# ── Interactive loop ──────────────────────────────────────────────────────────

_HELP = """\
Commands
────────
  <person> <emotion>          run one turn   e.g.  alice happy
  boost <person> [r] [t]      TEST AID: seed rapport/trust  e.g.  boost alice 0.75 0.75
  robot <chatbox|ellebot>     switch active robot
  who                         list all known people and their tiers
  graph                       export graph snapshot right now
  help                        show this message
  q / quit                    exit (exports graph before quitting)

Emotions:  happy  sad  neutral  angry  calm  fear  disgust  surprise
"""


def run_interactive(robot_id: str = "chatbox", obsidian: bool = False) -> None:
    h = Harness(robot_id=robot_id, obsidian=obsidian)

    print("=" * 72)
    print("  KG × PAD Integration Harness  —  interactive mode")
    print(f"  robot: {robot_id}   |   type 'help' for commands")
    print("=" * 72)
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            break

        if cmd == "help":
            print(_HELP)
            continue

        if cmd == "who":
            h.show_people()
            print()
            continue

        if cmd == "graph":
            h.export(".")
            print()
            continue

        if cmd == "robot":
            if len(parts) < 2 or parts[1].lower() not in VALID_ROBOTS:
                print(f"  Usage: robot <{'|'.join(sorted(VALID_ROBOTS))}>")
            else:
                h.robot_id = parts[1].lower()
                print(f"  Robot switched to: {h.robot_id}")
            print()
            continue

        if cmd == "boost":
            if len(parts) < 2:
                print("  Usage: boost <person> [rapport=0.5] [trust=0.5]")
                print()
                continue
            person_id = parts[1]
            try:
                rapport = float(parts[2]) if len(parts) > 2 else 0.5
                trust   = float(parts[3]) if len(parts) > 3 else 0.5
            except ValueError:
                print("  rapport and trust must be floats in [0, 1]")
                print()
                continue
            seed_relationship(h.store, person_id, h.robot_id, rapport, trust)
            print()
            continue

        # Two-token: <person> <emotion>
        if len(parts) == 2:
            person_id, emotion = parts[0], parts[1].lower()
            if emotion not in VALID_EMOTIONS:
                print(f"  Unknown emotion '{emotion}'.")
                print(f"  Valid: {', '.join(sorted(VALID_EMOTIONS))}")
                print()
                continue
            h.run_turn(person_id, h.robot_id, emotion)
            print()
            continue

        # One token: just a person name → prompt for emotion
        if len(parts) == 1 and cmd not in VALID_ROBOTS:
            person_id = parts[0]
            try:
                emotion = input(f"  emotion for {person_id}? ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if emotion not in VALID_EMOTIONS:
                print(f"  Unknown emotion '{emotion}'.")
                print(f"  Valid: {', '.join(sorted(VALID_EMOTIONS))}")
                print()
                continue
            h.run_turn(person_id, h.robot_id, emotion)
            print()
            continue

        print(f"  Unrecognised input: {raw!r}  —  type 'help' for commands")
        print()

    print()
    print("Exiting — writing final graph …")
    h.export(".")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="KG × PAD integration harness — proves per-person KG adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--scripted",
        action="store_true",
        help="Run the fixed proof sequence (alice×6, bob×2, boost, casey) then exit",
    )
    ap.add_argument(
        "--robot",
        choices=sorted(VALID_ROBOTS),
        default="chatbox",
        metavar="ROBOT",
        help="Robot persona to use: chatbox (default) or ellebot",
    )
    ap.add_argument(
        "--obsidian",
        action="store_true",
        help="Also write an Obsidian-compatible markdown vault to ./vault/",
    )
    args = ap.parse_args(argv)

    if args.scripted:
        run_scripted(robot_id=args.robot, obsidian=args.obsidian)
    else:
        run_interactive(robot_id=args.robot, obsidian=args.obsidian)


if __name__ == "__main__":
    main()
