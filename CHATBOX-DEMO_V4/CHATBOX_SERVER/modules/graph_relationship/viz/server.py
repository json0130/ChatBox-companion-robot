"""
Standalone, decoupled live visualizer server for the graph_relationship KG.

Design contract (do not break — keeps graph_relationship/ copy-pasteable):
  * The ONLY data source is the on-disk kg_state.json written by
    InMemoryGraphStore.save(path). This server never imports store.py,
    schema.py, kg_bridge.py, pad_persona, or any adapter. It only reads and
    parses JSON off disk.
  * No in-process hooks, observers, events, or websockets. The browser polls
    /graph.json; this server re-reads the file on each request.
  * Never returns 500 for a missing / partially-written file. It caches the
    last successfully parsed graph and serves that (or an empty graph) instead.

kg_state.json shape (from InMemoryGraphStore.save):
  {
    "nodes": [ { "id", "node_type": person|robot|topic|event,
                 "name"|"display_name"|"label", ... }, ... ],
    "edges": [ { "id", "source_id", "target_id", "edge_type",
                 "provenance": {...}, "weight"|"value"|"count", ... }, ... ]
  }

Run:
    python3 -m graph_relationship.viz.server --kg-path kg_state.json --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_HTML = os.path.join(_HERE, "index.html")

# --------------------------------------------------------------------------
# edge_type -> timescale bucket, extracted from schema.py.
#   FAST         : mood, attention, current_topic   (decay within a session)
#   SLOW         : trait, preference                 (stable across sessions)
#   RELATIONSHIP : rapport, trust, disclosure_depth, interaction_count
# Hardcoded (not imported) so this viz folder stays fully self-contained.
# --------------------------------------------------------------------------
_TIMESCALE_BY_EDGE_TYPE = {
    "mood": "FAST",
    "attention": "FAST",
    "current_topic": "FAST",
    "trait": "SLOW",
    "preference": "SLOW",
    "rapport": "RELATIONSHIP",
    "trust": "RELATIONSHIP",
    "disclosure_depth": "RELATIONSHIP",
    "interaction_count": "RELATIONSHIP",
    # Event participation links person+robot through a turn (rerouted interaction).
    "participated_in": "RELATIONSHIP",
    # Authored identity edges (seed.py) — SLOW, cross-session.
    "has_persona": "SLOW",
    "has_role": "SLOW",
    "has_style": "SLOW",
    "has_capability": "SLOW",
}

# node_type -> display type (frontend maps this to a shape)
_NODE_TYPE_DISPLAY = {
    "person": "Person",
    "robot": "Robot",
    "topic": "Topic",
    "event": "Event",
    # Authored-attribute subnodes (seed.py).
    "persona": "Persona",
    "role": "Role",
    "style": "Style",
    "capability": "Capability",
}


def _node_label(node: dict) -> str:
    """Human-readable label: name (Robot), display_name (Person), label/descriptor."""
    for key in ("display_name", "name", "label", "descriptor"):
        val = node.get(key)
        if val:
            return str(val)
    return str(node.get("id", "?"))


def _edge_weight(edge: dict) -> float:
    """Single numeric magnitude for thickness/label: weight, else count, else |value|."""
    if "weight" in edge and edge["weight"] is not None:
        return float(edge["weight"])
    if "count" in edge and edge["count"] is not None:
        return float(edge["count"])
    if "value" in edge and edge["value"] is not None:
        return float(edge["value"])
    return 1.0


def transform(raw: dict) -> dict:
    """Turn the raw kg_state.json dict into {nodes:[...], edges:[...]} for the UI.

    Event (session) nodes carry their `turns` transcript and `turn_count`, and
    their label shows the turn count so the click panel can render the session.
    """
    node_ids = set()
    nodes = []
    for n in raw.get("nodes", []):
        nid = n.get("id")
        if not nid:
            continue
        node_ids.add(nid)
        node_type = n.get("node_type")
        obj = {
            "id": nid,
            "type": _NODE_TYPE_DISPLAY.get(node_type, "Topic"),
            "label": _node_label(n),
        }
        if node_type == "event":
            turns = n.get("turns", []) or []
            obj["turns"] = turns
            obj["turn_count"] = n.get("turn_count", len(turns))
            obj["label"] = f"{obj['label']} ({obj['turn_count']} turns)"
        nodes.append(obj)

    edges = []
    for e in raw.get("edges", []):
        src, tgt = e.get("source_id"), e.get("target_id")
        # Drop dangling edges — the frontend force layout needs both endpoints.
        if src not in node_ids or tgt not in node_ids:
            continue
        et = e.get("edge_type", "")
        edges.append({
            "source": src,
            "target": tgt,
            "type": et,
            "timescale": _TIMESCALE_BY_EDGE_TYPE.get(et, "RELATIONSHIP"),
            "weight": round(_edge_weight(e), 3),
        })

    return {"nodes": nodes, "edges": edges}


def build_history(raw: dict) -> dict:
    """Per-person transcript aggregated from session Event nodes: reconstructed
    purely from the graph JSON (no separate transcript file), so kg_state.json
    stays the single source of truth.

    Returns { person_id: [ {turn, ts, emotion, child, reply, session}, ... ] }.
    """
    nodes = {n["id"]: n for n in raw.get("nodes", []) if n.get("id")}
    history: dict = {}
    for e in raw.get("edges", []):
        if e.get("edge_type") != "participated_in":
            continue
        participant = nodes.get(e.get("source_id"))
        event = nodes.get(e.get("target_id"))
        if not participant or not event or event.get("node_type") != "event":
            continue
        if participant.get("node_type") != "person":
            continue  # attribute each session to the person, not the robot
        pid = participant["id"]
        session = event.get("label", "session")
        started = event.get("timestamp", "")
        for t in event.get("turns", []) or []:
            history.setdefault(pid, []).append({**t, "session": session, "_started": started})
    # Chronological: by session start, then turn index.
    for pid, turns in history.items():
        turns.sort(key=lambda t: (t.get("_started", ""), t.get("turn", 0)))
        for t in turns:
            t.pop("_started", None)
    return history


class GraphState:
    """Reads kg_state.json on demand, caching the last good parse. History is
    derived from the same file (session Event nodes)."""

    def __init__(self, kg_path: str):
        self.kg_path = kg_path
        self._last_good_raw: dict = {"nodes": [], "edges": []}

    def _raw(self) -> dict:
        try:
            with open(self.kg_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._last_good_raw = raw
            return raw
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            # Missing or mid-write (partial JSON): use the last good parse.
            return self._last_good_raw

    def read(self) -> dict:
        return transform(self._raw())

    def read_history(self) -> dict:
        return build_history(self._raw())


def make_handler(state: GraphState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep console clean
            pass

        def _send(self, code, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                try:
                    with open(_INDEX_HTML, "rb") as fh:
                        body = fh.read()
                    self._send(200, body, "text/html; charset=utf-8")
                except OSError:
                    self._send(404, b"index.html not found", "text/plain")
            elif path == "/graph.json":
                body = json.dumps(state.read()).encode("utf-8")
                self._send(200, body, "application/json")
            elif path == "/history.json":
                body = json.dumps(state.read_history()).encode("utf-8")
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Live browser visualizer for the KG.")
    ap.add_argument("--kg-path", default="kg_state.json",
                    help="Path to kg_state.json written by InMemoryGraphStore.save "
                         "(default: kg_state.json in the current directory)")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    args = ap.parse_args()

    kg_path = os.path.abspath(args.kg_path)
    state = GraphState(kg_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    print(f"[viz] serving KG visualizer at  http://{args.host}:{args.port}/")
    print(f"[viz] polling KG file:          {state.kg_path}")
    if not os.path.exists(state.kg_path):
        print("[viz] (file not present yet — will show empty graph until it appears)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
