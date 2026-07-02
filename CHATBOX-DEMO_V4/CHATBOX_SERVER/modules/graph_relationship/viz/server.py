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
}

# node_type -> display type (frontend maps this to a shape)
_NODE_TYPE_DISPLAY = {
    "person": "Person",
    "robot": "Robot",
    "topic": "Topic",
    "event": "Event",
}


def _node_label(node: dict) -> str:
    """Human-readable label: name (Robot), display_name (Person), label (Topic/Event)."""
    for key in ("display_name", "name", "label"):
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
    """Turn the raw kg_state.json dict into {nodes:[...], edges:[...]} for the UI."""
    node_ids = set()
    nodes = []
    for n in raw.get("nodes", []):
        nid = n.get("id")
        if not nid:
            continue
        node_ids.add(nid)
        nodes.append({
            "id": nid,
            "type": _NODE_TYPE_DISPLAY.get(n.get("node_type"), "Topic"),
            "label": _node_label(n),
        })

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


def conversation_path(kg_path: str) -> str:
    """Derive the transcript sidecar path from the KG path (mirrors the writer).

    kg_state.json -> kg_state_conversations.json  (same directory).
    """
    base, ext = os.path.splitext(kg_path)
    return f"{base}_conversations{ext or '.json'}"


class GraphState:
    """Reads kg_state.json (+ transcript sidecar) on demand, caching last good."""

    def __init__(self, kg_path: str, conv_path: str):
        self.kg_path = kg_path
        self.conv_path = conv_path
        self._last_good = {"nodes": [], "edges": []}
        self._last_good_history: dict = {}

    def read(self) -> dict:
        try:
            with open(self.kg_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            graph = transform(raw)
            self._last_good = graph
            return graph
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            # Missing or mid-write (partial JSON): serve the last good graph.
            return self._last_good

    def read_history(self) -> dict:
        """Per-person transcript { person_id: [ {turn, ts, child, reply, ...} ] }.

        Optional: absent sidecar just yields an empty history (panel shows
        nothing), so a copied graph_relationship/ that only writes kg_state.json
        still works.
        """
        try:
            with open(self.conv_path, "r", encoding="utf-8") as fh:
                hist = json.load(fh)
            if isinstance(hist, dict):
                self._last_good_history = hist
            return self._last_good_history
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            return self._last_good_history


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
    ap.add_argument("--conv-path", default=None,
                    help="Path to the conversation transcript sidecar "
                         "(default: derived from --kg-path, e.g. "
                         "kg_state_conversations.json)")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    args = ap.parse_args()

    kg_path = os.path.abspath(args.kg_path)
    conv_path = os.path.abspath(args.conv_path) if args.conv_path else conversation_path(kg_path)
    state = GraphState(kg_path, conv_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    print(f"[viz] serving KG visualizer at  http://{args.host}:{args.port}/")
    print(f"[viz] polling KG file:          {state.kg_path}")
    print(f"[viz] polling transcript file:  {state.conv_path}")
    if not os.path.exists(state.kg_path):
        print("[viz] (file not present yet — will show empty graph until it appears)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
