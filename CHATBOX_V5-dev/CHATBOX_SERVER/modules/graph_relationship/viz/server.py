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

Chat (the ONLY part that imports CHATBOX code): the server also answers POST /chat via
modules.persona_chat — a robot-only conversation (role + persona, no information about
the visitor). Backend: the lab AI server by default ($AI_HOST or the lab IP),
--ai-server HOST[:PORT] to pick another, --ollama [MODEL] for a local model, --no-chat
to switch it off (then nothing here imports anything, as above).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_HTML = os.path.join(_HERE, "index.html")

# --------------------------------------------------------------------------
# edge_type -> timescale bucket, extracted from schema.py.
#   FAST         : mood, attention   (decay within a session)
#   SLOW         : trait, preference                 (stable across sessions)
#   RELATIONSHIP : rapport, trust, disclosure_depth, interaction_count
# Hardcoded (not imported) so this viz folder stays fully self-contained.
# --------------------------------------------------------------------------
_TIMESCALE_BY_EDGE_TYPE = {
    "mood": "FAST",
    "attention": "FAST",
    "trait": "SLOW",
    "preference": "SLOW",
    "rapport": "RELATIONSHIP",
    "trust": "RELATIONSHIP",
    "disclosure_depth": "RELATIONSHIP",
    "interaction_count": "RELATIONSHIP",
    # Interaction abstraction: person+robot <-> Interaction, Interaction -> Session.
    "has_interaction": "RELATIONSHIP",
    "has_session": "RELATIONSHIP",
    "has_conversation": "FAST",
    # Authored identity edges (seed.py) — SLOW, cross-session.
    "has_persona": "SLOW",
    "has_role": "SLOW",
    "has_capability": "SLOW",
    # Shared-topic layer (seed.py) — SLOW.
    "has_interest": "SLOW",
    "about": "SLOW",
    # Topic <-> Topic semantic relation (Feature-2c) — SLOW.
    "related_topic": "SLOW",
    # Culture layer (Command A) — SLOW, cross-session.
    "knows_culture": "SLOW",
    "belongs_to_culture": "SLOW",
    "culture_prior": "SLOW",
    # Role layer (roles.py) — SLOW.
    "role_knows": "SLOW",
}

# node_type -> display type (frontend maps this to a shape)
_NODE_TYPE_DISPLAY = {
    "person": "Person",
    "robot": "Robot",
    "topic": "Topic",
    # Authored-attribute subnodes (seed.py).
    "persona": "Persona",
    "role": "Role",
    "capability": "Capability",
    "interest": "Interest",
    # Interaction abstraction (interactions.py).
    "interaction": "Interaction",
    "session": "Session",
    # Live conversation-status node (topics.py update_conversation).
    "conversation": "Conversation",
    # Culture background node (cultures.py) — Command A.
    "culture": "Culture",
    # Robot-owned cultural-knowledge topic (cultures.py) — Command A.
    "culture_topic": "CultureTopic",
    # Knowledge that comes with a role (roles.py).
    "role_topic": "RoleTopic",
}


def _node_label(node: dict) -> str:
    """Human-readable label for each node type."""
    nt = node.get("node_type")
    if nt == "capability":
        items = node.get("items") or []
        return ", ".join(str(i) for i in items) if items else "capabilities"
    if nt == "interaction":
        return (f"interaction  rapport {node.get('rapport', 0):.2f} · "
                f"trust {node.get('trust', 0):.2f} · {node.get('interaction_count', 0)} turns")
    for key in ("display_name", "name", "label", "descriptor"):
        val = node.get(key)
        if val:
            return str(val)
    return str(node.get("id", "?"))


def _edge_weight(edge: dict) -> float:
    """Single numeric magnitude for thickness/label: weight/prior, else count, else |value|."""
    if "weight" in edge and edge["weight"] is not None:
        return float(edge["weight"])
    if "prior" in edge and edge["prior"] is not None:      # culture_prior edges
        return float(edge["prior"])
    if "count" in edge and edge["count"] is not None:
        return float(edge["count"])
    if "value" in edge and edge["value"] is not None:
        return float(edge["value"])
    return 1.0


def transform(raw: dict) -> dict:
    """Turn the raw kg_state.json dict into {nodes:[...], edges:[...]} for the UI.

    Session nodes carry their `turns` transcript and `turn_count`, and their
    label shows the turn count so the click panel can render the session.
    """
    # Pre-scan FAST self-edges (mood/attention are person→person) so the current
    # mood + emotion can be folded onto the person's label instead of drawn as an
    # invisible zero-length self-loop.
    mood_by_person: dict = {}
    for e in raw.get("edges", []):
        if e.get("edge_type") == "mood" and e.get("source_id") == e.get("target_id"):
            mood_by_person[e.get("source_id")] = {
                "value": e.get("value"), "label": e.get("label"),
            }

    def _mood_face(val) -> str:
        return "🙂" if (val or 0) > 0.15 else ("🙁" if (val or 0) < -0.15 else "😐")

    def _mood_bits(val, emo) -> str:
        bits = [_mood_face(val)]
        if emo:
            bits.append(str(emo))
        if val is not None:
            bits.append(f"({float(val):+.2f})")
        return " ".join(bits)

    def _mood_suffix(nid: str) -> str:
        m = mood_by_person.get(nid)
        if not m:
            return ""
        return "  " + _mood_bits(m.get("value"), m.get("label"))

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
        if node_type == "person":
            obj["label"] = obj["label"] + _mood_suffix(nid)
        elif node_type == "conversation":
            # Live status node: "▶ topic1 · topic2 · topic3   🙂 emotion (+0.55)".
            topics = n.get("topics", []) or []
            head = "▶ " + " · ".join(str(t) for t in topics) if topics else "▶ (talking…)"
            mood = _mood_bits(n.get("mood"), n.get("emotion")) \
                if (n.get("mood") is not None or n.get("emotion")) else ""
            obj["label"] = head + (("   " + mood) if mood else "")
            obj["topics"] = topics
            obj["current"] = True
        if node_type == "session":
            turns = n.get("turns", []) or []
            obj["turns"] = turns
            obj["turn_count"] = n.get("turn_count", len(turns))
            obj["label"] = f"{obj['label']} ({obj['turn_count']} turns)"
        elif node_type == "topic":
            notes = n.get("notes", []) or []
            obj["notes"] = notes
            obj["category"] = n.get("category", "other")   # fine-grained topic type
            obj["topicLabel"] = n.get("label", "")          # clean label for /history
            if notes:
                obj["label"] = f"{obj['label']} ({len(notes)})"
        elif node_type == "culture_topic":
            obj["category"] = n.get("category", "other")   # robot-owned culture topic
            obj["facts"] = n.get("facts", []) or []         # shareable cultural facts
        elif node_type == "culture":
            obj["styleHint"] = n.get("style_hint", "") or ""  # static manner hint
        elif node_type == "role_topic":
            obj["kind"] = n.get("kind", "knowledge")        # goal/manner/do/avoid/phrase/knowledge
            obj["facts"] = n.get("items", []) or []         # its lines (shown in the panel)
        nodes.append(obj)

    edges = []
    for e in raw.get("edges", []):
        src, tgt = e.get("source_id"), e.get("target_id")
        # Drop dangling edges — the frontend force layout needs both endpoints.
        if src not in node_ids or tgt not in node_ids:
            continue
        # Skip self-edges (mood/attention) — folded onto the person label above;
        # a zero-length self-loop would not render usefully anyway.
        if src == tgt:
            continue
        et = e.get("edge_type", "")
        obj = {
            "source": src,
            "target": tgt,
            "type": et,
            "timescale": _TIMESCALE_BY_EDGE_TYPE.get(et, "RELATIONSHIP"),
            "weight": round(_edge_weight(e), 3),
        }
        if e.get("label"):          # capability→topic 'knows jazz' etc.
            obj["elabel"] = str(e["label"])
        if et == "related_topic" and (
                str(src).startswith("ck:") != str(tgt).startswith("ck:")):
            obj["cross"] = True     # Step-2 person topic: ↔ culture ck: bridge
        if et == "about":           # person interest→topic affinity/confidence
            # affinity is stored internally in [0,1]; show it on the human 0–10 scale
            # (display-only mirror of scales.aff10_from_01, kept inline so this
            # self-contained viz server needs no package import).
            aff = e.get("affinity")
            if aff is not None:
                obj["affinity10"] = round(max(0.0, min(1.0, float(aff))) * 10)
            conf = e.get("confidence")
            if conf is not None:
                obj["confidence"] = round(float(conf), 2)
        edges.append(obj)

    return {"nodes": nodes, "edges": edges}


class GraphState:
    """Reads kg_state.json on demand, caching the last good parse."""

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
        g = transform(self._raw())
        g["active"] = self.read_active_state()   # {person, culture} for viz highlight
        return g

    def read_active_state(self) -> dict:
        """The loop's current focus sidecar — display-only. Includes `live` (the loop
        wrote it within the last few seconds — else it's not running, so the viz must
        NOT dim) and `present` (a face, even unknown, is on camera). {} when absent."""
        import time
        p = os.path.join(os.path.dirname(os.path.abspath(self.kg_path)),
                         "active_state.json")
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            live = (time.time() - float(d.get("ts", 0))) < 5.0
            return {"person": d.get("person"), "culture": d.get("culture"),
                    "present": bool(d.get("present")), "live": live}
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError, TypeError):
            return {}

    def _read_fresh(self) -> dict:
        """Read the file directly (no cache) for a read-modify-write delete."""
        with open(self.kg_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, raw: dict) -> None:
        with open(self.kg_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, default=str)
        self._last_good_raw = raw

    # ── culture override (testing knob shared with the webcam loop) ───────────
    # A tiny sidecar file next to kg_state.json: the loop reads it each turn to
    # force the ACTIVE culture ('korean'/'maori'), turn culture off ('generic'), or
    # follow the person ('auto'). Kept OUT of kg_state.json so the two processes
    # never fight over that file.

    def _override_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.kg_path)),
                            "culture_override.json")

    def get_culture_override(self) -> str:
        try:
            with open(self._override_path(), "r", encoding="utf-8") as fh:
                return (json.load(fh).get("active_culture") or "auto").strip().lower()
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            return "auto"

    def set_culture_override(self, value: str) -> str:
        v = (value or "auto").strip().lower()
        with open(self._override_path(), "w", encoding="utf-8") as fh:
            json.dump({"active_culture": v}, fh)
        return v

    # ── role / persona assignment (sidecar, read by the webcam loop) ───────────
    # {robot_id: {"role": "role:tutor", "ocean": {O,C,E,A,N}, "ts": iso}}. The loop
    # applies it to the graph (attach the role, build the persona); this server
    # only validates and writes the request, so it still never edits schema objects.

    _TRAITS = ("O", "C", "E", "A", "N")

    def _persona_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.kg_path)),
                            "persona_assignment.json")

    def persona_choices(self) -> dict:
        """Robots, library roles and what each robot currently plays (from the KG)."""
        raw = self._raw()
        nodes = {n.get("id"): n for n in raw.get("nodes", [])}
        robots = sorted(i for i, n in nodes.items() if n.get("node_type") == "robot")
        roles = {i: {"id": i, "label": n.get("descriptor", i),
                     "brief": n.get("brief", ""), "parts": {}}
                 for i, n in nodes.items()
                 if n.get("node_type") == "role" and str(i).startswith("role:")}
        current = {r: {} for r in robots}
        for e in raw.get("edges", []):
            src, tgt = e.get("source_id"), e.get("target_id")
            if e.get("edge_type") == "role_knows" and src in roles and tgt in nodes:
                part = nodes[tgt]                     # role subnode → summary
                roles[src]["parts"].setdefault(part.get("kind", "knowledge"), []).extend(
                    part.get("items", []) if part.get("kind") != "knowledge"
                    else [part.get("label", "")])
                continue
            if src not in current or tgt not in nodes:
                continue
            if e.get("edge_type") == "has_role":
                current[src]["role"] = tgt
                current[src]["role_label"] = nodes[tgt].get("descriptor", tgt)
            elif e.get("edge_type") == "has_persona":
                current[src]["persona"] = nodes[tgt].get("descriptor", "")
                current[src]["ocean"] = nodes[tgt].get("ocean")
        try:                                   # what was last REQUESTED (chat uses this)
            with open(self._persona_path(), "r", encoding="utf-8") as fh:
                assigned = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            assigned = {}
        return {"robots": robots, "roles": sorted(roles.values(), key=lambda r: r["id"]),
                "current": current, "assigned": assigned}

    def set_persona(self, robot: str, role: str, ocean: dict) -> dict:
        from datetime import datetime, timezone
        choices = self.persona_choices()
        if robot not in choices["robots"]:
            raise ValueError(f"unknown robot {robot!r}")
        if role not in {r["id"] for r in choices["roles"]}:
            raise ValueError(f"unknown role {role!r}")
        try:
            ocean = {k: max(-1.0, min(1.0, float(ocean[k]))) for k in self._TRAITS}
        except (KeyError, TypeError, ValueError):
            raise ValueError("ocean needs numeric O/C/E/A/N")
        path = self._persona_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            data = {}
        data[robot] = {"role": role, "ocean": ocean,
                       "ts": datetime.now(timezone.utc).isoformat()}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)          # atomic: the loop never reads half a file
        return data[robot]

    def culture_labels(self) -> list:
        """Labels of CultureNodes the robot knows — the choices for the selector."""
        return sorted(n.get("label", "") for n in self._raw().get("nodes", [])
                      if n.get("node_type") == "culture" and n.get("label"))

    def delete_node(self, node_id: str) -> dict:
        """Remove a node and every edge touching it. CASCADES for a person: also
        removes their interests / interaction / conversation subnodes (and thus the
        fast has_conversation link to the robot), so no orphans are left behind.
        Returns removed counts."""
        raw = self._read_fresh()
        victims = {node_id}
        node = next((n for n in raw.get("nodes", []) if n.get("id") == node_id), None)
        if node is not None and node.get("node_type") == "person":
            for n in raw.get("nodes", []):
                nid = n.get("id", "")
                if (nid.startswith(f"interest:{node_id}:")
                        or nid.startswith(f"interaction:{node_id}:")
                        or nid.startswith(f"conversation:{node_id}:")):
                    victims.add(nid)
        nodes = [n for n in raw.get("nodes", []) if n.get("id") not in victims]
        edges = [e for e in raw.get("edges", [])
                 if e.get("source_id") not in victims and e.get("target_id") not in victims]
        removed = {"nodes": len(raw.get("nodes", [])) - len(nodes),
                   "edges": len(raw.get("edges", [])) - len(edges)}
        raw["nodes"], raw["edges"] = nodes, edges
        self._write(raw)
        return removed

    def delete_edge(self, source: str, target: str, edge_type: str) -> dict:
        """Remove one edge by (source, target, edge_type). Returns removed counts."""
        raw = self._read_fresh()
        kept, removed = [], 0
        for e in raw.get("edges", []):
            if (e.get("source_id") == source and e.get("target_id") == target
                    and e.get("edge_type") == edge_type):
                removed += 1
            else:
                kept.append(e)
        raw["edges"] = kept
        self._write(raw)
        return {"nodes": 0, "edges": removed}


class HistoryProvider:
    """Serves conversation history for a clicked topic from the SQLite transcript
    store (Phase 1) — semantically via RAG when an embedding model is reachable,
    otherwise by keyword (turns tagged with that topic). Read-only; if the DB is
    absent the viz simply shows no history.
    """

    def __init__(self, sessions_db: str, embed_model: Optional[str] = None):
        self.store = None
        self.rag = None
        try:
            from modules.session_store import SessionStore
            if sessions_db and os.path.exists(sessions_db):
                self.store = SessionStore(sessions_db)
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] session store unavailable ({exc}) — history disabled")
        if self.store is not None and embed_model:
            try:
                from modules.graph_relationship.embedding import ollama_embed_fn
                from modules.session_rag import SessionRAG
                self.rag = SessionRAG(self.store, ollama_embed_fn(model=embed_model))
            except Exception as exc:  # noqa: BLE001
                print(f"[viz] RAG unavailable ({exc}) — history falls back to keyword")

    def history(self, topic: str, person: Optional[str] = None, limit: int = 12) -> list:
        if self.store is None or not topic:
            return []
        if self.rag is not None:
            try:
                hits = self.rag.search(topic, top_k=limit, person_id=person)
                if hits:
                    return hits
            except Exception:  # noqa: BLE001
                pass
        return self.store.turns_for_topic(topic, person, limit=limit)


def make_handler(state: GraphState, history: Optional["HistoryProvider"] = None,
                 chat=None, robot=None):
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
            elif path == "/history":
                qs = urllib.parse.parse_qs(self.path.split("?", 1)[1]
                                           if "?" in self.path else "")
                topic = (qs.get("topic") or [""])[0]
                person = (qs.get("person") or [None])[0]
                turns = history.history(topic, person) if history else []
                self._send(200, json.dumps({"topic": topic, "turns": turns}).encode(),
                           "application/json")
            elif path == "/culture":
                # Current override + the cultures the robot knows (selector choices).
                body = json.dumps({"active": state.get_culture_override(),
                                   "cultures": state.culture_labels()}).encode()
                self._send(200, body, "application/json")
            elif path == "/persona":
                self._send(200, json.dumps(state.persona_choices()).encode(),
                           "application/json")
            elif path == "/chat":
                body = {"available": chat is not None,
                        "backend": chat.backend if chat is not None else None,
                        "hint": None if chat is not None else
                        "the viz server was started with --no-chat; restart it without that flag"}
                self._send(200, json.dumps(body).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if path == "/culture":
                    active = state.set_culture_override(str(body.get("active_culture", "auto")))
                    self._send(200, json.dumps({"ok": True, "active": active}).encode(),
                               "application/json")
                    return
                if path == "/chat":
                    if chat is None:
                        self._send(503, json.dumps({"ok": False, "error":
                            "chat is off (the viz server was started with --no-chat)"}
                        ).encode(), "application/json")
                        return
                    try:
                        hist = [(str(u), str(b)) for u, b in (body.get("history") or [])]
                        out = chat.reply(str(body.get("robot", "chatbox")),
                                         str(body.get("message", "")), hist)
                    except ValueError as exc:
                        self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(),
                                   "application/json")
                        return
                    except Exception as exc:    # backend failure: say so, don't 500
                        self._send(502, json.dumps({"ok": False, "error": str(exc)}).encode(),
                                   "application/json")
                        return
                    # robot = Esp32Link or None: the reply's action tag drives the real robot
                    out["robot"] = robot.show(out.get("tag", "")) if robot is not None else None
                    self._send(200, json.dumps({"ok": True, **out}).encode(), "application/json")
                    return
                if path == "/persona":
                    try:
                        a = state.set_persona(str(body.get("robot", "")),
                                              str(body.get("role", "")),
                                              body.get("ocean") or {})
                    except ValueError as exc:
                        self._send(400, json.dumps({"ok": False, "error": str(exc)})
                                   .encode(), "application/json")
                        return
                    self._send(200, json.dumps({"ok": True, **a}).encode(),
                               "application/json")
                    return
                if path != "/delete":
                    self._send(404, b"not found", "text/plain")
                    return
                kind = body.get("kind")
                if kind == "node":
                    removed = state.delete_node(str(body["id"]))
                elif kind == "edge":
                    removed = state.delete_edge(
                        str(body["source"]), str(body["target"]), str(body["edge_type"]))
                else:
                    self._send(400, b'{"ok":false,"error":"kind must be node|edge"}',
                               "application/json")
                    return
                self._send(200, json.dumps({"ok": True, "removed": removed}).encode(),
                           "application/json")
            except Exception as exc:  # never 500 — report as JSON
                self._send(200, json.dumps({"ok": False, "error": str(exc)}).encode(),
                           "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Live browser visualizer for the KG.")
    ap.add_argument("--kg-path", default="kg_culture.json",
                    help="Path to the KG JSON written by InMemoryGraphStore.save "
                         "(default: kg_culture.json — the culture branch's isolated KG)")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    ap.add_argument("--sessions-db", default="sessions.db",
                    help="SQLite transcript DB for topic-click history (default: sessions.db)")
    ap.add_argument("--embed-model", default="nomic-embed-text",
                    help="Ollama embedding model for RAG history (blank to use keyword only)")
    ap.add_argument("--ai-server", default=None, metavar="HOST[:PORT]",
                    help="persona chat backend: this AI server (default: the lab AI server)")
    ap.add_argument("--ollama", nargs="?", const="qwen2.5:7b", default=None, metavar="MODEL",
                    help="persona chat backend: local Ollama instead (default model qwen2.5:7b)")
    ap.add_argument("--no-chat", action="store_true", help="turn the persona chat off")
    ap.add_argument("--esp32-host", default="", metavar="IP",
                    help="ChatBox ESP32: each chat reply's action tag is sent as a robot expression")
    ap.add_argument("--esp32-port", type=int, default=8888, help="ESP32 TCP port (default 8888)")
    args = ap.parse_args()

    kg_path = os.path.abspath(args.kg_path)
    state = GraphState(kg_path)
    chat = None
    if not args.no_chat:
        from modules.persona_chat import (
            DEFAULT_AI_SERVER, OllamaChat, PersonaChat, ai_server_chat,
        )
        llm = (OllamaChat(args.ollama) if args.ollama
               else ai_server_chat(args.ai_server or DEFAULT_AI_SERVER))
        chat = PersonaChat(kg_path, state._persona_path(), llm)
    history = HistoryProvider(os.path.abspath(args.sessions_db),
                              embed_model=(args.embed_model or None))
    robot = None
    if args.esp32_host:
        from modules.esp32_link import Esp32Link
        robot = Esp32Link(args.esp32_host, args.esp32_port)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, history, chat, robot))

    print(f"[viz] serving KG visualizer at  http://{args.host}:{args.port}/")
    print(f"[viz] polling KG file:          {state.kg_path}")
    print(f"[viz] persona chat:             {chat.backend if chat else 'off (--no-chat)'}")
    print(f"[viz] robot (ESP32):            {f'{robot.host}:{robot.port}' if robot else 'off (--esp32-host)'}")
    if history.store is not None:
        kind = "RAG" if history.rag is not None else "keyword"
        print(f"[viz] topic history ({kind}) from:  {os.path.abspath(args.sessions_db)}")
    if not os.path.exists(state.kg_path):
        print("[viz] (file not present yet — will show empty graph until it appears)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
