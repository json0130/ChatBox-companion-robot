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
import queue
import threading
import time
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

    def __init__(self, kg_path: str, persona_file: str = "persona_assignment.json"):
        self.kg_path, self.persona_file = kg_path, persona_file
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
                            self.persona_file)

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
        return {**data[robot], "graph": self._attach_in_graph(robot, role, ocean)}

    def _attach_in_graph(self, robot: str, role: str, ocean: dict) -> bool:
        """Attach the role + persona in the KG file now, so the graph shows it even with
        no webcam loop running. A running loop applies the same assignment itself."""
        from modules.graph_relationship.roles import apply_assignment
        from modules.graph_relationship.store import InMemoryGraphStore
        store = InMemoryGraphStore()
        try:
            if not store.load(self.kg_path, quiet=True):
                return False
            apply_assignment(store, robot, role, ocean, source="viz")
            tmp = self.kg_path + ".tmp"
            store.save(tmp)
            os.replace(tmp, self.kg_path)      # atomic: the poller never reads half a file
        except (OSError, ValueError, KeyError) as exc:
            print(f"[viz] could not attach the role in the graph: {exc}")
            return False
        return True

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


def seed_robot_graph(path: str, robot_id: str) -> None:
    """A graph file for a robot other than the ChatBox: the robot and the role library (no people),
    plus, for Navel, the topics it declares. Safe to run on every start: an existing file loses the
    capability nodes an earlier version added to the Navel graph, and gets the topics if it has none."""
    from modules.graph_relationship.schema import Embodiment, RobotNode
    from modules.graph_relationship.store import InMemoryGraphStore
    from modules.role_seed import seed_role_library
    store = InMemoryGraphStore()
    existed = os.path.exists(path) and store.load(path, quiet=True)
    legacy = [n.id for n in list(store._nodes.values())
              if robot_id == "navel" and n.node_type == "capability" and n.id.startswith("navel:")]
    for nid in legacy:
        store.delete_node(nid)
    if store.get_node(robot_id) is None:
        store.upsert_node(RobotNode(id=robot_id, name=robot_id.capitalize(), embodiment=Embodiment.CAT))
        seed_role_library(store)
    has_topics = any(e.edge_type == "about" and e.source_id == "navel" for e in store._edges.values())
    if existed and not legacy and (robot_id != "navel" or has_topics):
        return
    if robot_id == "navel" and not has_topics:
        from modules.navel_profile import seed_navel
        seed_navel(store)
    store.save(path)


PRIMARY = "chatbox"      # the robot of the main graph (--kg-path); `others` are extra robots, each with its own graph


def make_handler(state: GraphState, history: Optional["HistoryProvider"] = None,
                 chat=None, hub=None, tagger=None, stt=None, others=None):
    # robot id -> (its GraphState, its PersonaChat): one tab each, nothing shared between them
    spaces = {PRIMARY: (state, chat), **(others or {})}

    def space(robot):
        if robot not in spaces:
            raise ValueError(f"unknown robot {robot!r}")
        return spaces[robot]

    def merged_graph():
        """ONE graph for every robot: each robot's file is read on its own (so Navel never depends on
        the ChatBox's Jay graph) and the views are joined. Shared nodes (the role library) have the same
        ids in each file, so they appear once and both robots' edges attach to them."""
        nodes, edges, seen_n, seen_e, active = [], [], set(), set(), {}
        for rid, (st, _) in spaces.items():
            g = st.read()
            if rid == PRIMARY:
                active = g.get("active") or {}
            for n in g["nodes"]:
                if n["id"] not in seen_n:
                    seen_n.add(n["id"]); nodes.append(n)
            for e in g["edges"]:
                k = (e["source"], e["target"], e["type"])
                if k not in seen_e:
                    seen_e.add(k); edges.append(e)
        return {"nodes": nodes, "edges": edges, "active": active}

    def speak(texts, robot=PRIMARY):
        """`robot` says `texts` in order (in a thread: it waits on acks), then returns to DEFAULT."""
        threading.Thread(target=hub.say_steps, args=(texts,), kwargs={"robot": robot}, daemon=True).start()
        n = len(hub.robots(robot))
        return {"robots": n, "sent": bool(n), "steps": len(texts)}

    def converse(robot, message, hist):
        """One turn: persona reply, then the robot (if any) speaks it. Shared by /chat and the mic."""
        t0 = time.time()
        out = space(robot)[1].reply(robot, message, hist)
        out["timing"] = {"reply_s": round(time.time() - t0, 2)}
        # hub = RobotHub or None: the connected robot speaks the reply (its tag drives the face)
        if hub is not None and tagger is not None and robot == PRIMARY:
            # Method B: one gesture tag per sentence; each sentence is spoken as soon as ITS tag is
            # chosen, while the next ones are still being planned
            from modules.action_tag_generator import step_texts
            q, queued = queue.Queue(), []
            def steps():
                while (t := q.get()) is not None:
                    yield t
            threading.Thread(target=hub.say_steps, args=(steps(),), kwargs={"robot": robot}, daemon=True).start()
            def on_step(i):
                if not queued:
                    out["timing"]["first_step_s"] = round(time.time() - t0, 2)
                queued.append(i); q.put(step_texts([i])[0])
            t0 = time.time()
            try:
                out["plan"] = tagger.generate(out["reply"], message, on_step=on_step)
                out["timing"]["tags_s"] = round(time.time() - t0, 2)
            except Exception as exc:    # planner failed: say what is left as one plain reply
                out["plan_error"] = str(exc)
                if not queued:
                    q.put(out["reply"])
            finally:
                q.put(None)
            n = len(hub.robots(robot))
            out["robot"] = {"robots": n, "sent": bool(n), "steps": len(queued) or 1}
        elif hub is not None and robot == PRIMARY:
            out["robot"] = speak([f"[{out['tag']}] {out['reply']}" if out.get("tag") else out["reply"]], robot)
        elif hub is not None:       # another robot: its client's tags are unknown here, so plain speech
            out["robot"] = speak([out["reply"]], robot)
        else:
            out["robot"] = None
        return out

    voice_log, vlock = [], threading.Lock()
    voice_hist, voice_ts = {}, {}           # per robot: its own spoken conversation, restarted on a new setup

    def on_speech(audio_b64, robot_name):
        """Mic audio from the robot: transcribe, show it like a typed message, reply, speak."""
        robot = robot_name.lower().replace(" ", "")
        item = {"id": 0, "robot": robot, "text": "", "error": None, "out": None, "done": False}
        with vlock:
            item["id"] = (voice_log[-1]["id"] + 1) if voice_log else 1
            voice_log.append(item); del voice_log[:-50]
        try:
            t0 = time.time()
            ok, text, _ = stt.transcribe_audio_base64(audio_b64)
            item["stt_s"] = round(time.time() - t0, 2)
            if not ok or not text.strip():
                item["error"] = text if not ok else "nothing heard"
                return
            item["text"] = text.strip()
            r_state, r_chat = space(robot)
            if r_chat is None:
                item["error"] = "chat is off (--no-chat)"
                return
            a = (r_state.persona_choices().get("assigned") or {}).get(robot)
            hist = voice_hist.setdefault(robot, [])
            if (a or {}).get("ts") != voice_ts.get(robot):  # a new setup was applied: restart the conversation
                hist.clear(); voice_ts[robot] = (a or {}).get("ts")
            item["out"] = out = converse(robot, item["text"], list(hist))
            print(f"[voice] stt {item['stt_s']}s · reply {out['timing']['reply_s']}s"
                  f" · tags {out['timing'].get('tags_s', '-')}s · {item['text']!r}")
            hist.append((item["text"], out["reply"]))
        except Exception as exc:
            item["error"] = str(exc)
        finally:
            item["done"] = True

    if hub is not None and stt is not None:
        hub.on_speech = on_speech

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
            qs = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            robot = (qs.get("robot") or [PRIMARY])[0]
            try:
                r_state, r_chat = space(robot)
            except ValueError as exc:
                self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(), "application/json")
                return
            if path == "/" or path == "/index.html":
                try:
                    with open(_INDEX_HTML, "rb") as fh:
                        body = fh.read()
                    self._send(200, body, "text/html; charset=utf-8")
                except OSError:
                    self._send(404, b"index.html not found", "text/plain")
            elif path == "/graph.json":
                body = json.dumps(merged_graph()).encode("utf-8")
                self._send(200, body, "application/json")
            elif path == "/history":
                topic = (qs.get("topic") or [""])[0]
                person = (qs.get("person") or [None])[0]
                turns = history.history(topic, person) if history and robot == PRIMARY else []   # transcripts are the ChatBox's
                self._send(200, json.dumps({"topic": topic, "turns": turns}).encode(),
                           "application/json")
            elif path == "/culture":
                # Current override + the cultures the robot knows (selector choices).
                body = json.dumps({"active": r_state.get_culture_override(),
                                   "cultures": r_state.culture_labels()}).encode()
                self._send(200, body, "application/json")
            elif path == "/persona":
                self._send(200, json.dumps(r_state.persona_choices()).encode(),
                           "application/json")
            elif path == "/voice":
                self._send(200, json.dumps({"enabled": stt is not None, "items": list(voice_log)}).encode(),
                           "application/json")
            elif path == "/robots":
                from modules.chatbox_actions import ACTIONS
                body = {"hub": hub is not None, "port": hub.port if hub else None,
                        "robots": hub.robots() if hub else [], "actions": ACTIONS,
                        "workspaces": list(spaces)}
                self._send(200, json.dumps(body).encode(), "application/json")
            elif path == "/chat":
                body = {"available": r_chat is not None,
                        "backend": r_chat.backend if r_chat is not None else None,
                        "hint": None if r_chat is not None else
                        "the viz server was started with --no-chat; restart it without that flag"}
                self._send(200, json.dumps(body).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                robot = str(body.get("robot") or PRIMARY)
                try:
                    r_state, r_chat = space(robot)
                except ValueError as exc:
                    self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(), "application/json")
                    return
                if path == "/culture":
                    active = r_state.set_culture_override(str(body.get("active_culture", "auto")))
                    self._send(200, json.dumps({"ok": True, "active": active}).encode(),
                               "application/json")
                    return
                if path == "/action":               # a button: play one gesture on the connected robot(s)
                    from modules.chatbox_actions import ACTIONS
                    tag = str(body.get("tag", "")).upper()
                    if robot != PRIMARY:
                        self._send(400, json.dumps({"ok": False, "error": f"no actions defined for {robot} yet"}
                                                   ).encode(), "application/json")
                    elif tag not in ACTIONS:
                        self._send(400, json.dumps({"ok": False, "error": f"unknown action {tag!r}"}).encode(),
                                   "application/json")
                    elif hub is None or not hub.robots(robot):
                        self._send(409, json.dumps({"ok": False, "error": "no robot connected"}).encode(),
                                   "application/json")
                    else:
                        self._send(200, json.dumps({"ok": True, "tag": tag, **hub.say(f"[{tag}]", robot)}).encode(),
                                   "application/json")
                    return
                if path == "/chat":
                    if r_chat is None:
                        self._send(503, json.dumps({"ok": False, "error":
                            "chat is off (the viz server was started with --no-chat)"}
                        ).encode(), "application/json")
                        return
                    try:
                        hist = [(str(u), str(b)) for u, b in (body.get("history") or [])]
                        out = converse(robot, str(body.get("message", "")), hist)
                    except ValueError as exc:
                        self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(),
                                   "application/json")
                        return
                    except Exception as exc:    # backend failure: say so, don't 500
                        self._send(502, json.dumps({"ok": False, "error": str(exc)}).encode(),
                                   "application/json")
                        return
                    self._send(200, json.dumps({"ok": True, **out}).encode(), "application/json")
                    return
                if path == "/persona":
                    try:
                        a = r_state.set_persona(robot,
                                              str(body.get("role", "")),
                                              body.get("ocean") or {})
                    except ValueError as exc:
                        self._send(400, json.dumps({"ok": False, "error": str(exc)})
                                   .encode(), "application/json")
                        return
                    from modules.persona_voice import voice_for
                    role = next((r for r in r_state.persona_choices()["roles"]
                                 if r["id"] == a["role"]), {})
                    label = role.get("label", a["role"])
                    voice = voice_for(label, a["ocean"])
                    sent = None
                    if hub is not None:                     # tell THIS robot about its new setup
                        sent = hub.push_persona(label,      # the robot speaks "Persona updated to <name>."
                                                role.get("brief") or label, a["ocean"], voice, robot)
                    self._send(200, json.dumps({"ok": True, **a, "voice": voice, "robot": sent}).encode(),
                               "application/json")
                    return
                if path != "/delete":
                    self._send(404, b"not found", "text/plain")
                    return
                kind = body.get("kind")
                states = [st for st, _ in spaces.values()]     # the graph is joined: a node may live in either file
                if kind == "node":
                    removed = any([st.delete_node(str(body["id"])) for st in states])
                elif kind == "edge":
                    removed = any([st.delete_edge(
                        str(body["source"]), str(body["target"]), str(body["edge_type"])) for st in states])
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
    ap.add_argument("--navel-kg-path", default="kg_navel.json",
                    help="graph file of the Navel tab (created with the role library if missing; it shares "
                         "nothing with the ChatBox graph)")
    ap.add_argument("--no-navel", action="store_true", help="no Navel tab")
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
    ap.add_argument("--robot-hub", nargs="?", const=5000, type=int, default=None, metavar="PORT",
                    help="serve a hub the ChatBox Jetson / Navel client connects to (set its server_url "
                         "to http://THIS_IP:PORT, default 5000): Apply pushes the persona, chat replies "
                         "are spoken by the robot")
    ap.add_argument("--action-tags", action="store_true",
                    help="with --robot-hub: Method B -- split each chat reply into sentences and pick a "
                         "gesture tag per sentence (modules/action_tag_generator.py) instead of the "
                         "reply's single tag")
    ap.add_argument("--no-voice", action="store_true",
                    help="with --robot-hub: don't transcribe the robot's microphone (faster-whisper)")
    ap.add_argument("--whisper-device", default="cuda", choices=["cuda", "cpu"],
                    help="with --robot-hub: where faster-whisper runs (default cuda, falls back to cpu "
                         "if it cannot load)")
    ap.add_argument("--whisper-model", default="base", metavar="SIZE",
                    help="with --robot-hub: faster-whisper model for the robot's microphone (default base)")
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
    others = {}
    if not args.no_navel:
        navel_path = os.path.abspath(args.navel_kg_path)
        seed_robot_graph(navel_path, "navel")
        navel = GraphState(navel_path, "persona_assignment_navel.json")
        others["navel"] = (navel, PersonaChat(navel_path, navel._persona_path(), llm) if chat else None)
    history = HistoryProvider(os.path.abspath(args.sessions_db),
                              embed_model=(args.embed_model or None))
    hub = None
    from modules.robot_hub import first_free
    if args.robot_hub is not None:
        from modules.robot_hub import RobotHub
        hub = RobotHub(args.robot_hub)
        hub.start()
        if hub.port != args.robot_hub:
            print(f"[viz] hub port {args.robot_hub} is taken -> using {hub.port}")
    tagger = None
    if args.action_tags:
        if hub is None or chat is None:
            ap.error("--action-tags needs --robot-hub and the chat (not --no-chat)")
        from modules.action_tag_generator import ActionTagGenerator, ChatLLM
        tagger = ActionTagGenerator(ChatLLM(chat.llm))
    stt = None
    if hub is not None and not args.no_voice:
        from modules.speech_processor import SpeechProcessor
        stt = SpeechProcessor({"whisper_model_size": args.whisper_model, "whisper_device": args.whisper_device,
                               "whisper_compute_type": "float16"})
        threading.Thread(target=stt.initialize, daemon=True).start()   # load in the background
    handler = make_handler(state, history, chat, hub, tagger, stt, others)
    port = first_free(args.port, args.host)
    if port != args.port:
        print(f"[viz] port {args.port} is taken -> using {port}")
    args.port = port
    server = ThreadingHTTPServer((args.host, port), handler)

    print(f"[viz] serving KG visualizer at  http://{args.host}:{args.port}/")
    print(f"[viz] polling KG file:          {state.kg_path}")
    print(f"[viz] navel tab:                {others['navel'][0].kg_path if others else 'off (--no-navel)'}")
    print(f"[viz] persona chat:             {chat.backend if chat else 'off (--no-chat)'}")
    print(f"[viz] action tags (Method B):   {'on' if tagger else 'off (--action-tags)'}")
    print(f"[viz] robot microphone:         {f'faster-whisper {args.whisper_model} on {args.whisper_device} (loading in the background)' if stt else 'off (needs --robot-hub, not --no-voice)'}")
    print(f"[viz] robot hub:                {f'port {hub.port} (robot server_url -> http://THIS_IP:{hub.port})' if hub else 'off (--robot-hub)'}")
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
