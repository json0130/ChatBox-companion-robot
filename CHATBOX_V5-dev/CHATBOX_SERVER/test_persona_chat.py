"""
Persona chat: the viz /chat endpoint talks to the robot as it is currently set up.

Run:  python3 -m pytest test_persona_chat.py     (stub LLM — no server, no Ollama)

Covers: Apply → the very next chat message uses it (no webcam loop involved); the prompt
is IDENTITY + reply rules + role/persona blocks and says nothing about the visitor;
history is capped; the reply tag is split off; error paths (chat off / empty message /
backend failure) answer with JSON instead of a 500.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from modules.graph_relationship.schema import Embodiment, RobotNode
from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.viz.server import GraphState, make_handler
from modules.persona_chat import (
    MAX_HISTORY_TURNS, REPLY_RULES, ChatError, PersonaChat, split_tag,
)
from modules.role_seed import seed_role_library

SHY_TUTOR = {"O": 0.0, "C": 0.8, "E": -0.9, "A": 0.6, "N": 0.1}
LOUD_STORY = {"O": 0.9, "C": -0.6, "E": 0.9, "A": 0.9, "N": -0.5}


class StubLLM:
    label = "stub"

    def __init__(self, reply="[HAPPY] Hello there!"):
        self.reply, self.calls = reply, []

    def respond(self, system_prompt, user_msg, history, max_tokens=140):
        self.calls.append({"system": system_prompt, "user": user_msg, "history": list(history)})
        return self.reply


@pytest.fixture
def env(tmp_path):
    kg = str(tmp_path / "kg.json")
    store = InMemoryGraphStore()
    store.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    seed_role_library(store)
    store.save(kg)
    state, llm = GraphState(kg), StubLLM()
    chat = PersonaChat(kg, state._persona_path(), llm)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, None, chat))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_port}", "state": state, "llm": llm,
           "kg": kg, "tmp": tmp_path}
    server.shutdown()


def call(env, path, body=None):
    req = urllib.request.Request(env["url"] + path, None if body is None else json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def test_apply_then_chat_uses_it_without_a_loop(env):
    assert call(env, "/chat")[1] == {"available": True, "backend": "stub", "hint": None}
    s, a = call(env, "/persona", {"robot": "chatbox", "role": "role:tutor", "ocean": SHY_TUTOR})
    assert s == 200 and a["ok"]
    s, out = call(env, "/chat", {"robot": "chatbox", "message": "hi"})
    assert s == 200 and out["ok"] and out["role"] == "tutor"
    assert out["persona"] == "organised, quiet, warm" and out["assigned_ts"] == a["ts"]
    assert (out["tag"], out["reply"]) == ("HAPPY", "Hello there!")
    system = env["llm"].calls[-1]["system"]
    assert "a tutor robot" in system and "HOW YOU TALK AS A TUTOR" in system
    assert "short and low-key" in system                         # OCEAN → style
    assert "Giving the answer straight away" in system            # tutor's avoid subnode

    # a different setup changes the prompt on the very next message
    call(env, "/persona", {"robot": "chatbox", "role": "role:storyteller", "ocean": LOUD_STORY})
    call(env, "/chat", {"robot": "chatbox", "message": "hi"})
    system2 = env["llm"].calls[-1]["system"]
    assert "a storyteller robot" in system2 and "Giving the answer" not in system2
    assert system2 != system
    assert call(env, "/persona")[1]["assigned"]["chatbox"]["role"] == "role:storyteller"


def test_prompt_is_about_the_robot_only(env):
    call(env, "/persona", {"robot": "chatbox", "role": "role:receptionist", "ocean": SHY_TUTOR})
    call(env, "/chat", {"robot": "chatbox", "message": "hi"})
    system = env["llm"].calls[-1]["system"]
    assert system.index("IDENTITY") < system.index("HOW TO REPLY") < system.index("HOW YOU TALK")
    for forbidden in ("WHO YOU'RE TALKING TO", "remember", "rapport", "recognise", "culture",
                      "Cultural", "memory below —"):
        assert forbidden not in system, forbidden
    assert REPLY_RULES in system


def test_history_is_capped_and_passed_through(env):
    hist = [[f"q{i}", f"a{i}"] for i in range(MAX_HISTORY_TURNS + 4)]
    call(env, "/chat", {"robot": "chatbox", "message": "next", "history": hist})
    sent = env["llm"].calls[-1]["history"]
    assert len(sent) == MAX_HISTORY_TURNS and sent[-1] == ("q9", "a9") and sent[0] == ("q4", "a4")
    assert env["llm"].calls[-1]["user"] == "next"


def test_error_paths(env, tmp_path):
    assert call(env, "/chat", {"robot": "chatbox", "message": "   "})[0] == 400
    env["llm"].reply = "[LLM error: connection refused]"
    s, out = call(env, "/chat", {"robot": "chatbox", "message": "hi"})
    assert s == 502 and "connection refused" in out["error"]
    # chat not enabled at all
    off = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(env["state"], None, None))
    threading.Thread(target=off.serve_forever, daemon=True).start()
    try:
        env2 = {"url": f"http://127.0.0.1:{off.server_port}"}
        assert call(env2, "/chat")[1]["available"] is False
        s, out = call(env2, "/chat", {"message": "hi"})
        assert s == 503 and "--no-chat" in out["error"]
    finally:
        off.shutdown()


def test_works_without_a_seeded_role_library(env, tmp_path):
    """A KG file that never ran the loop has no library roles: chat seeds them in memory."""
    bare = str(tmp_path / "bare.json")
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    s.save(bare)
    state = GraphState(bare)
    state.set_persona  # noqa: B018 — set_persona validates against the KG, so write the file directly
    with open(state._persona_path(), "w", encoding="utf-8") as fh:
        json.dump({"chatbox": {"role": "role:tutor", "ocean": SHY_TUTOR, "ts": "t"}}, fh)
    llm = StubLLM()
    out = PersonaChat(bare, state._persona_path(), llm).reply("chatbox", "hi", [])
    assert out["role"] == "tutor" and "HOW YOU TALK AS A TUTOR" in llm.calls[-1]["system"]


class _LateServer(StubLLM):
    """An AI-server client that was unreachable at start-up and comes up later."""
    label = "ai-server:late"

    def __init__(self):
        super().__init__()
        self.available, self.connects = False, 0

    def connect(self):
        self.connects += 1
        self.available = True
        return True

    def respond(self, *a, **k):
        if not self.available:
            return "[LLM not connected — run with --ai-server]"
        return super().respond(*a, **k)


def test_reconnects_when_the_server_comes_up_later(env):
    late = _LateServer()
    chat = PersonaChat(env["kg"], env["state"]._persona_path(), late)
    out = chat.reply("chatbox", "hi", [])              # start-up connect had failed; this retries
    assert late.connects == 1 and out["reply"] == "Hello there!"


def test_unreachable_server_message_points_at_the_fix(env):
    class Down(StubLLM):
        available = False
        label = "ai-server:x:7898"

        def connect(self):
            return False

        def respond(self, *a, **k):
            return "[LLM not connected — run with --ai-server]"
    with pytest.raises(ChatError, match="--ollama"):
        PersonaChat(env["kg"], env["state"]._persona_path(), Down()).reply("chatbox", "hi", [])


def test_default_backend_is_the_lab_ai_server(monkeypatch):
    import importlib
    import modules.persona_chat as pc
    assert pc.DEFAULT_AI_SERVER == "172.24.47.255"
    monkeypatch.setenv("AI_HOST", "10.0.0.9")
    assert importlib.reload(pc).DEFAULT_AI_SERVER == "10.0.0.9"      # $AI_HOST wins
    monkeypatch.delenv("AI_HOST")
    importlib.reload(pc)


def test_split_tag():
    assert split_tag("[CURIOUS] Tell me more.") == ("CURIOUS", "Tell me more.")
    assert split_tag("No tag here.") == ("", "No tag here.")
    assert split_tag("[x] [y] nested") == ("X", "[y] nested")


def test_reply_rules_still_match_the_loops_prompt():
    """Drift: every REPLY_RULES line except the (identical) header is a line of the loop's own."""
    pytest.importorskip("cv2")
    import os
    import threading as th
    from modules.face_webcam.webcam_loop import WebcamKGLoop
    from modules.graph_relationship.seed import seed_all
    store = InMemoryGraphStore()
    seed_all(store, os.path.join(os.path.dirname(__file__), "modules", "graph_relationship", "specs"))
    seed_role_library(store)
    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store, L.robot_id, L._robot_display = store, "chatbox", "ChatBox"
    L._store_lock, L._chat_history, L._persona_since = th.RLock(), {}, None
    real = L._build_system_prompt(None).splitlines()
    rules = REPLY_RULES.splitlines()
    assert rules[0] == "━━━ HOW TO REPLY ━━━"
    missing = [r for r in rules[1:] if r not in real]
    assert not missing, missing
