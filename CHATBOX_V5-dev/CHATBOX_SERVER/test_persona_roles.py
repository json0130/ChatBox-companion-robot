"""
Headless check for role + persona assignment, viz → loop → prompt.

Run:  python3 -m pytest test_persona_roles.py      (no camera, no LLM, no network)

The viz server writes persona_assignment.json; the loop adopts it, attaches the
role subgraph + OCEAN persona, clears its chat history, and the prompt then
carries the role brief, the style rules and ONLY the attached role's knowledge —
with the robot's pre-assignment replies dropped from the recent conversation.
"""

from __future__ import annotations

import os
import time

from modules.graph_relationship.schema import Embodiment, RobotNode
from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.viz.server import GraphState
from modules.face_webcam.webcam_loop import WebcamKGLoop
from modules.role_seed import seed_role_library

SHY_TUTOR = {"O": 0.0, "C": 0.8, "E": -0.9, "A": 0.6, "N": 0.1}


class _Turns:
    def __init__(self, turns):
        self.turns = turns

    def recent_turns(self, pid, n):
        return self.turns[-n:]


def _setup(tmp_path):
    import threading
    store = InMemoryGraphStore()
    store.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    seed_role_library(store)
    kg = str(tmp_path / "kg.json")
    store.save(kg)
    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store, L.robot_id, L._robot_display, L.kg_path = store, "chatbox", "ChatBox", kg
    L._store_lock, L._chat_history = threading.RLock(), {"jay": ["old"]}
    L._persona_mtime = L._persona_since = None
    L._kg_dirty = False
    L._session_store = _Turns([
        {"ts": "2000-01-01T00:00:00+00:00", "child": "hi", "reply": "OLD STYLE REPLY"}])
    return L, GraphState(kg)


def test_viz_assignment_reaches_prompt(tmp_path):
    L, viz = _setup(tmp_path)
    viz.set_persona("chatbox", "role:tutor", SHY_TUTOR)
    L._sync_persona_assignment()

    assert L._chat_history == {}
    prompt = L._build_system_prompt(None)
    assert "a tutor robot" in prompt
    assert "Your job in this role: help the child practise maths" in prompt
    assert "Personality: organised, quiet, warm." in prompt
    assert "HOW YOU TALK AS A TUTOR" in prompt
    assert "Goal in every conversation: The child works" in prompt   # role: goal
    assert "short and low-key" in prompt                     # OCEAN → style
    assert "Give a hint, let them try" in prompt               # role: do
    assert "Avoid:" in prompt and "Giving the answer straight away" in prompt
    assert '"What do you think comes next?"' in prompt         # role: phrase
    assert "WHAT YOU KNOW AS A TUTOR" in prompt and "fractions" in prompt

    # switch role: the tutor subgraph detaches from the prompt
    time.sleep(0.01)
    viz.set_persona("chatbox", "role:storyteller", SHY_TUTOR)
    os.utime(L._persona_assignment_path(), None)
    L._persona_mtime = None                                    # coarse-mtime filesystems
    L._sync_persona_assignment()
    prompt = L._build_system_prompt(None)
    assert "a storyteller robot" in prompt and "fractions" not in prompt
    assert "HOW YOU TALK AS A STORYTELLER" in prompt and "Giving the answer" not in prompt


def test_old_replies_hidden_after_assignment(tmp_path):
    L, viz = _setup(tmp_path)
    assert "OLD STYLE REPLY" in L._build_system_prompt("jay")  # before: shown
    viz.set_persona("chatbox", "role:tutor", SHY_TUTOR)
    L._sync_persona_assignment()
    p = L._build_system_prompt("jay")
    assert "OLD STYLE REPLY" not in p and "them: hi" in p      # their words kept


def test_viz_rejects_unknown_role(tmp_path):
    _L, viz = _setup(tmp_path)
    for robot, role in (("chatbox", "role:pilot"), ("nobot", "role:tutor")):
        try:
            viz.set_persona(robot, role, SHY_TUTOR)
        except ValueError:
            continue
        raise AssertionError(f"accepted {robot}/{role}")
    assert {r["id"] for r in viz.persona_choices()["roles"]} >= {"role:tutor",
                                                                "role:companion"}
