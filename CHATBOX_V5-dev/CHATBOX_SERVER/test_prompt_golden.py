"""
Golden test: the loop's system prompt is byte-identical to the recorded one.

Guards refactors of the prompt path (e.g. moving role_persona_blocks into roles.py
for rolesim). Covers: no assignment (spec role/persona), an assigned role + OCEAN
persona, and a recognised person with recent turns.

Run:            python3 -m pytest test_prompt_golden.py
Re-record:      python3 test_prompt_golden.py --write    (only for an INTENDED change)
"""

from __future__ import annotations

import os
import sys
import threading

from modules.face_webcam.webcam_loop import WebcamKGLoop
from modules.graph_relationship.roles import apply_assignment
from modules.graph_relationship.seed import seed_all
from modules.graph_relationship.store import InMemoryGraphStore
from modules.role_seed import seed_role_library

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "golden")
_SPECS = os.path.join(_HERE, "modules", "graph_relationship", "specs")


class _Turns:
    def recent_turns(self, pid, n):
        return [{"ts": "2026-01-01T00:00:00+00:00", "child": "hi", "reply": "Hello!"}][-n:]


def _loop(assign: bool) -> WebcamKGLoop:
    store = InMemoryGraphStore()
    seed_all(store, _SPECS)
    seed_role_library(store)
    if assign:
        apply_assignment(store, "chatbox", "role:tutor",
                         {"O": 0.9, "C": 0.2, "E": 0.9, "A": 0.7, "N": -0.6})
    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store, L.robot_id, L._robot_display = store, "chatbox", "ChatBox"
    L._store_lock, L._chat_history = threading.RLock(), {}
    L._persona_since, L._session_store = None, _Turns()
    return L


CASES = {
    "spec_no_person":     lambda: _loop(False)._build_system_prompt(None),
    "tutor_no_person":    lambda: _loop(True)._build_system_prompt(None),
    "tutor_with_person":  lambda: _loop(True)._build_system_prompt("jay"),
}


def _path(name: str) -> str:
    return os.path.join(_GOLDEN, f"prompt_{name}.txt")


def test_prompts_match_golden():
    for name, make in CASES.items():
        with open(_path(name), encoding="utf-8") as fh:
            assert make() == fh.read(), f"prompt changed: {name}"


if __name__ == "__main__" and "--write" in sys.argv:
    os.makedirs(_GOLDEN, exist_ok=True)
    for name, make in CASES.items():
        with open(_path(name), "w", encoding="utf-8") as fh:
            fh.write(make())
        print("wrote", _path(name))
