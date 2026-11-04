"""The chat only ever sends the robot tags it can perform (modules/chatbox_actions.py).

Run:  python3 -m pytest test_chatbox_actions.py
"""
import ast
from pathlib import Path

from modules.action_tag_generator import step_texts
from modules.chatbox_actions import ACTIONS, ALIASES, normalize_tag, tag_rule
from modules.persona_chat import REPLY_RULES, split_tag
from modules.project56.robot_profile.chatbox_profile import CHATBOX_NATIVE_TAG

CLIENT = Path(__file__).resolve().parents[2] / "FINAL_DEMO" / "CHATBOX_CLIENT" / "robot.py"


def _emotion_map():
    for node in ast.parse(CLIENT.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EMOTION_MAP":
            return ast.literal_eval(node.value)
    raise AssertionError("EMOTION_MAP not found in robot.py")


def test_excited_and_other_feeling_words_become_real_actions():
    assert normalize_tag("EXCITED") == "HANDS_CLAP"
    assert normalize_tag("[Curious]") == "CONFUSED"
    assert normalize_tag("welcome") == "GREETING"
    assert normalize_tag("something new") == "DEFAULT"      # the client would do this anyway
    assert normalize_tag("") == ""                          # no tag = plain speech
    assert normalize_tag("wave") == "WAVE" and normalize_tag("EARS-WIGGLE") == "EARS_WIGGLE"


def test_every_alias_lands_on_a_listed_action():
    assert set(ALIASES.values()) <= set(ACTIONS)
    assert not set(ALIASES) & set(ACTIONS)                  # an action is never also an alias


def test_the_prompt_lists_every_action_and_nothing_else():
    assert tag_rule() in REPLY_RULES
    for tag in ACTIONS:
        assert f"[{tag}]" in tag_rule()
    assert "[CURIOUS]" not in tag_rule() and "e.g." not in tag_rule()


def test_actions_exist_on_the_jetson_and_move_differently():
    if not CLIENT.exists():
        import pytest
        pytest.skip("FINAL_DEMO client not in this checkout")
    emap = _emotion_map()
    assert set(ACTIONS) <= set(emap)                        # the client knows every tag we send
    moves = [emap[t] for t in ACTIONS]
    assert len(set(moves)) == len(moves)                    # no two of them are the same motion


def test_the_method_b_generator_only_uses_listed_actions():
    assert set(CHATBOX_NATIVE_TAG.values()) <= set(ACTIONS)
    items = [{"text": "Hi.", "tag": "WAVE"}, {"text": "ok", "tag": None}]
    assert step_texts(items) == ["[WAVE] Hi.", "ok"]
    assert split_tag("[EXCITED] Great!") == ("EXCITED", "Great!")
