"""
Action-tag generator (Method B on the ChatBox): reply -> one native gesture tag per sentence, sent
to the Jetson client one demo_step at a time.

Run:  python3 -m pytest test_action_tag_generator.py     (scripted LLM + fake Socket.IO robot)
"""
from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from modules.action_tag_generator import ActionTagGenerator, ChatLLM, step_texts
from modules.graph_relationship.viz.server import make_handler
from modules.persona_chat import PersonaChat
from test_persona_chat import _FakeRobot, _post, StubLLM, env, SHY_TUTOR  # noqa: F401  (fixtures)

REPLY = "Hello! Welcome to the lab. I am sorry that was hard. Well done, great job! Over there is the arm."
PICK = {"Welcome": "GREET", "sorry": "SAD", "Well done": "HAPPY", "Over there": "POINT"}


class ScriptedLLM:
    """Answers the picker's {"tag","confidence"} request by keyword in the sentence being scored."""
    available = True

    def __init__(self, junk=False):
        self.junk = junk

    def respond(self, system, user, history, max_tokens=140):
        if self.junk:
            return "no idea"
        sentence = user.split("Generated robot dialogue:\n")[1].split("\n")[0]
        tag = next((t for k, t in PICK.items() if k in sentence), "DEFAULT")
        return json.dumps({"tag": tag, "confidence": 0.9})


def test_each_sentence_gets_its_native_tag():
    items = ActionTagGenerator(ChatLLM(ScriptedLLM())).generate(REPLY, "hi")
    assert [i["text"] for i in items][:2] == ["Hello!", "Welcome to the lab."]
    got = {i["text"]: (i["action"], i["tag"]) for i in items}
    assert got["Welcome to the lab."] == ("GREET", "WAVE")            # planner action -> Jetson tag
    assert got["I am sorry that was hard."] == ("SAD", "SAD")
    assert got["Well done, great job!"] == ("HAPPY", "HANDS_CLAP")
    assert got["Over there is the arm."] == ("POINT", "POINT")
    assert step_texts(items)[1] == "[WAVE] Welcome to the lab."


def test_a_broken_llm_degrades_to_the_neutral_pose_not_an_error():
    llm = ChatLLM(ScriptedLLM(junk=True))
    items = ActionTagGenerator(llm).generate("Hello! Welcome to the lab.")
    assert items and all(i["tag"] == "DEFAULT" for i in items)


def test_say_steps_waits_for_each_ack(hub_env):
    robot = _FakeRobot(hub_env["hub"].port); hub_env["robots"].append(robot)
    robot.wait_for("client_init_response")
    t = threading.Thread(target=hub_env["hub"].say_steps, args=(["[WAVE] a", "[SAD] b"],), daemon=True)
    t.start()
    first = robot.wait_for("demo_step")[0]
    assert first["text"] == "[WAVE] a" and first["require_ack"] is True
    time.sleep(0.3)
    assert len(robot.wait_for("demo_step")) == 1                     # b is held back until a is acked
    robot.sio.emit("ack", {"type": "ack", "step_id": first["step_id"]})
    second = robot.wait_for("demo_step", 2)[1]
    assert second["text"] == "[SAD] b"
    time.sleep(0.3)
    assert len(robot.wait_for("demo_step", 2)) == 2                  # no default pose while b is still being spoken
    robot.sio.emit("ack", {"type": "ack", "step_id": second["step_id"]})
    assert robot.wait_for("demo_step", 3)[2]["text"] == "[DEFAULT]"  # speech over -> back to the default pose
    t.join(2); assert not t.is_alive()


@pytest.fixture
def hub_env(env):
    from modules.robot_hub import RobotHub
    hub = RobotHub(port=0, host="127.0.0.1"); hub.start()
    chat = PersonaChat(env["kg"], env["state"]._persona_path(), StubLLM(REPLY))
    tagger = ActionTagGenerator(ChatLLM(ScriptedLLM()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(env["state"], None, chat, hub, tagger))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    robots = []
    yield {**env, "hub": hub, "url": f"http://127.0.0.1:{server.server_port}", "robots": robots}
    time.sleep(0.5)             # let the last ack's polling request finish: disconnecting mid-request can hang ~25 s
    for r in robots:
        r.close()
    server.shutdown(); hub.stop()


def test_chat_reply_reaches_the_robot_as_tagged_steps(hub_env):
    robot = _FakeRobot(hub_env["hub"].port); hub_env["robots"].append(robot)
    robot.wait_for("client_init_response")
    out = _post(hub_env["url"], "/chat", {"message": "hi"})
    assert [p["tag"] for p in out["plan"]][1:3] == ["WAVE", "SAD"]
    assert out["robot"]["sent"] and out["robot"]["steps"] == len(out["plan"])
    heard = []
    for n in range(1, len(out["plan"]) + 1):                          # act like the Jetson: speak, then ack
        step = robot.wait_for("demo_step", n)[n - 1]
        heard.append(step["text"])
        robot.sio.emit("ack", {"type": "ack", "step_id": step["step_id"]})
    assert heard == step_texts(out["plan"])
    assert "[WAVE] Welcome to the lab." in heard and "[HANDS_CLAP] Well done, great job!" in heard
