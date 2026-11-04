import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from modules.graph_relationship.viz.server import GraphState, make_handler, seed_robot_graph
from modules.navel_link import NavelLink
from modules.persona_chat import PersonaChat
from modules.robot_hub import RobotHub
from modules.robot_registry import RobotRegistry
from test_navel_link import FakeNavel, wait
from test_persona_chat import SHY_TUTOR, StubLLM, _FakeRobot, env  # noqa: F401  (fixtures)


def test_registry_saves_validates_and_colours(tmp_path):
    p = str(tmp_path / "robots.json")
    reg = RobotRegistry(p)
    assert [r["id"] for r in reg.robots] == ["chatbox", "navel"]
    stu = reg.add("Stuart", "dial-out", "192.168.0.11", 8765)
    assert stu["id"] == "stuart" and stu["tts"] == "navel" and stu["color"] not in ("#ff8c42", "#4fc3f7")
    for bad in (dict(name="Stuart"), dict(name="9x"), dict(name="Q", kind="dial-out"), dict(name="Zed", color="red")):
        with pytest.raises(ValueError):
            reg.add(**bad)
    reg.set_voice_adjust("stuart", {"rate": 99, "pitch": -2})
    assert reg.voice_adjust("stuart") == {"rate": 30.0, "pitch": -2.0, "volume": 0.0}   # clamped to the slider range
    again = RobotRegistry(p)                                                        # survives a restart
    assert again.get("stuart")["host"] == "192.168.0.11" and again.voice_adjust("stuart")["pitch"] == -2.0
    with pytest.raises(ValueError):
        again.remove("chatbox")                                                     # built in
    again.remove("stuart"); assert RobotRegistry(p).get("stuart") is None


@pytest.fixture
def viz(env, tmp_path):  # noqa: F811
    hub = RobotHub(port=0, host="127.0.0.1"); hub.start()
    reg = RobotRegistry(str(tmp_path / "robots.json"))
    llm = env["llm"]

    def attach(spec):                       # what main() does for a robot
        path = str(tmp_path / f"kg_{spec['id']}.json")
        seed_robot_graph(path, spec["id"])
        st = GraphState(path, f"persona_assignment_{spec['id']}.json")
        if spec["kind"] == "dial-out":
            hub.add_link(spec["id"], NavelLink(spec["host"], spec["port"], name=spec["name"], retry_s=0.2,
                                               tts=spec["tts"] == "navel"))
        return st, PersonaChat(path, st._persona_path(), llm)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
        env["state"], None, PersonaChat(env["kg"], env["state"]._persona_path(), llm), hub, None, None, {},
        reg, attach, lambda rid: hub.remove_link(rid)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"

    def call(path, body=None):
        req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)
    yield call, hub, reg
    srv.shutdown(); hub.stop()


def test_add_a_dial_out_robot_from_the_page_assign_a_role_and_change_its_voice(viz):
    call, hub, reg = viz
    fake = FakeNavel()
    code, out = call("/registry/add", {"name": "Stuart", "kind": "dial-out", "host": "127.0.0.1", "port": fake.port})
    assert code == 200 and out["id"] == "stuart" and out["has_graph"]
    assert wait(lambda: [r for r in call("/registry")[1]["robots"] if r["id"] == "stuart"][0]["connected"])
    assert "stuart" in call("/robots")[1]["workspaces"] and call("/robots")[1]["colors"]["stuart"] == out["color"]
    # its own graph: role library + robot, nothing of the ChatBox's Jay
    assert call("/persona?robot=stuart")[1]["robots"] == ["stuart"]
    # assign a role from the page (same endpoint as the graph page): only Stuart is told
    code, a = call("/persona", {"robot": "stuart", "role": "role:tutor", "ocean": SHY_TUTOR})
    assert code == 200 and a["robot"] == {"robots": 1, "sent": True}
    assert wait(lambda: any(m.get("event") == "persona_update" for m in fake.got))
    # a voice tweak is applied at once, as Navel TTS tags in front of what Stuart says
    code, v = call("/registry/voice", {"id": "stuart", "rate": 10, "pitch": -2, "volume": 0})
    assert code == 200 and v["voice"]["pitch"] == "-3.6st" and v["adjust"]["rate"] == 10
    call("/registry/say", {"id": "stuart", "text": "Testing."})
    assert wait(lambda: any(m.get("text", "").endswith("Testing.") and m["text"].startswith("<vol,") for m in fake.got))
    row = [r for r in call("/registry")[1]["robots"] if r["id"] == "stuart"][0]
    assert row["role_label"] == "tutor" and row["voice_adjust"]["rate"] == 10 and row["connected"]
    # remove it again: graph tab and link go, ChatBox is protected
    assert call("/registry/remove", {"id": "stuart"})[0] == 200
    assert "stuart" not in call("/robots")[1]["workspaces"] and "stuart" not in hub.links
    assert call("/registry/remove", {"id": "chatbox"})[0] == 400


def test_add_a_robot_that_connects_in_and_reject_bad_input(viz):
    call, hub, reg = viz
    code, out = call("/registry/add", {"name": "Stu Art", "kind": "connects-in"})
    assert code == 200 and out["id"] == "stuart" and not out["connected"] and "robot_name" in out["detail"]
    robot = _FakeRobot(hub.port, "Stu Art", "stu_1"); robot.wait_for("client_init_response")      # name with a space still matches
    assert wait(lambda: [r for r in call("/registry")[1]["robots"] if r["id"] == "stuart"][0]["connected"])
    for bad in ({"name": "Stu Art"}, {"name": "x!"}, {"name": "Navel2", "kind": "dial-out"}):
        assert call("/registry/add", bad)[0] == 400
    robot.close()
