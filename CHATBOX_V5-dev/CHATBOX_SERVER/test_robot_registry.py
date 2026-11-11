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


def test_navel_gets_method_b_gesture_tags_and_its_own_action_buttons(env, tmp_path):  # noqa: F811
    from modules.action_tag_generator import ChatLLM, navel_tagger
    from test_action_tag_generator import REPLY, ScriptedLLM
    fake, hub = FakeNavel(), RobotHub(port=0, host="127.0.0.1"); hub.start()
    link = NavelLink("127.0.0.1", fake.port, retry_s=0.2); hub.add_link("navel", link)
    assert wait(lambda: link.connected)
    path = str(tmp_path / "kg_navel.json"); seed_robot_graph(path, "navel")
    st = GraphState(path, str(tmp_path / "pa_navel.json"))
    tagger = {"navel": navel_tagger(ChatLLM(ScriptedLLM()), overrun_seconds=10)}   # long gestures allowed: keep every pick
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
        env["state"], None, PersonaChat(env["kg"], env["state"]._persona_path(), env["llm"]), hub, tagger, None,
        {"navel": (st, PersonaChat(path, st._persona_path(), StubLLM(REPLY)))}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"

    def post(p, body):
        req = urllib.request.Request(url + p, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)
    try:
        code, out = post("/chat", {"message": "hi", "robot": "navel"})
        assert code == 200 and [p["tag"] for p in out["plan"]][1:4] == ["GREET", "SAD", "HAPPY"]   # Navel's own tag names
        texts = lambda: [m["text"] for m in fake.got if m.get("event") == "demo_step"]
        assert wait(lambda: len(texts()) == len(out["plan"]))                   # spoken step by step, each acked
        texts = texts()
        assert "[GREET] Welcome to the lab." in texts and "[POINT] Over there is the arm." in texts
        assert not any(t.strip() == "[DEFAULT]" for t in texts)                 # the DEFAULT pose reset is the ChatBox's
        assert "actions" in json.load(urllib.request.urlopen(url + "/robots"))
        assert post("/action", {"robot": "navel", "tag": "agree"})[0] == 200
        assert wait(lambda: any(m.get("text") == "[AGREE]" for m in fake.got))
        assert post("/action", {"robot": "navel", "tag": "GREETING"})[0] == 400      # a ChatBox tag
    finally:
        srv.shutdown(); link.stop(); hub.stop()


def test_the_merged_graph_keeps_jay_apart_from_the_robots(viz, env):  # noqa: F811
    from modules.graph_relationship.interactions import get_or_create_interaction
    from modules.graph_relationship.schema import (AboutEdge, CapabilityNode, HasCapabilityEdge, InterestNode,
                                                   PersonNode, Provenance)
    from modules.graph_relationship.store import InMemoryGraphStore
    from modules.graph_relationship.topics import resolve_topic
    call, hub, reg = viz
    prov = Provenance(source="t", confidence=1.0)
    store = InMemoryGraphStore(); store.load(env["kg"], quiet=True)
    store.upsert_node(PersonNode(id="jay", display_name="Jay"))
    get_or_create_interaction(store, "jay", "chatbox")
    store.upsert_node(CapabilityNode(id="chatbox:capability"))
    store.upsert_edge(HasCapabilityEdge(source_id="chatbox", target_id="chatbox:capability", provenance=prov))
    jazz = resolve_topic(store, "jazz")                                  # a topic the robot AND Jay's interest reach
    store.upsert_edge(AboutEdge(source_id="chatbox:capability", target_id=jazz.id, provenance=prov))
    store.upsert_node(InterestNode(id="interest:jay:music", label="music"))
    store.upsert_edge(AboutEdge(source_id="interest:jay:music", target_id=jazz.id, provenance=prov))
    store.save(env["kg"])
    g = call("/graph.json")[1]
    ids = {n["id"] for n in g["nodes"]}
    assert {"jay", "chatbox", "interaction:jay:chatbox", "chatbox:capability", jazz.id, "interest:jay:music"} <= ids
    edges = {(e["source"], e["type"], e["target"]) for e in g["edges"]}
    assert ("chatbox", "has_capability", "chatbox:capability") in edges             # the robot keeps its sub nodes
    assert ("chatbox:capability", "about", jazz.id) in edges
    assert ("jay", "has_interaction", "interaction:jay:chatbox") in edges           # Jay keeps his own
    assert ("chatbox", "has_interaction", "interaction:jay:chatbox") not in edges   # nothing joins the two
    assert ("interest:jay:music", "about", jazz.id) not in edges


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def test_a_chatbox_on_the_shared_client_protocol_gets_the_full_persona_and_the_default_pose():
    """The shared (v6) client for the ChatBox, dialled by link: Socket.IO behaviour kept (robot_role + voice_config
    in persona_update, [DEFAULT] pose on connect and after speech); Navel still gets name + OCEAN only, no pose."""
    box, nav, hub = FakeNavel(), FakeNavel(), RobotHub(port=0, host="127.0.0.1"); hub.start()
    lb = NavelLink("127.0.0.1", box.port, name="ChatBox", retry_s=0.2, tts=False)
    ln = NavelLink("127.0.0.1", nav.port, name="Navel", retry_s=0.2, tts=True)
    hub.add_link("chatbox", lb); hub.add_link("navel", ln)
    try:
        assert wait(lambda: lb.connected and ln.connected)
        assert wait(lambda: any(m.get("text") == "[DEFAULT]" for m in box.got))        # starting pose
        out = hub.push_persona("Tutor", "You are a tutor.", {"E": 0.5}, voice={"rate": "+5%"}, robot="chatbox")
        assert out["sent"]
        assert wait(lambda: any(m.get("event") == "persona_update" for m in box.got))
        pu = [m for m in box.got if m.get("event") == "persona_update"][0]
        assert pu["robot_role"] == "You are a tutor." and pu["voice_config"]["rate"] == "+5%" and pu["voice_config"]["provider"] == "gtts"
        hub.push_persona("Tutor", "You are a tutor.", {"E": 0.5}, robot="navel")
        assert wait(lambda: any(m.get("event") == "persona_update" for m in nav.got))
        nu = [m for m in nav.got if m.get("event") == "persona_update"][0]
        assert set(nu) == {"event", "persona_name", "personality"}
        hub.say_steps(["[GREET] Hi."], robot="chatbox"); hub.say_steps(["[GREET] Hi."], robot="navel")
        assert wait(lambda: [m["text"] for m in box.got if m.get("event") == "demo_step"][-1] == "[DEFAULT]")
        assert not any(m.get("text") == "[DEFAULT]" for m in nav.got)
    finally:
        lb.stop(); ln.stop(); hub.stop()


def test_a_robot_that_dials_in_with_a_hello_is_spoken_to_and_its_acks_count():
    """connect_to_server mode of the shared client: it registers as inbound, then opens ws://hub:PORT and says hello."""
    import websocket
    hub = RobotHub(port=0, host="127.0.0.1"); hub.start()
    port = _free_port()
    hub.inbound["chatbox_jetson_001"] = {"robot_id": "chatbox", "name": "ChatBox", "tts": False}
    hub.start_inbound(port, "127.0.0.1")
    try:
        assert wait(lambda: _can_connect(port))
        refused = websocket.create_connection(f"ws://127.0.0.1:{port}", timeout=3)
        refused.send(json.dumps({"type": "hello", "client_id": "stranger"}))
        assert not refused.recv()                                                   # closed on us: not registered
        ws = websocket.create_connection(f"ws://127.0.0.1:{port}", timeout=3)
        ws.send(json.dumps({"type": "hello", "client_id": "chatbox_jetson_001"}))
        assert json.loads(ws.recv())["text"] == "[DEFAULT]"                         # greeted like a Socket.IO ChatBox
        assert wait(lambda: hub.links.get("chatbox") and hub.links["chatbox"].connected)
        hub.push_persona("Tutor", "You are a tutor.", {"E": 0.5}, robot="chatbox")
        assert json.loads(ws.recv())["robot_role"] == "You are a tutor."
        done = []
        threading.Thread(target=lambda: done.append(hub.say_steps(["[GREET] Hi."], robot="chatbox")), daemon=True).start()
        step = json.loads(ws.recv()); assert step["text"] == "[GREET] Hi." and step["require_ack"]
        ws.send(json.dumps({"type": "ack", "step_id": step["step_id"]}))
        assert wait(lambda: done)
        assert json.loads(ws.recv())["text"] == "[DEFAULT]"                         # pose reset after speech
        ws.close()
        assert wait(lambda: "chatbox" not in hub.links)                             # a dropped robot is gone, not stale
    finally:
        hub.stop()


def _can_connect(port):
    import websocket
    try:
        websocket.create_connection(f"ws://127.0.0.1:{port}", timeout=1).close(); return True
    except Exception:
        return False
