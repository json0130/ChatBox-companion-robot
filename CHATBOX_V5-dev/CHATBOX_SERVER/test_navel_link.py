"""NavelLink + hub against a fake Navel: a plain WebSocket SERVER like navel-client/client.py."""
import asyncio
import json
import threading
import time

import pytest
import websockets

from modules.navel_link import NavelLink
from modules.robot_hub import RobotHub


class FakeNavel:
    def __init__(self):
        self.got, self.ws, self.loop = [], None, asyncio.new_event_loop()
        started = threading.Event()
        threading.Thread(target=self._run, args=(started,), daemon=True).start()
        started.wait(5)

    def _run(self, started):
        asyncio.set_event_loop(self.loop)

        async def handler(ws):
            self.ws = ws
            async for raw in ws:
                data = json.loads(raw)
                self.got.append(data)
                if data.get("event") == "demo_step" and data.get("require_ack"):     # speak, then ack
                    await ws.send(json.dumps({"type": "ack", "step_id": data["step_id"]}))

        async def main():
            async with websockets.serve(handler, "127.0.0.1", 0) as srv:
                self.port = srv.sockets[0].getsockname()[1]
                started.set()
                await asyncio.Future()
        self.loop.run_until_complete(main())

    def send(self, data):
        asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(data)), self.loop).result(3)


def wait(cond, t=4.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


@pytest.fixture
def navel_hub():
    fake = FakeNavel()
    hub = RobotHub(port=0, host="127.0.0.1"); hub.start()
    link = NavelLink("127.0.0.1", fake.port, retry_s=0.2)
    hub.add_link("navel", link)
    assert wait(lambda: link.connected)
    yield fake, hub, link
    link.stop(); hub.stop()


def test_navel_counts_as_a_connected_robot(navel_hub):
    fake, hub, link = navel_hub
    assert [r["robot_name"] for r in hub.robots("navel")] == ["Navel"] and hub.robots("chatbox") == []
    assert link.status()["connected"] is True


def test_say_steps_speaks_one_step_at_a_time_and_waits_for_navels_ack(navel_hub):
    fake, hub, link = navel_hub
    out = hub.say_steps(["[GREET] Hello!", "Nice to meet you."], robot="navel")
    steps = [m for m in fake.got if m.get("event") == "demo_step"]
    assert [s["text"] for s in steps] == ["[GREET] Hello!", "Nice to meet you."]
    assert all(s["require_ack"] for s in steps) and out == {"robots": 1, "sent": True, "steps": 2}
    assert not any(m.get("text") == "[DEFAULT]" for m in fake.got)       # DEFAULT pose is the ChatBox's tag list


def test_persona_sends_only_the_name_and_personality(navel_hub):
    fake, hub, link = navel_hub
    hub.push_persona("Tutor", "help with maths", {"O": 0, "C": 1, "E": 0, "A": 0, "N": 0},
                     {"rate": "+5%", "pitch": "+1.0st"}, robot="navel")
    assert wait(lambda: any(m.get("event") == "persona_update" for m in fake.got))
    msg = next(m for m in fake.got if m.get("event") == "persona_update")
    assert msg == {"event": "persona_update", "persona_name": "Tutor",
                   "personality": {"O": 0.5, "C": 1.0, "E": 0.5, "A": 0.5, "N": 0.5}}   # never robot_role / voice_config


def test_speech_from_navel_reaches_the_hub_as_navels(navel_hub):
    fake, hub, link = navel_hub
    heard = []
    hub.on_speech = lambda audio, name: heard.append((audio, name))
    fake.send({"type": "speech", "audio": "QVVESU8="})
    assert wait(lambda: heard) and heard == [("QVVESU8=", "Navel")]


def test_link_reconnects_and_reports_why_it_cannot_connect():
    link = NavelLink("127.0.0.1", 1, retry_s=0.2)                        # nothing listens on port 1
    link.start()
    assert wait(lambda: link.last_error) and link.connected is False
    link.stop()


def test_voice_spec_goes_to_navel_as_tts_tags_in_front_of_every_step(navel_hub):
    fake, hub, link = navel_hub
    hub.push_persona("Receptionist", "x", {k: 0 for k in "OCEAN"}, {"rate": "-3%", "pitch": "-5.0st", "volume": "+1dB"},
                     robot="navel")
    hub.say_steps(["[GREET] Hello!"], robot="navel")
    steps = [m for m in fake.got if m.get("event") == "demo_step"]
    assert steps[0]["text"] == "<vol,67><rpit,75><rspd,97>[GREET] Hello!"      # pitch -5 st = 75 %, volume +1 dB of 60
    hub.say("Plain", robot="navel")
    assert wait(lambda: any(m.get("text", "").endswith("Plain") and m["text"].startswith("<vol,") for m in fake.got))
    hub.set_voice("navel", None)
    hub.say("Bare", robot="navel")
    assert wait(lambda: any(m.get("text") == "Bare" for m in fake.got))


def test_a_robot_registers_with_the_hub_port_like_navels_client_does(navel_hub):
    import urllib.error
    import urllib.request
    fake, hub, link = navel_hub

    def register(body):
        req = urllib.request.Request(f"http://127.0.0.1:{hub.port}/robots/register", data=json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)
    seen = []
    hub.on_register = lambda d: seen.append(d) or "ok"
    body = {"robot_name": "Navel", "client_id": "navel_001", "ip_address": "10.0.0.5", "ws_port": 8765}
    assert register(body) == (200, {"message": "ok"}) and seen == [body]
    assert register({"robot_name": "Navel"})[0] == 400                    # no ip to dial
    hub.on_register = lambda d: (_ for _ in ()).throw(ValueError("unknown robot"))
    assert register(body) == (400, {"error": "unknown robot"})
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://127.0.0.1:{hub.port}/nothing", timeout=5)
    assert link.connected                                                 # Socket.IO side unaffected
