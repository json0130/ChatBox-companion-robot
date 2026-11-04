"""Robot hub: a tiny server the ChatBox Jetson client and the Navel client connect OUT to.

It speaks the same Socket.IO contract as the central servers, so a robot only needs its
client_config.json `server_url` pointed here (http://<this machine>:<port>); nothing changes on the
Jetson / Navel. We use two server -> client events the clients already handle:
  persona_update  role + OCEAN personality; the robot says "Persona updated to <name>."
  demo_step       {text: "[TAG] words"}: the robot speaks the words and the tag drives its face
                  (on the ChatBox the Jetson forwards the tag to the ESP32).
Events the robot sends us (camera frames, ...) are ignored; `speech` goes to `on_speech` if set.
"""
import errno
import socket
import threading
from typing import Dict, Optional

import socketio

from modules.persona_voice import navel_tts_tags
from werkzeug.serving import make_server


# the ChatBox client's own voice_config keys (client_config.json); persona offsets go on top
BASE_VOICE = {"language": "en", "gender": "female", "provider": "gtts"}


# robots whose client knows the `[DEFAULT]` pose tag (ChatBox: EMOTION_MAP); other robots just get speech
DEFAULT_POSE_ROBOTS = {"chatbox"}


def ocean_to_personality(ocean: Dict[str, float]) -> Dict[str, float]:
    """viz OCEAN is -1..1; the client_config `personality` block is 0..1."""
    return {k: round((float(v) + 1) / 2, 3) for k, v in ocean.items()}


def require_free(port: int, host: str = "0.0.0.0", what: str = "port") -> int:
    """Ports are FIXED (viz 8765, robot hub 5000): the robots' server_url points at them, so quietly moving
    to another port would leave a robot talking to nothing (or to a stale copy). If it is taken, stop with a
    clear message instead. Probes by binding because werkzeug's make_server exits on EADDRINUSE with a vaguer one."""
    with socket.socket() as probe:
        try:
            probe.bind((host, port))
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            raise SystemExit(f"[viz] {what} {port} is already in use - another viz (or the central server) is "
                             f"running. Stop it first: fuser -k {port}/tcp")
    return port


class RobotHub:
    def __init__(self, port: int = 5000, host: str = "0.0.0.0") -> None:
        self.sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
        self.clients: Dict[str, dict] = {}           # sid -> {robot_name, client_id}
        self.last_persona: Dict[str, dict] = {}      # robot id -> last persona, replayed to a robot that connects after Apply
        self._step = 0
        self._acks: Dict[str, threading.Event] = {}  # step_id -> set when the robot finished speaking it
        self._lock = threading.Lock()
        self.voices: Dict[str, dict] = {}            # robot id -> voice spec, applied as TTS tags on robots we dial
        self.links: Dict[str, "NavelLink"] = {}      # robot id -> a robot we dial OUT to (Navel runs its own server)
        self.on_speech = None                        # callable(audio_b64, robot_name), set by the viz: mic audio -> text -> reply
        self._server = make_server(host, require_free(port, host, "robot hub port"), socketio.WSGIApp(self.sio), threaded=True)
        self.port = self._server.server_port
        self._wire()

    def _wire(self) -> None:
        @self.sio.on("client_init")
        def client_init(sid, data):
            data = data or {}
            if not data.get("robot_name") or not data.get("client_id"):
                self.sio.emit("client_init_response", {"success": False,
                              "message": "robot_name and client_id are required"}, to=sid)
                return
            with self._lock:
                self.clients[sid] = {"robot_name": data["robot_name"], "client_id": data["client_id"]}
                persona = self.last_persona.get(data["robot_name"].lower().replace(" ", ""))
            self.sio.emit("client_init_response", {"success": True, "client_id": data["client_id"],
                          "robot_name": data["robot_name"],
                          "message": f"persona hub: {data['robot_name']} connected"}, to=sid)
            # tag-only step (no words): the robot starts in its default pose
            if data["robot_name"].lower().replace(" ", "") in DEFAULT_POSE_ROBOTS:
                self.sio.emit("demo_step", {"step_id": "connect", "text": "[DEFAULT]", "require_ack": False}, to=sid)
            if persona:
                self.sio.emit("persona_update", persona, to=sid)

        @self.sio.on("ack")
        def ack(sid, data):
            self._ack_step((data or {}).get("step_id", ""))

        @self.sio.on("speech")
        def speech(sid, data):
            audio, cb = (data or {}).get("audio", ""), self.on_speech
            if cb and audio:        # own thread: transcribing + speaking must not block this client's acks
                with self._lock:
                    name = self.clients.get(sid, {}).get("robot_name", "chatbox")
                threading.Thread(target=cb, args=(audio, name), daemon=True).start()

        @self.sio.on("disconnect")
        def disconnect(sid):
            with self._lock:
                self.clients.pop(sid, None)

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()

    def _ack_step(self, step_id: str) -> None:
        ev = self._acks.get(step_id)
        if ev:
            ev.set()

    def add_link(self, robot_id: str, link) -> None:
        """Attach a robot we dial out to (see navel_link.py); it then behaves like a connected client."""
        link.robot_id = robot_id
        link.on_ack = self._ack_step
        link.on_speech = lambda audio, name: self.on_speech and self.on_speech(audio, name)
        self.links[robot_id] = link
        link.start()

    def set_voice(self, robot_id: str, voice: Optional[dict]) -> None:
        """The voice spec (rate / pitch / tone / volume, see persona_voice) a dialled robot speaks with."""
        if voice:
            self.voices[robot_id] = voice
        else:
            self.voices.pop(robot_id, None)

    def remove_link(self, robot_id: str) -> None:
        link = self.links.pop(robot_id, None)
        if link:
            link.stop()
        self.voices.pop(robot_id, None)

    def _spoken(self, link, text: str) -> str:
        """`text` for a robot we dial: with its voice spec as the robot's own TTS tags, if it takes them."""
        voice = self.voices.get(getattr(link, "robot_id", ""))
        return (navel_tts_tags(voice) + text) if (voice and getattr(link, "tts", False)) else text

    def _links(self, robot: Optional[str] = None) -> list:
        return [l for r, l in self.links.items() if l.connected and (robot is None or r == robot)]

    def _sids(self, robot: Optional[str] = None) -> list:
        """Connected clients; `robot` (a robot id like "navel", matched to the client's robot_name,
        case-insensitive) limits it to that robot, None = every robot."""
        with self._lock:
            return [sid for sid, c in self.clients.items()
                    if robot is None or c["robot_name"].lower().replace(" ", "") == robot]

    def robots(self, robot: Optional[str] = None) -> list:
        with self._lock:
            out = [dict(c) for c in self.clients.values()
                   if robot is None or c["robot_name"].lower().replace(" ", "") == robot]
        return out + [{"robot_name": l.name, "client_id": l.client_id, "via": l.url} for l in self._links(robot)]

    def push_persona(self, persona_name: str, robot_role: str, ocean: Dict[str, float],
                     voice: Optional[Dict[str, str]] = None, robot: Optional[str] = None) -> dict:
        payload = {"persona_name": persona_name, "robot_role": robot_role,
                   "personality": ocean_to_personality(ocean)}
        if voice:           # the client REPLACES its whole voice_config, so keep the base voice keys
            payload["voice_config"] = {**BASE_VOICE, **voice}
        with self._lock:
            self.last_persona[robot or "*"] = payload
        targets = self._sids(robot)
        for sid in targets:
            self.sio.emit("persona_update", payload, to=sid)
        # Navel writes a persona_update into its own client_config.json (robot_role, tags, voice...), and our
        # voice offsets are the ChatBox client's units, so it only gets the name and the OCEAN personality.
        if robot and voice:
            self.set_voice(robot, voice)
        nav = {"event": "persona_update", "persona_name": persona_name, "personality": payload["personality"]}
        sent = sum(l.send(nav) for l in self._links(robot))
        return {"robots": len(targets) + sent, "sent": bool(targets) or bool(sent)}

    def say(self, text: str, robot: Optional[str] = None) -> dict:
        with self._lock:
            self._step += 1
            step = f"persona-{self._step}"
        targets = self._sids(robot)
        for sid in targets:
            self.sio.emit("demo_step", {"step_id": step, "text": text, "require_ack": False}, to=sid)
        sent = sum(l.send({"event": "demo_step", "step_id": step, "text": self._spoken(l, text),
                           "require_ack": False}) for l in self._links(robot))
        return {"robots": len(targets) + sent, "sent": bool(targets) or bool(sent)}

    def say_steps(self, texts, timeout: float = 30.0, robot: Optional[str] = None) -> dict:
        """Speak `texts` one after another: each is its own demo_step, sent once the robot has
        acked (= finished speaking) the previous one, so each step's [TAG] fires its own gesture.
        `texts` may be a generator that blocks until the next sentence is ready, so speech can start
        before the whole reply is planned. Only `robot` hears it (None = all). Blocks; call from a
        thread. A robot that never acks costs `timeout` per step."""
        targets, links = self._sids(robot), self._links(robot)
        n = 0
        for text in texts:
            n += 1
            with self._lock:
                self._step += 1
                step = f"persona-{self._step}"
            ev = self._acks[step] = threading.Event()
            for sid in targets:
                self.sio.emit("demo_step", {"step_id": step, "text": text, "require_ack": True}, to=sid)
            heard = sum(l.send({"event": "demo_step", "step_id": step, "text": self._spoken(l, text),
                                "require_ack": True}) for l in links)
            if targets or heard:
                ev.wait(timeout)
            self._acks.pop(step, None)
        if n:                       # speech is over: tag-only step, no words, so the robot returns to its default pose
            with self._lock:
                self._step += 1
                step = f"persona-{self._step}"
                knows = [sid for sid in targets if sid in self.clients
                         and self.clients[sid]["robot_name"].lower().replace(" ", "") in DEFAULT_POSE_ROBOTS]
            for sid in knows:
                self.sio.emit("demo_step", {"step_id": step, "text": "[DEFAULT]", "require_ack": False}, to=sid)
        return {"robots": len(targets) + len(links), "sent": bool(targets or links), "steps": n}
