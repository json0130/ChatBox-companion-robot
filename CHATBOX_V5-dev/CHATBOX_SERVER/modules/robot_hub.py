"""Robot hub: a tiny server the ChatBox Jetson client and the Navel client connect OUT to.

It speaks the same Socket.IO contract as the central servers, so a robot only needs its
client_config.json `server_url` pointed here (http://<this machine>:<port>); nothing changes on the
Jetson / Navel. We use two server -> client events the clients already handle:
  persona_update  role + OCEAN personality; the robot says "Persona updated to <name>."
  demo_step       {text: "[TAG] words"}: the robot speaks the words and the tag drives its face
                  (on the ChatBox the Jetson forwards the tag to the ESP32).
Events the robot sends us (camera frames, speech, ...) are ignored.
"""
import threading
from typing import Dict, Optional

import socketio
from werkzeug.serving import make_server


# the ChatBox client's own voice_config keys (client_config.json); persona offsets go on top
BASE_VOICE = {"language": "en", "gender": "female", "provider": "gtts"}


def ocean_to_personality(ocean: Dict[str, float]) -> Dict[str, float]:
    """viz OCEAN is -1..1; the client_config `personality` block is 0..1."""
    return {k: round((float(v) + 1) / 2, 3) for k, v in ocean.items()}


class RobotHub:
    def __init__(self, port: int = 5000, host: str = "0.0.0.0") -> None:
        self.sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
        self.clients: Dict[str, dict] = {}           # sid -> {robot_name, client_id}
        self.last_persona: Optional[dict] = None     # replayed to a robot that connects after Apply
        self._step = 0
        self._acks: Dict[str, threading.Event] = {}  # step_id -> set when the robot finished speaking it
        self._lock = threading.Lock()
        self._server = make_server(host, port, socketio.WSGIApp(self.sio), threaded=True)
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
                persona = self.last_persona
            self.sio.emit("client_init_response", {"success": True, "client_id": data["client_id"],
                          "robot_name": data["robot_name"],
                          "message": f"persona hub: {data['robot_name']} connected"}, to=sid)
            if persona:
                self.sio.emit("persona_update", persona, to=sid)

        @self.sio.on("ack")
        def ack(sid, data):
            ev = self._acks.get((data or {}).get("step_id", ""))
            if ev:
                ev.set()

        @self.sio.on("disconnect")
        def disconnect(sid):
            with self._lock:
                self.clients.pop(sid, None)

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()

    def robots(self) -> list:
        with self._lock:
            return [dict(c) for c in self.clients.values()]

    def push_persona(self, persona_name: str, robot_role: str, ocean: Dict[str, float],
                     voice: Optional[Dict[str, str]] = None) -> dict:
        payload = {"persona_name": persona_name, "robot_role": robot_role,
                   "personality": ocean_to_personality(ocean)}
        if voice:           # the client REPLACES its whole voice_config, so keep the base voice keys
            payload["voice_config"] = {**BASE_VOICE, **voice}
        with self._lock:
            self.last_persona = payload
            targets = list(self.clients)
        for sid in targets:
            self.sio.emit("persona_update", payload, to=sid)
        return {"robots": len(targets), "sent": bool(targets)}

    def say(self, text: str) -> dict:
        with self._lock:
            self._step += 1
            step, targets = f"persona-{self._step}", list(self.clients)
        for sid in targets:
            self.sio.emit("demo_step", {"step_id": step, "text": text, "require_ack": False}, to=sid)
        return {"robots": len(targets), "sent": bool(targets)}

    def say_steps(self, texts, timeout: float = 30.0) -> dict:
        """Speak `texts` one after another: each is its own demo_step, sent once the robot has
        acked (= finished speaking) the previous one, so each step's [TAG] fires its own gesture.
        Blocks; call from a thread. A robot that never acks costs `timeout` per step."""
        with self._lock:
            targets = list(self.clients)
        for text in texts:
            with self._lock:
                self._step += 1
                step = f"persona-{self._step}"
            ev = self._acks[step] = threading.Event()
            for sid in targets:
                self.sio.emit("demo_step", {"step_id": step, "text": text, "require_ack": True}, to=sid)
            if targets:
                ev.wait(timeout)
            self._acks.pop(step, None)
        return {"robots": len(targets), "sent": bool(targets), "steps": len(texts)}
