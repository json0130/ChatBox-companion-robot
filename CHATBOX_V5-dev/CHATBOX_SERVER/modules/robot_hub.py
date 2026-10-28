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


def ocean_to_personality(ocean: Dict[str, float]) -> Dict[str, float]:
    """viz OCEAN is -1..1; the client_config `personality` block is 0..1."""
    return {k: round((float(v) + 1) / 2, 3) for k, v in ocean.items()}


class RobotHub:
    def __init__(self, port: int = 5000, host: str = "0.0.0.0") -> None:
        self.sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
        self.clients: Dict[str, dict] = {}           # sid -> {robot_name, client_id}
        self.last_persona: Optional[dict] = None     # replayed to a robot that connects after Apply
        self._step = 0
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

    def push_persona(self, persona_name: str, robot_role: str, ocean: Dict[str, float]) -> dict:
        payload = {"persona_name": persona_name, "robot_role": robot_role,
                   "personality": ocean_to_personality(ocean)}
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
