"""Dial OUT to a Navel robot (the opposite of the ChatBox Jetson, which dials in to the viz hub).

The Navel client (robot-robot-interaction/navel-client/client.py) runs a plain WebSocket SERVER on
ws_port (8765) and waits for a server to connect; it never calls out. So the viz opens
ws://<navel-ip>:8765 itself and keeps it open, reconnecting when the link drops:

    to Navel   {"event": "demo_step", "step_id", "text", "require_ack"}   speak (a leading [TAG] is a gesture)
               {"event": "persona_update", "persona_name", "personality"}
    from Navel {"type": "ack", "step_id"}   {"type": "speech", "audio": <base64 wav>}   {"type": "tts_state", ...}
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Callable, Optional

import websocket


class NavelLink:
    def __init__(self, host: str, port: int = 8765, name: str = "Navel", client_id: str = "navel_001",
                 retry_s: float = 3.0, tts: bool = True) -> None:
        self.host, self.port, self.name, self.client_id, self.retry_s = host, port, name, client_id, retry_s
        self.tts = tts                  # takes Navel TTS tags (<vol,..><rpit,..><rspd,..>) in front of its text
        self.robot_id = name.lower()
        self.url = f"ws://{host}:{port}"
        self.connected = False
        self.last_error = ""
        self.on_ack: Optional[Callable[[str], None]] = None          # (step_id)
        self.on_speech: Optional[Callable[[str, str], None]] = None  # (audio_b64, robot name)
        self.on_open: Optional[Callable[[], None]] = None            # () the link just came up (hub: starting pose, last persona)
        self._ws: Optional[websocket.WebSocketApp] = None
        self._stop = False

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="navel-link").start()

    def stop(self) -> None:
        self._stop = True
        if self._ws:
            self._ws.close()

    def send(self, data: dict) -> bool:
        if not (self._ws and self.connected):
            return False
        try:
            self._ws.send(json.dumps(data))
            return True
        except Exception as exc:
            self.last_error = f"send failed: {exc}"
            return False

    def status(self) -> dict:
        return {"url": self.url, "connected": self.connected, "error": self.last_error}

    # ---- internals ----
    def _run(self) -> None:
        while not self._stop:
            self._ws = websocket.WebSocketApp(
                self.url, on_open=self._open, on_message=self._message,
                on_error=lambda ws, e: setattr(self, "last_error", str(e) or type(e).__name__),
                on_close=self._close)
            self._ws.run_forever(ping_interval=20, ping_timeout=10)
            self.connected = False
            if not self._stop:
                time.sleep(self.retry_s)

    def _open(self, ws) -> None:
        self.connected, self.last_error = True, ""
        if self.on_open:
            self.on_open()

    def _close(self, ws, code, msg) -> None:
        self.connected = False
        if not self.last_error:
            self.last_error = f"closed ({code})"

    def _message(self, ws, raw) -> None:
        try:
            data = json.loads(raw)
        except ValueError:
            return
        kind = data.get("type")
        if kind == "ack" and self.on_ack:
            self.on_ack(str(data.get("step_id", "")))
        elif kind == "speech" and self.on_speech and data.get("audio"):
            threading.Thread(target=self.on_speech, args=(data["audio"], self.name), daemon=True).start()


class InboundLink(NavelLink):
    """A robot that dialled US (client_config `connect_to_server`, for a robot behind NAT / on Tailscale).
    Same messages as NavelLink, the socket is just opened from the other side: the hub's inbound server
    (RobotHub.start_inbound) owns the socket and feeds what it receives to `_message`."""

    def __init__(self, loop, sock, name: str, client_id: str, tts: bool = False) -> None:
        super().__init__("inbound", 0, name=name, client_id=client_id, tts=tts)
        self._loop, self._sock = loop, sock
        self.url, self.connected = f"inbound:{client_id}", True

    def start(self) -> None:        # the socket is already open
        pass

    def stop(self) -> None:
        self._stop = self.connected = False
        asyncio.run_coroutine_threadsafe(self._sock.close(), self._loop)

    def send(self, data: dict) -> bool:
        if not self.connected:
            return False
        asyncio.run_coroutine_threadsafe(self._sock.send(json.dumps(data)), self._loop)
        return True
