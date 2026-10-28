"""ChatBox ESP32 link: LLM action tag -> expression name -> one TCP line (firmware: validExpressions[]).

Shared by the webcam loop and the viz persona chat. Stdlib only, so the viz needs no cv2/numpy.
"""
import socket
from typing import Optional

# Map LLM action tags → ESP32 validExpressions[]
TAG_TO_ESP32: dict = {
    "GREETING": "greeting",
    "WAVE":     "wave",
    "NOD":      "head_nod",
    "CONFUSED": "confused",
    "SAD":      "sad",
    "ANGRY":    "angry",
    "SHRUG":    "shrug",
    "POINT":    "point",
    "DANCE":    "seq_dance",
    "SLEEP":    "sleep",
    "IDLE":     "idle",
    "HAPPY":    "ears_wiggle",
    "SURPRISE": "ears_perk",
    "EARS":     "ears_perk",
}


def send_esp32(expression: str, host: str, port: int = 8888, timeout: float = 0.5) -> bool:
    """Send one expression; False (and a log line) if the robot is unreachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall((expression + "\n").encode())
        print(f"[ESP32] → {expression!r}")
        return True
    except OSError as exc:
        print(f"[ESP32] send failed ({host}:{port}): {exc}")
        return False


class Esp32Link:
    """A configured robot. `show(tag)` is what the viz chat calls after each reply."""

    def __init__(self, host: str, port: int = 8888) -> None:
        self.host, self.port = host, port

    def show(self, tag: str) -> dict:
        expr: Optional[str] = TAG_TO_ESP32.get((tag or "").upper())
        sent = send_esp32(expr, self.host, self.port) if expr else False
        return {"host": f"{self.host}:{self.port}", "expression": expr, "sent": sent}
