"""ChatBox ESP32 link: LLM action tag -> expression name -> one TCP line (firmware: validExpressions[]).

Used by the webcam loop when it runs next to the ESP32. The viz does NOT use this: it reaches the
robot through the Jetson client (modules/robot_hub.py).
"""
import socket

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

