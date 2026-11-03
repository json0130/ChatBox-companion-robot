"""Persona -> how the robot SOUNDS: speed, pitch, tone, volume on top of its base voice.

The base voice (gTTS on the ChatBox Jetson) is unchanged; these are offsets the client applies
with ffmpeg (FINAL_DEMO/CHATBOX_CLIENT/OutputModules/edge_tts_output.py). Sent to the robot as
`voice_config` in persona_update. All numbers are a starting point to tune by ear.

    rate    "+8%"    speed vs the client's normal pace
    pitch   "+1.5st" semitones; speed is not affected
    tone    "+2dB"   > 0 brighter (treble up, bass down), < 0 warmer
    volume  "+1dB"
"""

from __future__ import annotations

from typing import Dict

# role label -> (rate %, pitch st, tone dB, volume dB); added to the OCEAN part below
ROLE_VOICE = {
    "companion":     (-4, 0.0, -1, 0),
    "reading buddy": (-8, 1.0, 0, 0),
    "receptionist":  (6, -0.5, 2, 1),
    "storyteller":   (-10, 0.5, -1, 0),
    "tutor":         (0, 0.5, 1, 0),
}
LIMITS = {"rate": 25, "pitch": 4, "tone": 6, "volume": 3}   # per side; keeps ffmpeg's atempo in range


def _clamp(v: float, key: str) -> float:
    return max(-LIMITS[key], min(LIMITS[key], v))


def voice_for(role_label: str, ocean: Dict[str, float]) -> Dict[str, str]:
    """OCEAN in -1..1. Extraverts talk faster, higher and brighter; neurotic voices a touch
    faster and higher but quieter; agreeable voices warmer; conscientious ones a little slower."""
    r0, p0, t0, v0 = ROLE_VOICE.get(role_label, (0, 0.0, 0, 0))
    o = {k: float(ocean.get(k, 0.0)) for k in "OCEAN"}
    rate = _clamp(r0 + 14 * o["E"] + 5 * o["N"] - 3 * o["C"], "rate")
    pitch = _clamp(p0 + 2.5 * o["E"] + 1.5 * o["N"], "pitch")
    tone = _clamp(t0 + 3 * o["E"] - 2 * o["A"], "tone")
    volume = _clamp(v0 + 2 * o["E"] - 1 * o["N"], "volume")
    return {"rate": f"{rate:+.0f}%", "pitch": f"{pitch:+.1f}st",
            "tone": f"{tone:+.0f}dB", "volume": f"{volume:+.0f}dB"}
