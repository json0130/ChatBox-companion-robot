"""
robots/chatbox — ChatBox (Jetson): piper voice on the USB speaker, wired ESP32 gestures
=========================================================================================
    python3 run.py chatbox

The shared client does the protocol. This adapter adds only the body:
  Speech.say()               speak one sentence with piper, played by aplay (cut short on barge-in)
  Robot.on_emotion_detected  tag -> ESP32 command over the USB cable (arduino_output.py)
"""

import logging
import os
import re
import subprocess
import tempfile

from client import BasicClient
from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.sim_tts_output import SimTTSOutputModule
from .arduino_output import ArduinoOutputModule

logger = logging.getLogger(__name__)

# LLM tag -> ESP32 command string (same map as FINAL_DEMO/CHATBOX_CLIENT/robot.py)
EMOTION_MAP = {
    "GREETING": "greeting", "GREET": "greeting", "POINT": "point",
    "CONFUSED": "confused", "SHRUG": "shrug", "ANGRY": "angry",
    "SAD": "sad", "SLEEP": "sleep", "DEFAULT": "default",
    "POSE": "pose", "HAPPY": "greeting", "FEAR": "sad",
    "SURPRISE": "confused", "NEUTRAL": "default",
    "HANDS_CLAP": "hands_clap", "EARS_WIGGLE": "ears_wiggle",
}


def find_usb_speaker() -> str:
    """First non-Tegra playback card from `aplay -l` as plughw:N,D. Card numbers swap between boots."""
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True).stdout
        for m in re.finditer(r"^card (\d+): (\S+) .*?device (\d+):", out, re.M):
            if "tegra" not in m.group(2).lower():
                return f"plughw:{m.group(1)},{m.group(3)}"
    except Exception as e:
        logger.warning(f"[Speech] speaker auto-detect failed: {e}")
    return "default"


class Speech(SimTTSOutputModule):
    """piper -> wav -> aplay. Without piper or its model, a sentence is timed but silent."""

    def __init__(self, name="speech", config=None):
        super().__init__(name, config)
        self.model = os.path.expanduser(self.config.get("piper_model", "~/piper-voices/en_US-amy-medium.onnx"))
        self.length_scale = float(self.config.get("piper_length_scale", 1.0))
        cmd = list(self.config.get("audio_cmd", ["aplay", "-D", "auto"]))
        self.audio_cmd = [find_usb_speaker() if a == "auto" else a for a in cmd]

    def say(self, sentence: str):
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            r = subprocess.run(
                ["piper", "--model", self.model, "--length_scale", str(self.length_scale),
                 "--noise_scale", str(self.config.get("piper_noise_scale", 0.667)),
                 "--sentence_silence", "0.2", "--output_file", wav],
                input=sentence.encode(), capture_output=True, timeout=20)
            if r.returncode != 0 or not os.path.getsize(wav):
                raise RuntimeError(r.stderr.decode(errors="ignore")[:200] or "empty output")
            player = subprocess.Popen(self.audio_cmd + [wav], stderr=subprocess.DEVNULL)
            while player.poll() is None:
                if self._interrupt_event.wait(0.05):     # barge-in: stop mid-sentence
                    player.terminate()
                    break
        except Exception as e:
            logger.warning(f"[Speech] piper failed ({e}); timing the sentence silently")
            super().say(sentence)
        finally:
            if os.path.exists(wav):
                os.unlink(wav)

    def update_voice_config(self, voice_config: dict):
        # persona_update: "rate": "+20%" -> faster = smaller piper length_scale
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)%$", str(voice_config.get("rate", "")))
        if m:
            self.length_scale = 1.0 / max(0.3, 1 + float(m.group(1)) / 100)
        logger.info(f"[Speech] voice config {voice_config} -> length_scale {self.length_scale:.2f}")


class Robot(BasicClient):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.arduino = None
        if "speech" in self.config.get("modules", []):
            self.register_input_module(VoiceInputModule("voice_input", self.config.get("voice_config", {})))
        self.register_output_module(ConsoleOutputModule("console_output", self.config.get("console_config", {})))
        self.register_output_module(Speech("speech", self.config.get("edge_tts_config", {})))
        if self.config.get("features", {}).get("arduino_integration", True):
            self.arduino = ArduinoOutputModule("arduino_output", self.config.get("arduino_output", {}))
            if not self.register_output_module(self.arduino):
                logger.warning("[Setup] ESP32 link failed to register")
                self.arduino = None

    def on_emotion_detected(self, emotion_tag: str):
        tag = emotion_tag.strip().strip("[]").upper()
        command = EMOTION_MAP.get(tag, "default")
        logger.info(f"[Emotion] {tag} -> {command}")
        if self.arduino and self.arduino.is_connected():
            self.arduino.send_command(command)
        else:
            logger.warning(f"[ESP32] not connected, skipped '{command}'")
