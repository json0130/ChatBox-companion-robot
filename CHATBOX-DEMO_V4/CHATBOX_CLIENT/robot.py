# robot.py — Robot client (v6)
import sys
import re
import logging
from typing import Optional

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule
from OutputModules.arduino_output import ArduinoOutputModule

logger = logging.getLogger(__name__)


# Maps emotion tags emitted by the LLM ([GREETING], [SAD], etc.)
# to the command names the Arduino firmware understands.
EMOTION_MAP = {
    "GREETING": "greeting", "WAVE":   "wave",      "POINT":    "point",
    "CONFUSED": "confused", "SHRUG":  "shrug",     "ANGRY":    "angry",
    "SAD":      "sad",      "SLEEP":  "sleep",     "DEFAULT":  "default",
    "POSE":     "pose",     "HAPPY":  "greeting",  "FEAR":     "sad",
    "SURPRISE": "confused", "NEUTRAL": "default",
}


class SimpleConcurrentClient(BasicClient):
    """
    Robot client (v6) — verbal interaction + Arduino gestures.

    • Parses [EMOTION] tags from chat_response and forwards to Arduino.
    • Handles persona_update event from server → updates TTS voice config live.
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        self.arduino_module: Optional[ArduinoOutputModule] = None
        self.setup_all_modules()
        self._register_custom_event_handlers()

    # ── Arduino helpers ───────────────────────────────────────────────────────

    def _on_arduino_connected(self):
        logger.info("[Arduino] Connected")

    def _on_arduino_disconnected(self):
        logger.warning("[Arduino] Disconnected")

    def _on_arduino_error(self, error_msg: str):
        logger.error(f"[Arduino] Error: {error_msg}")

    def send_robot_emotion(self, emotion: str) -> bool:
        """Map an emotion tag to an Arduino command and send it."""
        command = EMOTION_MAP.get(emotion.upper(), "default")
        if self.arduino_module and self.arduino_module.is_connected():
            return self.arduino_module.send_command(command)
        logger.warning(f"[Arduino] Cannot send '{emotion}' — not connected")
        return False

    # ── Emotion hook ──────────────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        """Log the emotion and forward it to the Arduino."""
        tag = emotion_tag.strip().upper()
        logger.info(f"[Emotion] {tag}")
        self.send_robot_emotion(tag)

    # ── WebSocket event handlers ──────────────────────────────────────────────

    def _register_custom_event_handlers(self):
        self.server_connection.register_handler("chat_response",   self.on_chat_response)
        self.server_connection.register_handler("speech_response", self.on_speech_response)
        self.server_connection.register_handler("persona_update",  self.on_persona_update)
        logger.info("[Client] Event handlers registered")

    @staticmethod
    def _split_sentences(text: str) -> list:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in parts if len(s.strip()) > 2]

    def on_chat_response(self, data: dict):
        response_text = data.get("response", "")
        if not response_text:
            return

        match      = re.search(r"\[(.*?)\]", response_text)
        emotion    = match.group(1) if match else None
        clean_text = re.sub(r"\[.*?\]", "", response_text).strip()

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(response_text)

        tts = self.output_modules.get("edge_tts_output")
        if not tts:
            if emotion:
                self.on_emotion_detected(emotion)
            return

        sentences = self._split_sentences(clean_text)
        if not sentences:
            return

        # First sentence carries the emotion callback — fires when audio starts
        start_cb = (lambda e: lambda: self.on_emotion_detected(e))(emotion) if emotion else None
        tts.process_output_synced(sentences[0], start_callback=start_cb)

        # Remaining sentences queued individually — TTS plays them back-to-back
        for sentence in sentences[1:]:
            tts.process_output(sentence)

    def on_speech_response(self, data: dict):
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] '{transcription}'")
        if data.get("response"):
            self.on_chat_response(data)

    def on_persona_update(self, data: dict):
        """
        Called when the server assigns a new persona to this robot.
        Updates TTS voice config live — no restart needed.
        """
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] Switching to: '{persona_name}'")

        voice_config = data.get("voice_config", {})
        if voice_config:
            tts = self.output_modules.get("edge_tts_output")
            if tts and hasattr(tts, "update_voice_config"):
                tts.update_voice_config(voice_config)
                logger.info(f"[Persona] TTS voice updated: {voice_config}")

        capabilities = data.get("capabilities", {})
        if capabilities:
            active = [k for k, v in capabilities.items() if v]
            if active:
                logger.info(f"[Persona] Active capabilities: {', '.join(active)}")

        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(f"Persona updated to {persona_name}.")

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(
                f"[PERSONA] Switched to: {persona_name}"
            )

    # ── Module setup ──────────────────────────────────────────────────────────

    def setup_all_modules(self):

        # ── INPUT: Voice ──────────────────────────────────────────────────────
        if "speech" in self.config.get("modules", []):
            logger.info("[Setup] Voice input...")
            voice_config = self.config.get("voice_config", {
                "sample_rate": 48000, "channels": 1,
                "input_device_index": 11, "max_record_time": 30,
            })
            voice = VoiceInputModule("voice_input", voice_config)
            self.register_input_module(voice)
            voice.start()

        # ── OUTPUT: Console ───────────────────────────────────────────────────
        logger.info("[Setup] Console output...")
        console = ConsoleOutputModule("console_output", self.config.get("console_config", {}))
        self.register_output_module(console)
        console.start()

        # ── OUTPUT: TTS ───────────────────────────────────────────────────────
        logger.info("[Setup] Edge TTS...")
        edge_cfg = self.config.get("edge_tts_config", {
            "voice": "en-US-AriaNeural", "rate": "+0%",
            "pitch": "+0Hz", "remove_emotion_tags": True,
        })
        edge = EdgeTTSOutputModule("edge_tts_output", edge_cfg)
        if self.register_output_module(edge):
            edge.start()
        else:
            logger.warning("[Setup] Edge TTS failed — check gtts/ffmpeg")

        # ── OUTPUT: Arduino (TCP) ─────────────────────────────────────────────
        if self.config.get("features", {}).get("arduino_integration", True):
            logger.info("[Setup] Arduino TCP output...")
            arduino_cfg = self.config.get("arduino_output", {})

            self.arduino_module = ArduinoOutputModule("arduino_output", arduino_cfg)
            self.arduino_module.on_connected        = self._on_arduino_connected
            self.arduino_module.on_disconnected     = self._on_arduino_disconnected
            self.arduino_module.on_connection_error = self._on_arduino_error

            if not self.register_output_module(self.arduino_module):
                logger.warning("[Setup] Arduino failed to register")
                self.arduino_module = None

    # ── Startup info ──────────────────────────────────────────────────────────

    def print_startup_info(self):
        print("\n" + "=" * 60)
        print(f"  {self.config.get('robot_name', 'Robot')} — connecting to server")
        print("=" * 60)
        print(f"  Robot    : {self.config.get('robot_name', 'Unknown')}")
        print(f"  ID       : {self.config.get('client_id', 'Unknown')}")
        print(f"  Server   : {self.config.get('server_url', 'Unknown')}")
        print(f"  Modules  : {', '.join(self.config.get('modules', []))}")
        print()
        print("  Input modules :")
        for n in self.input_modules:   print(f"    {n}")
        print("  Output modules:")
        for n in self.output_modules:  print(f"    {n}")
        if self.arduino_module:
            host = self.arduino_module.config.get("host", "chatbox.local")
            port = self.arduino_module.config.get("port", 8888)
            print(f"  Arduino  : {host}:{port}")
        print("=" * 60 + "\n")


def main():
    try:
        client = SimpleConcurrentClient("client_config.json")
        client.print_startup_info()
        client.run()
        return 0
    except FileNotFoundError:
        print("Error: client_config.json not found")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
