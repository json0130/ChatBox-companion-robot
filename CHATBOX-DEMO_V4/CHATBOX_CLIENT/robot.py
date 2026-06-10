# robot.py — Robot client (sign-language-demo branch)
import sys
import re
import logging
import subprocess
import time
from enum import Enum
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
    "GREETING":    "greeting",   "WAVE":    "wave",    "POINT":    "point",
    "CONFUSED":    "confused",   "SHRUG":   "shrug",   "ANGRY":    "angry",
    "SAD":         "sad",        "SLEEP":   "sleep",   "DEFAULT":  "default",
    "POSE":        "pose",       "HAPPY":   "greeting","FEAR":     "sad",
    "SURPRISE":    "confused",   "NEUTRAL": "default",
    "HANDS_CLAP":  "hands_clap",
    "EARS_WIGGLE": "ears_wiggle",
}

# ── Pepeha pipeline ───────────────────────────────────────────────────────────

class PepehaState(Enum):
    IDLE             = "idle"
    AWAITING_CONSENT = "awaiting_consent"

INTRO_PATTERN = re.compile(
    r'\b(introduce\s+yourself|who are you|tell me about yourself|what is your name|k[oō]rero)',
    re.IGNORECASE
)
YES_PATTERN = re.compile(
    r'\b(yes|yeah|sure|[aā]e|okay|ok|please|go ahead|absolutely|of course)\b',
    re.IGNORECASE
)
NO_PATTERN = re.compile(
    r'\b(no|nah|nope|k[aā]o|english|skip|just english)\b',
    re.IGNORECASE
)

CONSENT_QUESTION = "Would it be okay if I introduced myself in te reo Māori?"

PEPEHA_LINES = [
    "Tena kotoh, kahtoa.",
    "Ko Rangitoto te maunga.",
    "Ko Waitemata te moana.",
    "Ko Tamaki Makoh-ro, toku ka-inga.",
    "Ko Chat Box, toku ingoah.",
    "Tena kotoh, tena kotoh, tena kotoh kahtoa.",
]


class SimpleConcurrentClient(BasicClient):
    """
    Robot client — verbal interaction + Arduino gestures + Māori pepeha pipeline.
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        self.arduino_module: Optional[ArduinoOutputModule] = None
        self._pepeha_state = PepehaState.IDLE
        self.setup_all_modules()
        self._register_custom_event_handlers()

    # ── Arduino helpers ───────────────────────────────────────────────────────

    def _on_arduino_connected(self):
        logger.info("[Arduino] Connected")
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output("Arduino connected.")

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
        self.server_connection.register_handler("chat_response",        self.on_chat_response)
        self.server_connection.register_handler("speech_response",      self.on_speech_response)
        self.server_connection.register_handler("persona_update",       self.on_persona_update)
        self.server_connection.register_handler("client_init_response", self._on_server_ready)
        logger.info("[Client] Event handlers registered")

    def _on_server_ready(self, data: dict):
        """Fires when the server confirms client_init succeeded. Replaces the
        internal BasicClient handler (which only logs). Plays startup greeting."""
        if data.get("success"):
            logger.info(f"[WS] Server ready: {data.get('message', 'OK')}")
            tts = self.output_modules.get("edge_tts_output")
            if tts:
                tts.process_output("Hello! I'm online and ready to chat.")
            self.on_emotion_detected("GREETING")
        else:
            logger.error(f"[WS] Server init failed: {data.get('message')}")

    @staticmethod
    def _split_sentences(text: str) -> list:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in parts if len(s.strip()) > 2]

    # ── Pepeha pipeline ───────────────────────────────────────────────────────

    def _build_english_intro_context(self) -> str:
        name = self.config.get('robot_name', 'ChatBox')
        role = self.config.get('robot_role', 'a friendly companion robot')
        return (
            f"Introduce yourself. Your name is {name}. {role} "
            f"Give a warm, natural self-introduction in 1-2 sentences."
        )

    def _trigger_pepeha_pipeline(self):
        self._pepeha_state = PepehaState.AWAITING_CONSENT
        logger.info("[Pepeha] Awaiting consent")
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(CONSENT_QUESTION)
        self.on_emotion_detected("GREETING")

    def _handle_pepeha_consent(self, transcription: str):
        tts = self.output_modules.get("edge_tts_output")
        if YES_PATTERN.search(transcription):
            self._pepeha_state = PepehaState.IDLE
            logger.info("[Pepeha] Delivering pepeha")
            self.on_emotion_detected("GREETING")
            if tts:
                for line in PEPEHA_LINES:
                    tts.process_output(line)
        elif NO_PATTERN.search(transcription):
            self._pepeha_state = PepehaState.IDLE
            logger.info("[Pepeha] English intro via LLM")
            self.on_emotion_detected("GREETING")
            self.send_to_server('chat', self._build_english_intro_context())
        else:
            logger.info("[Pepeha] Unclear consent response — re-asking")
            if tts:
                tts.process_output("Sorry, I didn't catch that. Would you like me to introduce myself in Māori?")

    # ── Chat / speech handlers ────────────────────────────────────────────────

    def on_chat_response(self, data: dict):
        if self._pepeha_state != PepehaState.IDLE:
            return
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

        start_cb = (lambda e: lambda: self.on_emotion_detected(e))(emotion) if emotion else None
        tts.process_output_synced(sentences[0], start_callback=start_cb)

        for sentence in sentences[1:]:
            tts.process_output(sentence)

    def on_speech_response(self, data: dict):
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] '{transcription}'")

        if self._pepeha_state == PepehaState.AWAITING_CONSENT:
            self._handle_pepeha_consent(transcription)
            return

        if transcription and INTRO_PATTERN.search(transcription):
            self._trigger_pepeha_pipeline()
            return

        if data.get("response"):
            self.on_chat_response(data)

    def on_persona_update(self, data: dict):
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
            self.output_modules["console_output"].process_output(f"[PERSONA] Switched to: {persona_name}")


    # ── Auto-Speaker Search & Setup ───────────────────────────────────────────

    def _configure_usb_speaker(self):
        """Dynamically find the USB speaker's ALSA card ID and inject it into config."""
        try:
            output = subprocess.check_output(['aplay', '-l'], text=True)
            for line in output.split('\n'):
                # Look for the specific USB Audio tags seen in your hardware
                if line.startswith("card") and ("UACDemoV10" in line or "USB Audio" in line):
                    match = re.search(r"card (\d+):", line)
                    if match:
                        card_num = match.group(1)
                        logger.info(f"[Setup] Auto-detected USB Speaker on card {card_num}")
                        
                        if "edge_tts_config" not in self.config:
                            self.config["edge_tts_config"] = {}
                            
                        # Override whatever is in client_config.json with the live hardware ID
                        self.config["edge_tts_config"]["audio_cmd"] = ["aplay", "-D", f"plughw:{card_num},0"]
                        return
                        
            logger.warning("[Setup] USB Speaker not found. Falling back to default audio configuration.")
        except Exception as e:
            logger.error(f"[Setup] Error searching for USB speaker: {e}")

    def setup_all_modules(self):
        # 1. Search for and map the USB speaker dynamically first
        self._configure_usb_speaker()

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

        # ── OUTPUT: TTS (Now using dynamic audio_cmd if found) ────────────────
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

    # ── Startup Lifecycle & Info ──────────────────────────────────────────────

    def print_startup_info(self):
        print("\n" + "=" * 60)
        print(f"  {self.config.get('robot_name', 'Robot')} — Online and Connected")
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
            host = self.arduino_module.config.get("host", "Unknown")
            port = self.arduino_module.config.get("port", 8888)
            print(f"  ESP32/Arduino : {host}:{port}")
        
        # Optionally show dynamically found audio config
        tts_cfg = self.config.get("edge_tts_config", {})
        if "audio_cmd" in tts_cfg:
            print(f"  Audio Output  : {' '.join(tts_cfg['audio_cmd'])}")
        print("=" * 60 + "\n")

    def run(self):
        """
        Override default run() to start threads, wait for hardware/server 
        connections to lock in, and THEN print out the final setup state.
        """
        try:
            # 1. Start all modules and socket.io background threads
            if not self.start():
                return
            
            # 2. Block and wait for central server to connect (Timeout 15s)
            logger.info("[Startup] Waiting for Central Server connection...")
            if self.server_connection.wait_for_server(timeout=15):
                logger.info("[Startup] Server connection verified.")
            else:
                logger.warning("[Startup] Server connection timed out. (Will retry in background)")

            # 3. Block and wait for ESP32 connection
            if self.arduino_module:
                logger.info("[Startup] Waiting for ESP32 Robot connection...")
                for _ in range(15):  # Wait up to 15 seconds
                    if self.arduino_module.is_connected():
                        break
                    time.sleep(1)
                else:
                    logger.warning("[Startup] ESP32 connection timed out. (Will retry in background)")

            # 4. Now that we verified connections, print the splash screen
            self.print_startup_info()

            # 5. Keep alive main thread
            logger.info("[Client] Running — press Ctrl+C to stop")
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("[Client] Ctrl+C received")
        except Exception as e:
            logger.error(f"[Client] Runtime error: {e}", exc_info=True)
        finally:
            self.stop()


def main():
    try:
        client = SimpleConcurrentClient("client_config.json")
        # Notice print_startup_info() was removed from here; 
        # it is now safely nested inside client.run() after connections establish.
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
