# robot.py — Robot client (v6)
import argparse
import sys
import re
import logging
from enum import Enum
from typing import Optional

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from InputModules.camera_input import CameraInputModule
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
    "HANDS_CLAP": "hands_clap",
    "EARS_WIGGLE": "ears_wiggle",
}


# ── Pepeha (te reo Māori introduction) ────────────────────────────────────────
# Asking for an intro puts the robot in AWAITING_CONSENT: it offers to introduce
# itself in te reo Māori and waits for the next utterance to be yes/no.

class PepehaState(Enum):
    IDLE             = "idle"
    AWAITING_CONSENT = "awaiting_consent"


INTRO_PATTERN = re.compile(
    r'\b(introduce\s+(yourself|you)|introduction|intro\b|who\s+are\s+you|'
    r'tell\s+me\s+about\s+yourself|what(?:\'s| is)\s+your\s+name|'
    r'pepeha|mihi\b|k[oō]rero)',
    re.IGNORECASE
)
YES_PATTERN = re.compile(
    r'\b(yes|yeah|yep|sure|[aā]e|okay|ok|please|go ahead|absolutely|of course)\b',
    re.IGNORECASE
)
NO_PATTERN = re.compile(
    r'\b(no|nah|nope|don\'?t|k[aā]o|english|skip|maybe later)\b',
    re.IGNORECASE
)

CONSENT_QUESTION = "Would it be okay if I introduced myself in te reo Māori?"

# Spelled phonetically — the English TTS voice mispronounces the macronised
# spelling, and "Chat Box" as two words stops it reading the name as letters.
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
    Robot client (v6) — verbal interaction + Arduino gestures.

    • Parses [EMOTION] tags from chat_response and forwards to Arduino.
    • Handles persona_update event from server → updates TTS voice config live.
    • With --pepeha, intercepts intro requests and offers the Māori pepeha.
    """

    def __init__(self, config_file: str = "client_config.json",
                 pepeha: bool = False):
        super().__init__(config_file)
        self.arduino_module: Optional[ArduinoOutputModule] = None

        # Off unless --pepeha was passed. Nothing else can turn it on.
        self.pepeha_enabled = bool(pepeha)
        self._pepeha_state = PepehaState.IDLE
        logger.info(f"[Pepeha] Māori intro {'enabled' if self.pepeha_enabled else 'disabled'}")

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

    # ── Pepeha pipeline ───────────────────────────────────────────────────────

    def _build_english_intro_context(self) -> str:
        name = self.config.get("robot_name", "ChatBox")
        role = self.config.get("robot_role", "a friendly companion robot")
        return (
            f"Introduce yourself. Your name is {name}. {role} "
            f"Give a warm, natural self-introduction in 1-2 sentences."
        )

    def _trigger_pepeha_pipeline(self):
        """Ask for consent before switching into te reo Māori."""
        self._pepeha_state = PepehaState.AWAITING_CONSENT
        logger.info("[Pepeha] Awaiting consent")
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(CONSENT_QUESTION)
        self.on_emotion_detected("GREETING")

    def _handle_pepeha_consent(self, transcription: str):
        tts = self.output_modules.get("edge_tts_output")

        # "no" is checked first — a decline often carries a polite "please"/"ok"
        # that would otherwise match YES_PATTERN ("no thanks, English please").
        if NO_PATTERN.search(transcription):
            self._pepeha_state = PepehaState.IDLE
            logger.info("[Pepeha] Declined — English intro via LLM")
            self.on_emotion_detected("GREETING")
            self.send_to_server("chat", self._build_english_intro_context())
        elif YES_PATTERN.search(transcription):
            self._pepeha_state = PepehaState.IDLE
            logger.info("[Pepeha] Delivering pepeha")
            self.on_emotion_detected("GREETING")
            if tts:
                for line in PEPEHA_LINES:
                    tts.process_output(line)
        else:
            logger.info(f"[Pepeha] Unclear consent ('{transcription}') — re-asking")
            if tts:
                tts.process_output(
                    "Sorry, I didn't catch that. "
                    "Would you like me to introduce myself in Māori?"
                )

    # ── Chat / speech handlers ────────────────────────────────────────────────

    def on_chat_response(self, data: dict):
        # While waiting on consent, drop server chatter so it can't talk over
        # the consent question or the pepeha.
        if self._pepeha_state != PepehaState.IDLE:
            logger.debug("[Pepeha] Suppressed chat_response while awaiting consent")
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

        if self._pepeha_state == PepehaState.AWAITING_CONSENT:
            self._handle_pepeha_consent(transcription)
            return

        if self.pepeha_enabled and transcription and INTRO_PATTERN.search(transcription):
            self._trigger_pepeha_pipeline()
            return

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

        # ── INPUT: Camera (emotion) ───────────────────────────────────────────
        # Frames go to the server's emotion processor; the detected emotion then
        # rides along with the next chat message as "text (happy-0.7)".
        if "emotion" in self.config.get("modules", []):
            logger.info("[Setup] Camera input (emotion)...")
            camera_config = self.config.get("camera_config", {
                "camera_index": 0, "width": 640, "height": 480,
                "fps": 30, "send_fps": 5, "jpeg_quality": 85,
            })
            camera = CameraInputModule("camera_input", camera_config)
            if self.register_input_module(camera):
                camera.start()
            else:
                logger.warning("[Setup] Camera failed to register — emotion disabled")

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
        print(f"  Pepeha   : {'on' if self.pepeha_enabled else 'off (use --pepeha)'}")
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ChatBox robot client")
    parser.add_argument(
        "--pepeha", action="store_true",
        help="offer the te reo Māori pepeha when asked for an introduction",
    )
    parser.add_argument(
        "--config", default="client_config.json",
        help="path to the client config (default: client_config.json)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        client = SimpleConcurrentClient(args.config, pepeha=args.pepeha)
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
