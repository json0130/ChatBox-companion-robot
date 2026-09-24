# robot.py — Robot client (v6)
import argparse
import sys
import re
import subprocess
import threading
import time
import logging
from enum import Enum
from typing import Optional

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from InputModules.camera_input import CameraInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.face_tracking_output import FaceTrackingOutputModule
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
        self.server_connection.register_handler("chat_response",   self.on_chat_response)
        self.server_connection.register_handler("speech_response", self.on_speech_response)
        self.server_connection.register_handler("persona_update",  self.on_persona_update)
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

        # A response arrived: start turning toward the person straight away, but
        # keep tracking while TTS synthesises. Freezing here instead would leave
        # the head locked and silent for however long synthesis takes — about two
        # seconds on gTTS's network round trip — which reads as a crash rather
        # than as a robot about to speak.
        tracker = self.output_modules.get("face_tracking_output")
        if tracker:
            tracker.request_center()

        # Set once the hold has actually been taken, so the finally below cannot
        # release a hold this call never acquired — the callback runs on the TTS
        # worker thread and may never fire if synthesis fails outright.
        hold_taken = threading.Event()

        def _on_audio_start():
            """Runs on the TTS worker thread immediately before playback starts.

            Every TTS path (gTTS, piper warm, piper one-shot) invokes this right
            before handing the file to aplay, so blocking here holds back the
            audio itself: nothing is spoken until the subject is centred and the
            head is frozen. Centring that fails or times out does not gag the
            robot — talking slightly off-centre beats not talking at all.
            """
            if tracker:
                if not tracker.wait_until_centered():
                    logger.info("[FaceTracking] not centred — speaking anyway")
                tracker.request_hold()
                hold_taken.set()
                if not tracker.wait_until_held(timeout=2.0):
                    logger.warning("[FaceTracking] hold not confirmed — speaking anyway")
                self._settle_before_speaking(tracker)
            if emotion:
                self.on_emotion_detected(emotion)

        try:
            # First sentence carries the callback — it gates audio on centring and
            # fires the gesture at the moment the robot actually starts talking.
            tts.process_output_synced(sentences[0], start_callback=_on_audio_start)

            # Remaining sentences queued individually — TTS plays them back-to-back
            for sentence in sentences[1:]:
                tts.process_output(sentence)

            self._wait_for_speech_end()
        finally:
            if tracker and hold_taken.is_set():
                tracker.release_hold()

    @staticmethod
    def _settle_before_speaking(tracker) -> None:
        """Hold a beat between finishing the turn and opening its mouth.

        Measured from when centring completed rather than added after it, so a
        slow synthesiser absorbs the pause instead of stacking with it: gTTS
        already spends ~2s making the audio, and nobody wants that to become
        three. The head is already frozen by the time we get here, so the pause
        looks like the robot regarding you before it speaks.
        """
        delay = getattr(tracker, "speak_delay", 0.0)
        if delay <= 0:
            return
        centered_at = getattr(tracker, "centered_at", None)
        remaining = delay - (time.time() - centered_at) if centered_at else delay
        if remaining > 0:
            time.sleep(remaining)

    def _wait_for_speech_end(self, settle: float = 0.8, timeout: float = 120.0) -> bool:
        """Block until TTS has been quiet for `settle` seconds.

        is_speaking clears briefly *between* queued sentences, so a plain
        'not set' check would resume tracking mid-reply — hence the settle
        window. Returns False if `timeout` was hit first.
        """
        if not hasattr(self, "is_speaking"):
            return True
        self.is_speaking.wait(timeout=3.0)      # let the first audio start
        deadline = time.time() + timeout
        quiet_since = None
        while time.time() < deadline:
            if self.is_speaking.is_set():
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= settle:
                return True
            time.sleep(0.05)
        logger.warning("[FaceTracking] speech wait timed out — resuming tracking")
        return False

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

        # The announcement is speech too, so it gets the same treatment as a
        # reply: centre while it synthesises, freeze the instant audio starts,
        # resume when it finishes. Otherwise the head would pan mid-sentence.
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tracker = self.output_modules.get("face_tracking_output")
            if tracker:
                tracker.request_center()

            hold_taken = threading.Event()

            def _on_audio_start():
                if tracker:
                    tracker.wait_until_centered()
                    tracker.request_hold()
                    hold_taken.set()
                    tracker.wait_until_held(timeout=2.0)
                    self._settle_before_speaking(tracker)

            try:
                tts.process_output_synced(f"Persona updated to {persona_name}.",
                                          start_callback=_on_audio_start)
                if tracker:
                    self._wait_for_speech_end()
            finally:
                if tracker and hold_taken.is_set():
                    tracker.release_hold()

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(
                f"[PERSONA] Switched to: {persona_name}"
            )

    # ── Module setup ──────────────────────────────────────────────────────────

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
        # Find the USB speaker's ALSA card before the TTS module reads the config
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

        # ── OUTPUT: Face tracking (state machine, ticked by the main loop) ────
        if self.config.get("features", {}).get("face_tracking", False):
            logger.info("[Setup] Face tracking...")
            ft_cfg = self.config.get("face_tracking_config", {})
            tracker = FaceTrackingOutputModule("face_tracking_output", ft_cfg)
            if self.register_output_module(tracker):
                tracker.start()
            else:
                logger.warning("[Setup] Face tracking failed — check camera/serial")

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
        client.run()  # prints startup info once server + ESP32 connections settle
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
