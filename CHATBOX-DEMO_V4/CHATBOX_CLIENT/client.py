"""
client.py — Base Client (v7.0.0)
=================================
Key change from v6.x:
  The robot now connects OUT to the central server as a Socket.IO client.
  On connect it emits 'client_init' with its config so the server
  initialises immediately — no manual "Connect" step needed.
  Auto-reconnects with exponential backoff on any drop.
"""

import json
import re
import time
import threading
import requests
import base64
import logging
import socketio
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Abstract module base classes (unchanged) ──────────────────────────────────

class BaseModule(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = False
        self.client = None

    @abstractmethod
    def initialize(self) -> bool: ...

    @abstractmethod
    def start(self) -> bool: ...

    @abstractmethod
    def stop(self): ...

    def set_client(self, client):
        self.client = client


class InputModule(BaseModule):
    @abstractmethod
    def get_data(self) -> Optional[Any]: ...


class OutputModule(BaseModule):
    @abstractmethod
    def process_output(self, data: Any) -> bool: ...


# ── ServerConnection — Socket.IO CLIENT (robot dials out to server) ───────────

class ServerConnection:
    """
    Connects to the central server as a Socket.IO client.
    On connect, emits 'client_init' so the server initialises immediately.
    Auto-reconnects — no manual intervention needed after startup.

    Public interface is unchanged from v6 so robot.py needs no edits.
    """

    def __init__(self, server_url: str, client_config: Dict):
        self.server_url = server_url.rstrip("/")
        self.client_config = client_config
        self.client_id = client_config.get("client_id", "robot_001")

        self._sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,       # retry forever
            reconnection_delay=2,
            reconnection_delay_max=30,
            logger=False,
            engineio_logger=False,
        )
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._handlers: Dict[str, callable] = {}

        # Core lifecycle
        self._sio.on("connect",              self._on_connect)
        self._sio.on("disconnect",           self._on_disconnect)
        self._sio.on("connect_error",        self._on_connect_error)
        self._sio.on("client_init_response", self._on_init_response)
        # persona_update needs internal handling + forwarding to robot.py
        self._sio.on("persona_update",       self._on_persona_update_event)

    # ── Lifecycle events ──────────────────────────────────────────────────────

    def _on_connect(self):
        logger.info("[WS] Connected to server — sending client_init")
        self._connected.set()
        self._sio.emit("client_init", {
            "robot_name":  self.client_config.get("robot_name"),
            "client_id":   self.client_id,
            "robot_role":  self.client_config.get("robot_role", ""),
            "allowed_tags": self.client_config.get("allowed_tags", ["[DEFAULT]"]),
            "modules":     self.client_config.get("modules", []),
            "voice_config": self.client_config.get("voice_config", {}),
        })

    def _on_disconnect(self):
        logger.warning("[WS] Disconnected from server")
        self._connected.clear()

    def _on_connect_error(self, data):
        logger.error(f"[WS] Connection error: {data}")

    def _on_init_response(self, data):
        if data.get("success"):
            logger.info(f"[WS] Server ready: {data.get('message', 'OK')}")
        else:
            logger.error(f"[WS] Server init failed: {data.get('message')}")

    def _on_persona_update_event(self, data):
        threading.Thread(
            target=self._apply_persona_update, args=(data,), daemon=True
        ).start()

    # ── Handler registration ──────────────────────────────────────────────────

    def register_handler(self, event: str, callback: callable):
        """
        Register a callback for a named event pushed by the server.
        For 'persona_update', the internal config-update logic runs first,
        then the registered callback is invoked.
        """
        with self._lock:
            self._handlers[event] = callback

        if event == "persona_update":
            # Handled via _on_persona_update_event → _apply_persona_update
            # which calls self._handlers['persona_update'] at the end
            return

        def _threaded(data):
            threading.Thread(target=callback, args=(data,), daemon=True).start()

        self._sio.on(event, _threaded)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_server(self):
        """Connect to the central server in a background thread."""
        def _run():
            while True:
                try:
                    logger.info(f"[WS] Connecting to {self.server_url} ...")
                    self._sio.connect(
                        self.server_url,
                        transports=["websocket", "polling"],
                    )
                    self._sio.wait()   # blocks while connected
                except socketio.exceptions.ConnectionError as e:
                    logger.warning(f"[WS] Server unreachable: {e} — retrying in 5s")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"[WS] Unexpected error: {e} — retrying in 5s")
                    time.sleep(5)

        threading.Thread(target=_run, daemon=True, name="ws-client").start()

    # ── Sending ───────────────────────────────────────────────────────────────

    def send(self, message: dict) -> bool:
        """Send a message to the server. Thread-safe."""
        if not self.is_connected():
            logger.warning("[WS] Cannot send — not connected")
            return False
        try:
            # Map internal type strings to the server's Socket.IO event names
            _event_map = {
                "chat":        "chat_message",
                "speech":      "speech",
                "image_frame": "image_frame",
                "ack":         "ack",
            }
            msg_type = message.get("type", "")
            event = _event_map.get(msg_type, msg_type)
            self._sio.emit(event, message)
            return True
        except Exception as e:
            logger.error(f"[WS] Send error: {e}")
            return False

    # ── Status ────────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def wait_for_server(self, timeout: float = None) -> bool:
        """Block until the server is connected (or timeout). Returns True if connected."""
        return self._connected.wait(timeout=timeout)

    # ── Persona update ────────────────────────────────────────────────────────

    def _apply_persona_update(self, data: dict):
        """
        Handle persona_update pushed by the central server.
        1. Updates in-memory config immediately
        2. Writes updated client_config.json to disk
        3. Calls any registered persona_update handler (robot.py updates TTS)
        """
        import json as _json
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] Applying persona: '{persona_name}'")

        updatable = {
            "robot_role": "robot_role",
            "allowed_tags": "allowed_tags",
            "modules": "modules",
            "voice_config": "voice_config",
            "capabilities": "capabilities",
            "personality": "personality",
        }
        changed = {cfg_key: data[ws_key]
                   for ws_key, cfg_key in updatable.items() if ws_key in data}
        if not changed:
            return

        self.client_config.update(changed)

        config_path = self.client_config.get("_config_file", "client_config.json")
        try:
            with open(config_path, "r") as f:
                on_disk = _json.load(f)
            on_disk.update(changed)
            with open(config_path, "w") as f:
                _json.dump(on_disk, f, indent=4)
            logger.info(f"[Persona] Saved to {config_path}")
        except Exception as e:
            logger.error(f"[Persona] Failed to write config: {e}")

        with self._lock:
            handler = self._handlers.get("persona_update")
        if handler:
            threading.Thread(target=handler, args=(data,), daemon=True).start()


# ── BasicClient — identical public interface to v6.x ─────────────────────────

class BasicClient:
    """
    Main client class. Manages modules and server communication.
    Public interface unchanged from v6.x.
    """

    def __init__(self, config_file: str = "client_config.json"):
        self.config = self._load_config(config_file)
        if not self.config:
            raise RuntimeError(f"Failed to load config from {config_file}")

        server_url = self.config.get("server_url", "http://localhost:5000")

        self.server_connection = ServerConnection(server_url, self.config)

        self.server_connection.register_handler("chat_response",   self._default_chat_handler)
        self.server_connection.register_handler("speech_response", self._default_speech_handler)
        self.server_connection.register_handler("emotion_update",  self._default_emotion_handler)
        self.server_connection.register_handler("frame_result",    self._default_frame_result_handler)
        self.server_connection.register_handler("demo_step",       self._on_demo_step)

        # Last (emotion, confidence) printed — frame_result arrives at send_fps,
        # so only log when it actually changes to keep the terminal readable.
        self._last_emotion_print = None

        self.input_modules:  Dict[str, InputModule]  = {}
        self.output_modules: Dict[str, OutputModule] = {}

        self.running = False
        self.is_speaking       = threading.Event()
        self.tts_started_event = threading.Event()

        logger.info(f"[Client] {self.config.get('robot_name', 'Robot')} initialising")
        logger.info(f"         ID      : {self.config.get('client_id')}")
        logger.info(f"         Server  : {server_url}")
        logger.info(f"         Modules : {', '.join(self.config.get('modules', []))}")

    # ── Default server event handlers ─────────────────────────────────────────

    def _default_chat_handler(self, data: dict):
        self.process_server_response(data, "chat")

    def _default_speech_handler(self, data: dict):
        self.process_server_response(data, "speech")

    def _default_emotion_handler(self, data: dict):
        pass

    def _default_frame_result_handler(self, data: dict):
        """Print the server's emotion-recognition result for each camera frame.

        Payload: {'result': {'emotion', 'confidence', 'status', 'distribution'}}.
        Only logged when the emotion or confidence actually changes, since frames
        arrive continuously at `send_fps`.
        """
        result = (data or {}).get("result", {})
        if not result:
            return

        emotion = result.get("emotion", "unknown")
        conf    = result.get("confidence", 0.0) or 0.0
        status  = result.get("status", "")

        # 'no_faces' means nobody is in frame — say so once, don't repeat.
        key = (emotion, round(float(conf)), status == "no_faces")
        if key == self._last_emotion_print:
            return
        self._last_emotion_print = key

        if status == "no_faces":
            logger.info("[Emotion] no face in frame")
        else:
            logger.info(f"[Emotion] {emotion} ({float(conf):.1f}%)")

    def _on_demo_step(self, data: dict):
        step_id  = data.get("step_id", "")
        text     = data.get("text", "")
        need_ack = data.get("require_ack", True)

        if not text:
            if need_ack:
                self.send_ack(step_id)
            return

        logger.info(f"[Demo] Step '{step_id}': {text[:60]}{'...' if len(text) > 60 else ''}")

        match = re.search(r"\[(.*?)\]", text)
        if match:
            self.on_emotion_detected(match.group(1))

        tts_done = threading.Event()

        def _on_tts_done():
            logger.info(f"[Demo] TTS completed for '{step_id}'")
            tts_done.set()

        spoken = False
        for name, module in self.output_modules.items():
            try:
                if hasattr(module, "speak_with_callback"):
                    module.speak_with_callback(
                        text,
                        callback=_on_tts_done if need_ack else None,
                    )
                    spoken = True
                    break
            except Exception as e:
                logger.error(f"[Demo] TTS module '{name}' error: {e}")

        if not need_ack:
            return

        if not spoken:
            logger.warning(f"[Demo] No TTS module for step '{step_id}' — ACK now")
            self.send_ack(step_id)
            return

        completed = tts_done.wait(timeout=120)
        if not completed:
            logger.warning(f"[Demo] TTS wait timed out for '{step_id}' — ACK anyway")

        self.send_ack(step_id)

    def send_ack(self, step_id: str):
        sent = self.server_connection.send({"type": "ack", "step_id": step_id})
        if sent:
            logger.info(f"[Demo] ACK sent for step '{step_id}'.")
        else:
            logger.warning(f"[Demo] Could not send ACK for '{step_id}' — not connected.")

    # ── Module registration ───────────────────────────────────────────────────

    def register_input_module(self, module: InputModule) -> bool:
        try:
            module.set_client(self)
            if module.initialize():
                self.input_modules[module.name] = module
                logger.info(f"[Modules] Input  '{module.name}' registered")
                return True
            logger.error(f"[Modules] Input  '{module.name}' failed to initialize")
            return False
        except Exception as e:
            logger.error(f"[Modules] Input  '{module.name}' error: {e}")
            return False

    def register_output_module(self, module: OutputModule) -> bool:
        try:
            module.set_client(self)
            if module.initialize():
                self.output_modules[module.name] = module
                logger.info(f"[Modules] Output '{module.name}' registered")
                return True
            logger.error(f"[Modules] Output '{module.name}' failed to initialize")
            return False
        except Exception as e:
            logger.error(f"[Modules] Output '{module.name}' error: {e}")
            return False

    # ── Sending data to server ────────────────────────────────────────────────

    def send_to_server(self, data_type: str, data: Any) -> Optional[dict]:
        if data_type == "chat":
            self.server_connection.send({
                "type": "chat",
                "message": data,
            })
        elif data_type == "speech":
            audio_b64 = base64.b64encode(data).decode("utf-8")
            self.server_connection.send({
                "type": "speech",
                "audio": audio_b64,
            })
        elif data_type == "frame":
            if isinstance(data, dict):
                frame_b64 = data.get("color", data.get("frame", ""))
            else:
                frame_b64 = data
            self.server_connection.send({
                "type": "image_frame",
                "frame": frame_b64,
            })
        else:
            logger.warning(f"[Client] Unknown data_type: {data_type}")
        return None

    # ── Processing server responses ───────────────────────────────────────────

    def process_server_response(self, response_data: Optional[dict], response_type: str = "chat"):
        if not response_data:
            return

        response_text = response_data.get("response", "")
        if not response_text:
            transcription = response_data.get("transcription")
            if transcription:
                logger.info(f"[STT] Transcribed: '{transcription}'")
            return

        logger.info(f"[Server] {response_text[:80]}{'...' if len(response_text) > 80 else ''}")

        match = re.search(r"\[(.*?)\]", response_text)
        if match:
            self.on_emotion_detected(match.group(1))

        self.tts_started_event.clear()

        for name, module in self.output_modules.items():
            try:
                module.process_output({
                    "text": response_text,
                    "type": response_type,
                    "full_response": response_data,
                })
            except Exception as e:
                logger.error(f"[Modules] Output '{name}' error: {e}")

        if response_type == "speech":
            transcription = response_data.get("transcription", "")
            if transcription:
                logger.info(f"[STT] Transcribed: '{transcription}'")

    def on_emotion_detected(self, emotion_tag: str):
        pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start the client:
          1. Connect to the server (sends client_init automatically on connect)
          2. Start all registered modules
        """
        try:
            self.server_connection.start_server()
            self.running = True

            for name, module in self.input_modules.items():
                try:
                    module.start()
                    logger.info(f"[Modules] Started input '{name}'")
                except Exception as e:
                    logger.error(f"[Modules] Error starting input '{name}': {e}")

            for name, module in self.output_modules.items():
                try:
                    module.start()
                    logger.info(f"[Modules] Started output '{name}'")
                except Exception as e:
                    logger.error(f"[Modules] Error starting output '{name}': {e}")

            return True

        except Exception as e:
            logger.error(f"[Client] Start error: {e}")
            return False

    def stop(self):
        logger.info("[Client] Stopping...")
        self.running = False
        for module in list(self.input_modules.values()) + list(self.output_modules.values()):
            try:
                module.stop()
            except Exception as e:
                logger.error(f"[Client] Error stopping '{module.name}': {e}")
        logger.info("[Client] Stopped.")

    def run(self):
        """Start and block until Ctrl+C."""
        try:
            if not self.start():
                return
            logger.info("[Client] Running — press Ctrl+C to stop")

            # Modules exposing tick() run as state machines driven from here
            # (rather than their own threads), so they can be gated cleanly
            # against speech/gestures. Idle at 1 Hz when there are none.
            tickables = [m for m in list(self.input_modules.values()) +
                         list(self.output_modules.values())
                         if callable(getattr(m, "tick", None))]
            interval = 0.03 if tickables else 1.0
            if tickables:
                logger.info(f"[Client] Ticking {len(tickables)} module(s) "
                            f"at ~{1/interval:.0f} Hz")

            while self.running:
                for module in tickables:
                    try:
                        module.tick()
                    except Exception as e:
                        logger.error(f"[Modules] tick '{module.name}' error: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("[Client] Ctrl+C received")
        except Exception as e:
            logger.error(f"[Client] Runtime error: {e}")
        finally:
            self.stop()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_config(self, config_file: str) -> Optional[dict]:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            config["_config_file"] = config_file
            logger.info(f"[Client] Config loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"[Client] Config file not found: {config_file}")
            return None
        except Exception as e:
            logger.error(f"[Client] Config load error: {e}")
            return None
