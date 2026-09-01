"""
robot_link — the CHATBOX CLIENT's Socket.IO endpoint, bolted onto the webui.

The Jetson client (CHATBOX-DEMO_V4/CHATBOX_CLIENT/client.py on main) dials OUT to
a server and speaks Socket.IO, emitting `client_init` the moment it connects. This
module answers on exactly the events that client already sends, so an unmodified
client works against the V5 pipeline — only `server_url` in client_config.json has
to point here.

    client → server    client_init, chat_message, speech
    server → client    client_init_response, chat_response, speech_response

Mic audio arrives as base64 WAV on `speech` (VoiceInputModule records it at 16 kHz
mono after its own VAD) and is transcribed HERE with faster-whisper, the same
placement the v4 server used in Modules/speech_processor.py. The transcript is
then pushed into the SAME queue the web page's text box feeds, so a spoken turn
and a typed one take an identical path through face-reco, PAD, the KG and the
prompt builder — there is no second, divergent chat path to keep in step.

The reply goes back in main's wire format: a leading '[TAG] ' on the response
text, which is where the v4 client already looks — its TTS strips that span
before speaking and its Arduino output reads it for the gesture. The servo STYLE
line does NOT go out; this server drives the servos over its own ESP32 socket and
the client has nothing to do with it.

Note on identity: face recognition runs on the SERVER's camera, so a spoken turn
is attributed to whoever this machine can see — not to the client that sent it.
That is the same rule the typed path follows.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import threading
import time
import wave
from typing import Callable, Optional

# Matches the v4 client's recorder: 16 kHz mono int16 WAV.
MAX_AUDIO_SECONDS = 30.0
MIN_AUDIO_SECONDS = 0.1
DEFAULT_PORT = 5000


class SpeechToText:
    """faster-whisper wrapped so the caller can't be blocked by model load.

    `start()` loads in the background: the webui must keep serving frames while a
    multi-hundred-MB model comes off disk, and the first utterance usually arrives
    long after startup anyway. `transcribe_b64` reports not-ready rather than
    queueing, so a client that speaks too early gets an answer instead of a hang.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 language: Optional[str] = None):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._ready = threading.Event()
        self._failed = False
        # faster-whisper is not documented as thread-safe and the webui can have
        # several clients; one transcription at a time.
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._load, daemon=True,
                         name="whisper-load").start()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[STT] faster-whisper not installed — speech disabled "
                  "(pip install faster-whisper)", flush=True)
            self._failed = True
            return

        # int8 on CPU is what the v4 server settled on; float16 only makes sense
        # once the model is actually on the GPU.
        attempts = ([("cuda", "float16"), ("cpu", "int8")]
                    if self.device in ("auto", "cuda") else [("cpu", "int8")])
        for dev, compute in attempts:
            try:
                print(f"[STT] loading faster-whisper '{self.model_size}' "
                      f"on {dev}/{compute} …", flush=True)
                self.model = WhisperModel(self.model_size, device=dev,
                                          compute_type=compute)
                print(f"[STT] ready — {self.model_size} on {dev}", flush=True)
                self._ready.set()
                return
            except Exception as e:                       # noqa: BLE001
                print(f"[STT] {dev} load failed: {e}", flush=True)
        self._failed = True
        print("[STT] no usable backend — speech disabled", flush=True)

    @property
    def available(self) -> bool:
        return self._ready.is_set()

    @property
    def failed(self) -> bool:
        return self._failed

    def transcribe_b64(self, audio_b64: str) -> tuple[bool, str]:
        """(ok, text-or-reason) for one base64 WAV payload."""
        if not self.available:
            return False, ("speech-to-text is still loading"
                           if not self._failed else "speech-to-text unavailable")
        try:
            audio = base64.b64decode(audio_b64)
        except Exception as e:                           # noqa: BLE001
            return False, f"could not decode audio: {e}"

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                fh.write(audio)
                path = fh.name

            ok, why = _check_wav(path)
            if not ok:
                return False, why

            with self._lock:
                segments, info = self.model.transcribe(
                    path,
                    language=self.language,
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    # The client's VAD already decided this is speech, but it cuts
                    # on volume alone — a cough clears its threshold. Whisper's VAD
                    # drops those before they can be hallucinated into words.
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                text = " ".join(s.text.strip() for s in segments).strip()

            if len(text) < 2:
                return False, "no speech detected"
            print(f"[STT] '{text}'  ({info.language} "
                  f"{info.language_probability:.2f})", flush=True)
            return True, text
        except Exception as e:                           # noqa: BLE001
            return False, f"transcription failed: {e}"
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


class _HijackedSocketFilter(logging.Filter):
    """Drop werkzeug's traceback for every closed websocket.

    A websocket handler hijacks the socket, so werkzeug's run_wsgi never sees a
    start_response and asserts on teardown — once per disconnect, with a full
    traceback. The client reconnects forever by design, so left alone this buries
    the --trace-prompt output it shares a terminal with. Nothing is wrong and
    nothing is lost; only this exact assertion is filtered.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # werkzeug renders the traceback into the message itself rather than
        # passing exc_info, so the text is all there is to match on.
        msg = record.getMessage()
        return not ("Error on request" in msg
                    and "write() before start_response" in msg)


def _quiet_werkzeug() -> None:
    """Access lines for every polling GET would drown the console too."""
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    log.addFilter(_HijackedSocketFilter())


def _check_wav(path: str) -> tuple[bool, str]:
    """Reject anything Whisper would waste time on or choke over."""
    try:
        with wave.open(path, "rb") as wav:
            rate = wav.getframerate()
            secs = wav.getnframes() / float(rate or 1)
    except Exception as e:                               # noqa: BLE001
        return False, f"not a readable WAV: {e}"
    if secs > MAX_AUDIO_SECONDS:
        return False, f"audio too long ({secs:.1f}s > {MAX_AUDIO_SECONDS:.0f}s)"
    if secs < MIN_AUDIO_SECONDS:
        return False, f"audio too short ({secs:.2f}s)"
    return True, ""


class RobotLink:
    """Socket.IO server for the robot client, run alongside the webui's HTTP server.

    `enqueue(text, on_reply=...)` is the webui's own chat queue — the callback fires
    on the vision thread once the LLM has answered, and emits back to the sid that
    asked. Nothing here touches the pipeline directly.
    """

    def __init__(self, enqueue: Callable[..., None], *, port: int = DEFAULT_PORT,
                 stt: Optional[SpeechToText] = None, robot_name: str = "chatbox"):
        self.enqueue = enqueue
        self.port = port
        self.stt = stt
        self.robot_name = robot_name
        self.clients: dict[str, dict] = {}     # sid -> the client's init payload
        self._lock = threading.Lock()
        self._sio = None

    # ── wiring ────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            from flask import Flask, request
            from flask_socketio import SocketIO
        except ImportError:
            print("[robot] flask-socketio not installed — client link disabled "
                  "(pip install flask-socketio simple-websocket)", flush=True)
            return False

        _quiet_werkzeug()
        app = Flask(__name__)
        # threading mode needs no eventlet/gevent; simple-websocket gives it real
        # websockets, which is the client's preferred transport.
        sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                       logger=False, engineio_logger=False)
        self._sio = sio

        @sio.on("connect")
        def _connect():
            print(f"[robot] socket connected: {request.sid}", flush=True)
            return True

        @sio.on("disconnect")
        def _disconnect():
            with self._lock:
                info = self.clients.pop(request.sid, None)
            who = (info or {}).get("client_id", request.sid)
            print(f"[robot] disconnected: {who}", flush=True)

        @sio.on("client_init")
        def _client_init(data):
            data = data or {}
            name, cid = data.get("robot_name"), data.get("client_id")
            if not name or not cid:
                sio.emit("client_init_response",
                         {"success": False,
                          "message": "robot_name and client_id are required"},
                         to=request.sid)
                return
            with self._lock:
                self.clients[request.sid] = data
            mods = data.get("modules", [])
            print(f"[robot] '{name}' ({cid}) connected — modules: "
                  f"{', '.join(mods) or 'none'}", flush=True)
            sio.emit("client_init_response", {
                "success": True,
                "message": f"{name} connected to the V5 webui pipeline",
                "client_id": cid,
                "robot_name": name,
                "enabled_modules": mods,
            }, to=request.sid)

        @sio.on("chat_message")
        def _chat_message(data):
            info = self._authed(request.sid, "chat_response")
            if info is None:
                return
            text = ((data or {}).get("message") or "").strip()
            if not text:
                self._fail(request.sid, "chat_response", "no message provided")
                return
            self._ask(request.sid, info, text, "chat_response")

        @sio.on("speech")
        def _speech(data):
            info = self._authed(request.sid, "speech_response")
            if info is None:
                return
            if self.stt is None:
                self._fail(request.sid, "speech_response",
                           "speech-to-text is not enabled on this server")
                return
            audio_b64 = (data or {}).get("audio") or ""
            if not audio_b64:
                self._fail(request.sid, "speech_response", "no audio data provided")
                return

            # Transcription takes seconds; the handler must not hold the socket.
            threading.Thread(
                target=self._transcribe_and_ask, args=(request.sid, info, audio_b64),
                daemon=True, name="stt-turn").start()

        def _serve():
            try:
                sio.run(app, host="0.0.0.0", port=self.port,
                        allow_unsafe_werkzeug=True)
            except OSError as e:
                print(f"[robot] port {self.port} unavailable ({e}) — "
                      "client link disabled", flush=True)

        threading.Thread(target=_serve, daemon=True, name="robot-link").start()
        return True

    # ── turn handling ─────────────────────────────────────────────────────────

    def _authed(self, sid: str, reply_event: str) -> Optional[dict]:
        with self._lock:
            info = self.clients.get(sid)
        if info is None:
            self._fail(sid, reply_event, "not initialised — send client_init first")
        return info

    def _fail(self, sid: str, event: str, why: str) -> None:
        print(f"[robot] {event}: {why}", flush=True)
        self._sio.emit(event, {"error": why, "timestamp": time.time()}, to=sid)

    def _transcribe_and_ask(self, sid: str, info: dict, audio_b64: str) -> None:
        ok, text = self.stt.transcribe_b64(audio_b64)
        if not ok:
            self._fail(sid, "speech_response", text)
            return
        self._ask(sid, info, text, "speech_response", transcription=text)

    def _ask(self, sid: str, info: dict, text: str, reply_event: str,
             transcription: str = "") -> None:
        """Queue one turn and wire its reply back to `sid`.

        `response` carries the reply with its action tag at the head, main's
        format. No STYLE line — that is this server's own link to the ESP32. The
        transcript rides along on the speech path because the v4 client logs it,
        and hearing what it thought you said is how you debug a bad turn.
        """
        def on_reply(reply: str) -> None:
            payload = {
                "client_id": info.get("client_id"),
                "robot_name": info.get("robot_name", self.robot_name),
                "response": reply,
                "timestamp": time.time(),
            }
            if transcription:
                payload["transcription"] = transcription
            self._sio.emit(reply_event, payload, to=sid)

        self.enqueue(text, on_reply=on_reply)
