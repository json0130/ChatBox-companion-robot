# OutputModules/edge_tts_output.py
import json as _json
import subprocess
import re
import threading
import queue
import logging
import os
import tempfile
import time
from typing import Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)


class EdgeTTSOutputModule(OutputModule):
    """
    Piper TTS output (primary) with gTTS and espeak fallbacks.
    Call update_voice_config(dict) at any time — takes effect on the next utterance.

    TTS priority:  Piper (offline neural) → gTTS (online) → espeak (robotic)

    Piper runs as a persistent subprocess (model loaded once at start) to avoid
    the ~1.2s model-reload cost on every utterance.
    """

    def __init__(self, name: str = "edge_tts_output", config: Dict = None):
        super().__init__(name, config)
        self.max_length    = self.config.get('max_length', 500)
        self.talking_speed = "1.25"

        # Voice settings — readable/writable at runtime
        self._voice_lock = threading.Lock()
        self._language   = self.config.get('language', 'en')
        self._gender     = self.config.get('gender', 'female')
        self._rate       = self.config.get('rate', '+0%')

        # Speaker output (hardcoded to USB speaker; override in config if needed)
        self._audio_cmd = self.config.get('audio_cmd', ['aplay', '-D', 'plughw:3,0'])

        self.tts_queue  = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()

        # Persistent piper process — model loaded once, reused across utterances
        self._piper_proc    = None
        self._piper_out_dir = tempfile.mkdtemp(prefix='piper_out_')
        self._piper_seq     = 0   # mirrors piper's output-file counter (0.wav, 1.wav, …)

    # ── BaseModule interface ───────────────────────────────────────────────────

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            # Warm piper in background so model is ready before first response
            threading.Thread(target=self._start_piper_process, daemon=True).start()
            return True
        return False

    def stop(self):
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self.tts_queue.put(None)
            if self.tts_thread:
                self.tts_thread.join(timeout=2)
            if self._piper_proc:
                try:
                    self._piper_proc.stdin.close()
                except Exception:
                    pass
                self._piper_proc.terminate()
                self._piper_proc = None

    def process_output(self, data: Any) -> bool:
        if not self.enabled:
            return False
        try:
            text = data.get('text', '') if isinstance(data, dict) else str(data)
            text = self._prepare_text(text)
            if text and len(text.strip()) > 2:
                self.tts_queue.put((text, None, None))
                return True
            return False
        except Exception as e:
            logger.error(f"[TTS] Processing error: {e}")
            return False

    def process_output_synced(self, data: Any, start_callback=None) -> bool:
        """
        Like process_output but fires start_callback() at the moment audio playback
        begins — not after it finishes.  Used by robot.py to sync Arduino gestures
        with the start of speech rather than the arrival of the server response.
        """
        if not self.enabled:
            if start_callback:
                start_callback()
            return False
        try:
            text = data.get('text', '') if isinstance(data, dict) else str(data)
            text = self._prepare_text(text)
            if text and len(text.strip()) > 2:
                self.tts_queue.put((text, None, start_callback))
                return True
            if start_callback:
                start_callback()
            return False
        except Exception as e:
            logger.error(f"[TTS] Processing error: {e}")
            return False

    def speak_with_callback(self, text: str, callback=None) -> bool:
        """
        Queue text for TTS and fire callback() after playback finishes.
        Used by BasicClient._on_demo_step() to send ACK after speech.
        """
        if not self.enabled:
            if callback:
                callback()
            return False
        text = self._prepare_text(text)
        if text and len(text.strip()) > 2:
            self.tts_queue.put((text, callback, None))
            return True
        if callback:
            callback()
        return False

    # ── Runtime voice update (called by robot.py on persona_update) ───────────

    def update_voice_config(self, voice_config: dict):
        """
        Update voice settings at runtime — takes effect on the next utterance.
        Safe to call from any thread.

        Accepted keys:
          language  : str  e.g. 'en', 'es', 'fr', 'ja'
          gender    : str  'female' | 'male'
          rate      : str  e.g. '+0%', '+10%'
        """
        with self._voice_lock:
            if 'language' in voice_config:
                self._language = voice_config['language']
                logger.info(f"[TTS] Language → {self._language}")
            if 'gender' in voice_config:
                self._gender = voice_config['gender']
                logger.info(f"[TTS] Gender   → {self._gender}")
            if 'rate' in voice_config:
                self._rate = voice_config['rate']
                logger.info(f"[TTS] Rate     → {self._rate}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[{}"]', '', text)
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0] + '...'
        return text

    def _tts_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.tts_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            if isinstance(item, tuple):
                padded = item + (None,) * 3
                text, callback, start_cb = padded[0], padded[1], padded[2]
            else:
                text, callback, start_cb = str(item), None, None

            try:
                self._speak_text(text, start_cb)
            except Exception as e:
                logger.error(f"[TTS] Playback error: {e}")
            finally:
                self.tts_queue.task_done()
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"[TTS] Callback error: {e}")

    def _speak_text(self, text: str, start_callback=None):
        with self._voice_lock:
            language = self._language

        if self.client:
            if not hasattr(self.client, 'is_speaking'):
                self.client.is_speaking = threading.Event()
            self.client.is_speaking.set()
            if hasattr(self.client, 'tts_started_event'):
                self.client.tts_started_event.set()

        try:
            # Priority 1: gTTS — natural quality, requires internet
            if self._speak_gtts(text, language, start_callback):
                return
            # Priority 2: Piper — offline neural TTS, no internet needed
            if self._speak_piper(text, start_callback):
                return
            # No further fallback
            logger.warning(f"[TTS] All TTS methods failed for: {text[:60]}")
            if start_callback:
                try:
                    start_callback()
                except Exception:
                    pass
            time.sleep(max(1.0, len(text.split()) / 2.5))
        except Exception as e:
            logger.error(f"[TTS] Speak error: {e}")
            time.sleep(max(1.0, len(text.split()) / 2.5))
        finally:
            if self.client and hasattr(self.client, 'is_speaking'):
                self.client.is_speaking.clear()
            logger.debug("[TTS] is_speaking cleared")

    # ── Piper persistent process ───────────────────────────────────────────────

    def _start_piper_process(self) -> bool:
        """Start a persistent piper subprocess using --json-input mode.
        The model is loaded once here; subsequent calls avoid the ~1.2s reload cost."""
        model = self.config.get(
            'piper_model',
            os.path.expanduser('~/piper-voices/en_US-amy-medium.onnx')
        )
        if not os.path.exists(model):
            logger.warning(f"[TTS] Piper model not found: {model} — warm process not started")
            return False
        if not os.path.exists(model + '.json'):
            logger.warning(f"[TTS] Piper .onnx.json sidecar missing — warm process not started")
            return False
        try:
            self._piper_proc = subprocess.Popen(
                ['piper',
                 '--model',            model,
                 '--json-input',
                 '--length_scale',     str(self.config.get('piper_length_scale', 1.0)),
                 '--noise_scale',      str(self.config.get('piper_noise_scale', 0.667)),
                 '--sentence_silence', '0.2'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._piper_seq = 0
            logger.info("[TTS] Piper warm process started (model loaded once)")
            return True
        except Exception as e:
            logger.warning(f"[TTS] Could not start warm piper process: {e}")
            return False

    def _speak_piper(self, text: str, start_callback=None) -> bool:
        model = self.config.get(
            'piper_model',
            os.path.expanduser('~/piper-voices/en_US-amy-medium.onnx')
        )
        if not os.path.exists(model):
            logger.warning(f"[TTS] piper model not found: {model}")
            return False
        if not os.path.exists(model + '.json'):
            logger.warning(f"[TTS] piper .onnx.json sidecar not found: {model}.json")
            return False

        # Use warm persistent process if alive
        if self._piper_proc and self._piper_proc.poll() is None:
            return self._speak_piper_warm(text, start_callback)

        # Warm process not available — restart it in background and use one-shot this time
        if self._piper_proc is None:
            threading.Thread(target=self._start_piper_process, daemon=True).start()

        # One-shot fallback: spawns a new process (loads model each call)
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            proc = subprocess.run(
                ['piper',
                 '--model',            model,
                 '--length_scale',     str(self.config.get('piper_length_scale', 1.0)),
                 '--noise_scale',      str(self.config.get('piper_noise_scale', 0.667)),
                 '--sentence_silence', '0.2',
                 '--output_file',      tmp_wav],
                input=text.encode(),
                capture_output=True,
                timeout=20,
            )
            if proc.returncode != 0:
                logger.warning(f"[TTS] piper exited {proc.returncode}: {proc.stderr.decode()[:200]}")
                return False
            if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
                logger.warning("[TTS] piper produced empty output")
                return False
            logger.info(f"[TTS] Speaking (piper-oneshot): {text[:60]}{'...' if len(text) > 60 else ''}")
            if start_callback:
                try:
                    start_callback()
                except Exception:
                    pass
            result = subprocess.run(self._audio_cmd + [tmp_wav], capture_output=True)
            if result.returncode != 0:
                subprocess.run(['aplay', tmp_wav], capture_output=True)
            return True
        except FileNotFoundError:
            logger.warning("[TTS] piper not found in PATH — rebuild Docker image")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("[TTS] piper timed out (>20 s) — check available RAM")
            return False
        except Exception as e:
            logger.warning(f"[TTS] piper failed: {e}")
            return False
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def _speak_piper_warm(self, text: str, start_callback=None) -> bool:
        """Send text to the already-running piper process via JSON stdin."""
        seq      = self._piper_seq
        self._piper_seq += 1
        out_file = os.path.join(self._piper_out_dir, f'{seq}.wav')

        if os.path.exists(out_file):
            os.unlink(out_file)  # remove any stale file from a previous run

        try:
            self._piper_proc.stdin.write((_json.dumps({"text": text, "output_file": out_file}) + '\n').encode())
            self._piper_proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("[TTS] Piper pipe broken — restarting warm process")
            self._piper_proc = None
            threading.Thread(target=self._start_piper_process, daemon=True).start()
            return False

        # Poll until piper writes the output file
        deadline = time.time() + 20
        while time.time() < deadline:
            if self._piper_proc.poll() is not None:
                logger.warning("[TTS] Piper process died — restarting")
                self._piper_proc = None
                threading.Thread(target=self._start_piper_process, daemon=True).start()
                return False
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                time.sleep(0.05)  # let piper finish flushing the file
                break
            time.sleep(0.05)
        else:
            logger.warning("[TTS] Piper warm process timed out")
            return False

        logger.info(f"[TTS] Speaking (piper-warm): {text[:60]}{'...' if len(text) > 60 else ''}")
        if start_callback:
            try:
                start_callback()
            except Exception:
                pass

        result = subprocess.run(self._audio_cmd + [out_file], capture_output=True)
        if result.returncode != 0:
            subprocess.run(['aplay', out_file], capture_output=True)

        if os.path.exists(out_file):
            os.unlink(out_file)
        return True

    def _speak_gtts(self, text: str, language: str, start_callback=None) -> bool:
        """Online fallback using gTTS + ffmpeg. Returns False if no internet or gtts unavailable."""
        try:
            from gtts import gTTS
        except ImportError:
            return False

        tmp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            tts = gTTS(text=text, lang=language)
            tts.save(tmp_mp3)

            subprocess.run([
                'ffmpeg', '-i', tmp_mp3,
                '-filter:a', f'atempo={self.talking_speed}',
                '-ar', '22050', '-ac', '1', '-sample_fmt', 's16', '-y', tmp_wav,
            ], capture_output=True, check=True)

            logger.info(f"[TTS] Speaking (gTTS): {text[:60]}{'...' if len(text) > 60 else ''}")
            if start_callback:
                try:
                    start_callback()
                except Exception:
                    pass
            result = subprocess.run(self._audio_cmd + [tmp_wav], capture_output=True)
            if result.returncode != 0:
                subprocess.run(['aplay', tmp_wav], capture_output=True)
            return True

        except Exception as e:
            logger.warning(f"[TTS] gTTS failed ({e}) — falling back to espeak")
            return False
        finally:
            for f in [tmp_mp3, tmp_wav]:
                if os.path.exists(f):
                    os.unlink(f)
