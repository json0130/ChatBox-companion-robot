"""
Live webcam → FaceIdentifier + EmotionProcessor → KGBridge → PAD → LLM loop.

All interaction happens inside the OpenCV window — no terminal typing needed.

Run from CHATBOX_SERVER/:
    python3 -m modules.face_webcam.webcam_loop --mode enroll --name jay
    python3 -m modules.face_webcam.webcam_loop --mode run --llm

In-window keyboard controls:
    T       — open chat input box (type message, Enter=send, Esc=cancel)
    E       — open enroll box (type name, Enter=capture 12 frames, Esc=cancel)
    B       — boost current person's rapport+trust (+0.15 each)
    S       — save faces.npz
    Q / Esc — quit and auto-save faces

Tier progression (at 1 tick/sec, happy emotion):
    unknown → visitor : tick 1   (first interaction)
    visitor → known   : tick 6   (count > 5)
    known   → close   : ~34 s    (rapport+trust average > 0.70 via auto-increment)
    OR press B 5 ×               (instant +0.15 each press)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import socket
import sys
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


@contextlib.contextmanager
def _mute_stderr():
    """
    Redirect file-descriptor 2 to /dev/null for the duration of the block.

    Qt's QFontDatabase writes 'Cannot find font directory' via qWarning() which
    bypasses Python's sys.stderr and QT_LOGGING_RULES — only an fd-level
    redirect suppresses it.  Used only around cv2.namedWindow / cv2.imshow
    so real errors are never hidden.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved   = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# ── Import path resolution ────────────────────────────────────────────────────

def _add_server_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

_SERVER_ROOT = _add_server_root()

# ── KG + PAD ──────────────────────────────────────────────────────────────────

from modules.graph_relationship.kg_bridge import KGBridge
from modules.graph_relationship.store import InMemoryGraphStore
from modules.pad_persona.pipeline_adapter import PADPipelineAdapter
from modules.face_webcam.face_id import FaceIdentifier
from modules.face_webcam.emotion_detector import EmotionDetector
from modules.session_store import SessionStore, DEFAULT_DB as _DEFAULT_SESSIONS_DB

# ── LLM ───────────────────────────────────────────────────────────────────────

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FACES   = "faces.npz"
_DEFAULT_KG      = "kg_state.json"
_DEFAULT_ROBOT   = "chatbox"
_DEFAULT_TICK    = 1.0
_DEFAULT_THRESH  = 0.75
_DEFAULT_CAMERA  = 0
_DEFAULT_MODEL   = "qwen2.5:7b"

# Input mode states
_MODE_IDLE   = 0
_MODE_CHAT   = 1
_MODE_ENROLL = 2

# BGR colours
_C_CLOSE   = (80,  220, 60)
_C_KNOWN   = (60,  200, 200)
_C_VISITOR = (40,  170, 255)
_C_UNKNOWN = (60,  60,  230)
_C_WHITE   = (255, 255, 255)
_C_BLACK   = (0,   0,   0)
_C_YELLOW  = (0,   215, 215)
_C_GRAY    = (140, 140, 140)
_C_GREEN   = (70,  220, 90)
_C_PINK    = (200, 130, 255)
_C_CYAN    = (210, 230, 20)
_C_DARK    = (20,  20,  20)

_TIER_COL = {
    "close":   _C_CLOSE,
    "known":   _C_KNOWN,
    "visitor": _C_VISITOR,
    "unknown": _C_UNKNOWN,
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM action-tag parsing + ESP32 dispatch
# ─────────────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r'^\[([A-Z_]+)\]', re.ASCII)

# Map LLM action tags → ESP32 validExpressions[]
_TAG_TO_ESP32: dict[str, str] = {
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


_LEAK_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                 "\nuser", "\nUser", "\nassistant", "\nAssistant")


def _clean_reply(text: Optional[str]) -> str:
    """Trim a model reply at any leaked ChatML / next-turn marker (qwen sometimes
    keeps generating past its turn — Chinese text + a fake 'user:' turn)."""
    s = (text or "").strip()
    cut = len(s)
    for mark in _LEAK_MARKERS:
        i = s.find(mark)
        if i != -1:
            cut = min(cut, i)
    return s[:cut].strip()


def _parse_llm_response(text: str) -> tuple[str, str]:
    """Split '[TAG] body text' into ('TAG', 'body text'). Returns ('', text) if no tag."""
    m = _TAG_RE.match(text.strip())
    if m:
        return m.group(1), text[m.end():].strip()
    return "", text.strip()


def _send_esp32(expression: str, host: str, port: int = 8888,
                timeout: float = 0.5) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall((expression + "\n").encode())
        print(f"[ESP32] → {expression!r}")
    except OSError as exc:
        print(f"[ESP32] send failed ({host}:{port}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Ollama wrapper with optional rolling chat history."""

    def __init__(self, model: str = _DEFAULT_MODEL,
                 host: str = "127.0.0.1", port: int = 11434):
        self.model     = model
        self.available = False
        self._base_url = f"http://{host}:{port}/v1"
        self._client   = None

    def connect(self) -> bool:
        if not _OPENAI_AVAILABLE:
            print("[LLM] openai package not installed — run: pip install openai")
            return False
        try:
            self._client = _OpenAI(base_url=self._base_url, api_key="ollama")
            self._client.models.list()
            self.available = True
            print(f"[LLM] connected — model: {self.model}")
        except Exception as exc:
            print(f"[LLM] Ollama unavailable: {exc}")
        return self.available

    def respond(self, system_prompt: str, user_msg: str,
                history: list[tuple[str, str]] | None = None) -> str:
        """
        Args:
            history: list of (user_text, assistant_text) pairs from previous turns.
                     Injected as alternating user/assistant messages before user_msg.
        """
        if not self.available or self._client is None:
            return "[LLM not connected — run with --llm]"
        try:
            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            for u, b in (history or []):
                messages.append({"role": "user",      "content": u})
                messages.append({"role": "assistant", "content": b})
            messages.append({"role": "user", "content": user_msg})
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=140,
                temperature=0.7,
                # Stop if the model tries to continue past its turn / open a new one
                # (qwen sometimes leaks ChatML tokens or a fake "user:" turn).
                stop=["<|im_start|>", "<|im_end|>", "\nuser", "\nUser", "\nassistant"],
            )
            return _clean_reply(resp.choices[0].message.content)
        except Exception as exc:
            return f"[LLM error: {exc}]"


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def _text(frame, msg: str, pos: tuple,
          scale: float = 0.62, col=_C_WHITE, thickness: int = 2) -> None:
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, col, thickness, cv2.LINE_AA)


def _panel(frame, x1: int, y1: int, x2: int, y2: int,
           col=_C_DARK, alpha: float = 0.72) -> None:
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def _bar(frame, x: int, y: int, w: int, h: int,
         value: float, col, bg_col=_C_GRAY) -> None:
    """Draw a horizontal progress bar for a 0–1 float."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_col, 1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), col, -1)


def draw_overlay(
    frame: np.ndarray,
    *,
    # primary person (drives bottom panel)
    person_id:   Optional[str],
    sim:         float,
    box:         Optional[tuple],
    emotion:     str,
    e_conf:      float,
    tier:        str,
    pad_state:   Optional[tuple],
    descriptors: Optional[dict],
    rapport:     float,
    trust:       float,
    # emotion V/A values (weighted-blend from softmax)
    va:          tuple = (0.0, 0.0),
    # chat display
    last_user_msg: Optional[str],
    last_verbal:   Optional[str],
    # header
    robot_name: str,
    tick:       int,
    fps:        float,
    # input box
    input_mode:    int   = _MODE_IDLE,
    input_text:    str   = "",
    input_error:   str   = "",
    cursor_on:     bool  = True,
    # misc
    enroll_capturing: bool = False,
    enroll_progress:  int  = 0,
    enroll_total:     int  = 12,
    llm_on:           bool = False,
    # ALL detected faces this tick  ← new
    all_detections: list  = [],
) -> np.ndarray:
    frame = frame.copy()
    h, w  = frame.shape[:2]
    tc    = _TIER_COL.get(tier, _C_UNKNOWN)

    # ── Face bounding boxes — one per detected face ───────────────────────────
    draw_list = all_detections if all_detections else (
        [{"person_id": person_id, "sim": sim, "box": box,
          "emotion": emotion, "e_conf": e_conf, "tier": tier}]
        if box is not None else []
    )
    for det in draw_list:
        dbox = det.get("box")
        if dbox is None:
            continue
        x1, y1, x2, y2 = int(dbox[0]), int(dbox[1]), int(dbox[2]), int(dbox[3])
        dtier = det.get("tier", "unknown")
        dpid  = det.get("person_id")
        dcol  = _TIER_COL.get(dtier, _C_UNKNOWN) if dpid else _C_UNKNOWN
        cv2.rectangle(frame, (x1, y1), (x2, y2), dcol, 2)
        # Corner accents
        clen = 18
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*clen, cy), dcol, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy*clen), dcol, 3)
        # Name + similarity above box
        name_label = f"{dpid}  {det.get('sim',0):.2f}" if dpid else f"?  {det.get('sim',0):.2f}"
        _text(frame, name_label, (x1 + 4, max(y1 - 20, 18)), 0.58, dcol)
        # Emotion below name (inside top of box area)
        dem, dec = det.get("emotion", ""), det.get("e_conf", 0.0)
        if dem:
            emo_str = f"{dem} {dec:.0f}%" if dec > 1 else dem
            _text(frame, emo_str, (x1 + 4, max(y1 - 5, 34)), 0.46, dcol, 1)

    # ── Top bar ───────────────────────────────────────────────────────────────
    _panel(frame, 0, 0, w, 38)
    _text(frame, f"{robot_name}  |  tick {tick}  |  {fps:.1f} fps",
          (8, 26), 0.60, _C_YELLOW)

    # LLM status badge (right side of top bar)
    if llm_on:
        llm_label, llm_col = "LLM ON", _C_GREEN
    else:
        llm_label, llm_col = "no LLM", _C_UNKNOWN
    _text(frame, llm_label, (w - 100, 26), 0.50, llm_col, 1)

    # ── Bottom info panel ─────────────────────────────────────────────────────
    # Decide height based on how much chat text we have
    chat_extra = 0
    if last_user_msg:
        chat_extra += 22
    if last_verbal:
        chat_extra += 22 * max(1, (len(last_verbal) + 54) // 55)

    panel_top = h - (130 + chat_extra)
    _panel(frame, 0, panel_top, w, h)
    cv2.line(frame, (0, panel_top), (w, panel_top), tc, 1)

    y = panel_top + 22

    # Person + emotion row
    pname = person_id or "unknown"
    _text(frame, f"Person: {pname}", (8, y), 0.60, _C_WHITE)
    ec_str = f"Emotion: {emotion}  {e_conf:.0f}%" if e_conf > 1 else f"Emotion: {emotion}"
    _text(frame, ec_str, (w // 2, y), 0.60, _C_WHITE)
    y += 26

    # Tier label + rapport/trust bars
    _text(frame, f"Tier:  {tier.upper()}", (8, y), 0.68, tc, 2)
    bx, bw = w // 2, 90
    _text(frame, f"R {rapport:.2f}", (bx - 6, y), 0.50, _C_GRAY, 1)
    _bar(frame, bx + 40, y - 13, bw, 11, rapport, tc)
    _text(frame, f"T {trust:.2f}",  (bx - 6, y + 16), 0.50, _C_GRAY, 1)
    _bar(frame, bx + 40, y + 3,  bw, 11, trust,   tc)
    # Threshold marker at 0.70
    mx = bx + 40 + int(bw * 0.70)
    cv2.line(frame, (mx, y - 15), (mx, y + 16), _C_YELLOW, 1)
    y += 32

    # PAD values
    if pad_state is not None:
        p, a, d = pad_state
        _text(frame, f"PAD   P={p:+.2f}   A={a:+.2f}   D={d:+.2f}", (8, y), 0.60, _C_WHITE)
    y += 22

    # Emotion V/A (weighted softmax blend from camera)
    ev, ea = va
    _text(frame, f"V/A   V={ev:+.2f}   A={ea:+.2f}", (8, y), 0.56, _C_CYAN)
    y += 22

    # Mood descriptors
    if descriptors:
        mood = (f"Mood:  {descriptors.get('pleasure','?')} / "
                f"{descriptors.get('arousal','?')} / "
                f"{descriptors.get('dominance','?')}")
        _text(frame, mood, (8, y), 0.54, _C_GRAY)
    y += 22

    # Chat exchange
    if last_user_msg:
        short = last_user_msg[:52] + ("…" if len(last_user_msg) > 52 else "")
        _text(frame, f"[you]   \"{short}\"", (8, y), 0.52, _C_PINK, 1)
        y += 22
    if last_verbal:
        words, buf, lines = last_verbal.split(), "", []
        for ww in words:
            if len(buf) + len(ww) + 1 > 55:
                lines.append(buf); buf = ww
            else:
                buf = (buf + " " + ww).strip()
        if buf:
            lines.append(buf)
        for ln in lines[:3]:
            _text(frame, f"[{robot_name}]  \"{ln}\"", (8, y), 0.52, _C_GREEN, 1)
            y += 20

    # ── Input box (replaces controls hint when active) ────────────────────────
    BOX_H = 46
    box_y = h - BOX_H
    _panel(frame, 0, box_y, w, h, col=_C_DARK, alpha=0.88)
    cv2.line(frame, (0, box_y), (w, box_y),
             _C_CYAN if input_mode != _MODE_IDLE else _C_GRAY, 2)

    if input_mode == _MODE_IDLE:
        if enroll_capturing:
            pct = int(enroll_progress / enroll_total * 100)
            bar_w = int((w - 200) * enroll_progress / enroll_total)
            cv2.rectangle(frame, (8, box_y + 14), (8 + bar_w, box_y + 30), _C_GREEN, -1)
            _text(frame, f"Capturing …  {enroll_progress}/{enroll_total}  ({pct}%)",
                  (8, box_y + 34), 0.52, _C_GREEN, 1)
        elif llm_on:
            _text(frame, "T=chat   E=enroll   B=boost   S=save   Q=quit",
                  (8, box_y + 30), 0.50, _C_GRAY, 1)
        else:
            _text(frame, "T=chat   E=enroll   B=boost   S=save   Q=quit",
                  (8, box_y + 18), 0.50, _C_GRAY, 1)
            _text(frame, "restart with --llm flag to enable AI chat",
                  (8, box_y + 36), 0.45, _C_UNKNOWN, 1)

    else:
        label = "Chat:" if input_mode == _MODE_CHAT else "Enroll name:"
        cursor = "|" if cursor_on else " "
        display_text = input_text + cursor

        # Prompt label
        _text(frame, label, (8, box_y + 30), 0.56, _C_CYAN, 1)

        # Input text area with border
        tx = 130
        cv2.rectangle(frame, (tx - 4, box_y + 10), (w - 130, box_y + 40),
                      _C_GRAY, 1)
        # Clip displayed text to fit the box (show last N chars)
        max_chars = 52
        disp = display_text[-max_chars:] if len(display_text) > max_chars else display_text
        _text(frame, disp, (tx, box_y + 30), 0.56, _C_WHITE, 1)

        # Hints on the right
        hints = "Enter=send  Esc=cancel" if input_mode == _MODE_CHAT else "Enter=confirm  Esc=cancel"
        _text(frame, hints, (w - 260, box_y + 30), 0.44, _C_GRAY, 1)

        # Error message
        if input_error:
            _text(frame, input_error, (tx, box_y + 10), 0.45, _C_UNKNOWN, 1)

    # Enrollment progress overlay (centre)
    if enroll_capturing:
        pct = int(enroll_progress / enroll_total * 100)
        banner_w, banner_h = 340, 60
        bx0 = (w - banner_w) // 2
        by0 = h // 2 - banner_h // 2
        _panel(frame, bx0, by0, bx0 + banner_w, by0 + banner_h,
               col=(0, 80, 0), alpha=0.85)
        cv2.rectangle(frame, (bx0, by0), (bx0 + banner_w, by0 + banner_h),
                      _C_GREEN, 2)
        _text(frame, f"Capturing …  {enroll_progress}/{enroll_total}",
              (bx0 + 20, by0 + 22), 0.7, _C_GREEN, 2)
        bar_inner = int((banner_w - 40) * enroll_progress / enroll_total)
        cv2.rectangle(frame, (bx0 + 20, by0 + 34), (bx0 + banner_w - 20, by0 + 50),
                      _C_GRAY, 1)
        if bar_inner > 0:
            cv2.rectangle(frame, (bx0 + 20, by0 + 34),
                          (bx0 + 20 + bar_inner, by0 + 50), _C_GREEN, -1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# KG rapport/trust helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_rapport_trust(
    store: InMemoryGraphStore, person_id: str, robot_id: str
) -> tuple[float, float]:
    from modules.graph_relationship.interactions import get_interaction
    interaction = get_interaction(store, person_id, robot_id)
    if interaction is None:
        return 0.0, 0.0
    return interaction.rapport, interaction.trust


def _dump_kg(store: InMemoryGraphStore, robot_id: str) -> None:
    """Print all KG nodes and edges to stdout for debugging."""
    nodes = list(store._nodes.values())
    edges = list(store._edges.values())
    print("\n" + "=" * 60)
    print(f"  KG DUMP  ({len(nodes)} nodes, {len(edges)} edges)")
    print("=" * 60)
    for node in nodes:
        print(f"  [NODE] {node.node_type:8s}  id={node.id!r}")
    print()
    for edge in edges:
        val = getattr(edge, "value",  None)
        wt  = getattr(edge, "weight", None)
        cnt = getattr(edge, "count",  None)
        extra = ""
        if val  is not None: extra = f"  value={val:+.3f}"
        if wt   is not None: extra = f"  weight={wt:.3f}"
        if cnt  is not None: extra = f"  count={cnt}"
        print(f"  [EDGE] {edge.edge_type:18s}  {edge.source_id!r:12s} → {edge.target_id!r:12s}{extra}")
    print("=" * 60 + "\n")


def _update_rapport_trust(
    store: InMemoryGraphStore,
    person_id: str,
    robot_id:  str,
    delta: float,
    verbose: bool = False,
) -> None:
    from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment
    from modules.graph_relationship.interactions import set_closeness

    if store.get_node(person_id) is None:
        store.upsert_node(PersonNode(id=person_id))
    if store.get_node(robot_id) is None:
        store.upsert_node(RobotNode(id=robot_id, name=robot_id,
                                    embodiment=Embodiment.CAT))
    r_cur, t_cur = _read_rapport_trust(store, person_id, robot_id)
    r_new = min(1.0, r_cur + delta)
    t_new = min(1.0, t_cur + delta)
    # Closeness lives on the pair's InteractionNode.
    set_closeness(store, person_id, robot_id, rapport=r_new, trust=t_new,
                  source="webcam_loop")
    if verbose:
        print(f"[Boost] {person_id} → rapport={r_new:.2f}  trust={t_new:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Background detection worker
# ─────────────────────────────────────────────────────────────────────────────

class _DetectionWorker(threading.Thread):
    """
    Runs MTCNN + ResNet + emotion in a daemon thread.

    The main display loop submits a frame via submit() and reads
    get_results() without ever blocking — detection latency is completely
    hidden from the display frame-rate.

    Design:
    - Only the latest submitted frame is processed; stale frames are dropped.
    - Per-person emotion smoothers live here so the smoothing window is never
      shared between different identities.
    - Capped at max_faces (default 4) and downscaled by det_scale (default 0.5)
      for fast MTCNN inference.
    """

    def __init__(
        self,
        face_id,
        emotion_backend: str   = 'hsemotion',
        max_faces:       int   = 4,
        det_scale:       float = 0.5,
        detect_emotion:  bool  = True,
    ):
        super().__init__(daemon=True, name="detection-worker")
        self._face_id         = face_id
        self._emotion_backend = emotion_backend
        self._max_faces       = max_faces
        self._det_scale       = det_scale
        self._detect_emotion  = detect_emotion

        # Per-person smoothers (only ever touched from this thread — no lock needed).
        # Skipped entirely when emotion detection is disabled (no models loaded).
        self._per_emotion: dict[str, EmotionDetector] = {}
        self._unknown_emotion = (
            EmotionDetector.create(emotion_backend) if detect_emotion else None
        )

        self._lock    = threading.Lock()
        self._frame   = None
        self._results: list = []
        self._event   = threading.Event()
        # NOTE: must NOT be named `_stop` — that shadows threading.Thread._stop,
        # which Thread.join() calls internally (→ 'Event' object is not callable).
        self._stop_evt = threading.Event()

    # ── Public API (called from main thread) ──────────────────────────────────

    def submit(self, frame: np.ndarray) -> None:
        """Drop latest frame in; any unprocessed previous frame is discarded."""
        with self._lock:
            self._frame = frame
        self._event.set()

    def get_results(self) -> list:
        """Non-blocking — returns last completed detection list."""
        with self._lock:
            return list(self._results)

    def stop(self) -> None:
        self._stop_evt.set()
        self._event.set()  # unblock the wait

    # ── Worker loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop_evt.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()
            if self._stop_evt.is_set():
                break

            with self._lock:
                frame = self._frame
            if frame is None:
                continue

            raw = self._face_id.identify_all(
                frame,
                max_faces=self._max_faces,
                scale=self._det_scale,
            )

            results = []
            for person_id, sim, box in raw:
                if not self._detect_emotion:
                    # Emotion disabled for this pass — face identification only.
                    emo, e_conf, ev, ea = "neutral", 0.0, 0.0, 0.0
                elif person_id is not None:
                    if person_id not in self._per_emotion:
                        self._per_emotion[person_id] = EmotionDetector.create(
                            self._emotion_backend
                        )
                    emo, e_conf, ev, ea = self._per_emotion[person_id].detect(
                        frame, box=box, smooth=True
                    )
                else:
                    emo, e_conf, ev, ea = self._unknown_emotion.detect(
                        frame, box=box, smooth=False
                    )
                results.append({
                    "person_id": person_id,
                    "sim":       sim,
                    "box":       box,
                    "emotion":   emo,
                    "e_conf":    e_conf,
                    "va":        (ev, ea),
                })

            with self._lock:
                self._results = results


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

class WebcamKGLoop:
    """
    Webcam → face_id + emotion → KGBridge → PAD → LLM, with in-window chat UI.

    Press T in the OpenCV window to open the chat box.
    Press E to enroll a new face by typing a name in the window.
    """

    def __init__(
        self,
        robot_id:        str   = _DEFAULT_ROBOT,
        faces_path:      str   = _DEFAULT_FACES,
        kg_path:         str   = _DEFAULT_KG,
        threshold:       float = _DEFAULT_THRESH,
        tick_interval:   float = _DEFAULT_TICK,
        llm_client:      Optional[LLMClient] = None,
        show_window:     bool  = True,
        emotion_backend: str   = 'hsemotion',
        esp32_host:      str   = "",
        esp32_port:      int   = 8888,
        spec_dir:        Optional[str] = None,
        seed:            bool  = True,
        matcher                = None,
        embed_fn               = None,
        sessions_db:     str   = _DEFAULT_SESSIONS_DB,
        pad_enabled:     bool  = False,
        emotion_enabled: bool  = False,
    ):
        self.robot_id      = robot_id
        self.faces_path    = faces_path
        self.kg_path       = kg_path
        self.tick_interval = tick_interval
        self.llm           = llm_client
        self.show_window   = show_window

        self.face_id = FaceIdentifier(threshold=threshold)
        if os.path.exists(faces_path):
            self.face_id.load(faces_path)
        else:
            print(f"[WebcamLoop] No face DB at '{faces_path}' — starting empty")

        self.store   = InMemoryGraphStore()
        if os.path.exists(kg_path):
            self.store.load(kg_path)
        else:
            print(f"[WebcamLoop] No KG at '{kg_path}' — starting fresh")

        # Seed authored robot/human subgraphs from spec files so a recognized
        # person has KG info to retrieve immediately (idempotent — deterministic ids).
        if seed:
            spec_dir = spec_dir or os.path.join(
                _SERVER_ROOT, "modules", "graph_relationship", "specs"
            )
            try:
                from modules.graph_relationship.seed import seed_all
                seed_all(self.store, spec_dir)
                if self.kg_path:
                    self.store.save(self.kg_path)
            except Exception as exc:  # noqa: BLE001 — seeding is best-effort
                print(f"[WebcamLoop] seed skipped ({spec_dir}): {exc}")

        self.bridge  = KGBridge(self.store)
        self._adapters: dict[str, PADPipelineAdapter] = {}

        self._emotion_backend  = emotion_backend
        self._esp32_host       = esp32_host
        self._esp32_port       = esp32_port

        # PAD persona engine + emotion detection are disabled for this pass
        # (face-reco → KG-through-conversation only). Re-enable via CLI later.
        self._pad_enabled      = pad_enabled
        self._emotion_enabled  = emotion_enabled
        self._matcher          = matcher
        self._embed_fn         = embed_fn   # for on-demand topic consolidation (Feature 2)
        # Conversation transcripts live in SQLite (not the graph). The graph keeps
        # only Interaction (rapport/trust/count) + topics/interests.
        self._session_store    = SessionStore(sessions_db)
        # RAG over the transcript store (needs embeddings); None when --no-embed.
        self._session_rag = None
        if embed_fn is not None:
            from modules.session_rag import SessionRAG
            self._session_rag = SessionRAG(self._session_store, embed_fn)
        # person_id -> this run's session id (uuid). Populated on the first turn.
        self._run_sessions: dict[str, str] = {}
        # person_id -> (valence, emotion_label) last persisted — dirty-check so the
        # per-tick mood write only hits disk when it actually changes.
        self._last_mood: dict[str, tuple] = {}

        # Per-person rolling chat history (last 5 turns)
        self._chat_history: dict[str, deque] = {}

        self._robot_display = {"chatbox": "ChatBox", "ellebot": "ElleBot"}.get(
            robot_id.lower(), robot_id
        )
        self._last_pad_result: Optional[dict] = None

    def _adapter(self) -> PADPipelineAdapter:
        if self.robot_id not in self._adapters:
            self._adapters[self.robot_id] = PADPipelineAdapter(self.robot_id)
        return self._adapters[self.robot_id]

    def _pipeline_tick(self, person_id: str, emotion: str,
                       va: Optional[tuple] = None):
        bi = self.bridge.pre_turn(person_id, self.robot_id, emotion, camera_va=va)
        pad = self._adapter().process_turn(
            valence=bi.valence, arousal=bi.arousal,
            relationship_tier=bi.tier, memory_context=bi.structured_memory,
            rapport=bi.rapport, trust=bi.trust,
            interaction_count=bi.interaction_count,
        )
        self.bridge.post_turn(person_id, self.robot_id, pad, emotion=emotion)
        p = pad["pad_state"][0]
        if p > 0.05:
            _update_rapport_trust(self.store, person_id, self.robot_id,
                                  delta=0.025 * p)
        self._last_pad_result = pad
        return bi, pad

    # ── KG-only path (PAD/emotion disabled) ────────────────────────────────────

    def _ensure_interaction(self, pid: str) -> None:
        """Ensure the person, robot and their InteractionNode exist in the graph.
        No SessionNode — transcripts live in SQLite now."""
        from modules.graph_relationship.schema import (
            PersonNode, RobotNode, Embodiment,
        )
        from modules.graph_relationship.interactions import get_or_create_interaction
        if self.store.get_node(pid) is None:
            self.store.upsert_node(PersonNode(id=pid, display_name=pid))
        if self.store.get_node(self.robot_id) is None:
            emb = (Embodiment.CAT if self.robot_id.lower() == "chatbox"
                   else Embodiment.ELEPHANT)
            self.store.upsert_node(
                RobotNode(id=self.robot_id, name=self.robot_id, embodiment=emb))
        get_or_create_interaction(self.store, pid, self.robot_id, source=self.robot_id)

    def _run_session_id(self, pid: str) -> str:
        """This run's session id for `pid` (uuid), created on the first turn.
        Ensures the interaction exists. Groups the run's turns in the SQLite store."""
        import uuid
        self._ensure_interaction(pid)
        sid = self._run_sessions.get(pid)
        if sid is None:
            sid = str(uuid.uuid4())
            self._run_sessions[pid] = sid
            if self.kg_path:
                self.store.save(self.kg_path)   # persist the new interaction once
        return sid

    def _kg_tick(self, pid: str) -> tuple[str, float, float]:
        """Ensure the person's interaction exists and return (tier, rapport, trust)
        for the overlay. No PAD, no per-tick transcript writes, no SessionNode."""
        from modules.graph_relationship.kg_bridge import derive_tier
        self._ensure_interaction(pid)
        tier = derive_tier(pid, self.robot_id, self.store)
        r, t = _read_rapport_trust(self.store, pid, self.robot_id)
        return tier, r, t

    def _mood_tick(self, pid: str, emotion: Optional[str],
                   va: Optional[tuple]) -> bool:
        """Write the person's FAST MoodEdge from the current camera emotion
        (emotion → valence). No PAD. Returns True if it changed enough to save."""
        from datetime import datetime, timezone
        from modules.graph_relationship.schema import MoodEdge, Provenance
        if va is not None and va[0] is not None:
            valence = float(va[0])
        else:
            from modules.graph_relationship.kg_bridge import emotion_label_to_va
            valence, _a = emotion_label_to_va(emotion)
        valence = max(-1.0, min(1.0, valence))
        self.store.upsert_edge(MoodEdge(
            source_id=pid, target_id=pid,
            provenance=Provenance(source=self.robot_id, confidence=1.0,
                                  timestamp=datetime.now(timezone.utc)),
            value=valence, label=emotion,
        ))
        # Only treat as "changed" (→ a disk save) on an emotion-label change or a
        # real valence shift; the raw detector valence jitters frame-to-frame, so
        # a tight threshold here would save on almost every tick (log spam).
        prev = self._last_mood.get(pid)
        changed = (prev is None or prev[1] != emotion
                   or abs(prev[0] - valence) >= 0.15)
        if changed:
            self._last_mood[pid] = (valence, emotion)
            # Mirror onto the live conversation node if one exists (don't create
            # one just from being seen — only once a conversation has started).
            from modules.graph_relationship.topics import update_conversation
            update_conversation(self.store, pid, self.robot_id,
                                mood=valence, emotion=emotion, create=False,
                                source="live-mood")
        return changed

    def _detect_topic(self, user_msg: str, reply: str = "") -> Optional[str]:
        """Best-effort 1–3 word label of what's being discussed, via the LLM.
        Used to update the live conversation node. Returns None on any failure."""
        if not (self.llm and self.llm.available):
            return None
        sys = ("You label the topic of a short conversation snippet. Reply with "
               "ONLY the topic as 1-3 lowercase words (a noun phrase) — no "
               "punctuation, no sentence. Examples: 'jazz music', 'the stock "
               "market', 'space travel'.")
        try:
            raw = self.llm.respond(sys, f"Person said: {user_msg}\nRobot said: {reply}")
        except Exception:  # noqa: BLE001
            return None
        topic = (raw or "").strip().strip('".\'').lower()
        # Reject sentences / error strings — keep only short noun phrases.
        if not topic or topic.startswith("[") or len(topic.split()) > 4 or len(topic) > 40:
            return None
        return topic

    # Memory caps — keep the prompt bounded as the graph grows.
    _MAX_INTERESTS = 4
    _MAX_TOPICS_PER_INTEREST = 3
    _MAX_NOTES = 8
    _MAX_NOTES_PER_TOPIC = 2

    def _person_memory(self, pid: Optional[str]) -> str:
        """The 'WHO YOU'RE TALKING TO' body: key interests (capped), common
        ground, and the most recent notes (deduped). Returns "" if nothing known."""
        if not pid:
            return ""
        from modules.graph_relationship.topics import (
            person_interests, related_common_ground, person_related_pairs,
            topic_related,
        )
        interests = person_interests(self.store, pid)
        lines: list[str] = []

        if interests:
            # Richer interests (more topics) first, then cap.
            ranked = sorted(interests, key=lambda it: len(it[1]), reverse=True)
            parts = []
            for interest, topics in ranked[:self._MAX_INTERESTS]:
                if topics:
                    labels = [t.label for t in topics][:self._MAX_TOPICS_PER_INTEREST]
                    parts.append(f"{interest.label} ({', '.join(labels)})")
                else:
                    parts.append(interest.label)
            lines.append("Interests: " + " · ".join(parts))

        # Common ground — direct + RELATED bridges (Feature-2c, point 2): a topic
        # they like that relates to something the robot knows counts as connection.
        cg = related_common_ground(self.store, pid, self.robot_id)
        if cg["direct"]:
            lines.append("Common ground: " + ", ".join(cg["direct"]))
        if cg["bridges"]:
            bl = ", ".join(f"their {p} ~ your {r}" for p, r in cg["bridges"])
            lines.append("You can also connect via related topics: " + bl)
        # Their own related topics, so the robot can bridge/recall across them.
        rp = person_related_pairs(self.store, pid)
        if rp:
            lines.append("Related interests: "
                         + ", ".join(f"{a} ~ {b}" for a, b in rp))

        # Collect notes, then surface the ones with SPECIFIC facts first (proper
        # nouns / quoted titles like "Rafael Nadal", "SZA 'Open Arms'"), then by
        # recency — so concrete memories aren't crowded out by generic ones.
        def _specificity(text: str) -> int:
            score = 2 if ("'" in text or '"' in text) else 0
            words = str(text).split()
            score += sum(1 for w in words[1:] if w[:1].isupper())  # mid-sentence caps
            return score
        # Point 1: gather notes from the person's topics AND one hop across
        # related_topic edges, so related memories surface (e.g. a note on 'hiphop'
        # when they mention 'rap'). Deduped by topic id.
        note_topics: dict = {}
        for _interest, topics in interests:
            for t in topics:
                note_topics[t.id] = t
        for tid in list(note_topics):
            for r in topic_related(self.store, tid):
                note_topics.setdefault(r.id, r)
        collected: list = []
        for t in note_topics.values():
            for n in getattr(t, "notes", []) or []:
                if n.get("person") == pid and n.get("text"):
                    collected.append((_specificity(n["text"]), n.get("ts", ""),
                                      t.label, n["text"]))
        collected.sort(key=lambda x: (x[0], x[1]), reverse=True)  # specific + recent first
        collected = [(ts, label, text) for _s, ts, label, text in collected]
        per_topic: dict = {}
        note_lines: list[str] = []
        for _ts, label, text in collected:
            # Up to _MAX_NOTES_PER_TOPIC per topic so specific facts (e.g. a
            # favourite player/song) aren't hidden behind a generic note.
            if per_topic.get(label, 0) >= self._MAX_NOTES_PER_TOPIC:
                continue
            per_topic[label] = per_topic.get(label, 0) + 1
            note_lines.append(f"  – {label}: {text}")
            if len(note_lines) >= self._MAX_NOTES:
                break
        if note_lines:
            lines.append("What you remember about them:\n" + "\n".join(note_lines))

        return "\n".join(lines)

    def _build_system_prompt(self, pid: Optional[str], *,
                             rag_hits: Optional[list] = None) -> str:
        """Assemble the system prompt from the seeded RobotNode + retrieved memory,
        in three labelled blocks. Used when PAD is disabled (no PAD system_prompt).
        Mood/emotion is deliberately not injected (kept for the graph/viz only)."""
        from modules.graph_relationship.topics import robot_capability
        personas = [n.descriptor for _e, n in
                    self.store.query_neighbors(self.robot_id, "has_persona")
                    if n.node_type == "persona"]
        roles = [n.descriptor for _e, n in
                 self.store.query_neighbors(self.robot_id, "has_role")
                 if n.node_type == "role"]
        cap = robot_capability(self.store, self.robot_id)
        caps = cap.items if cap else []

        role_word = roles[0] if roles else "friendly companion"

        blocks: list[str] = []
        # ── IDENTITY ──
        ident = ["━━━ IDENTITY ━━━",
                 f"You are {self._robot_display}, a {role_word} robot chatting "
                 f"with someone through a webcam."]
        if personas:
            ident.append(f"Personality: {', '.join(personas)}.")
        if caps:
            ident.append(f"You can: {', '.join(caps)}.")
        blocks.append("\n".join(ident))

        # ── HOW TO REPLY ──
        blocks.append(
            "━━━ HOW TO REPLY ━━━\n"
            "• Reply in ENGLISH, in one or two short, warm, spoken sentences. Output "
            "ONLY your single reply — never write the user's next turn.\n"
            "• Begin every reply with an emotion tag in square brackets, e.g. "
            "[HAPPY], [CURIOUS].\n"
            "• ANSWER what they actually ask. If they ask about something they told "
            "you before (a favourite player, song, etc.) and it IS in the memory "
            "below, answer DIRECTLY and state the name — don't say you forgot. If it "
            "is NOT in the memory below, say you don't remember it — NEVER invent or "
            "guess a name.\n"
            "• Weave memories in naturally — don't list them back.\n"
            "• Reply to what they actually said or asked. Do not comment on how they "
            "seem to feel or offer emotional support unless they bring up their "
            "feelings themselves.")

        # ── WHO YOU'RE TALKING TO ──
        if pid:
            who = [f"━━━ WHO YOU'RE TALKING TO: {pid} ━━━"]
            mem = self._person_memory(pid)
            who.append(mem if mem else "You don't remember much about them yet.")
            # NOTE: the detected mood/emotion is intentionally NOT injected into the
            # prompt for now — it pulled replies into unsolicited emotional support.
            # It's still tracked on the graph/conversation node for the viz. Revisit
            # once the emotion model / weighting is improved.
            # RAG: what THEY said before that's relevant now (their own words only —
            # we don't feed the robot's past replies back, to avoid reinforcing any
            # earlier "I forgot" deflections).
            hit_lines = [f'  – ({h["ts"][:10]}) "{h["child"]}"'
                         for h in (rag_hits or []) if h.get("child")]
            if hit_lines:
                who.append("Relevant things they've told you before:\n"
                           + "\n".join(hit_lines))
            blocks.append("\n".join(who))
        else:
            blocks.append("━━━ WHO YOU'RE TALKING TO ━━━\n"
                          "You don't recognise this person yet.")

        return "\n\n".join(blocks)

    def _extract_session(self) -> None:
        """End-of-session knowledge extraction: distill each session's transcript
        into interests/topics + rapport/trust deltas, then persist. Needs the LLM."""
        if not (self.llm and self.llm.available):
            print("[WebcamLoop] extraction skipped — LLM not connected")
            return
        # Graph-aware typed TOPIC extraction lives in the app layer (kg_extraction);
        # closeness (rapport/trust) reuses the existing pure extractor for deltas
        # ONLY and the untouched adjust_closeness — its interest logic is not used.
        from modules.kg_extraction import extract_and_apply_topics
        from modules.graph_relationship.extraction import extract as _extract_closeness
        from modules.graph_relationship.interactions import adjust_closeness
        # Respect external edits (e.g. viz deletions) before extracting.
        if self.kg_path and os.path.exists(self.kg_path):
            self.store.reload(self.kg_path)
        people = list(self._run_sessions.keys())   # people talked to this run
        if not people:
            print("[WebcamLoop] no conversation this run — nothing to extract")
            return
        print("[WebcamLoop] extracting knowledge from this session …")
        for pid in people:
            sid = self._run_sessions.get(pid)
            # Un-extracted transcript turns come from the SQLite store now.
            turns = [t for t in self._session_store.unextracted_turns(pid)
                     if t.get("child") or t.get("reply")]
            if not turns:
                print(f"  {pid}: no conversation turns to extract")
                continue

            # (a) Graph-aware typed topics (reuse existing / add genuinely new).
            ts = extract_and_apply_topics(
                self.store, pid, self.robot_id, turns, self.llm.respond, session_id=sid)

            # (b) Closeness deltas — existing logic, untouched (deltas applied only).
            cu = _extract_closeness(turns, self.llm.respond)
            if cu.rapport_delta or cu.trust_delta:
                adjust_closeness(self.store, pid, self.robot_id,
                                 d_rapport=cu.rapport_delta, d_trust=cu.trust_delta,
                                 source=f"extraction:{sid}")

            self._session_store.mark_extracted(pid)

            reinf = ", ".join(lab for lab, _c in ts.get("reinforced", []))
            newt = ", ".join(f"{lab}[{cat}]" for lab, cat, _c in ts.get("added", []))
            print(f"  {pid}: Δrapport {cu.rapport_delta:+.2f}  Δtrust {cu.trust_delta:+.2f}")
            print(f"      reused: {reinf or '—'}   new: {newt or '—'}"
                  + (f"   dropped: {len(ts.get('dropped', []))}" if ts.get('dropped') else ""))
            if not ts.get("applied"):
                print("      (topic extraction skipped — LLM JSON parse failed)")
        # Consolidate near-duplicate topics + interests after EVERY extraction.
        self._auto_consolidate()
        if self.kg_path:
            self.store.save(self.kg_path)

    def _auto_consolidate(self) -> None:
        """Merge near-duplicate topics + interests after every extraction session.
        Applies merges (not a dry-run). No-op without embeddings."""
        if self._embed_fn is None:
            return
        from modules.kg_extraction import (
            consolidate_topics, consolidate_interests, link_related_topics,
        )
        merges = (consolidate_topics(self.store, self._embed_fn, source="auto-consolidate")["merges"]
                  + consolidate_interests(self.store, self._embed_fn, source="auto-consolidate")["merges"])
        if merges:
            print(f"[WebcamLoop] auto-consolidate — merged {len(merges)}:")
            for canon, dup in merges:
                print(f"    '{dup}'  →  '{canon}'")
        else:
            print("[WebcamLoop] auto-consolidate: no near-duplicate topics/interests")
        # Link related-but-distinct topics (rap ~ hiphop) rather than merging them.
        links = link_related_topics(self.store, self._embed_fn, source="auto-related")["links"]
        if links:
            print(f"[WebcamLoop] related-topic links (+{len(links)}):")
            for a, b, sim in links:
                print(f"    '{a}' ~ '{b}'  ({sim})")

    def _consolidate_preview(self) -> None:
        """Dry-run: print near-duplicate topics that WOULD merge (non-destructive).
        Applying is a separate, reviewable step: --mode consolidate."""
        if self._embed_fn is None:
            print("[WebcamLoop] consolidation needs embeddings (run without --no-embed)")
            return
        from modules.kg_extraction import consolidate_topics, consolidate_interests
        merges = (consolidate_topics(self.store, self._embed_fn, dry_run=True)["merges"]
                  + consolidate_interests(self.store, self._embed_fn, dry_run=True)["merges"])
        if not merges:
            print("[WebcamLoop] consolidation preview: no near-duplicate topics/interests")
            return
        print(f"[WebcamLoop] consolidation preview — {len(merges)} merge(s), DRY RUN:")
        for canon, dup in merges:
            print(f"    '{dup}'  →  '{canon}'")
        print("    apply with:  python3 -m modules.face_webcam.webcam_loop --mode consolidate")

    def run(self, camera_index: int = _DEFAULT_CAMERA) -> None:  # noqa: C901
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[WebcamLoop] Cannot open camera {camera_index}")
            return

        WIN = "ChatBox KG Loop"
        if self.show_window:
            with _mute_stderr():   # Qt prints QFontDatabase warnings here
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        llm_status = "LLM ready" if (self.llm and self.llm.available) else "no LLM (restart with --llm)"
        mode = ("PAD+emotion" if self._pad_enabled else "KG-only") + \
               (" (emotion on)" if self._emotion_enabled else " (no emotion)")
        print(f"\n[WebcamLoop] robot={self._robot_display}  tick={self.tick_interval}s  "
              f"cam={camera_index}  {llm_status}  mode={mode}")
        print("  T=chat  E=enroll  B=boost  K=dump KG  X=extract  C=consolidate?  "
              "S=save  Q=quit  (all in the OpenCV window)\n")

        # Embed any un-embedded transcript turns once up front so the first chat
        # doesn't stall building the RAG index.
        if self._session_rag is not None:
            added = self._session_rag.reindex()
            if added:
                print(f"[WebcamLoop] RAG: embedded {added} past turn(s)")

        # ── Background detection worker ───────────────────────────────────────
        worker = _DetectionWorker(
            self.face_id,
            emotion_backend=self._emotion_backend,
            max_faces=4,
            det_scale=0.5,
            detect_emotion=self._emotion_enabled,
        )
        worker.start()

        # ── KG state — updated every tick, survives between ticks ─────────────
        # person_id -> {tier, pad_state, descriptors, rapport, trust}
        _kg_state: dict[str, dict] = {}

        # ── Persistent display state ──────────────────────────────────────────
        tick_n          = 0
        last_tick_t     = 0.0
        last_person_id  : Optional[str]   = None
        last_sim        : float           = 0.0
        last_box        : Optional[tuple] = None
        last_emotion    : str             = "neutral"
        last_e_conf     : float           = 0.0
        last_tier       : str             = "unknown"
        last_pad_state  : Optional[tuple] = None
        last_descriptors: Optional[dict]  = None
        last_rapport    : float           = 0.0
        last_trust      : float           = 0.0
        last_user_msg   : Optional[str]   = None
        last_verbal     : Optional[str]   = None
        chat_expire_t   : float           = 0.0
        last_all_detections: list         = []
        last_va         : tuple           = (0.0, 0.0)

        # ── In-window input state ─────────────────────────────────────────────
        input_mode      : int  = _MODE_IDLE
        input_text      : str  = ""
        input_error     : str  = ""

        # ── Enroll capture state ──────────────────────────────────────────────
        enroll_capturing : bool = False
        enroll_name      : str  = ""
        enroll_progress  : int  = 0
        enroll_total     : int  = 12
        enroll_attempts  : int  = 0
        enroll_max_att   : int  = enroll_total * 5

        fps_t0, fps_frames, fps_display = time.time(), 0, 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WebcamLoop] Frame read failed — camera disconnected?")
                    break

                # FPS tracking
                fps_frames += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 2.0:
                    fps_display = fps_frames / elapsed
                    fps_frames  = 0
                    fps_t0      = time.time()

                # ── Enrollment capture (frame-by-frame) ───────────────────────
                if enroll_capturing:
                    if enroll_progress < enroll_total and enroll_attempts < enroll_max_att:
                        enroll_attempts += 1
                        if self.face_id.enroll(enroll_name, frame):
                            enroll_progress += 1
                    else:
                        # Finished
                        if enroll_progress > 0:
                            print(f"[Enroll] '{enroll_name}' captured {enroll_progress} frames. Saving …")
                            self.face_id.save(self.faces_path)
                            input_error = f"'{enroll_name}' enrolled!"
                        else:
                            print(f"[Enroll] No face found for '{enroll_name}'.")
                            input_error = "No face detected — try again"
                        enroll_capturing = False
                        enroll_name      = ""
                        enroll_progress  = 0
                        enroll_attempts  = 0

                # ── Submit frame to background worker ─────────────────────────
                if not enroll_capturing:
                    worker.submit(frame)

                # ── Read latest worker results (non-blocking, every frame) ─────
                raw_dets = worker.get_results()

                # Merge worker detections with last known KG state for the overlay
                last_all_detections = []
                for d in raw_dets:
                    pid = d["person_id"]
                    kg  = _kg_state.get(pid, {}) if pid else {}
                    last_all_detections.append({
                        **d,
                        "tier":        kg.get("tier",        "unknown"),
                        "pad_state":   kg.get("pad_state",   None),
                        "descriptors": kg.get("descriptors", None),
                        "rapport":     kg.get("rapport",     0.0),
                        "trust":       kg.get("trust",       0.0),
                    })

                # Update primary-person visual refs every frame
                # (boxes/names track live; tier/PAD come from last tick below)
                primary_det = None
                for d in last_all_detections:
                    if d["person_id"] is not None:
                        primary_det = d
                        break
                if primary_det:
                    last_person_id = primary_det["person_id"]
                    last_sim       = primary_det["sim"]
                    last_box       = primary_det["box"]
                    last_emotion   = primary_det["emotion"]
                    last_e_conf    = primary_det["e_conf"]
                    last_tier      = primary_det["tier"]
                    last_pad_state = primary_det["pad_state"]
                    last_descriptors = primary_det["descriptors"]
                    last_rapport   = primary_det["rapport"]
                    last_trust     = primary_det["trust"]
                    last_va        = primary_det.get("va", (0.0, 0.0))
                elif last_all_detections:
                    first = last_all_detections[0]
                    last_person_id = None
                    last_sim       = first["sim"]
                    last_box       = first["box"]
                    last_emotion   = first["emotion"]
                    last_e_conf    = first["e_conf"]
                else:
                    last_person_id = None
                    last_sim       = 0.0
                    last_box       = None

                # ── KG / PAD tick (every tick_interval) ───────────────────────
                now = time.time()
                if not enroll_capturing and now - last_tick_t >= self.tick_interval:
                    last_tick_t = now
                    tick_n += 1
                    mood_dirty = False

                    for d in raw_dets:
                        pid = d["person_id"]
                        if pid is None:
                            continue
                        if self._pad_enabled:
                            bi, pad = self._pipeline_tick(pid, d["emotion"], va=d.get("va"))
                            r, t    = _read_rapport_trust(self.store, pid, self.robot_id)
                            _kg_state[pid] = {
                                "tier":        bi.tier,
                                "pad_state":   pad["pad_state"],
                                "descriptors": pad["descriptors"],
                                "rapport":     r,
                                "trust":       t,
                            }
                            if pid == last_person_id:
                                self._last_pad_result = pad
                                last_tier        = bi.tier
                                last_pad_state   = pad["pad_state"]
                                last_descriptors = pad["descriptors"]
                                last_rapport     = r
                                last_trust       = t
                        else:
                            # KG-only tick: ensure session, refresh overlay state.
                            tier, r, t = self._kg_tick(pid)
                            _kg_state[pid] = {
                                "tier":        tier,
                                "pad_state":   None,
                                "descriptors": None,
                                "rapport":     r,
                                "trust":       t,
                            }
                            # Emotion drives the FAST MoodEdge only (no PAD).
                            if self._emotion_enabled and self._mood_tick(
                                    pid, d.get("emotion"), d.get("va")):
                                mood_dirty = True
                            if pid == last_person_id:
                                last_tier    = tier
                                last_rapport = r
                                last_trust   = t

                    # Persist after the tick so the live viz server
                    # (modules.graph_relationship.viz.server) can poll it within ~1s.
                    # PAD mode mutates every tick; the KG-only path otherwise saves
                    # on session creation + each chat turn, so we only add a per-tick
                    # save when the FAST mood actually changed (mood_dirty).
                    if (self.kg_path and (self._pad_enabled or mood_dirty)
                            and any(d["person_id"] for d in raw_dets)):
                        self.store.save(self.kg_path)

                # Clear stale chat
                if last_user_msg and time.time() > chat_expire_t:
                    last_user_msg = None
                    last_verbal   = None

                # ── Draw overlay ──────────────────────────────────────────────
                if self.show_window:
                    cursor_on = (time.time() % 1.0) < 0.5
                    display = draw_overlay(
                        frame,
                        person_id   = last_person_id,
                        sim         = last_sim,
                        box         = last_box,
                        emotion     = last_emotion,
                        e_conf      = last_e_conf,
                        tier        = last_tier,
                        pad_state   = last_pad_state,
                        descriptors = last_descriptors,
                        rapport     = last_rapport,
                        trust       = last_trust,
                        va          = last_va,
                        last_user_msg = last_user_msg,
                        last_verbal   = last_verbal,
                        robot_name  = self._robot_display,
                        tick        = tick_n,
                        fps         = fps_display,
                        input_mode      = input_mode,
                        input_text      = input_text,
                        input_error     = input_error,
                        cursor_on       = cursor_on,
                        enroll_capturing = enroll_capturing,
                        enroll_progress  = enroll_progress,
                        enroll_total     = enroll_total,
                        llm_on          = bool(self.llm and self.llm.available),
                        all_detections  = last_all_detections,
                    )
                    # _mute_stderr on first frame only — Qt finishes font init there
                    if tick_n <= 1:
                        with _mute_stderr():
                            cv2.imshow(WIN, display)
                    else:
                        cv2.imshow(WIN, display)

                # ── Key handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == 255:  # no key pressed
                    continue

                if input_mode != _MODE_IDLE:
                    # ── Text capture mode ─────────────────────────────────────
                    if key in (13, 10):  # Enter
                        if input_mode == _MODE_CHAT:
                            msg = input_text.strip()
                            if msg:
                                input_error = ""
                                last_user_msg  = msg
                                chat_expire_t  = time.time() + 45.0
                                print(f"\n  [you]  \"{msg}\"")
                                if self.llm and self.llm.available:
                                    hist = list(self._chat_history.get(
                                        last_person_id or "", []
                                    ))
                                    # Current mood (valence) + instantaneous emotion
                                    # label. Both feed the LLM: the mood line sits in
                                    # the prompt context (stable), the emotion label is
                                    # tagged on this user turn (per-moment) — so a wrong
                                    # RAG: relevant past turns for this message.
                                    rag_hits = []
                                    if self._session_rag and last_person_id:
                                        try:
                                            rag_hits = self._session_rag.search(
                                                msg, top_k=5, person_id=last_person_id)
                                        except Exception:  # noqa: BLE001
                                            rag_hits = []
                                    # PAD off → build the system prompt from the KG
                                    # (persona + retrieved person memory + RAG). Mood is
                                    # tracked on the graph/viz only, not injected here.
                                    if self._pad_enabled and self._last_pad_result:
                                        sys_prompt = self._last_pad_result["system_prompt"]
                                    else:
                                        sys_prompt = self._build_system_prompt(
                                            last_person_id, rag_hits=rag_hits)
                                    raw_reply = self.llm.respond(sys_prompt, msg, history=hist)
                                    tag, verbal = _parse_llm_response(raw_reply)
                                    last_verbal = verbal
                                    if tag:
                                        print(f"  [tag]   [{tag}]")
                                        if self._esp32_host:
                                            expr = _TAG_TO_ESP32.get(tag)
                                            if expr:
                                                _send_esp32(expr, self._esp32_host,
                                                            self._esp32_port)
                                            else:
                                                print(f"  [ESP32] no mapping for [{tag}]")
                                    print(f"  [{self._robot_display}]  \"{verbal}\"\n")
                                    # Save to per-person history
                                    pid_key = last_person_id or "__unknown__"
                                    if pid_key not in self._chat_history:
                                        self._chat_history[pid_key] = deque(maxlen=5)
                                    self._chat_history[pid_key].append((msg, verbal))
                                    # Record the turn (recognized persons only). The
                                    # transcript goes to the SQLite session store, NOT
                                    # the graph; the graph keeps only Interaction count.
                                    if last_person_id:
                                        from modules.graph_relationship.interactions import (
                                            set_interaction_count,
                                        )
                                        from modules.graph_relationship.topics import (
                                            update_conversation,
                                        )
                                        sid = self._run_session_id(last_person_id)
                                        topic = self._detect_topic(msg, verbal)
                                        self._session_store.append_turn(
                                            session_id=sid, person_id=last_person_id,
                                            robot_id=self.robot_id,
                                            emotion=(last_emotion
                                                     if self._emotion_enabled else None),
                                            child=msg, reply=verbal,
                                            topics=[topic] if topic else None)
                                        # interaction_count now comes from the store.
                                        set_interaction_count(
                                            self.store, last_person_id, self.robot_id,
                                            self._session_store.person_turn_count(
                                                last_person_id, self.robot_id),
                                            source="session-store")
                                        # Live conversation-status node (rolling topics + mood).
                                        conv = update_conversation(
                                            self.store, last_person_id, self.robot_id,
                                            topic=topic,
                                            mood=self._last_mood.get(
                                                last_person_id, (None,))[0],
                                            emotion=(last_emotion
                                                     if self._emotion_enabled else None),
                                            create=True, source="live-topic")
                                        if topic:
                                            print(f"  [topic]  recent → "
                                                  f"{', '.join(conv.topics)}")
                                        if self.kg_path:
                                            self.store.save(self.kg_path)
                                else:
                                    last_verbal = "[LLM not enabled — run with --llm]"
                                    print("  [chat] LLM not connected. Run with --llm.\n")
                            input_mode = _MODE_IDLE
                            input_text = ""

                        elif input_mode == _MODE_ENROLL:
                            name = input_text.strip()
                            if name:
                                input_error      = ""
                                enroll_name      = name
                                enroll_progress  = 0
                                enroll_attempts  = 0
                                enroll_capturing = True
                                print(f"[Enroll] Capturing frames for '{name}' …")
                            else:
                                input_error = "Name cannot be empty"
                            input_mode = _MODE_IDLE
                            input_text = ""

                    elif key == 27:  # Escape → cancel
                        input_mode  = _MODE_IDLE
                        input_text  = ""
                        input_error = ""

                    elif key in (8, 127):  # Backspace
                        input_text = input_text[:-1]
                        input_error = ""

                    elif 32 <= key <= 126:  # Printable ASCII
                        input_text += chr(key)
                        input_error = ""

                else:
                    # ── Hotkeys (idle mode) ───────────────────────────────────
                    if key in (ord("q"), 27):   # Q or Esc
                        break
                    elif key == ord("t") or key == ord("T"):
                        input_mode  = _MODE_CHAT
                        input_text  = ""
                        input_error = ""
                    elif key == ord("e") or key == ord("E"):
                        input_mode  = _MODE_ENROLL
                        input_text  = ""
                        input_error = ""
                    elif key in (ord("k"), ord("K")):
                        _dump_kg(self.store, self.robot_id)
                    elif key in (ord("x"), ord("X")):
                        self._extract_session()   # run extraction mid-session (testing)
                    elif key in (ord("c"), ord("C")):
                        self._consolidate_preview()   # dry-run: preview topic merges
                    elif key in (ord("b"), ord("B")) and last_person_id:
                        _update_rapport_trust(
                            self.store, last_person_id, self.robot_id,
                            delta=0.15, verbose=True,
                        )
                        if self.kg_path:
                            self.store.save(self.kg_path)
                    elif key in (ord("s"), ord("S")):
                        self.face_id.save(self.faces_path)

        except KeyboardInterrupt:
            print("\n[WebcamLoop] Interrupted.")
        finally:
            worker.stop()
            worker.join(timeout=2.0)
            if self.face_id.known_people():
                self.face_id.save(self.faces_path)
            # End-of-session knowledge extraction → update the graph.
            try:
                self._extract_session()
            except Exception as exc:  # noqa: BLE001 — never fail on shutdown
                print(f"[WebcamLoop] extraction failed: {exc}")
            self.store.save(self.kg_path)
            self._session_store.close()
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone enroll mode (--mode enroll CLI shortcut)
# ─────────────────────────────────────────────────────────────────────────────

def run_enroll_mode(name: str, faces_path: str,
                    camera_index: int, n_captures: int, threshold: float) -> None:
    fi = FaceIdentifier(threshold=threshold)
    if os.path.exists(faces_path):
        fi.load(faces_path)
    ok = fi.enroll_from_camera(name, camera_index=camera_index, n_captures=n_captures)
    if ok:
        fi.save(faces_path)
        print(f"Done. Known people: {fi.known_people()}")
    else:
        print("Enrollment failed — no faces captured.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone topic consolidation (--mode consolidate) — Feature 2
# ─────────────────────────────────────────────────────────────────────────────

def run_consolidate_mode(kg_path: str, embed_model: str,
                         merge_floor: float, dry_run: bool) -> None:
    """Merge near-duplicate topics in an existing KG by embedding similarity.
    Reviewable + deterministic; run with --dry-run first to preview."""
    store = InMemoryGraphStore()
    if not (os.path.exists(kg_path) and store.load(kg_path)):
        print(f"[consolidate] no KG at '{kg_path}'")
        return
    try:
        from modules.graph_relationship.embedding import ollama_embed_fn
        embed_fn = ollama_embed_fn(model=embed_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[consolidate] embeddings unavailable ({exc}) — cannot consolidate")
        return
    from modules.kg_extraction import (
        consolidate_topics, consolidate_interests, link_related_topics,
    )
    merges = (consolidate_topics(store, embed_fn, floor=merge_floor, dry_run=dry_run)["merges"]
              + consolidate_interests(store, embed_fn, floor=merge_floor, dry_run=dry_run)["merges"])
    links = link_related_topics(store, embed_fn, merge_floor=merge_floor, dry_run=dry_run)["links"]
    if not merges and not links:
        print(f"[consolidate] no near-duplicate or related topics/interests (floor {merge_floor})")
        return
    tag = "DRY RUN — no changes written" if dry_run else "APPLIED"
    print(f"[consolidate] {len(merges)} merge(s), {len(links)} related-link(s) — {tag}:")
    for canon, dup in merges:
        print(f"    merge  '{dup}'  →  '{canon}'")
    for a, b, sim in links:
        print(f"    link   '{a}' ~ '{b}'  ({sim})")
    if not dry_run:
        store.save(kg_path)
        print(f"[consolidate] saved → {kg_path}")


def run_migrate_sessions(kg_path: str, sessions_db: str) -> None:
    """One-off: move existing graph SessionNodes' transcripts into the SQLite store,
    then remove the SessionNodes (+ has_session edges) from the graph. Migrated turns
    are marked extracted (they already shaped the graph). interaction_count is
    refreshed from the store."""
    store = InMemoryGraphStore()
    if not (os.path.exists(kg_path) and store.load(kg_path)):
        print(f"[migrate] no KG at '{kg_path}'")
        return
    from modules.graph_relationship.interactions import set_interaction_count
    ss = SessionStore(sessions_db)
    sessions = [n for n in list(store._nodes.values()) if n.node_type == "session"]
    if not sessions:
        print("[migrate] no SessionNodes in the graph — nothing to move")
        ss.close()
        return
    moved_turns = 0
    pairs: set = set()
    for sess in sessions:
        # The interaction is the source of the has_session edge into this session.
        inter_id = None
        for edge, _nbr in store.query_neighbors(sess.id, "has_session"):
            if edge.target_id == sess.id:
                inter_id = edge.source_id
                break
        person = robot = None
        if inter_id and inter_id.startswith("interaction:"):
            parts = inter_id.split(":", 2)   # ['interaction', person, robot]
            if len(parts) == 3:
                person, robot = parts[1], parts[2]
        for t in (sess.turns or []):
            ss.append_turn(session_id=sess.id, person_id=person or "unknown",
                           robot_id=robot or "chatbox",
                           emotion=t.get("emotion"), child=t.get("child"),
                           reply=t.get("reply"))
            moved_turns += 1
        if person and robot:
            pairs.add((person, robot))
        store.delete_node(sess.id)   # also drops the has_session edge
    # Old history already shaped the graph → don't re-extract it.
    for person, _robot in pairs:
        ss.mark_extracted(person)
    for person, robot in pairs:
        set_interaction_count(store, person, robot,
                              ss.person_turn_count(person, robot), source="migrate")
    store.save(kg_path)
    ss.close()
    print(f"[migrate] moved {moved_turns} turn(s) from {len(sessions)} session(s) → "
          f"{sessions_db}; removed SessionNodes from the graph → {kg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Live webcam → KG relationship engine → PAD persona loop"
    )
    p.add_argument("--mode",
                   choices=["run", "enroll", "consolidate", "migrate-sessions"],
                   default="run",
                   help="run | enroll | consolidate (merge near-dup topics) | "
                        "migrate-sessions (move graph SessionNodes → SQLite)")
    p.add_argument("--sessions-db", default=_DEFAULT_SESSIONS_DB,
                   help=f"SQLite transcript DB path (default: {_DEFAULT_SESSIONS_DB})")
    p.add_argument("--name",       default=None,
                   help="Person name for enroll mode")
    p.add_argument("--faces",      default=_DEFAULT_FACES,
                   help=f"Face DB .npz path (default: {_DEFAULT_FACES})")
    p.add_argument("--kg",         default=_DEFAULT_KG,
                   help=f"KG state JSON path (default: {_DEFAULT_KG})")
    p.add_argument("--robot",      default=_DEFAULT_ROBOT,
                   choices=["chatbox", "ellebot"])
    p.add_argument("--camera",     type=int,   default=_DEFAULT_CAMERA)
    p.add_argument("--threshold",  type=float, default=_DEFAULT_THRESH,
                   help="Cosine similarity threshold (default: 0.75)")
    p.add_argument("--tick",       type=float, default=_DEFAULT_TICK,
                   help="Pipeline tick interval in seconds (default: 1.0)")
    p.add_argument("--n-captures", type=int,   default=15,
                   help="Frames to capture in enroll mode (default: 15)")
    p.add_argument("--llm",        action="store_true",
                   help="Enable LLM verbal responses via Ollama")
    p.add_argument("--model",      default=_DEFAULT_MODEL,
                   help=f"Ollama model (default: {_DEFAULT_MODEL})")
    p.add_argument("--emotion",    default="hsemotion",
                   choices=["hsemotion", "hsemotion-b2", "efficientnet"],
                   help="Emotion detection backend (default: hsemotion)")
    p.add_argument("--no-window",  action="store_true",
                   help="Headless — terminal output only")
    p.add_argument("--esp32-host", default="",
                   help="ESP32 IP address for TCP expression dispatch (blank=disabled)")
    p.add_argument("--esp32-port", type=int, default=8888,
                   help="ESP32 TCP port (default: 8888)")
    # ── KG integration options ────────────────────────────────────────────────
    p.add_argument("--no-seed", action="store_true",
                   help="Do not seed the KG from spec files on startup")
    p.add_argument("--spec-dir", default=None,
                   help="Directory of KG spec YAML/JSON files "
                        "(default: modules/graph_relationship/specs)")
    p.add_argument("--no-embed", action="store_true",
                   help="Use the keyword matcher instead of embeddings for "
                        "topic↔capability linking during extraction")
    p.add_argument("--embed-model", default="nomic-embed-text",
                   help="Ollama embedding model (default: nomic-embed-text)")
    p.add_argument("--embed-floor", type=float, default=0.62,
                   help="Min cosine similarity to link topic↔capability (default: 0.62). "
                        "Higher = fewer, stricter links (avoids e.g. tennis↔'good at math')")
    p.add_argument("--enable-pad", action="store_true",
                   help="Enable the PAD persona engine (disabled by default this pass)")
    p.add_argument("--enable-emotion", action="store_true",
                   help="Enable emotion detection (disabled by default this pass)")
    # ── Feature 2: topic consolidation (--mode consolidate) ────────────────────
    p.add_argument("--merge-floor", type=float, default=0.86,
                   help="Min cosine similarity to MERGE two near-duplicate topics "
                        "(default: 0.86; same-category only)")
    p.add_argument("--dry-run", action="store_true",
                   help="consolidate mode: preview merges without writing")
    args = p.parse_args()

    if args.mode == "enroll":
        if not args.name:
            p.error("--name is required for enroll mode")
        run_enroll_mode(args.name, args.faces, args.camera,
                        args.n_captures, args.threshold)
        return

    if args.mode == "consolidate":
        run_consolidate_mode(args.kg, args.embed_model, args.merge_floor, args.dry_run)
        return

    if args.mode == "migrate-sessions":
        run_migrate_sessions(args.kg, args.sessions_db)
        return

    llm = None
    if args.llm:
        llm = LLMClient(model=args.model)
        llm.connect()

    # Embedding matcher (default when --llm); degrades gracefully on failure.
    # embed_fn is also handed to the loop for the on-demand topic-merge preview (C).
    matcher = None
    embed_fn = None
    if not args.no_embed:
        try:
            from modules.graph_relationship.embedding import (
                make_embedding_matcher, ollama_embed_fn,
            )
            embed_fn = ollama_embed_fn(model=args.embed_model)
            matcher = make_embedding_matcher(embed_fn, floor=args.embed_floor)
            print(f"[WebcamLoop] topic matching via embeddings "
                  f"({args.embed_model}, floor {args.embed_floor})")
        except Exception as exc:  # noqa: BLE001
            print(f"[WebcamLoop] embedding matcher unavailable ({exc}) — "
                  "using keyword matcher")
            matcher = None
            embed_fn = None

    loop = WebcamKGLoop(
        robot_id         = args.robot,
        faces_path       = args.faces,
        kg_path          = args.kg,
        threshold        = args.threshold,
        tick_interval    = args.tick,
        llm_client       = llm,
        show_window      = not args.no_window,
        emotion_backend  = args.emotion,
        esp32_host       = args.esp32_host,
        esp32_port       = args.esp32_port,
        spec_dir         = args.spec_dir,
        seed             = not args.no_seed,
        matcher          = matcher,
        embed_fn         = embed_fn,
        sessions_db      = args.sessions_db,
        pad_enabled      = args.enable_pad,
        emotion_enabled  = args.enable_emotion,
    )
    loop.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
