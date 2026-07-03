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
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
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
    from modules.graph_relationship.store import _PERSON_ATTRIBUTE_TYPES, _RELATIONSHIP_TYPES
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
    ):
        super().__init__(daemon=True, name="detection-worker")
        self._face_id         = face_id
        self._emotion_backend = emotion_backend
        self._max_faces       = max_faces
        self._det_scale       = det_scale

        # Per-person smoothers (only ever touched from this thread — no lock needed)
        self._per_emotion: dict[str, EmotionDetector] = {}
        self._unknown_emotion = EmotionDetector.create(emotion_backend)

        self._lock    = threading.Lock()
        self._frame   = None
        self._results: list = []
        self._event   = threading.Event()
        self._stop    = threading.Event()

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
        self._stop.set()
        self._event.set()  # unblock the wait

    # ── Worker loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()
            if self._stop.is_set():
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
                if person_id is not None:
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

        self.bridge  = KGBridge(self.store)
        self._adapters: dict[str, PADPipelineAdapter] = {}

        self._emotion_backend = emotion_backend
        self._esp32_host      = esp32_host
        self._esp32_port      = esp32_port

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
        print(f"\n[WebcamLoop] robot={self._robot_display}  tick={self.tick_interval}s  cam={camera_index}  {llm_status}")
        print("  T=chat  E=enroll  B=boost  K=dump KG  S=save  Q=quit  (all in the OpenCV window)\n")

        # ── Background detection worker ───────────────────────────────────────
        worker = _DetectionWorker(
            self.face_id,
            emotion_backend=self._emotion_backend,
            max_faces=4,
            det_scale=0.5,
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

                    for d in raw_dets:
                        pid = d["person_id"]
                        if pid is None:
                            continue
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

                    # Persist after the tick so the live viz server
                    # (modules.graph_relationship.viz.server) can poll it within ~1s.
                    if self.kg_path and any(d["person_id"] for d in raw_dets):
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
                                if self._last_pad_result and self.llm and self.llm.available:
                                    hist = list(self._chat_history.get(
                                        last_person_id or "", []
                                    ))
                                    raw_reply = self.llm.respond(
                                        self._last_pad_result["system_prompt"],
                                        msg,
                                        history=hist,
                                    )
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
                                elif not (self.llm and self.llm.available):
                                    last_verbal = "[LLM not enabled — run with --llm]"
                                    print("  [chat] LLM not connected. Run with --llm.\n")
                                else:
                                    last_verbal = "[Waiting for face recognition …]"
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
            self.store.save(self.kg_path)
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Live webcam → KG relationship engine → PAD persona loop"
    )
    p.add_argument("--mode",       choices=["run", "enroll"], default="run")
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
    args = p.parse_args()

    if args.mode == "enroll":
        if not args.name:
            p.error("--name is required for enroll mode")
        run_enroll_mode(args.name, args.faces, args.camera,
                        args.n_captures, args.threshold)
        return

    llm = None
    if args.llm:
        llm = LLMClient(model=args.model)
        llm.connect()

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
    )
    loop.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
