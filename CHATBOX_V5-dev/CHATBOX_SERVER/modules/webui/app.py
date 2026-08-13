"""
webui — webcam, chat, PAD state and the knowledge graph in one browser page.

Runs the SAME pipeline the OpenCV loop runs (same FaceIdentifier, EmotionDetector,
KGBridge, PAD adapter, prompt builder, LLM client, servo wire) and serves it over
HTTP instead of drawing a cv2 window. One process, so the page can show live
frames, the affect state behind them, the graph they are writing to, and the
conversation — without the file-passing dance a separate visualiser needs.

    python3 -m modules.webui --llm --enable-emotion
    python3 -m modules.webui --llm --enable-emotion --robot ellebot --port 8100

Endpoints
    GET  /               the page
    GET  /stream.mjpg    annotated camera frames
    GET  /state.json     person, emotion, V/A, tier, PAD, words, style, wire
    GET  /graph.json     the knowledge graph, for the force layout
    POST /chat           {"message": "..."} -> queued; reply appears in /state.json
    POST /robot          {"robot": "chatbox"|"ellebot"} -> switch persona live

Switching persona switches the WHOLE relationship: rapport, trust and turn count
live on the (person, robot) InteractionNode, so ElleBot starts as a stranger to
someone ChatBox knows well. That is the point rather than an inconvenience — it
is the cleanest way to see tier -> Dominance -> amplitude -> wording move.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules.affect_bridge import affect  # noqa: E402
from modules.graph_relationship.kg_bridge import derive_tier  # noqa: E402
from modules.graph_relationship.schema import Embodiment, RobotNode  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.join(_HERE, "index.html")

ROBOTS = ("chatbox", "ellebot")


class LiveState:
    """Everything the page polls, guarded by one lock."""

    def __init__(self, robot: str):
        self._lock = threading.Lock()
        self.frame_jpeg: bytes | None = None
        self.data: dict = {
            "robot": robot, "person": None, "sim": 0.0, "faces": 0,
            "emotion": "neutral", "valence": 0.0, "arousal": 0.0,
            "tier": "unknown", "rapport": 0.0, "trust": 0.0, "interactions": 0,
            "pad": None, "shown": None, "words": None, "affect_name": None,
            "style": None, "wire": None,
            "chat": [], "thinking": False, "fps": 0.0,
            "known_people": [], "ocean": {},
        }

    def update(self, **kw):
        with self._lock:
            self.data.update(kw)

    def push_chat(self, who: str, text: str, meta: dict | None = None):
        with self._lock:
            self.data["chat"] = (self.data["chat"] + [
                {"who": who, "text": text, "t": time.strftime("%H:%M:%S"),
                 **(meta or {})}])[-40:]

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.data, default=float))

    def set_frame(self, jpeg: bytes):
        with self._lock:
            self.frame_jpeg = jpeg

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self.frame_jpeg


def graph_json(store, robot_id: str) -> dict:
    """The graph as nodes+links, showing only the ACTIVE robot's relationship.

    Both robots keep their own InteractionNode with its own rapport, trust and
    turn count, so both sets of history survive a persona switch. But drawing
    both at once implies the person is simultaneously in two relationships, so
    the inactive robot's interaction/conversation subtree is left out and its
    node is returned unconnected. Nothing is deleted — this is a view.
    """
    others = {r for r in ROBOTS if r != robot_id}

    def owned_by_other(node_id: str) -> bool:
        return any(node_id.endswith(f":{o}") and node_id.startswith(
            ("interaction:", "conversation:")) for o in others)

    hidden = {n.id for n in store._nodes.values()          # noqa: SLF001
              if owned_by_other(n.id)}

    nodes = [{
        "id": n.id,
        "type": n.node_type,
        "label": getattr(n, "display_name", None) or getattr(n, "label", None)
                 or getattr(n, "name", None) or n.id.split(":")[-1],
        "active": n.id == robot_id,
        "idle_robot": n.node_type == "robot" and n.id != robot_id,
    } for n in store._nodes.values() if n.id not in hidden]   # noqa: SLF001

    links = [{
        "source": e.source_id, "target": e.target_id, "type": e.edge_type,
        "value": round(float(getattr(e, "weight", 0.0) or 0.0), 3),
    } for e in store._edges.values()                        # noqa: SLF001
        if e.source_id not in hidden and e.target_id not in hidden]

    return {"nodes": nodes, "links": links, "robot": robot_id,
            "sig": f"{robot_id}|{len(nodes)}|{len(links)}|"
                   f"{hash(frozenset(n['id'] for n in nodes)) & 0xffffff}"}


_TIER_COL = {                       # BGR
    "close":   (120, 220, 120), "family":  (140, 210, 130),
    "known":   (255, 170,  90), "visitor": (60, 190, 235),
    "unknown": (90,  90, 245),
}


def draw(frame, dets, state: dict):
    """Boxes plus the identity/relationship panel.

    Drawn here rather than in the page so the overlay travels WITH the frame —
    what you see is what the pipeline actually decided for that image.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for d in dets:
        box = d.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        known = d.get("person_id") is not None
        col = (90, 220, 120) if known else (120, 160, 250)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        sim = d.get("sim", 0.0)
        tag = f"{d.get('person_id') or '?'}"
        tag += " held" if sim < 0 else f" {sim:.2f}"
        cv2.putText(out, tag, (x1 + 3, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        emo = d.get("emotion")
        if emo:
            cv2.putText(out, f"{emo} {d.get('e_conf', 0):.0f}%",
                        (x1 + 3, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # ── identity + relationship panel, bottom-left ────────────────────────────
    tier = state.get("tier") or "unknown"
    person = state.get("person") or "—"
    words = state.get("words")
    ph, pw = 96, 330
    y0 = h - ph - 6
    panel = out[y0:y0 + ph, 6:6 + pw].copy()
    out[y0:y0 + ph, 6:6 + pw] = cv2.addWeighted(
        panel, 0.25, np.zeros_like(panel), 0.75, 0)
    cv2.rectangle(out, (6, y0), (6 + pw, y0 + ph), (60, 70, 85), 1)

    tc = _TIER_COL.get(tier, (150, 150, 150))
    cv2.putText(out, f"{state.get('robot','').upper()}  <->  {person}",
                (14, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
    cv2.putText(out, tier.upper(), (14, y0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, tc, 2)
    off = affect.tier_offset(tier)[2]
    cv2.putText(out, f"D {off:+.2f}", (110, y0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, tc, 1)

    # rapport / trust bars
    for i, (lbl, val) in enumerate((("R", state.get("rapport") or 0.0),
                                    ("T", state.get("trust") or 0.0))):
        bx, by = 14 + i * 160, y0 + 60
        cv2.putText(out, f"{lbl} {val:.2f}", (bx, by + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1)
        cv2.rectangle(out, (bx + 52, by), (bx + 132, by + 9), (55, 60, 70), -1)
        cv2.rectangle(out, (bx + 52, by), (bx + 52 + int(80 * max(0.0, min(1.0, val))),
                                           by + 9), tc, -1)

    if words:
        cv2.putText(out, " / ".join(words), (14, y0 + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 255), 1)
    turns = state.get("interactions") or 0
    cv2.putText(out, f"{turns} turns", (pw - 66, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 155, 165), 1)
    return out


def make_handler(state: LiveState, loop, chat_q, switch_robot):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                  # keep the console for the loop
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                with open(_INDEX, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif path == "/state.json":
                self._send(200, json.dumps(state.snapshot()).encode(),
                           "application/json")
            elif path == "/graph.json":
                with loop._store_lock:              # noqa: SLF001
                    body = json.dumps(graph_json(loop.store, loop.robot_id))
                self._send(200, body.encode(), "application/json")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        jpg = state.get_frame()
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpg)}\r\n\r\n"
                                               .encode() + jpg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if path == "/chat":
                msg = str(body.get("message", "")).strip()
                if msg:
                    chat_q(msg)
                self._send(200, b'{"ok":true}', "application/json")
            elif path == "/robot":
                r = str(body.get("robot", "")).lower()
                ok = r in ROBOTS and switch_robot(r)
                self._send(200, json.dumps({"ok": bool(ok), "robot": loop.robot_id})
                           .encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
    return H


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="chatbox", choices=list(ROBOTS))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--kg", default="kg_state.json")
    p.add_argument("--faces", default="faces.npz")
    p.add_argument("--sessions-db", default="sessions.db")
    p.add_argument("--llm", action="store_true", help="connect Ollama")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--enable-emotion", action="store_true")
    p.add_argument("--esp32-host", default="")
    p.add_argument("--esp32-port", type=int, default=8888)
    args = p.parse_args(argv)

    from modules.face_webcam.webcam_loop import (
        LLMClient, WebcamKGLoop, _DetectionWorker, _parse_llm_response,
    )

    llm = None
    if args.llm:
        llm = LLMClient(model=args.model)
        llm.connect()

    loop = WebcamKGLoop(
        robot_id=args.robot, show_window=False, seed=True, llm_client=llm,
        kg_path=args.kg, faces_path=args.faces, sessions_db=args.sessions_db,
        embed_fn=None, pad_enabled=True, emotion_enabled=args.enable_emotion,
        esp32_host=args.esp32_host, esp32_port=args.esp32_port,
    )
    state = LiveState(args.robot)
    state.update(known_people=loop.face_id.known_people(),
                 ocean=affect.ROBOTS[args.robot.upper()]["ocean"])

    def ensure_robot_node(rid: str):
        with loop._store_lock:                       # noqa: SLF001
            if loop.store.get_node(rid) is None:
                loop.store.upsert_node(RobotNode(
                    id=rid, name=rid,
                    embodiment=Embodiment.ELEPHANT if rid == "ellebot"
                    else Embodiment.CAT))
    ensure_robot_node(args.robot)

    def switch_robot(rid: str) -> bool:
        """Swap persona live. The relationship swaps with it: rapport/trust/count
        live on the (person, robot) InteractionNode, so the tier is re-derived
        against the new robot and Dominance moves with it."""
        ensure_robot_node(rid)
        loop.robot_id = rid
        loop._robot_display = rid.capitalize()       # noqa: SLF001
        loop._last_pad_result = None                 # noqa: SLF001
        state.update(robot=rid, ocean=affect.ROBOTS[rid.upper()]["ocean"])
        state.push_chat("system", f"— now talking to {rid.upper()} —")
        print(f"[webui] persona -> {rid}")
        return True

    pending: list[str] = []
    plock = threading.Lock()

    def enqueue(msg: str):
        state.push_chat("you", msg)
        state.update(thinking=True)
        with plock:
            pending.append(msg)

    worker = _DetectionWorker(
        loop.face_id, emotion_backend="hsemotion", max_faces=4, det_scale=0.5,
        detect_emotion=args.enable_emotion,
        id_check_interval=loop._id_check_interval,   # noqa: SLF001
        id_sample_window=loop._id_sample_window,     # noqa: SLF001
        id_confirm_ratio=loop._id_confirm_ratio,     # noqa: SLF001
    )
    worker.start()

    handler = make_handler(state, loop, enqueue, switch_robot)
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"\n  webui → http://localhost:{args.port}\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[webui] cannot open camera {args.camera}")
        return 1

    last_tick, fps_t, frames = 0.0, time.time(), 0
    overlay = state.snapshot()   # refreshed on the 1 Hz tick, not per frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frames += 1
            worker.submit(frame)
            dets = worker.get_results()

            annotated = draw(frame, dets, overlay)
            ok2, buf = cv2.imencode(".jpg", annotated,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok2:
                state.set_frame(buf.tobytes())

            if time.time() - fps_t >= 1.0:
                state.update(fps=round(frames / (time.time() - fps_t), 1))
                frames, fps_t = 0, time.time()

            primary = dets[0] if dets else None
            pid = primary["person_id"] if primary else None

            # ── the affect tick, once a second ────────────────────────────────
            now = time.time()
            if now - last_tick >= 1.0:
                last_tick = now
                emo = (primary or {}).get("emotion", "neutral")
                va = (primary or {}).get("va") or (0.0, 0.0)
                upd = {"person": pid, "faces": len(dets),
                       "sim": (primary or {}).get("sim", 0.0),
                       "emotion": emo, "valence": va[0], "arousal": va[1],
                       "known_people": loop.face_id.known_people()}
                if pid:
                    with loop._store_lock:           # noqa: SLF001
                        bi, pad = loop._pipeline_tick(pid, emo, va=va)  # noqa: SLF001
                    loop._maybe_send_style(pad.get("style"))            # noqa: SLF001
                    loop._last_pad_result = pad                         # noqa: SLF001
                    from modules.graph_relationship.interactions import get_interaction
                    it = get_interaction(loop.store, pid, loop.robot_id)
                    upd.update({
                        "tier": bi.tier, "pad": pad["pad_state"],
                        "shown": pad["shown"], "words": pad["words"],
                        "affect_name": pad["name"], "style": pad["style"],
                        "wire": pad["wire"],
                        "rapport": it.rapport if it else 0.0,
                        "trust": it.trust if it else 0.0,
                        "interactions": it.interaction_count if it else 0,
                    })
                state.update(**upd)
                overlay = state.snapshot()
                loop._drain_adapt_events(worker)     # noqa: SLF001
                loop._flush_kg()                     # noqa: SLF001

            # ── chat ──────────────────────────────────────────────────────────
            with plock:
                msg = pending.pop(0) if pending else None
            if msg is not None:
                if loop.llm is None or not loop.llm.available:
                    state.push_chat("robot", "[no LLM — start with --llm]")
                    state.update(thinking=False)
                else:
                    with loop._store_lock:           # noqa: SLF001
                        prompt = loop._build_system_prompt(   # noqa: SLF001
                            pid, rag_hits=[], pad=loop._last_pad_result)
                    hist = list(loop._chat_history.get(pid or "", []))  # noqa: SLF001
                    raw = loop.llm.respond(prompt, msg, history=hist)
                    tag, verbal = _parse_llm_response(raw)
                    st = state.snapshot()
                    state.push_chat("robot", verbal or raw, {
                        "tag": tag, "tier": st["tier"], "words": st["words"],
                        "wire": st["wire"], "robot": loop.robot_id})
                    state.update(thinking=False)
                    with loop._store_lock:           # noqa: SLF001
                        loop._apply_chat_result({    # noqa: SLF001
                            "pid": pid, "msg": msg, "verbal": verbal,
                            "tag": tag, "emotion": state.snapshot()["emotion"]})
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[webui] stopping")
    finally:
        worker.stop()
        cap.release()
        loop._flush_kg(force=True)                   # noqa: SLF001
        loop._session_store.close()                  # noqa: SLF001
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
