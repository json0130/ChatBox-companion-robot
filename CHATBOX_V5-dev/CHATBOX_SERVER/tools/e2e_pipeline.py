"""
e2e_pipeline.py — the whole chain, once, with every stage printed.

    webcam frame
      -> face recognition            who is this?
      -> emotion model               valence / arousal
      -> knowledge graph             rapport+trust+count -> tier -> DOMINANCE
      -> PAD                         three influences -> one coordinate
      -> three words + five style values
      -> system prompt (KG memory + PAD manner) -> Ollama -> verbal reply
      -> robot client                STYLE line, then the gesture tag
      -> knowledge graph             conversation history + closeness written back

Every stage uses the REAL component — the same FaceIdentifier, EmotionDetector,
KGBridge, PAD adapter, prompt builder and LLM client the live loop uses. Nothing
here reimplements the pipeline; it drives it and narrates it, so a failure lands
in the same place it would during a real conversation.

The robot is stood in by a local socket that captures the bytes that would have
crossed the wire, so the servo half is verified even when the ESP32 is off. Point
it at real hardware with --esp32-host.

    python3 -m tools.e2e_pipeline --message "hey, what have you been up to?"
    python3 -m tools.e2e_pipeline --camera 1 --turns 3
    python3 -m tools.e2e_pipeline --no-camera --emotion sad     # inject an emotion
    python3 -m tools.e2e_pipeline --esp32-host 10.42.0.100      # drive the robot
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules.affect_bridge import affect  # noqa: E402
from modules.graph_relationship.interactions import get_interaction  # noqa: E402
from modules.graph_relationship.kg_bridge import derive_tier  # noqa: E402

_RULE = "─" * 78


def _stage(n: int, title: str) -> None:
    print(f"\n{_RULE}\n {n}. {title}\n{_RULE}")


class _RobotStub:
    """Captures what the robot client would have received."""

    def __init__(self):
        self.lines: list[str] = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        self._srv.listen(16)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self._srv.accept()
                with c:
                    data = c.recv(512).decode(errors="replace").strip()
                    if data:
                        self.lines.append(data)
            except OSError:
                return

    def close(self):
        self._srv.close()


def _grab_frame(camera: int, warmup: int = 8):
    """One frame from the camera, after letting auto-exposure settle."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        return None
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    return frame


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="chatbox")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-camera", action="store_true",
                   help="skip capture; use --emotion and --person instead")
    p.add_argument("--emotion", default=None,
                   help="force an emotion label instead of reading a face")
    p.add_argument("--person", default=None,
                   help="force the person id instead of recognising a face")
    p.add_argument("--message", default="hey, what have you been up to?")
    p.add_argument("--turns", type=int, default=1)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--esp32-host", default=None,
                   help="real robot; default is a local capture socket")
    p.add_argument("--kg", default=None, help="graph to use (default: a temp copy)")
    p.add_argument("--faces", default="faces.npz")
    args = p.parse_args(argv)

    from modules.face_webcam.webcam_loop import (
        LLMClient, WebcamKGLoop, _TAG_TO_ESP32,
    )

    stub = None
    host, port = args.esp32_host, 8888
    if host is None:
        stub = _RobotStub()
        host, port = "127.0.0.1", stub.port

    # Work on a COPY of the live graph unless one is named, so a test run can
    # never corrupt the real relationship history.
    import shutil
    tmp = tempfile.mkdtemp(prefix="e2e_")
    kg_path = args.kg
    if kg_path is None:
        kg_path = os.path.join(tmp, "kg.json")
        if os.path.exists("kg_state.json"):
            shutil.copy2("kg_state.json", kg_path)
    # The graph's interaction_count is SET from the transcript store each turn, so
    # the two must be copied together — a fresh sessions.db beside a real graph
    # would reset everyone's count (and with it their tier) on the first write.
    sessions_db = os.path.join(tmp, "sessions.db")
    if os.path.exists("sessions.db"):
        shutil.copy2("sessions.db", sessions_db)

    print(f"\n{'='*78}\n  END-TO-END PIPELINE — {args.robot}\n{'='*78}")
    print(f"  graph  {kg_path}{'  (copy of kg_state.json)' if args.kg is None else ''}")
    print(f"  faces  {args.faces}")
    print(f"  robot  {host}:{port}{'  (local capture socket)' if stub else '  (REAL)'}")

    llm = LLMClient(model=args.model)
    llm.connect()          # LLMClient is lazy: nothing works until this runs
    print(f"  llm    {args.model}  {'available' if llm.available else 'UNAVAILABLE'}")

    loop = WebcamKGLoop(
        robot_id=args.robot, show_window=False, seed=True, llm_client=llm,
        kg_path=kg_path, faces_path=args.faces,
        sessions_db=sessions_db, embed_fn=None,
        pad_enabled=True, emotion_enabled=True,
        esp32_host=host, esp32_port=port,
    )

    # ── 1. capture ────────────────────────────────────────────────────────────
    frame = None
    if not args.no_camera:
        _stage(1, "CAPTURE — webcam")
        frame = _grab_frame(args.camera)
        if frame is None:
            print(f"  camera {args.camera} unavailable — continuing without it")
        else:
            print(f"  frame {frame.shape[1]}x{frame.shape[0]}")
    else:
        _stage(1, "CAPTURE — skipped (--no-camera)")

    # ── 2. face recognition ───────────────────────────────────────────────────
    _stage(2, "FACE RECOGNITION — who is this?")
    pid, sim, box = args.person, 1.0, None
    if frame is not None and args.person is None:
        dets = loop.face_id.identify_all(frame, max_faces=4, scale=0.5)
        print(f"  faces detected: {len(dets)}   known: {loop.face_id.known_people()}")
        for who, s, b in dets:
            na, nd = loop.face_id.gallery_size(who) if who else (0, 0)
            print(f"    {str(who):>10}  sim={s:.3f}  box={b}"
                  + (f"  ({na} enrolled + {nd} learned views)" if who else "  (unknown)"))
        if dets:
            pid, sim, box = dets[0]
        if pid is None:
            pid = loop._auto_enroll(frame)
            print(f"  unknown face -> auto-enrolled as {pid!r}")
    if pid is None:
        pid = "jay"
        print(f"  no face available — using {pid!r}")
    print(f"  => PERSON = {pid!r}")

    # ── 3. emotion ────────────────────────────────────────────────────────────
    _stage(3, "EMOTION — valence / arousal")
    if args.emotion:
        emo, e_conf = args.emotion, 100.0
        v, a = affect.category_to_va(emo)
        print(f"  forced: {emo}  ->  V={v:+.2f}  A={a:+.2f}")
    elif frame is not None and box is not None:
        from modules.face_webcam.emotion_detector import EmotionDetector
        det = EmotionDetector.create("hsemotion")
        emo, e_conf, v, a = det.detect(frame, box=box, smooth=False)
        print(f"  hsemotion: {emo} ({e_conf:.0f}%)  ->  V={v:+.2f}  A={a:+.2f}")
        print("  (V/A are a weighted blend over the full 8-class softmax)")
    else:
        emo, e_conf = "neutral", 0.0
        v, a = 0.0, 0.0
        print("  no face crop — neutral")
    print(f"  => EMOTION = {emo}  V={v:+.3f}  A={a:+.3f}")

    for turn in range(1, args.turns + 1):
        if args.turns > 1:
            print(f"\n\n{'#'*78}\n#  TURN {turn} of {args.turns}\n{'#'*78}")

        # ── 4. knowledge graph -> tier -> Dominance ───────────────────────────
        _stage(4, "KNOWLEDGE GRAPH — relationship -> Dominance")
        inter = get_interaction(loop.store, pid, args.robot)
        if inter is None:
            print(f"  no InteractionNode for ({pid}, {args.robot}) yet")
            r = t = 0.0
            c = 0
        else:
            r, t, c = inter.rapport, inter.trust, inter.interaction_count
        tier = derive_tier(pid, args.robot, loop.store)
        print(f"  rapport={r:.3f}  trust={t:.3f}  interactions={c}")
        print(f"  score=(rapport+trust)/2 = {(r+t)/2:.3f}")
        print(f"  => TIER = {tier!r}   -> Dominance offset "
              f"{affect.tier_offset(tier)[2]:+.2f}")

        # ── 5. PAD ────────────────────────────────────────────────────────────
        _stage(5, "PAD — persona + face + relationship")
        bi, pad = loop._pipeline_tick(pid, emo, va=(v, a))
        base = affect.to_pad(affect.ROBOTS[args.robot.upper()]["ocean"])
        fp, fa, fd = pad["pad_state"]
        print(f"  persona OCEAN      {affect.ROBOTS[args.robot.upper()]['ocean']}")
        print(f"  baseline PAD       P={base['P']:+.3f} Ar={base['Ar']:+.3f} D={base['D']:+.3f}")
        print(f"  + face (empathy {affect.EMPATHY})   moves P and Ar only")
        print(f"  + tier {tier!r:9}   moves D only")
        print(f"  = felt PAD         P={fp:+.3f} Ar={fa:+.3f} D={fd:+.3f}")
        sp, sa, sd = pad["shown"]
        print(f"  x show {affect.ROBOTS[args.robot.upper()]['show']:<4}        "
              f"P={sp:+.3f} Ar={sa:+.3f} D={sd:+.3f}   ({pad['name']})")
        print(f"  => WORDS  {' / '.join(pad['words'])}")
        st = pad["style"]
        print(f"  => STYLE  amplitude={st['amplitude']:.3f} tempo={st['tempo']:.3f} "
              f"posture={st['posture']:+.3f} droop={st['droop']:+.3f} idle={st['idle']:.2f}")

        # ── 6. prompt ─────────────────────────────────────────────────────────
        _stage(6, "SYSTEM PROMPT — KG memory + PAD manner")
        prompt = loop._build_system_prompt(pid, rag_hits=[], pad=pad)
        for block in ("IDENTITY", "HOW TO REPLY", "WHO YOU'RE TALKING TO"):
            print(f"  [{'x' if block in prompt else ' '}] {block}")
        for line in prompt.splitlines():
            if "manner right now" in line or line.startswith(("You know", "You have met", "You do not",
                                                              "You are meeting", "You recognise")):
                print(f"  PAD-> {line.strip()[:110]}")
        print(f"  prompt is {len(prompt)} chars")

        # ── 7. LLM ────────────────────────────────────────────────────────────
        _stage(7, f"OLLAMA — {args.model}")
        if not llm.available:
            print("  unavailable; skipping")
            tag, verbal = "", ""
        else:
            t0 = time.time()
            raw = llm.respond(prompt, args.message)
            from modules.face_webcam.webcam_loop import _parse_llm_response
            tag, verbal = _parse_llm_response(raw)
            print(f"  you   : {args.message!r}")
            print(f"  robot : [{tag}] {verbal!r}")
            print(f"  {int((time.time()-t0)*1000)} ms")

        # ── 8. robot client ───────────────────────────────────────────────────
        _stage(8, "ROBOT CLIENT — servo style + gesture")
        before = len(stub.lines) if stub else 0
        sent = loop._maybe_send_style(st, force=True)
        expr = _TAG_TO_ESP32.get(tag) if tag else None
        if expr:
            from modules.face_webcam.webcam_loop import _send_esp32
            _send_esp32(expr, host, port)
        time.sleep(0.25)
        if stub:
            for line in stub.lines[before:]:
                kind = "style " if line.startswith("STYLE") else "gesture"
                print(f"  -> {kind}: {line}")
            if not stub.lines[before:]:
                print("  (nothing sent)")
        print(f"  style sent: {sent}   tag: {tag or '—'} -> {expr or '—'}")

        # ── 9. KG write-back ──────────────────────────────────────────────────
        _stage(9, "KNOWLEDGE GRAPH — write-back")
        if verbal:
            loop._apply_chat_result({"pid": pid, "msg": args.message,
                                     "verbal": verbal, "tag": tag, "emotion": emo})
        after = get_interaction(loop.store, pid, args.robot)
        if after is not None:
            print(f"  rapport {r:.3f} -> {after.rapport:.3f}   "
                  f"trust {t:.3f} -> {after.trust:.3f}   "
                  f"interactions {c} -> {after.interaction_count}")
            print(f"  tier now: {derive_tier(pid, args.robot, loop.store)!r}")
        conv = loop.store.get_node(f"conversation:{pid}:{args.robot}")
        if conv is not None:
            print(f"  conversation node: topics={getattr(conv,'topics',[])[-3:]} "
                  f"mood={getattr(conv,'mood',None)} emotion={getattr(conv,'emotion',None)}")
        rows = loop._session_store.recent_turns(pid, limit=3)
        print(f"  transcript rows for {pid}: {len(rows)}")
        for row in rows[-2:]:
            print(f"    - {str(row.get('child'))[:60]!r}")

    loop._flush_kg(force=True)
    loop._session_store.close()
    print(f"\n{_RULE}\n  graph saved -> {kg_path}")
    if stub:
        print(f"  robot received {len(stub.lines)} line(s) in total")
        stub.close()
    print(_RULE + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
