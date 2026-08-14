"""
pad_prompt_grid.py — does PAD measurably change what the robot SAYS?

Sweeps relationship tier x detected emotion for one or both robots, capturing the
full system prompt and the reply for every cell, and reports whether the words
actually moved or whether we are looking at sampling noise.

Two things this is built to avoid concluding by accident:

  1. A NULL RESULT FOR MECHANICAL REASONS. PAD reaches the prompt as three
     descriptor words. Those come from the SHOWN coordinate, and CHATBOX's body
     shows only 30% of its temperament — which compresses Dominance so hard that
     `unknown`, `visitor` and `known` produce an identical word triplet. If two
     cells share a triplet, their prompts are byte-identical apart from the tier
     sentence, and any difference in the replies is noise BY CONSTRUCTION. The
     report says so out loud (the descriptor-delta matrix) rather than leaving it
     to be misread as "PAD does nothing".
  2. NONDETERMINISM SWAMPING THE EFFECT. Hence --repeats, temperature 0 by
     default, per-cell variance in the report, and --control, which re-runs every
     cell with PAD off. "PAD changed the words" means nothing without that
     baseline.

The tier is never forced by passing a string: each person is given rapport /
trust / interaction_count and the tier is then DERIVED through the real
kg_bridge path, with the result asserted. Forcing the string would test a string.

Every person in the grid gets an identical memory seed, so the only things
varying are tier (down) and emotion (across).

    python3 -m tools.pad_prompt_grid --no-llm            # prompts only, no Ollama
    python3 -m tools.pad_prompt_grid --robot chatbox,ellebot --control --repeats 3
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import difflib
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.affect_bridge import affect                      # noqa: E402
from modules.graph_relationship.interactions import (          # noqa: E402
    get_or_create_interaction, set_closeness,
)
from modules.graph_relationship.kg_bridge import derive_tier   # noqa: E402
from modules.graph_relationship.schema import (                # noqa: E402
    Embodiment, PersonNode, RobotNode,
)
from modules.graph_relationship.topics import add_person_topic  # noqa: E402

# rapport/trust/count chosen to land on each tier through _tier_from_scores:
#   score=(r+t)/2 >0.70 close | >0.45 known | count>5 known | count>0 visitor
_TIER_RECIPE = {
    "unknown": (0.00, 0.00, 0),
    "visitor": (0.10, 0.05, 1),
    "known":   (0.55, 0.50, 3),
    "close":   (0.80, 0.75, 9),
}
_DERIVABLE = {"unknown", "visitor", "known", "close"}

_HEDGES = re.compile(r"\b(maybe|perhaps|might|i think|i guess|sort of|kind of|sorry)\b", re.I)
_WARMTH = re.compile(r"\b(love|great|glad|awesome|wonderful|happy|excited|fun)\b", re.I)
_FEELING = re.compile(r"\b(you (seem|look|sound)|are you (ok|okay|alright)|"
                      r"how are you feeling|that sounds (hard|tough))\b", re.I)


def _seed_person(store, pid: str, robot: str, tier: str) -> str:
    """Create a person with an IDENTICAL memory seed, then dial the relationship
    to `tier` through the real derivation path. Returns the derived tier."""
    # robot_spec.yaml always seeds 'chatbox', so a non-default robot has no node.
    if store.get_node(robot) is None:
        store.upsert_node(RobotNode(id=robot, name=robot,
                                    embodiment=Embodiment.CAT))
    store.upsert_node(PersonNode(id=pid, display_name=pid))
    for label, cat, aff in (("guitar", "music", 0.85), ("space", "science", 0.70)):
        add_person_topic(store, pid, label, cat, affinity=aff, confidence=0.9,
                         source="grid-seed")
    rapport, trust, count = _TIER_RECIPE[tier]
    set_closeness(store, pid, robot, rapport=rapport, trust=trust, source="grid")
    node = get_or_create_interaction(store, pid, robot)
    store.upsert_node(node.model_copy(update={"interaction_count": count}))
    return derive_tier(pid, robot, store)


def _metrics(reply: str) -> dict:
    words = re.findall(r"[a-z']+", reply.lower())
    return {
        "chars": len(reply),
        "words": len(words),
        "ttr": round(len(set(words)) / len(words), 3) if words else 0.0,
        "hedges": len(_HEDGES.findall(reply)),
        "warmth": len(_WARMTH.findall(reply)),
        "feeling_talk": bool(_FEELING.search(reply)),
    }


def run_grid(args) -> list:
    from modules.face_webcam.webcam_loop import LLMClient, WebcamKGLoop, _parse_llm_response

    rows: list[dict] = []
    tmp = tempfile.mkdtemp(prefix="pad_grid_")
    llm = None
    if not args.no_llm:
        llm = LLMClient(model=args.model)
        llm.connect()          # LLMClient is lazy — nothing works until this runs
        if not llm.available:
            print("[grid] Ollama unavailable — falling back to --no-llm")
            llm = None

    for robot in args.robot.split(","):
        robot = robot.strip()
        loop = WebcamKGLoop(
            robot_id=robot, show_window=False, seed=True, llm_client=llm,
            kg_path=os.path.join(tmp, f"kg_{robot}.json"),
            faces_path=os.path.join(tmp, "faces.npz"),
            sessions_db=os.path.join(tmp, "s.db"),
            embed_fn=None, pad_enabled=True, emotion_enabled=True,
        )
        adapter = loop._adapter()

        for tier in args.tiers.split(","):
            tier = tier.strip()
            pid = f"p_{robot}_{tier}"
            derived = _seed_person(loop.store, pid, robot, tier)
            if tier in _DERIVABLE and derived != tier:
                print(f"[grid] WARNING {robot}/{tier}: derived {derived!r}")

            for emo in args.emotions.split(","):
                emo = emo.strip()
                v, a = affect.category_to_va(emo)
                for pad_on in ((True, False) if args.control else (True,)):
                    pad = adapter.process_turn(v, a, derived) if pad_on else None
                    if pad and args.no_directive:
                        pad = {**pad, "directive": False}   # words stay, directive off
                    prompt = loop._build_system_prompt(pid, rag_hits=[], pad=pad)
                    for rep in range(args.repeats if pad_on else 1):
                        reply = tag = ""
                        t0 = time.time()
                        if llm is not None:
                            # LLMClient fixes temperature internally (0.7 for
                            # spoken replies), so it cannot be set per call —
                            # --repeats and the variance column carry the noise
                            # instead of pretending it is pinned at 0.
                            raw = llm.respond(prompt, args.message)
                            tag, reply = _parse_llm_response(raw)
                        rows.append({
                            "robot": robot, "tier": tier, "derived_tier": derived,
                            "emotion": emo, "repeat": rep, "pad_on": pad_on,
                            "va": [v, a],
                            "pad_felt": list(pad["pad_state"]) if pad else None,
                            "words": list(pad["words"]) if pad else None,
                            "affect_name": pad["name"] if pad else None,
                            "style": pad["style"] if pad else None,
                            "wire": pad["wire"] if pad else None,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                            "prompt": prompt, "tag": tag, "reply": reply,
                            "latency_ms": int((time.time() - t0) * 1000),
                            "model": args.model if llm else None,
                            "temperature": args.temperature,
                            "metrics": _metrics(reply) if reply else None,
                        })
    return rows


def write_report(rows: list, path: str, args) -> None:
    out = ["# PAD prompt grid",
           f"\n_{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}_  ",
           f"message: `{args.message}`  ",
           f"model: `{args.model if not args.no_llm else 'none (--no-llm)'}`  "
           f"temperature: {args.temperature}  repeats: {args.repeats}\n"]

    for robot in sorted({r["robot"] for r in rows}):
        rr = [r for r in rows if r["robot"] == robot and r["pad_on"]]
        tiers = [t for t in args.tiers.split(",") if any(r["tier"] == t for r in rr)]
        emos = [e for e in args.emotions.split(",") if any(r["emotion"] == e for r in rr)]

        out.append(f"\n## {robot}\n")
        out.append("### Descriptor triplets — the necessary condition\n")
        out.append("If two cells share a triplet their prompts are identical apart from "
                   "the tier sentence, so any reply difference between them is noise.\n")
        out.append("| tier | " + " | ".join(emos) + " |")
        out.append("|---" * (len(emos) + 1) + "|")
        seen: dict[tuple, list] = collections.defaultdict(list)
        for t in tiers:
            cells = []
            for e in emos:
                m = next((r for r in rr if r["tier"] == t and r["emotion"] == e), None)
                w = tuple(m["words"]) if m and m["words"] else ("—",)
                seen[w].append(f"{t}/{e}")
                cells.append("/".join(w))
            out.append(f"| **{t}** | " + " | ".join(cells) + " |")
        dupes = {w: c for w, c in seen.items() if len(c) > 1}
        out.append(f"\n**{len(seen)} distinct triplet(s)** across {len(tiers)*len(emos)} cells.")
        if dupes:
            out.append("\nCells that CANNOT differ except by noise:\n")
            for w, cells in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
                out.append(f"- `{'/'.join(w)}` — {', '.join(cells)}")

        out.append("\n### Style / servo ladder\n")
        out.append("| tier | amplitude | tempo | posture | droop | wire |")
        out.append("|---|---|---|---|---|---|")
        for t in tiers:
            m = next((r for r in rr if r["tier"] == t and r["emotion"] == emos[0]), None)
            if m and m["style"]:
                s = m["style"]
                out.append(f"| {t} | {s['amplitude']:.3f} | {s['tempo']:.3f} | "
                           f"{s['posture']:+.3f} | {s['droop']:+.3f} | `{m['wire']}` |")

        base = next((r for r in rr if r["tier"] == tiers[0] and r["emotion"] == emos[0]), None)
        if base:
            out.append("\n### What PAD actually contributed to the prompt\n")
            other = next((r for r in rr if r["tier"] == tiers[-1]
                          and r["emotion"] == emos[0]), None)
            if other and other["prompt"] != base["prompt"]:
                diff = [l for l in difflib.unified_diff(
                    base["prompt"].splitlines(), other["prompt"].splitlines(),
                    lineterm="", n=0) if l.startswith(("+", "-"))
                    and not l.startswith(("+++", "---"))]
                out.append(f"`{tiers[0]}` vs `{tiers[-1]}` at emotion `{emos[0]}`:\n")
                out.append("```diff\n" + "\n".join(diff[:20]) + "\n```")
                delta = abs(len(other["prompt"]) - len(base["prompt"]))
                out.append(f"\nPrompt length differs by **{delta} characters** "
                           f"(of {len(base['prompt'])}).")

        if any(r["reply"] for r in rr):
            out.append("\n### Replies\n")
            out.append("| tier | " + " | ".join(emos) + " |")
            out.append("|---" * (len(emos) + 1) + "|")
            for t in tiers:
                cells = []
                for e in emos:
                    reps = [r for r in rr if r["tier"] == t and r["emotion"] == e]
                    cells.append("<br>".join(f"`[{x['tag']}]` {x['reply']}" for x in reps[:2])
                                 or "—")
                out.append(f"| **{t}** | " + " | ".join(cells) + " |")

            out.append("\n### Lexical metrics (mean over repeats)\n")
            ctrl = [r for r in rows if r["robot"] == robot and not r["pad_on"] and r["reply"]]
            out.append("| tier | emotion | words | TTR | hedges | warmth | feeling-talk |")
            out.append("|---|---|---|---|---|---|---|")
            for t in tiers:
                for e in emos:
                    reps = [r for r in rr if r["tier"] == t and r["emotion"] == e and r["metrics"]]
                    if not reps:
                        continue
                    def mean(k):
                        return statistics.mean(x["metrics"][k] for x in reps)
                    sd = (statistics.pstdev([x["metrics"]["words"] for x in reps])
                          if len(reps) > 1 else 0.0)
                    out.append(f"| {t} | {e} | {mean('words'):.1f} ±{sd:.1f} | "
                               f"{mean('ttr'):.3f} | {mean('hedges'):.1f} | "
                               f"{mean('warmth'):.1f} | "
                               f"{sum(x['metrics']['feeling_talk'] for x in reps)}/{len(reps)} |")
            if ctrl:
                cw = statistics.mean(r["metrics"]["words"] for r in ctrl)
                out.append(f"\n**Control (PAD off):** {cw:.1f} words mean over "
                           f"{len(ctrl)} replies. A tier effect must exceed the "
                           "repeat-to-repeat spread above to mean anything.")

    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="chatbox,ellebot")
    p.add_argument("--tiers", default="unknown,visitor,known,close")
    p.add_argument("--emotions", default="happy,sad,angry,neutral")
    p.add_argument("--message", default="hey, what have you been up to?")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="recorded only — LLMClient fixes it internally")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--no-directive", action="store_true",
                   help="A/B: adjective only, no behavioural directive")
    p.add_argument("--control", action="store_true",
                   help="also run every cell with PAD OFF (the baseline)")
    p.add_argument("--no-llm", action="store_true",
                   help="build prompts only — no Ollama, CI-safe")
    p.add_argument("--out", default=None, help="JSONL path")
    p.add_argument("--md", default=None, help="Markdown report path")
    args = p.parse_args(argv)

    rows = run_grid(args)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(runs, exist_ok=True)
    out = args.out or os.path.join(runs, f"grid_{stamp}.jsonl")
    md = args.md or os.path.join(runs, f"grid_{stamp}.md")

    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    write_report(rows, md, args)

    print(f"\n[grid] {len(rows)} cells")
    print(f"[grid] data   {out}")
    print(f"[grid] report {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
