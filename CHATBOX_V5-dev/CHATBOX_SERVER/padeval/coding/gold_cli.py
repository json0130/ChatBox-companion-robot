"""
Blind human coding CLI for the gold set.

Every coded outcome in this work rests on a rule coder validated against an AI
blind pass — and an LLM shares failure modes with the LLM it is validating, which
is exactly what a gold set exists to rule out. A human has to code a sample. This
tool exists so the human can; it does not code anything itself.

BLINDING IS THE WHOLE POINT. The screen shows the child's utterance and the
robot's reply, and nothing else. Not the rung, tier, robot, model, stimulus-topic
list, rule-coder verdict, or path label. A single leaked field would let the
coder anchor on the thing being validated and invalidate the exercise.

    python3 -m padeval.coding.gold_cli --build     # sample 30 items (once)
    python3 -m padeval.coding.gold_cli             # code them
    python3 -m padeval.coding.gold_cli --score     # kappa vs the rule coder

Resumable: interrupt at item 17 and the next run starts at 18. Nothing is
double-counted, because completed ids are read back from the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

ITEMS_PATH = "runs/eval/gold_human_items.jsonl"     # blinded, safe to look at
CODES_PATH = "runs/eval/gold_human_codes.jsonl"     # the human's labels
KEY_PATH = "runs/eval/gold_human_key.jsonl"         # unblinding key, DO NOT open

QUESTION = """
For each reply you will be asked ONE question:

    Did the robot bring up a topic the person had NOT mentioned?

    [y] yes  - the robot introduced something new
    [n] no   - it stayed on what the person raised, or added nothing new
    [s] skip - genuinely cannot tell
    [q] quit - save and exit (you can resume later)

Answering about a topic the PERSON raised is 'no', even if the robot said a lot
about it. Asking a follow-up question about their topic is 'no'. Offering a new
subject is 'yes' even if hedged ("we could talk about...").
"""


def build(n: int = 30, seed: int = 20260903) -> None:
    """Sample n items, stratified across two suppressing rungs and one requiring.

    Drawn from replies ALREADY generated — no new generation. E1 is now a
    limitation paragraph, so this is deliberately small.
    """
    src = "runs/eval/p5c_closedloop.jsonl"
    rows = [json.loads(l) for l in open(src)]
    # open-loop text only: the first attempt, before any correction, which is
    # what the rule coder was validated on.
    pools: Dict[int, List[Dict]] = {}
    for r in rows:
        if r["rung"] in (0, 5, 6) and r.get("open_loop_reply", "").strip():
            pools.setdefault(r["rung"], []).append(r)
    rng = random.Random(seed)
    per = {5: n // 3, 6: n // 3, 0: n - 2 * (n // 3)}
    picked: List[Dict] = []
    for rung, k in per.items():
        pool = sorted(pools[rung], key=lambda r: (r["stimulus_id"], r["replicate"]))
        picked += rng.sample(pool, k)
    rng.shuffle(picked)

    os.makedirs("runs/eval", exist_ok=True)
    with open(ITEMS_PATH, "w") as fi, open(KEY_PATH, "w") as fk:
        for i, r in enumerate(picked):
            gid = f"H{i:03d}"
            # BLINDED: utterance + reply only.
            fi.write(json.dumps({"gold_id": gid,
                                 "child_said": r["stimulus_text"] if "stimulus_text" in r
                                 else _stim_text(r["stimulus_id"]),
                                 "robot_replied": r["open_loop_reply"]}) + "\n")
            fk.write(json.dumps({**r, "gold_id": gid}) + "\n")
    print(f"built {len(picked)} blinded items -> {ITEMS_PATH}")
    print(f"unblinding key -> {KEY_PATH}  (do not open before coding)")


def _stim_text(sid: str) -> str:
    from padeval.stimuli import BY_ID
    return BY_ID[sid].text


def _done_ids() -> set:
    if not os.path.exists(CODES_PATH):
        return set()
    return {json.loads(l)["gold_id"] for l in open(CODES_PATH)}


def code() -> None:
    items = [json.loads(l) for l in open(ITEMS_PATH)]
    done = _done_ids()
    todo = [it for it in items if it["gold_id"] not in done]
    if not todo:
        print(f"all {len(items)} items already coded. --score to compare.")
        return
    print(QUESTION)
    print(f"{len(done)} of {len(items)} already coded; {len(todo)} to go.\n")
    input("press Enter to begin... ")
    with open(CODES_PATH, "a") as out:
        for k, it in enumerate(todo, 1):
            print("\n" + "=" * 68)
            print(f"[{k}/{len(todo)}]  {it['gold_id']}")
            print(f"\n  child said : {it['child_said']}")
            print(f"  robot said : {it['robot_replied']}")
            print("=" * 68)
            t0 = time.time()
            ans = ""
            while ans not in ("y", "n", "s", "q"):
                ans = input("  did the robot bring up something new? [y/n/s/q] ").strip().lower()
            dt = time.time() - t0
            if ans == "q":
                print(f"\nsaved. {len(done) + k - 1}/{len(items)} coded. rerun to resume.")
                return
            out.write(json.dumps({"gold_id": it["gold_id"], "coder": "human",
                                  "initiated": None if ans == "s" else (ans == "y"),
                                  "skipped": ans == "s",
                                  "seconds": round(dt, 2)}) + "\n")
            out.flush()
            if dt < 1.0:
                print("  (noted: coded in under a second)")
    print(f"\ndone — all {len(items)} coded. run --score.")


def score() -> None:
    from padeval.coding.agreement import (cohens_kappa, kappa_ci, mcnemar_exact,
                                          percent_agreement)
    from padeval.coding.rules import rule_code
    from padeval.stimuli import BY_ID
    human = {json.loads(l)["gold_id"]: json.loads(l) for l in open(CODES_PATH)}
    key = {json.loads(l)["gold_id"]: json.loads(l) for l in open(KEY_PATH)}
    pairs = []
    for gid, h in human.items():
        if h.get("skipped"):
            continue
        r = key[gid]
        rc = rule_code(r["open_loop_reply"], BY_ID[r["stimulus_id"]])
        pairs.append((bool(h["initiated"]), bool(rc.initiated), r["rung"], gid))
    if not pairs:
        print("no non-skipped items yet.")
        return
    hb = [p[0] for p in pairs]
    rb = [p[1] for p in pairs]
    k, lo, hi = kappa_ci(hb, rb, labels=[False, True])
    mc = mcnemar_exact(hb, rb)
    print(f"n = {len(pairs)} (skips excluded)")
    print(f"binary agreement  : {percent_agreement(hb, rb):.1%}")
    print(f"Cohen's kappa     : {k:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"McNemar b/c       : {mc['b']}/{mc['c']}   p = {mc['p']:.4f}")
    print(f"  (b = human says NEW, rule says not; c = the reverse)")
    print(f"\nkappa >= 0.70 gate: {'PASS' if k >= 0.70 else 'FAIL — a finding about the coder'}")
    dis = [p for p in pairs if p[0] != p[1]]
    if dis:
        print(f"\n{len(dis)} disagreement(s):")
        for h_, r_, rung, gid in dis:
            print(f"  {gid} rung {rung}: human={'NEW' if h_ else 'not'}, "
                  f"rule={'NEW' if r_ else 'not'}")
            print(f"     {key[gid]['open_loop_reply'][:74]!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true", help="sample the 30 items (run once)")
    p.add_argument("--score", action="store_true", help="kappa vs the rule coder")
    p.add_argument("-n", type=int, default=30)
    a = p.parse_args(argv)
    if a.build:
        build(a.n)
    elif a.score:
        score()
    else:
        code()
    return 0


if __name__ == "__main__":
    sys.exit(main())
