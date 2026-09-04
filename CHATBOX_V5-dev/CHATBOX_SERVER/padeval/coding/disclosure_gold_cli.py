"""
Blind human coding CLI for the DISCLOSURE detector.

WHY A SECOND GOLD SET
---------------------
8a's only unbiased number is kappa = 0.739 on n = 32 with 8 positives, 95% CI
[0.389, 1.000]. That interval establishes agreement beats chance and almost
nothing else. It is wide because the sample is small, and the fix is more items,
not more argument.

SAMPLE PROVENANCE — THE PART THAT MATTERS
------------------------------------------
The items are the 75 unique non-empty CHILD utterances in `sessions.db`: real
development and test traffic, typed and spoken at the running system, including
its ASR garble. They were never used to build, tune or validate the detector.

**Every one of them is included. There is no sampling step at all.**

That is deliberate and it is stronger than sampling carefully. A stratified draw
— even a well-intentioned one — would have to decide what to stratify ON, and the
only signal available is the detector's own output. Validating a detector against
a sample it helped select is exactly the circularity a gold set exists to break.
Taking the entire pool removes the question: there is no selection to be biased.

The base rate is whatever the traffic happens to contain, which is also the right
answer — a deployment's base rate is not something the evaluator gets to choose.

BLINDING
--------
The screen shows the child's utterance and nothing else. Not the detector's
verdict, not its depth, not the route that fired, not the session or person it
came from. The utterance is the only thing a human needs to answer the question,
and it is the only thing they get.

    python3 -m padeval.coding.disclosure_gold_cli --build   # once
    python3 -m padeval.coding.disclosure_gold_cli           # code them
    python3 -m padeval.coding.disclosure_gold_cli --score   # kappa vs detector

Resumable: interrupt at item 40 and the next run starts at 41. Completed ids are
read back from the output file, so nothing is double-counted.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import Dict, List

ITEMS_PATH = "runs/eval/disclosure_gold_items.jsonl"   # blinded, safe to open
CODES_PATH = "runs/eval/disclosure_gold_codes.jsonl"   # the human's labels
DB_PATH = "sessions.db"

QUESTION = """
For each thing the child said you will be asked ONE question:

    Did the child tell the robot something about THEMSELVES that a camera
    pointed at their face could not already have seen?

    [y] yes  - a preference, a fact about their life, or how they have been
    [n] no   - a question, a greeting, small talk, or nothing personal
    [s] skip - genuinely cannot tell (garbled, truncated, ambiguous)
    [q] quit - save and exit (you can resume later)

Guidance, so the question means the same thing every time:

  * "I like jazz"            -> yes (a preference)
  * "my dad works nights"    -> yes (a fact about their life)
  * "I felt left out"        -> yes (how they have been)
  * "I'm happy" / "I'm sad"  -> NO. A camera can see that right now. Only count
                                it if they tie it to something the camera did
                                not see: "I was sad YESTERDAY", "...after school".
  * "you're funny"           -> no (about the robot, not about them)
  * "what's my name?"        -> no (a question)
  * "I don't like maths"     -> yes. Negative statements still disclose.

Some items are ASR garble or fragments. Skip those rather than guessing; skips
are excluded from the score rather than counted against either side.
"""


def build() -> None:
    """Take EVERY unique non-empty child utterance. No sampling, no strata."""
    if not os.path.exists(DB_PATH):
        print(f"{DB_PATH} not found — run from CHATBOX_SERVER/", file=sys.stderr)
        return
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT child FROM turns WHERE child IS NOT NULL AND TRIM(child) <> ''"
    ).fetchall()
    seen, items = set(), []
    for (text,) in rows:
        t = text.strip()
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        items.append({"gold_id": f"DG{len(items):03d}", "child_said": t})
    os.makedirs(os.path.dirname(ITEMS_PATH), exist_ok=True)
    with open(ITEMS_PATH, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items -> {ITEMS_PATH}")
    print("the entire pool, unsampled: no selection step, so no selection bias.")
    print(f"\nnow run:  python3 -m {__spec__.name}")


def _done_ids() -> set:
    if not os.path.exists(CODES_PATH):
        return set()
    return {json.loads(l)["gold_id"] for l in open(CODES_PATH)}


def code() -> None:
    if not os.path.exists(ITEMS_PATH):
        print("no items yet — run --build first.", file=sys.stderr)
        return
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
            print("=" * 68)
            t0 = time.time()
            ans = ""
            while ans not in ("y", "n", "s", "q"):
                ans = input("  did they disclose something about themselves? "
                            "[y/n/s/q] ").strip().lower()
            dt = time.time() - t0
            if ans == "q":
                print(f"\nsaved. {len(done) + k - 1}/{len(items)} coded. "
                      f"rerun to resume.")
                return
            out.write(json.dumps({"gold_id": it["gold_id"], "coder": "human",
                                  "disclosed": None if ans == "s" else (ans == "y"),
                                  "skipped": ans == "s",
                                  "seconds": round(dt, 2)}) + "\n")
            out.flush()
            if dt < 1.0:
                print("  (noted: coded in under a second)")
    print(f"\ndone — all {len(items)} coded. run --score.")


def score() -> None:
    from padeval.coding.agreement import (cohens_kappa, kappa_ci, mcnemar_exact,
                                          percent_agreement)
    from padeval.coding.disclosure import detect
    items = {json.loads(l)["gold_id"]: json.loads(l) for l in open(ITEMS_PATH)}
    human = [json.loads(l) for l in open(CODES_PATH)]
    pairs, abstained = [], []
    for h in human:
        if h.get("skipped"):
            continue
        text = items[h["gold_id"]]["child_said"]
        r = detect(text)
        if r.abstained:
            abstained.append((bool(h["disclosed"]), h["gold_id"], text,
                              r.features.get("abstain_reason", "")))
            continue
        pairs.append((bool(h["disclosed"]), r.disclosed, h["gold_id"], text))
    if not pairs:
        print("no non-skipped items yet.")
        return
    hb = [p[0] for p in pairs]
    rb = [p[1] for p in pairs]
    k, lo, hi = kappa_ci(hb, rb, labels=[False, True])
    mc = mcnemar_exact(hb, rb)
    n_skip = sum(1 for h in human if h.get("skipped"))
    n_human = len(human) - n_skip
    coverage = len(pairs) / n_human if n_human else 0.0
    print(f"n = {len(pairs)} scored ({n_skip} human-skipped, "
          f"{len(abstained)} detector-abstained)")
    print(f"COVERAGE          : {coverage:.1%} "
          f"({len(pairs)}/{n_human} of human-coded items were called)")
    print("  kappa on a subset means nothing without this. A detector abstaining")
    print("  on 80% and scoring 0.9 on the rest is not better than one scoring")
    print("  0.5 at full coverage. Both numbers or neither.")
    print(f"human base rate   : {sum(hb)}/{len(hb)} = {sum(hb)/len(hb):.1%} positive")
    print(f"binary agreement  : {percent_agreement(hb, rb):.1%}")
    print(f"Cohen's kappa     : {k:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"McNemar b/c       : {mc['b']}/{mc['c']}   p = {mc['p']:.4f}")
    print(f"  (b = human says DISCLOSED, detector says not; c = the reverse)")
    print(f"\n8a held-out reference: kappa 0.739, 95% CI [0.389, 1.000], n=32")
    print(f"kappa >= 0.70 gate: {'PASS' if k >= 0.70 else 'FAIL — a finding about the detector'}")
    if mc["b"] > mc["c"]:
        print("NOTE: errors skew to FALSE NEGATIVES (detector misses disclosure) "
              "— the safe direction, as designed.")
    elif mc["c"] > mc["b"]:
        print("NOTE: errors skew to FALSE POSITIVES — this is the direction 8a "
              "argued must NOT happen, since it lets trust drift up on ordinary "
              "chat and re-collapse onto rapport. Worth investigating.")
    if abstained:
        n_pos = sum(1 for a in abstained if a[0])
        print(f"\nabstained on {len(abstained)} item(s) "
              f"({n_pos} of which the human called DISCLOSED):")
        for h_, gid, text, why in abstained:
            print(f"  {gid}: human={'DISCLOSED' if h_ else 'not'}  [{why}]")
            print(f"     {text[:70]!r}")

    dis = [p for p in pairs if p[0] != p[1]]
    if dis:
        print(f"\n{len(dis)} disagreement(s):")
        for h_, d_, gid, text in dis:
            print(f"  {gid}: human={'DISCLOSED' if h_ else 'not'}, "
                  f"detector={'DISCLOSED' if d_ else 'not'}")
            print(f"     {text[:74]!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true",
                   help="write the item file (run once)")
    p.add_argument("--score", action="store_true",
                   help="kappa vs the disclosure detector")
    a = p.parse_args(argv)
    if a.build:
        build()
    elif a.score:
        score()
    else:
        code()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
