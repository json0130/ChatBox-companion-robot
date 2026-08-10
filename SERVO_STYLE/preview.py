#!/usr/bin/env python3
"""
preview.py — what each robot's servos actually do, for a tag and a mood.

Answers the question directly: if the person looks sad and CHATBOX plays
GREETING, what angles come out, and how do they differ from ELLEBOT's?

    python preview.py greeting --emotion sad
    python preview.py wave --emotion happy --servo RShoulder
    python preview.py greeting --emotion sad --wire
    python preview.py --tags

Pulls the persona and affect maths from ../AFFECT_LAB so there is one source of
truth for PAD, and converts it here into the five style values the firmware
consumes.
"""

import argparse
import os
import sys

import servo_style as S

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "AFFECT_LAB"))
try:
    import affect as A
except ImportError:
    A = None


def style_from_affect(robot: str, emotion: str) -> dict:
    """Persona + detected emotion -> the five servo-style values.

    amplitude/tempo/posture/idle come from AFFECT_LAB's gesture_style, which is
    the paper's four parameters. `droop` is the extra one: a signed valence tint,
    because none of the paper's four can make a happy gesture read as sad.

    Amplitude is additionally multiplied by the body's `show` fraction. That is
    where the two robots diverge mechanically — CHATBOX physically travels less
    of the authored gesture than ELLEBOT does.
    """
    if A is None:
        raise SystemExit("cannot import AFFECT_LAB/affect.py — run from "
                         "SERVO_STYLE with AFFECT_LAB alongside it")

    valence, arousal = A.category_to_va(emotion)
    out = A.pipeline(A.ROBOTS[robot]["ocean"], valence, arousal, robot)
    four = out["style"]
    travel = S.TRAVEL.get(robot, 1.0)

    # Droop follows what the robot *feels*, not the display-scaled coordinate —
    # then the body's travel fraction decides how much of that reaches a servo.
    # Using the already-scaled coordinate would apply the same restraint twice
    # and leave CHATBOX barely tinted at all.
    droop = -out["felt"]["P"] * 1.4

    return {
        "amplitude": four["amplitude"] * travel,
        "tempo": four["tempo"],
        "posture": four["posture"],
        "droop": max(-1.0, min(1.0, droop * travel)),
        "idle": four["idle"],
    }, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", nargs="?", default="greeting")
    ap.add_argument("--emotion", default="neutral",
                    help="what the camera saw: happy, sad, angry, fear, ...")
    ap.add_argument("--servo", help="show only this servo")
    ap.add_argument("--wire", action="store_true",
                    help="also print the STYLE line the Jetson would send")
    ap.add_argument("--tags", action="store_true", help="list known tags and exit")
    args = ap.parse_args()

    if args.tags:
        print("tags:  " + "  ".join(sorted(S.MOVE_SETS)))
        print("servos:" + "  ".join(S.SERVOS))
        if A:
            print("emotions: " + "  ".join(sorted(set(A.CATEGORY_VA))))
        return 0

    if args.tag not in S.MOVE_SETS:
        print(f"unknown tag {args.tag!r}. known: {'  '.join(sorted(S.MOVE_SETS))}")
        return 1

    robots = list(A.ROBOTS) if A else ["(stock)"]
    print(f"\ntag: {args.tag}    person looks: {args.emotion}\n")

    resolved = {}
    for robot in robots:
        style, out = style_from_affect(robot, args.emotion)
        resolved[robot] = S.resolve_gesture(args.tag, style)
        st = resolved[robot]["style"]
        print(f"  {robot}")
        print(f"    feels        P {out['shown']['P']:+.2f}  "
              f"Ar {out['shown']['Ar']:+.2f}  D {out['shown']['D']:+.2f}"
              f"   ({out['name']})")
        print(f"    style        amp {st['amplitude']:.2f}   "
              f"tempo {st['tempo']:.2f}   posture {st['posture']:+.2f}   "
              f"droop {st['droop']:+.2f}")
        print(f"    step timing  {resolved[robot]['step_ms']} ms  "
              f"(stock {S.BASE_STEP_MS} ms)")
        if args.wire:
            print(f"    wire         {S.wire_message(st)}")
        print()

    # Angle table: stock alongside each robot, so the difference is the point.
    stock = S.resolve_gesture(args.tag, S.NEUTRAL_STYLE)
    servos = [args.servo] if args.servo else list(S.SERVOS)
    if args.servo and args.servo not in S.SERVOS:
        print(f"unknown servo {args.servo!r}. known: {'  '.join(S.SERVOS)}")
        return 1

    # Five 3-digit angles separated by spaces is 19 characters; leave room.
    width = 22
    header = f"  {'servo':<11}{'stock':<{width}}" + \
             "".join(f"{r:<{width}}" for r in robots)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for servo in servos:
        cells = [" ".join(f"{a:3d}" for a in stock["angles"][servo])]
        for robot in robots:
            cells.append(" ".join(f"{a:3d}" for a in resolved[robot]["angles"][servo]))
        row = f"  {servo:<11}" + "".join(f"{c:<{width}}" for c in cells)
        # Mark the rows where the robots actually differ from stock.
        differs = any(resolved[r]["angles"][servo] != stock["angles"][servo]
                      for r in robots)
        print(row + ("  *" if differs else ""))
    print("\n  * differs from the stock gesture      five columns = the five steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
