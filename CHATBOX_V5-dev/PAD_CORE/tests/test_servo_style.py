"""
test_servo_style.py â€” checks the angle maths before anything gets flashed.

Run:  python test_servo_style.py
"""

import os
import sys

# tests/ -> PAD_CORE/ on the path, so the suite runs from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pad_core import servo_style as S

failures = 0


def check(label, got, want):
    global failures
    ok = got == want
    if not ok:
        failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<54} got {got!r}")


def check_true(label, cond, detail=""):
    global failures
    if not cond:
        failures += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {label:<54} {detail}")


print("\n=== 1. Neutral style reproduces the stock firmware exactly ===")
# The whole design rests on this: style 1.0/0/0 must change nothing, so the
# robot behaves identically until someone deliberately styles it.
for tag in S.MOVE_SETS:
    out = S.resolve_gesture(tag, S.NEUTRAL_STYLE)
    mismatches = []
    for servo, column, per_step in S.CHANNELS:
        raw = S.MOVE_SETS[tag][column]
        seq = raw if per_step else [raw] * S.STEPS
        for i, sym in enumerate(seq):
            expect = S.SERVOS[servo]["angles"].get(sym, S.SERVOS[servo]["rest"])
            if out["angles"][servo][i] != expect:
                mismatches.append((servo, i, out["angles"][servo][i], expect))
    check(f"{tag} unchanged at neutral style", mismatches, [])
check("step timing unchanged", S.resolve_gesture("wave")["step_ms"], 900)

print("\n=== 2. Amplitude scales travel, never the rest pose ===")
for amp in (0.30, 0.60, 1.00):
    out = S.resolve_gesture("wave", {"amplitude": amp})
    # RShoulder is U for all five steps in wave: rest 140 -> target 170.
    got = out["angles"]["RShoulder"][0]
    want = round(140 + amp * 30)
    check(f"amplitude {amp:.2f}: RShoulder U", got, want)
out = S.resolve_gesture("greeting", {"amplitude": 0.3})
check_true("a servo already at rest does not move",
           out["angles"]["RHand"] == [90] * 5, "RHand is M throughout greeting")

print("\n=== 2b. Amplitude cannot exceed the authored gesture ===")
# Each servo has three symbols, so D and U *are* the ends of its stock range.
# Amplitude above 1.0 could only saturate against the clamp, so the limit is
# 1.0 and the stock move sets are treated as full expression.
check("amplitude is capped at 1.0", S.STYLE_LIMITS["amplitude"][1], 1.00)
check("a request for 1.3 is clamped", S.clamp_style({"amplitude": 1.3})["amplitude"], 1.00)
full = S.resolve_gesture("wave", {"amplitude": 1.0})["angles"]
lo, hi = S.servo_range("RShoulder")
check(f"full amplitude reaches the authored pose ({hi})",
      full["RShoulder"][0], hi)
damped = S.resolve_gesture("wave", {"amplitude": 0.30})["angles"]
check_true("damped stays short of it",
           damped["RShoulder"][0] < full["RShoulder"][0],
           f"{full['RShoulder'][0]} -> {damped['RShoulder'][0]}")

print("\n=== 3. Droop is what makes a happy gesture look sad ===")
# The point of the whole exercise: same tag, unpleasant coordinate.
happy = S.resolve_gesture("greeting", {"droop": -0.6})
neutral = S.resolve_gesture("greeting", {"droop": 0.0})
sadly = S.resolve_gesture("greeting", {"droop": +0.8})
print(f"    ears     step0   droop-0.6 {happy['angles']['Ears'][0]}   "
      f"0.0 {neutral['angles']['Ears'][0]}   +0.8 {sadly['angles']['Ears'][0]}")
print(f"    RNeck    step0   droop-0.6 {happy['angles']['RNeck'][0]}   "
      f"0.0 {neutral['angles']['RNeck'][0]}   +0.8 {sadly['angles']['RNeck'][0]}")
check_true("droop lowers the ears",
           sadly["angles"]["Ears"][0] < neutral["angles"]["Ears"][0], "")
check_true("droop dips the head",
           sadly["angles"]["RNeck"][0] < neutral["angles"]["RNeck"][0], "")
check_true("droop slumps the shoulders",
           sadly["angles"]["RShoulder"][0] < neutral["angles"]["RShoulder"][0], "")
check_true("negative droop lifts instead",
           happy["angles"]["Ears"][0] >= neutral["angles"]["Ears"][0], "")
check_true("hands are left alone",
           sadly["angles"]["RHand"] == neutral["angles"]["RHand"],
           "a drooping hand reads as a fault, not a mood")

print("\n=== 4. Left and right stay mirrored ===")
# Both sides sit at 90 +/- offset, so a mood must move them oppositely or the
# robot ends up lopsided.
out = S.resolve_gesture("greeting", {"droop": 0.8})
for a, b, rest_a, rest_b in (("RBrow", "LBrow", 120, 60),
                             ("REyelid", "LEyelid", 110, 70),
                             ("RShoulder", "LShoulder", 140, 40)):
    da = out["angles"][a][0] - S.resolve_gesture("greeting")["angles"][a][0]
    db = out["angles"][b][0] - S.resolve_gesture("greeting")["angles"][b][0]
    check_true(f"{a}/{b} move in opposite directions",
               da == 0 or db == 0 or (da > 0) != (db > 0),
               f"delta {da:+d} / {db:+d}")

print("\n=== 5. Posture only touches neck and shoulders ===")
base = S.resolve_gesture("greeting")
open_ = S.resolve_gesture("greeting", {"posture": +1.0})
shut = S.resolve_gesture("greeting", {"posture": -1.0})
for servo in ("Ears", "RBrow", "REyelid", "RHand"):
    check(f"{servo} ignores posture",
          open_["angles"][servo], base["angles"][servo])
check_true("posture opens the neck",
           open_["angles"]["RNeck"][0] > shut["angles"]["RNeck"][0],
           f"{shut['angles']['RNeck'][0]} -> {open_['angles']['RNeck'][0]}")

print("\n=== 6. Tempo changes timing, not a single angle ===")
slow = S.resolve_gesture("wave", {"tempo": 0.5})
fast = S.resolve_gesture("wave", {"tempo": 1.6})
check("slow step_ms", slow["step_ms"], 1800)
check("fast step_ms", fast["step_ms"], 562)
check("angles identical across tempo",
      slow["angles"] == fast["angles"], True)

print("\n=== 7. Nothing can leave the stock pose range ===")
# Every tag, every corner of style space. This is the safety property: styling
# must never command an angle the unstyled firmware would not have commanded.
violations = []
for tag in S.MOVE_SETS:
    for amp in (0.3, 0.65, 1.0):
        for dr in (-1.0, 0.0, 1.0):
            for po in (-1.0, 0.0, 1.0):
                out = S.resolve_gesture(tag, {"amplitude": amp, "droop": dr,
                                              "posture": po})
                for servo, seq in out["angles"].items():
                    lo, hi = S.servo_range(servo)
                    for ang in seq:
                        if not (lo <= ang <= hi):
                            violations.append((tag, servo, ang, lo, hi))
check(f"{len(S.MOVE_SETS) * 27} tag/style combinations in range",
      violations, [])
check_true("all angles are servo-legal (0-180)",
           all(0 <= a <= 180 for name in S.SERVOS
               for a in S.SERVOS[name]["angles"].values()), "")

print("\n=== 7b. Home poses reach their position in full ===")
# Amplitude scales toward rest, which on a home pose half-undoes it: default puts
# the arms down at 50 and amp 0.34 would drag them back to 102.
damped = {"amplitude": 0.34, "droop": 0.29, "posture": -0.54}
for tag in sorted(S.HOME_TAGS & set(S.MOVE_SETS)):
    plain = S.resolve_gesture(tag, S.NEUTRAL_STYLE)["angles"]
    styled = S.resolve_gesture(tag, damped)["angles"]
    # Amplitude ignored, so the arm still goes all the way down...
    check_true(f"{tag}: RShoulder reaches its pose",
               abs(styled["RShoulder"][0] - plain["RShoulder"][0]) <= 12,
               f"{plain['RShoulder'][0]} -> {styled['RShoulder'][0]} "
               f"(would be 102 if amplitude applied)")
    # ...but droop and posture still show, so the mood persists at rest.
    check_true(f"{tag}: droop still tints it",
               styled["Ears"][0] < plain["Ears"][0],
               f"ears {plain['Ears'][0]} -> {styled['Ears'][0]}")
gesture = S.resolve_gesture("greeting", damped)["angles"]
check_true("a real gesture is still damped",
           gesture["RShoulder"][0] < S.resolve_gesture("greeting")["angles"]["RShoulder"][0],
           "greeting is not a home pose")

print("\n=== 7c. The head comes back level at rest ===")
# Droop and posture both lift the neck for a bright, dominant robot, together
# eating 12 of its 30 degrees — it would park off-centre and stay there.
bright = {"posture": 0.49, "droop": -0.91}
for tag in sorted(S.HOME_TAGS & set(S.MOVE_SETS)):
    home = S.resolve_gesture(tag, bright)["angles"]
    plain = S.resolve_gesture(tag, S.NEUTRAL_STYLE)["angles"]
    for servo in ("RNeck", "LNeck"):
        check(f"{tag}: {servo} returns level", home[servo], plain[servo])
    # The rest of the body still carries the mood at rest. Checked with a sad
    # style: 'default' already holds the ears at 165, which is their maximum, so
    # a bright mood cannot lift them further and correctly clamps.
    sad_home = S.resolve_gesture(tag, {"posture": -0.54, "droop": 0.29})["angles"]
    check_true(f"{tag}: ears still droop at rest",
               sad_home["Ears"][0] < plain["Ears"][0],
               f"{plain['Ears'][0]} -> {sad_home['Ears'][0]}")
    check_true(f"{tag}: shoulders still tinted",
               home["RShoulder"][0] != plain["RShoulder"][0],
               f"{plain['RShoulder'][0]} -> {home['RShoulder'][0]}")
    check(f"{tag}: neck level under a sad style too",
          sad_home["RNeck"], plain["RNeck"])
# During a gesture the neck must still move with the mood.
g_bright = S.resolve_gesture("greeting", bright)["angles"]
g_plain = S.resolve_gesture("greeting")["angles"]
check_true("greeting: neck still moves with the mood",
           g_bright["RNeck"][0] != g_plain["RNeck"][0],
           f"{g_plain['RNeck'][0]} -> {g_bright['RNeck'][0]}")

print("\n=== 8. Out-of-range style values are clamped, not trusted ===")
wild = S.clamp_style({"amplitude": 99, "tempo": -5, "posture": 42,
                      "droop": -99, "idle": 100})
check("amplitude clamped", wild["amplitude"], 1.00)
check("tempo clamped", wild["tempo"], 0.50)
check("posture clamped", wild["posture"], 1.00)
check("droop clamped", wild["droop"], -1.00)
check("missing keys filled from neutral",
      S.clamp_style({})["amplitude"], 1.0)

print("\n=== 9. Wire format round-trips ===")
style = {"amplitude": 0.95, "tempo": 1.18, "posture": 0.49,
         "droop": -0.35, "idle": 0.65}
line = S.wire_message(style)
print(f"    {line}")
back = S.parse_wire(line)
for k, v in style.items():
    check(f"{k} survives the round trip", round(back[k], 2), round(v, 2))
for bad in ("STYLE 1 2 3", "", "TAG greeting", "STYLE a b c d e"):
    try:
        S.parse_wire(bad)
        check(f"rejects {bad!r}", "accepted", "rejected")
    except ValueError:
        check_true(f"rejects {bad!r}", True, "")

print("\n=== 10. Unknown tag fails loudly ===")
try:
    S.resolve_gesture("not_a_tag")
    check("unknown tag raises", "no", "KeyError")
except KeyError:
    check_true("unknown tag raises KeyError", True, "")

print(f"\n{'All checks passed.' if not failures else f'{failures} FAILED.'}\n")
raise SystemExit(1 if failures else 0)

