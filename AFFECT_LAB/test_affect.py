"""
test_affect.py — checks the affect maths without needing a camera or a model.

Run:  python test_affect.py
"""

import affect as A

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(label, got, want, tol=0.005):
    global failures
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        failures += 1
    print(f"  [{PASS if ok else FAIL}] {label:<52} got {got!r}")


def fmt(v):
    return f"{v:+.2f}"


print("\n=== 1. Table I of the paper reproduces exactly ===")
# CHATBOX  -.5 +.2 -.6 +.6 +.2  ->  P +.27  Ar -.01  D -.64
# ELLEBOT  +.5 +.4 +.7 +.6 -.4  ->  P +.42  Ar +.48  D +.42
for name, want in (("CHATBOX", (0.27, -0.01, -0.64)),
                   ("ELLEBOT", (0.42, 0.48, 0.42))):
    pad = A.to_pad(A.ROBOTS[name]["ocean"])
    check(f"{name} pleasure", round(pad["P"], 2), want[0])
    check(f"{name} arousal", round(pad["Ar"], 2), want[1])
    check(f"{name} dominance", round(pad["D"], 2), want[2])

print("\n=== 2. Descriptors match the paper's worked example ===")
# Section II.A gives "warm, calm, reserved" for CHATBOX.
words = A.descriptors(A.to_pad(A.ROBOTS["CHATBOX"]["ocean"]))
check("CHATBOX descriptors", ", ".join(words), "warm, calm, reserved")

print("\n=== 3. The face never moves Dominance ===")
base = A.to_pad(A.ROBOTS["CHATBOX"]["ocean"])
for v, a in ((0.8, 0.5), (-0.7, 0.65), (0.0, 0.0), (-0.7, -0.38)):
    felt = A.feel(base, v, a)
    check(f"D unchanged for face ({v:+.1f},{a:+.1f})",
          round(felt["D"], 6), round(base["D"], 6))

print("\n=== 4. Neutral face pulls toward neutral, not away ===")
felt = A.feel(base, 0.0, 0.0)
check("|P| shrinks toward 0", abs(felt["P"]) < abs(base["P"]), True)

print("\n=== 5. A body shows a fraction of what it feels ===")
felt = A.feel(base, 0.8, 0.5)
cb = A.show(felt, A.ROBOTS["CHATBOX"]["show"])
eb = A.show(felt, A.ROBOTS["ELLEBOT"]["show"])
check("CHATBOX shows less pleasure than ELLEBOT", cb["P"] < eb["P"], True)
check("ELLEBOT at show=1.0 shows all of it", round(eb["P"], 6),
      round(felt["P"], 6))

print("\n=== 6. Affect naming ===")
for (p, a), want in (((0.0, 0.0), "neutral"),
                     ((0.05, 0.05), "neutral"),
                     ((0.8, 0.5), "strongly elated"),
                     ((-0.7, 0.65), "strongly tense"),
                     ((0.2, 0.05), "mildly pleased"),
                     ((-0.7, -0.38), "strongly dejected")):
    check(f"({p:+.2f},{a:+.2f})", A.affect_name(p, a), want)

print("\n=== 7. Same persona, two bodies — the demo's headline ===")
traits = A.ROBOTS["CHATBOX"]["ocean"]
print(f"  {'detected':<12}{'CHATBOX':<26}{'ELLEBOT'}")
for label in ("happy", "angry", "sad", "fear", "neutral", "surprise"):
    v, a = A.category_to_va(label)
    out = {r: A.pipeline(traits, v, a, r) for r in A.ROBOTS}
    cells = [f"{fmt(out[r]['shown']['P'])},{fmt(out[r]['shown']['Ar'])} "
             f"{out[r]['name']}" for r in ("CHATBOX", "ELLEBOT")]
    print(f"  {label:<12}{cells[0]:<26}{cells[1]}")

print("\n=== 8. Unknown labels degrade to neutral ===")
check("garbage label", A.category_to_va("not_an_emotion"), (0.0, 0.0))

print("\n=== 9. Gesture style reproduces the paper's own claim ===")
# "a broad, brisk wave from ELLEBOT, a slow, gentle one from CHATBOX"
cb = A.gesture_style(A.to_pad(A.ROBOTS["CHATBOX"]["ocean"]))
eb = A.gesture_style(A.to_pad(A.ROBOTS["ELLEBOT"]["ocean"]))
for k in A.STYLE_LIMITS:
    print(f"    {k:<10} CHATBOX {cb[k]:+.2f}   ELLEBOT {eb[k]:+.2f}")
check("ELLEBOT waves broader", eb["amplitude"] > cb["amplitude"], True)
check("ELLEBOT waves brisker", eb["tempo"] > cb["tempo"], True)
check("ELLEBOT stands more open", eb["posture"] > cb["posture"], True)
check("ELLEBOT stirs more often", eb["idle"] > cb["idle"], True)
check("CHATBOX posture is withdrawn (negative)", cb["posture"] < 0, True)

print("\n=== 10. Style stays inside the servo-safe limits ===")
# Every extreme of the affective space, including corners a real face cannot
# reach — the firmware must never be handed an out-of-range scale factor.
worst = []
for p in (-1, 0, 1):
    for ar in (-1, 0, 1):
        for d in (-1, 0, 1):
            s = A.gesture_style({"P": p, "Ar": ar, "D": d})
            for k, (lo, hi) in A.STYLE_LIMITS.items():
                if not (lo <= s[k] <= hi):
                    worst.append((p, ar, d, k, s[k]))
check("all 27 corners within limits", worst, [])

print("\n=== 11. Arousal drives speed, Dominance drives carriage ===")
calm = A.gesture_style({"P": 0.0, "Ar": -0.8, "D": 0.0})
lively = A.gesture_style({"P": 0.0, "Ar": 0.8, "D": 0.0})
check("high arousal is faster", lively["tempo"] > calm["tempo"], True)
check("high arousal fidgets more", lively["idle"] > calm["idle"], True)
low_d = A.gesture_style({"P": 0.0, "Ar": 0.0, "D": -0.8})
high_d = A.gesture_style({"P": 0.0, "Ar": 0.0, "D": 0.8})
check("dominance opens posture", high_d["posture"] > low_d["posture"], True)
check("arousal alone leaves posture neutral",
      round(A.gesture_style({"P": 0, "Ar": 0.9, "D": 0})["posture"], 6), 0.0)

print(f"\n{'All checks passed.' if not failures else f'{failures} FAILED.'}\n")
raise SystemExit(1 if failures else 0)
