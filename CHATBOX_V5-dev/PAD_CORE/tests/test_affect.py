"""
test_affect.py — checks the affect maths without needing a camera or a model.

Run:  python test_affect.py
"""

import os
import sys

# tests/ -> PAD_CORE/ on the path, so the suite runs from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pad_core import affect as A

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

print("\n=== 9. The published equation matches the paper's own anchors ===")
# Hagane & Venture (2022) Machines 10(12):1118 state four values for fv in
# their Eq. 10. If the transcription of Eq. A1 is right, all four reproduce
# exactly. This is the check that the published half is really published.
for name, pad, want in (("Hostile",   (-1.0, 1.0, 1.0),   1.0),
                        ("Exuberant", (1.0, 1.0, 1.0),    0.5),
                        ("Anxious",   (-1.0, 1.0, -1.0),  0.5),
                        ("Bored",     (-1.0, -1.0, -1.0), 0.0)):
    coord = dict(zip(("P", "Ar", "D"), pad))
    check(f"fv({name}) reproduces the paper", round(A._velocity(coord), 9),
          round(want, 9))
# Eq. 9: Sp = (D + 1) / 2, on [0, 1], Dominance only.
for d, want in ((-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)):
    check(f"Sp(D={d:+.0f}) = (D+1)/2",
          A._spatial_extent({"P": 0, "Ar": 0, "D": d}), want)
check("Sp ignores Pleasure and Arousal",
      A._spatial_extent({"P": 0.9, "Ar": -0.9, "D": 0.3})
      == A._spatial_extent({"P": -0.9, "Ar": 0.9, "D": 0.3}), True)

print("\n=== 10. Gesture style reproduces the deck's own claim ===")
# "a broad, brisk wave from ELLEBOT, a slow, gentle one from CHATBOX" — now
# derived from PERFORM rather than fitted to the sentence.
cb = A.gesture_style(A.to_pad(A.ROBOTS["CHATBOX"]["ocean"]))
eb = A.gesture_style(A.to_pad(A.ROBOTS["ELLEBOT"]["ocean"]))
for k in A.STYLE_LIMITS:
    print(f"    {k:<10} CHATBOX {cb[k]:+.2f}   ELLEBOT {eb[k]:+.2f}")
check("ELLEBOT waves broader", eb["amplitude"] > cb["amplitude"], True)
check("ELLEBOT waves brisker", eb["tempo"] > cb["tempo"], True)
check("ELLEBOT stands more open", eb["posture"] > cb["posture"], True)
check("ELLEBOT stirs more often", eb["idle"] > cb["idle"], True)
check("CHATBOX posture is withdrawn (negative)", cb["posture"] < 0, True)

print("\n=== 11. Style stays inside the servo-safe limits ===")
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

print("\n=== 12. What the published tempo equation actually does ===")
# Recorded rather than asserted, because it is the one place where the
# published equation and the deck's stated intent disagree. Hagane & Venture's
# fv rises as Pleasure FALLS (their Eq. A1 term is 4 - Pn), following Wallbott's
# finding that high kinetic energy reads as anger. So an unpleasant face comes
# out slightly faster, not slower.
base = A.to_pad(A.ROBOTS["CHATBOX"]["ocean"])
speeds = {}
for label in ("happy", "angry", "sad", "neutral"):
    v, ar = A.category_to_va(label)
    speeds[label] = A.gesture_style(A.feel(base, v, ar))["tempo"]
    print(f"    tempo {label:<8}{speeds[label]:.2f}")
check("anger is the fastest (the paper's Hostile anchor)",
      max(speeds, key=speeds.get), "angry")
check("arousal raises tempo at fixed valence",
      A.gesture_style({"P": 0, "Ar": 0.9, "D": 0})["tempo"]
      > A.gesture_style({"P": 0, "Ar": -0.9, "D": 0})["tempo"], True)
check("NOTE: unpleasant is faster, not slower — see the module comment",
      speeds["sad"] >= speeds["happy"] - 0.02, True)

print("\n=== 13. Amplitude is Dominance-only, so the face cannot resize ===")
# A direct consequence of Sp = (D+1)/2 plus the rule that the face never moves
# Dominance. Worth pinning: it is a real behavioural property, not a bug.
amps = {}
for label in ("happy", "angry", "sad", "neutral"):
    v, ar = A.category_to_va(label)
    amps[label] = A.gesture_style(A.feel(base, v, ar))["amplitude"]
check("amplitude identical across every detected emotion",
      len({round(a, 9) for a in amps.values()}), 1)
check("but it still separates the two personas",
      A.gesture_style(A.to_pad(A.ROBOTS["ELLEBOT"]["ocean"]))["amplitude"]
      > A.gesture_style(A.to_pad(A.ROBOTS["CHATBOX"]["ocean"]))["amplitude"],
      True)

print("\n=== 14. Valence drives droop, monotonically (the deck's equation) ===")
droops = [A.gesture_style(A.feel(base, v, 0.0))["droop"]
          for v in (-1.0, -0.5, 0.0, 0.5, 1.0)]
print("    droop across valence -1..+1: " +
      "  ".join(f"{d:+.2f}" for d in droops))
check("droop falls as valence rises",
      all(a > b for a, b in zip(droops, droops[1:])), True)
check("an unpleasant face actually sags (droop > 0)", droops[0] > 0, True)
check("a pleasant face lifts (droop < 0)", droops[-1] < 0, True)

print("\n=== 15. The design deck's published numbers reproduce exactly ===")
# The "Same tag, two styles" slide states baseline PAD, felt PAD, amplitude and
# droop for both robots on a sad face. If any equation drifts from the deck,
# this is the check that catches it. Nothing downstream may rescale these —
# there is no `travel` fraction any more.
for robot, base_w, felt_w, amp_w, droop_w in (
        ("CHATBOX", (0.27, -0.01, -0.64), (-0.31, -0.23, -0.64), 0.42, 0.44),
        ("ELLEBOT", (0.43, 0.48, 0.42), (-0.25, -0.04, 0.42), 0.80, 0.35)):
    b = A.to_pad(A.ROBOTS[robot]["ocean"])
    f = A.feel(b, *A.category_to_va("sad"))
    s = A.gesture_style(f)
    for label, got, want in (
            ("baseline P", b["P"], base_w[0]), ("baseline Ar", b["Ar"], base_w[1]),
            ("baseline D", b["D"], base_w[2]),
            ("felt P", f["P"], felt_w[0]), ("felt Ar", f["Ar"], felt_w[1]),
            ("felt D", f["D"], felt_w[2]),
            ("amplitude", s["amplitude"], amp_w), ("droop", s["droop"], droop_w)):
        # 0.006 so the deck's 2-decimal rounding is satisfied either way
        check(f"{robot} {label} matches the deck ({want:+.2f})",
              abs(got - want) < 0.006, True)

print("\n=== 16. A neutral face relaxes back toward the persona ===")
# Decay with no timer, inherited from feel()'s gap-closure form.
persona = A.gesture_style(base)
sad = A.gesture_style(A.feel(base, *A.category_to_va("sad")))
back = A.gesture_style(A.feel(base, 0.0, 0.0))
check("sad pulls droop away from the persona",
      abs(sad["droop"] - persona["droop"]) > 0.1, True)
check("neutral sits closer to the persona than sad does",
      abs(back["droop"] - persona["droop"]) < abs(sad["droop"] - persona["droop"]),
      True)
check("empathy 0 pins the style to the persona exactly",
      round(A.gesture_style(A.feel(base, -0.9, 0.9, empathy=0.0))["droop"], 9),
      round(persona["droop"], 9))

print("\n=== 17. 'known' is the reference tier — the identity holds ===")
# The whole relationship path is a strict SUPERSET of the old model: with the
# reference tier it must reproduce the previous numbers bit for bit, so every
# figure published before this feature is still valid.
for robot in ("CHATBOX", "ELLEBOT"):
    b = A.to_pad(A.ROBOTS[robot]["ocean"])
    for emo in ("happy", "sad", "neutral", "anger"):
        v, ar = A.category_to_va(emo)
        old = A.feel(b, v, ar)
        new = A.feel_with_relationship(b, v, ar, "known")
        for axis in ("P", "Ar", "D"):
            check(f"{robot}/{emo} {axis}: known == no-tier",
                  round(new[axis], 12), round(old[axis], 12))

print("\n=== 18. The face still never moves Dominance ===")
# Section 4's invariant, now that a second influence writes to D: whatever the
# camera reports, D is exactly baseline + the tier's offset and nothing else.
for robot in ("CHATBOX", "ELLEBOT"):
    b = A.to_pad(A.ROBOTS[robot]["ocean"])
    for tier in A.TIERS:
        want = max(-1.0, min(1.0, b["D"] + A.tier_offset(tier)[2]))
        ds = {round(A.feel_with_relationship(b, v, ar, tier)["D"], 12)
              for v in (-0.9, 0.0, 0.9) for ar in (-0.9, 0.0, 0.9)}
        check(f"{robot}/{tier}: D independent of the face", len(ds), 1)
        check(f"{robot}/{tier}: D == baseline + offset", ds.pop(), round(want, 12))

print("\n=== 19. The tier ladder is monotone in amplitude ===")
# This is the payoff: Eq. 9 reads Dominance alone, so before the relationship
# existed amplitude was fixed per persona and no detected emotion could resize a
# gesture. Feeding D a social signal makes amplitude respond to WHO is present.
ladder = ("unknown", "visitor", "known", "family", "close")
for robot in ("CHATBOX", "ELLEBOT"):
    b = A.to_pad(A.ROBOTS[robot]["ocean"])
    amps = [A.gesture_style(A.feel_with_relationship(b, 0.0, 0.0, t))["amplitude"]
            for t in ladder]
    check(f"{robot}: amplitude rises with familiarity {[round(a,3) for a in amps]}",
          all(x < y for x, y in zip(amps, amps[1:])), True)

print("\n=== 20. Documented saturation at the bottom of the ladder ===")
# CHATBOX's baseline D (-0.643) plus the 'unknown' offset (-0.40) lands outside
# [-1,1], so it clamps -- and the clamped D puts amplitude exactly on its floor.
# Pinned rather than left to be rediscovered as a bug: at 'unknown' CHATBOX
# plays every gesture at minimum size. There is no tier below it.
b = A.to_pad(A.ROBOTS["CHATBOX"]["ocean"])
u = A.feel_with_relationship(b, 0.0, 0.0, "unknown")
check("CHATBOX @ unknown: D saturates at -1", round(u["D"], 6), -1.0)
check("CHATBOX @ unknown: amplitude sits on the 0.30 clamp floor",
      round(A.gesture_style(u)["amplitude"], 6), 0.30)

print("\n=== 21. What the tier does to the LLM's descriptor words ===")
# The experiment in the prompt grid depends on these words differing across
# tiers. They differ much less on CHATBOX, whose show=0.30 compresses D by 70%
# before the bands are read -- so expressive bandwidth gates language, not just
# servo travel. Asserted so the compression is a known fact, not a surprise.
for robot, want_distinct in (("CHATBOX", 2), ("ELLEBOT", 3)):
    b = A.to_pad(A.ROBOTS[robot]["ocean"])
    words = []
    for t in ladder:
        shown = A.show(A.feel_with_relationship(b, 0.0, 0.0, t),
                       A.ROBOTS[robot]["show"])
        words.append(A.descriptors(shown)[2])
    print(f"    {robot:8s} " + "  ".join(f"{t}={w}" for t, w in zip(ladder, words)))
    check(f"{robot}: {want_distinct} distinct dominance words across the ladder",
          len(set(words)), want_distinct)

print(f"\n{'All checks passed.' if not failures else f'{failures} FAILED.'}\n")
raise SystemExit(1 if failures else 0)
