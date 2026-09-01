```
branch: feature/pad-affect-core
HEAD:   838eda65d9ae2dd3f8ec972dc1896a3f50acbecd
date:   2026-08-31T15:50:26+12:00

working tree at audit time (not clean):
  M CHATBOX_V5-dev/CHATBOX_SERVER/modules/face_webcam/webcam_loop.py
  M CHATBOX_V5-dev/CHATBOX_SERVER/modules/webui/app.py
  ?? CHATBOX_V5-dev/CHATBOX_SERVER/modules/webui/robot_link.py
  (all quotes below are from the WORKING TREE, not from HEAD)

files read:
  CHATBOX_V5-dev/PAD_CORE/pad_core/affect.py
  CHATBOX_V5-dev/PAD_CORE/pad_core/adapter.py
  CHATBOX_V5-dev/PAD_CORE/pad_core/prompt.py
  CHATBOX_V5-dev/PAD_CORE/pad_core/stream.py
  CHATBOX_V5-dev/PAD_CORE/pad_core/servo_style.py
  CHATBOX_V5-dev/PAD_CORE/tests/test_affect.py
  CHATBOX_V5-dev/PAD_CORE/tests/test_pad_core.py
  CHATBOX_V5-dev/PAD_CORE/tests/test_servo_style.py
  CHATBOX_V5-dev/PAD_CORE/docs/CONCEPT.md
  CHATBOX_V5-dev/PAD_CORE/docs/RELATIONSHIP_TO_DOMINANCE.md
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/kg_bridge.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/interactions.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/extraction.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/store.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/schema.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/scales.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/specs/ellebot_spec.yaml
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/face_webcam/webcam_loop.py
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/face_webcam/emotion_detector.py
  CHATBOX_V5-dev/CHATBOX_SERVER/tools/pad_prompt_grid.py
  CHATBOX_V5-dev/CHATBOX_SERVER/test_pad_prompt.py
  CHATBOX_V5-dev/CHATBOX_SERVER/runs/B_dir.md
  CHATBOX-DEMO_V4/CHATBOX_CLIENT/client_config.json
```

---

## A1 — Trait → PAD

Live table, `CHATBOX_V5-dev/PAD_CORE/pad_core/affect.py:39-43`:

```python
WEIGHTS = {
    "P":  {"E": 0.21, "A":  0.59, "N":  0.19},
    "Ar": {"O": 0.15, "A":  0.30, "N": -0.57},
    "D":  {"O": 0.25, "C": 0.17, "E":  0.60, "A": -0.32},
}
```

**Coefficient count.** Ten, not twelve. `P` has three terms (E, A, N), `Ar` has
three (O, A, N), `D` has four (O, C, E, A). `P` and `Ar` carry no `C` term; `D`
carries no `N` term. Absent keys are absent from the dict, not present as zero —
`to_pad` iterates `terms.items()`, so a missing trait contributes nothing.

`affect.py:45`:

```python
TRAIT_KEYS = ("O", "C", "E", "A", "N")
```

`affect.py:48-51`:

```python
def to_pad(traits: Dict[str, float]) -> Dict[str, float]:
    """OCEAN vector -> baseline (P, Ar, D)."""
    return {axis: sum(w * traits[t] for t, w in terms.items())
            for axis, terms in WEIGHTS.items()}
```

**Fifth factor identifier and label.** The identifier is the single character
`"N"` (`affect.py:40`, `:41`, `:45`, and in each persona dict at `:65`, `:70`).
There is **no** long-form name in code — not `neuroticism`, not
`emotional_stability`. `to_pad`'s docstring (`affect.py:49`) is one line and does
not name it: `"""OCEAN vector -> baseline (P, Ar, D)."""`. The module docstring
at `affect.py:10` says only `OCEAN persona   -> baseline PAD           (Mehrabian
regressions, via ALMA)`.

The only place the fifth factor is spelled out is a comment in the module
docstring of the retired file and in prose docs. On this branch, the nearest
in-code gloss is `affect.py:78`:

```python
# Neuroticism carries Arousal *negatively* (-0.57).
```

— **NOT FOUND**: that line does not exist in `affect.py`. The sentence appears
only in `CHATBOX_V5-dev/PAD_CORE/docs/CONCEPT.md:94`:

```
- **Neuroticism** carries Arousal *negatively* (−0.57). See the caveat in §9.
```

So: the code names the factor `N` only; the expansion to "Neuroticism" exists in
documentation, not in `affect.py`.

**Expected input range and remapping.** `[-1, +1]`, and **no remapping happens
inside the function**. `affect.py:36-38`:

```python
# Mehrabian's temperament regressions as used by the ALMA model. Traits are on
# [-1, +1] with 0 as the population mean; a config storing them on [0, 1] must
# be remapped with (2v - 1) first or every coordinate lands too positive.
```

`to_pad` (`affect.py:48-51`, quoted above) performs no scaling of `traits[t]`.

**Clamping.** `to_pad` does **not** clamp. Its return value is a raw weighted
sum. The clamp lives downstream in `_clamp` / `feel_with_relationship`,
`affect.py:119-120` and `:147-149`:

```python
def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))
```

```python
    return {"P":  _clamp(felt["P"] + dP),
            "Ar": _clamp(felt["Ar"] + dAr),
            "D":  _clamp(felt["D"] + dD)}
```

**Call ordering.** `pipeline` (`affect.py:372-374`):

```python
    baseline = to_pad(traits)
    felt = feel_with_relationship(baseline, valence, arousal, tier, empathy)
    shown = show(felt, ROBOTS[robot]["show"])
```

So the order is `to_pad` (unclamped) → `feel` (unclamped, `affect.py:89-93`) →
tier offset added → `_clamp` → `show`. An out-of-range **baseline** produced by
`to_pad` is therefore carried unclamped through `feel` and is only clamped after
the tier offset is added. See A12.2 for the reachable range.

**Agreeableness coefficient in the Arousal regression.** It is **`0.30`**, not
`0.20`. `affect.py:41`:

```python
    "Ar": {"O": 0.15, "A":  0.30, "N": -0.57},
```

---

## A2 — Persona definitions

Grepped `*.py`, `*.yaml`, `*.yml`, `*.json`, `*.toml` across the repository.

**1. `CHATBOX_V5-dev/PAD_CORE/pad_core/affect.py:63-74` — the live definition.**

```python
ROBOTS = {
    "CHATBOX": {
        "ocean": {"O": -0.5, "C": 0.2, "E": -0.6, "A": 0.6, "N":  0.2},
        "show": 0.30,
        "body": "fixed tabletop, 12-DOF upper face",
    },
    "ELLEBOT": {
        "ocean": {"O":  0.5, "C": 0.4, "E":  0.7, "A": 0.6, "N": -0.4},
        "show": 1.00,
        "body": "wheeled base, fan ears, 12-DOF upper face",
    },
}
```

Scale: `[-1, +1]`. This is the only definition consumed by `pad_core`; every
other reader dereferences it (`adapter.py:53`, `webui/app.py:434`, `:438`,
`:458`, `tools/e2e_pipeline.py:232`, `bench/*.py`, `tests/*`).

**2. `CHATBOX-DEMO_V4/CHATBOX_CLIENT/client_config.json:89-95` — a second trait
vector, on a different scale.**

```json
    "personality": {
        "A": 0.9,
        "C": 0.7,
        "E": 0.8,
        "N": 0.3,
        "O": 0.6
    }
```

`CONTRADICTION` (scale + values). All five values lie in `[0, 1]`, and none
matches either persona in `affect.ROBOTS`. Per `affect.py:36-38` such a config
"must be remapped with (2v - 1) first or every coordinate lands too positive" —
no remapping call site was found for this file.

Status of this file: it **is tracked on this branch** (`git ls-files
--error-unmatch` succeeds; `CHATBOX-DEMO_V4` carries 36 tracked files). Grepping
`*.py` for the consuming key found only
`CHATBOX-DEMO_V4/CHATBOX_CLIENT/client.py:223`:

```python
            "personality": "personality",
```

No call path from this JSON block into `to_pad` or `affect.ROBOTS` was found.
Recorded as a contradictory definition; whether it is live is listed under OPEN.

**3. `CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/specs/ellebot_spec.yaml`
— trait values in a comment only, no machine-readable OCEAN block.**

Lines 5-14:

```yaml
# ElleBot is CHATBOX's deliberate opposite on the trait that matters. Their
# OCEAN vectors are held equal on Agreeableness (+0.6, both are warm) and run
# opposite on Extraversion (CHATBOX -0.6, ELLEBOT +0.7), which swings baseline
# Dominance by more than a full unit (-0.643 vs +0.421) — that single trait is
# why the two read as different creatures before either one speaks.
#
# The persona words below are the LANGUAGE face of that vector: what goes in the
# prompt's IDENTITY block. They must agree with the OCEAN values in
# pad_core/affect.py ROBOTS["ELLEBOT"], or the robot will describe itself as one
# thing while behaving as another.
```

Lines 15-18 carry only string persona fields:

```yaml
id: ellebot
name: ElleBot
embodiment: ELEPHANT     # one of the schema Embodiment enum values (CAT | ELEPHANT)
persona: "extroverted, outgoing, playful"
```

The quoted trait values (`A +0.6`, `CHATBOX E -0.6`, `ELLEBOT E +0.7`) and the
quoted baselines (`-0.643`, `+0.421`) **agree** with `affect.ROBOTS` and with the
recomputation in A12.1. No disagreement. The other two specs in that directory
are `human_spec.yaml` and `robot_spec.yaml`.

**4. Retired definition — not on this branch.** `config.py`, holding
`CHATBOX_PERSONA` / `ELLEBOT_PERSONA`, was deleted in commit `8742a04`. That
commit's message records the disagreement it carried:

```
- config.py — its personas CONTRADICTED affect.ROBOTS. CHATBOX's neuroticism
  was -0.5 there and +0.2 in AFFECT_LAB, and +0.2 is the value CONCEPT.md's
  table publishes and its "quirk, still open" section argues about.
```

It survives on branches `feature/integration`, `feature/cultural-awareness`,
`first-impression` (path
`CHATBOX_V5-dev/CHATBOX_SERVER/modules/pad_persona/config.py`) and
`KG-knowledge-extraction`, `feature/pad-persona-engine` (path
`CHATBOX-DEMO_V4/...`). Not checked out.

---

## A3 — Tier → Dominance, and application order

`affect.py:109-114`:

```python
TIER_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "close":   (0.00, +0.10, +0.40),
    "known":   (0.00,  0.00,  0.00),   # reference tier: no displacement at all
    "visitor": (0.00,  0.00, -0.20),
    "unknown": (0.00,  0.00, -0.40),
}
```

Four tiers. `close` displaces **Arousal as well as Dominance** (`+0.10` on Ar).
No tier displaces Pleasure.

`affect.py:123-125`:

```python
def tier_offset(tier: str) -> Tuple[float, float, float]:
    """(dP, dAr, dD) for a relationship tier. Unrecognised -> 'unknown'."""
    return TIER_OFFSETS.get(tier, TIER_OFFSETS["unknown"])
```

**Exact ordering, trait vector → servo.** Each step with its line reference:

| # | operation | location |
|---|---|---|
| 1 | `baseline = to_pad(traits)` — unclamped weighted sum | `affect.py:372`, defn `:48-51` |
| 2 | `felt = feel(baseline, v, a, empathy)` — empathy fusion on P and Ar only; D passed through untouched; **no clamp** | `affect.py:145`, defn `:89-93` |
| 3 | `dP, dAr, dD = tier_offset(tier)` | `affect.py:146` |
| 4 | `_clamp(felt[axis] + d…)` — tier offset added **and** clamped to `[-1,+1]` in the same expression | `affect.py:147-149` |
| 5 | `shown = show(felt, ROBOTS[robot]["show"])` | `affect.py:374`, defn `:152-154` |
| 6 | `style = gesture_style(felt)` — reads **felt**, i.e. step 4, **not** step 5 | `affect.py:385`; adapter `adapter.py:107` |
| 7 | `wire = servo_style.wire_message(style)` → `clamp_style` to `STYLE_LIMITS` | `adapter.py:117`; `servo_style.py:247-253` |
| 8 | `angle = rest + amplitude*(target-rest)`, `+= droop*DROOP_DEG`, `+= posture*POSTURE_DEG`, clamp to stock servo range | `servo_style.py:266-276` |

Step 2 is quoted here, `affect.py:89-93`:

```python
    return {
        "P":  baseline["P"] + empathy * (valence - baseline["P"]),
        "Ar": baseline["Ar"] + empathy * (arousal - baseline["Ar"]),
        "D":  baseline["D"],
    }
```

Note the ordering inside `feel_with_relationship` (`affect.py:145-149`): empathy
fusion runs **first**, then the tier offset is added. The header sequence
proposed in the audit brief (`baseline → tier offset → empathy fusion → clamp`)
is **not** the implemented order; the real order is `baseline → empathy fusion →
tier offset → clamp`. Because `feel` does not touch D and the tier offsets are
all zero on P, the two orders coincide numerically on P and D, but **not on Ar at
tier `close`**, whose `+0.10` is added after the empathy fusion rather than being
fused.

**Is the tier offset applied before or after `show_fraction`?** **Before.**
Step 4 precedes step 5. `affect.py:138-143` states the intent:

```python
    Applied BEFORE `show`, deliberately: `gesture_style` reads the *felt*
    coordinate, not the shown one, so a tier applied after `show` would never
    reach a servo at all. The design also admits exactly one embodiment scaling.
```

**Is there a clamp between the tier offset and the style computation?** **Yes**,
and it is the same expression that applies the offset — `_clamp` at
`affect.py:147-149`. `gesture_style` then applies a second, different clamp to
`STYLE_LIMITS` (`affect.py:358-359`).

**`show_fraction` does not reach servos, words, or the prompt.** `shown` is
computed at `adapter.py:96` and consumed at `adapter.py:115` for `affect_name`
only:

```python
            "name":        affect.affect_name(shown["P"], shown["Ar"]),
```

`words` come from `felt` (`adapter.py:103`) and `style` from `felt`
(`adapter.py:107`). Downstream, `pad["shown"]` is read only by display code:
`webui/app.py:272`, `:621`, `:645`; `bench/webcam_demo.py:425`;
`bench/preview.py:85-86`; `bench/live_demo.py:343`; and two assertions in
`tests/test_pad_core.py:69` and `tests/test_affect.py:80`. No servo path and no
prompt path reads it.

`CONTRADICTION`. `CONCEPT.md:162-171` presents `show` as a pipeline stage between
felt PAD and the outputs:

```
shown = show_fraction · felt
```

and `CONCEPT.md:61` places it in the flow chart as `E --> S["shown PAD<br/>scaled
by what the body expresses"] --> W["3 words<br/>-> LLM prompt"]`. In the code the
three words are derived from `felt`, not from `shown` (`adapter.py:103`,
`affect.py:381`). `adapter.py:98-102` and `affect.py:381` both record the change;
the CONCEPT.md flow chart does not.

---

## A4 — Tier derivation

`kg_bridge.py:104-124`, in full:

```python
def _tier_from_scores(rapport: float, trust: float, count: int) -> str:
    """
    Core tier thresholds (unchanged behaviour).

    score = (rapport + trust) / 2
      score > 0.70  → "close"
      score > 0.45  → "known"
      count > 5     → "known"   (even with low score — seen enough turns)
      count > 0     → "visitor"
      else          → "unknown"
    """
    score = (rapport + trust) / 2.0
    if score > 0.70:
        return "close"
    if score > 0.45:
        return "known"
    if count > 5:
        return "known"
    if count > 0:
        return "visitor"
    return "unknown"
```

`kg_bridge.py:127-139`, in full:

```python
def _tier_from_edges(relationship_edges: List[AnyEdge]) -> str:
    """Tier from a pre-fetched edge list (used by tests that build edges directly)."""
    rapport = 0.0
    trust = 0.0
    count = 0
    for edge in relationship_edges:
        if edge.edge_type == "rapport":
            rapport = edge.weight
        elif edge.edge_type == "trust":
            trust = edge.weight
        elif edge.edge_type == "interaction_count":
            count = edge.count
    return _tier_from_scores(rapport, trust, count)
```

`kg_bridge.py:142-155`, in full:

```python
def derive_tier(person_id: str, robot_id: str, store: GraphStore) -> str:
    """
    Derive the relationship tier from the pair's InteractionNode.

    Closeness (rapport, trust) and the interaction count are now FIELDS on the
    single InteractionNode — there are no direct person→robot rapport/trust
    edges. Scoring thresholds are unchanged (_tier_from_scores).
    """
    interaction = get_interaction(store, person_id, robot_id)
    if interaction is None:
        return "unknown"
    return _tier_from_scores(
        interaction.rapport, interaction.trust, interaction.interaction_count,
    )
```

**Branches and thresholds.** Five returns, evaluated in order:
`score > 0.70` → `close`; `score > 0.45` → `known`; `count > 5` → `known`;
`count > 0` → `visitor`; fallthrough → `unknown`. Plus the null guard at
`kg_bridge.py:151-152` returning `unknown` when no `InteractionNode` exists.
`score = (rapport + trust) / 2.0` (`kg_bridge.py:115`).

**Is `close` reachable?** **Yes.** `kg_bridge.py:116-117`:

```python
    if score > 0.70:
        return "close"
```

Both `rapport` and `trust` are clamped to `[0,1]` (`interactions.py:81-83`,
`:97-98`), so `score` ranges over `[0,1]` and `score > 0.70` is satisfiable.
Exercised in practice by `tools/pad_prompt_grid.py:67`, which reaches it with
`"close": (0.80, 0.75, 9)`.

`CONTRADICTION`. `RELATIONSHIP_TO_DOMINANCE.md:74-80` prints the same function
**without the `close` branch**:

```
score = (rapport + trust) / 2

    score > 0.45          →  known
    interaction_count > 5 →  known      (familiarity without warmth)
    interaction_count > 0 →  visitor
    otherwise             →  unknown
```

The `score > 0.70 → close` line is absent from the document's §4, while §5, §6
and §7 of the same document all tabulate `close`.

**Where `interaction_count` is read from, and set vs incremented.** Read at
`kg_bridge.py:154` off the `InteractionNode` field. It is **set (recomputed),
never incremented**. Two writers, both assignment:

`interactions.py:185-192`:

```python
def sync_interaction_count(store: GraphStore, person_id: str, robot_id: str) -> InteractionNode:
    """Recompute interaction_count (total turns) onto the InteractionNode."""
    node = get_or_create_interaction(store, person_id, robot_id)
    total = sum(s.turn_count for s in sessions_of(store, node.id))
    if total != node.interaction_count:
        node = node.model_copy(update={"interaction_count": total})
        store.upsert_node(node)
    return node
```

`interactions.py:195-204`:

```python
def set_interaction_count(store: GraphStore, person_id: str, robot_id: str,
                          count: int, *, source: Optional[str] = None) -> InteractionNode:
    """Set interaction_count directly (e.g. from an external transcript store when
    SessionNodes no longer live in the graph). Pure; no LLM/DB imports here."""
    node = get_or_create_interaction(store, person_id, robot_id, source=source)
    count = max(0, int(count))
    if count != node.interaction_count:
        node = node.model_copy(update={"interaction_count": count})
        store.upsert_node(node)
    return node
```

`sync_interaction_count` is called once per turn from `kg_bridge.py:359`:

```python
        sync_interaction_count(self._store, person_id, robot_node_id)
```

**Is a `family` tier still present anywhere?** **Not in live code or
configuration.** `TIER_OFFSETS` (`affect.py:109-114`) has four keys and
`_tier_from_scores` has no `family` branch. Grep over `*.py`, `*.yaml`, `*.json`,
`*.md` under `CHATBOX_V5-dev/` returns only:

- `CHATBOX_V5-dev/PAD_CORE/docs/RELATIONSHIP_TO_DOMINANCE.md:86-87` — the removal
  note itself.
- `CHATBOX_V5-dev/CHATBOX_SERVER/runs/grid_20260813-160631.md:19`, `:26`, `:36`,
  `:65`, `:72`, `:73`, `:84` — an archived experiment report from 2026-08-13,
  predating the removal.
- `CHATBOX_V5-dev/CHATBOX_SERVER/docs/PROGRESS.md:256` — the word "family" in an
  unrelated sentence about biographical facts.

---

## A5 — Style vector

`affect.py:328-334`:

```python
STYLE_LIMITS = {
    "amplitude": (0.30, 1.00),   # scale on each servo's travel from neutral
    "tempo":     (0.50, 1.60),   # multiplier on playback speed
    "posture":   (-1.00, 1.00),  # neck/shoulder carriage, withdrawn -> open
    "droop":     (-1.00, 1.00),  # signed valence tint, sagging -> lifted
    "idle":      (0.05, 1.00),   # how often it stirs between gestures
}
```

`affect.py:338`:

```python
DROOP_GAIN = 1.4
```

`affect.py:341-344`:

```python
def _onto(value: float, name: str) -> float:
    """Map a published [0,1] index onto the range this firmware documents."""
    lo, hi = STYLE_LIMITS[name]
    return lo + value * (hi - lo)
```

`affect.py:347-359`, the five computations and their clamp:

```python
def gesture_style(coord: Dict[str, float]) -> Dict[str, float]:
    """PAD -> the five movement parameters the firmware consumes."""
    raw = {
        # published — Hagane & Venture (2022)
        "amplitude": _onto(_spatial_extent(coord), "amplitude"),
        "tempo":     _onto(_velocity(coord), "tempo"),
        # ours — the deck, p.15 and p.16
        "posture":   0.70 * coord["D"] + 0.30 * coord["P"],
        "idle":      0.45 + 0.40 * coord["Ar"],
        "droop":     -coord["P"] * DROOP_GAIN,
    }
    return {k: max(lo, min(hi, raw[k]))
            for k, (lo, hi) in STYLE_LIMITS.items()}
```

`affect.py:323-325`:

```python
def _spatial_extent(coord: Dict[str, float]) -> float:
    """Hagane & Venture (2022) Eq. 9 — Sp, on [0, 1]. Dominance only."""
    return (coord["D"] + 1) / 2
```

`affect.py:305-320`:

```python
def _velocity(coord: Dict[str, float]) -> float:
    """Hagane & Venture (2022) Eq. A1 — the fv mapping, on [0, 1].
    ...
    """
    Pn = coord["P"] + 1
    An = coord["Ar"] + 1
    Dn = coord["D"] + 1
    r = math.hypot(An, Dn)
    if r == 0:                      # Bored corner; the paper's fv is 0 there
        return 0.0
    beta = math.acos(max(-1.0, min(1.0, Dn / r)))
    S = 2 + _HALF_ROOT_TWO_GAP * math.sin(2 * beta + math.pi)
    return 0.125 * r / S * (4 - Pn)
```

A second, independent clamp to the same limits is applied on the wire,
`servo_style.py:247-253`:

```python
def clamp_style(style: Dict[str, float]) -> Dict[str, float]:
    """Fill in missing keys and clamp the rest. Never trust the wire."""
    out = dict(NEUTRAL_STYLE)
    out.update({k: v for k, v in (style or {}).items() if k in STYLE_LIMITS})
    for k, (lo, hi) in STYLE_LIMITS.items():
        out[k] = max(lo, min(hi, float(out[k])))
    return out
```

**Computed — CHATBOX, neutral face (v=0, a=0), default empathy 0.6, all four
tiers.** Run against the live module:

| tier | felt D | amplitude | ref | tempo | ref | posture | ref | droop | idle |
|---|---|---|---|---|---|---|---|---|---|
| unknown | −1.0000 | 0.3000 | 0.300 | 0.6982 | 0.698 | −0.6681 | −0.67 | −0.1490 | 0.4486 |
| visitor | −0.8430 | 0.3549 | 0.355 | 0.7205 | 0.721 | −0.5582 | −0.56 | −0.1490 | 0.4486 |
| known | −0.6430 | 0.4249 | 0.425 | 0.7587 | 0.759 | −0.4182 | −0.42 | −0.1490 | 0.4486 |
| close | −0.2430 | 0.5649 | 0.565 | 0.8650 | 0.865 | −0.1382 | −0.14 | −0.1490 | 0.4886 |

All three reference series **agree** with the live code to the precision the
references are quoted at.

**Which candidate equation the code implements.** Neither of the two as written —
it implements the second one exactly, once `_onto` is expanded. `_onto` with
`STYLE_LIMITS["amplitude"] = (0.30, 1.00)` gives
`0.30 + Sp × 0.70` where `Sp = (D+1)/2`, i.e. `amplitude = 0.30 + 0.70·(D+1)/2`.
Verified numerically at all four tiers:

| tier | live | `0.75 + 0.45·Ar + 0.20·D` | `0.30 + 0.70·(D+1)/2` |
|---|---|---|---|
| unknown | 0.3000 | 0.5484 ✗ | 0.3000 ✓ |
| visitor | 0.3549 | 0.5798 ✗ | 0.3549 ✓ |
| known | 0.4249 | 0.6198 ✗ | 0.4249 ✓ |
| close | 0.5649 | 0.7448 ✗ | 0.5649 ✓ |

Candidate 2 matches to machine precision (`abs(live - cand) < 1e-9`) at every
tier. Candidate 1 matches at none. Arousal does not enter amplitude at all.

At tier `unknown` the value 0.3000 sits exactly on the `STYLE_LIMITS` floor — see
A12.3 for the clamp that produces it.

---

## A6 — Rapport / trust accrual

**Write sites.** Three, all reaching the `InteractionNode` through
`interactions.py`.

**(1) Live per-tick accrual.** `webcam_loop.py:1522-1525`:

```python
        p = pad["pad_state"][0]
        if p > 0.05:
            _update_rapport_trust(self.store, person_id, self.robot_id,
                                  delta=0.025 * p)
```

Gate: felt Pleasure `> 0.05`. Magnitude: `0.025 × P`. Trigger: every
`_pipeline_tick` call. The helper, `webcam_loop.py:690-695`:

```python
    r_cur, t_cur = _read_rapport_trust(store, person_id, robot_id)
    r_new = min(1.0, r_cur + delta)
    t_new = min(1.0, t_cur + delta)
    # Closeness lives on the pair's InteractionNode.
    set_closeness(store, person_id, robot_id, rapport=r_new, trust=t_new,
                  source="webcam_loop")
```

The **same** `delta` is added to both. `min(1.0, …)` only — no lower bound here,
though `set_closeness` clamps to `[0,1]` at `interactions.py:81-83`.

**(2) End-of-session LLM extraction.** `extraction.py:178-181`:

```python
    if update.rapport_delta or update.trust_delta:
        adjust_closeness(store, person_id, robot_id,
                         d_rapport=update.rapport_delta, d_trust=update.trust_delta,
                         source=source)
```

Called from `webcam_loop.py:1961-1964`. Cap, `extraction.py:35`:

```python
_MAX_DELTA = 0.2          # per-session cap on rapport/trust change
```

applied at `extraction.py:146-147`:

```python
        rapport_delta=_clamp_delta(obj.get("rapport_delta")),
        trust_delta=_clamp_delta(obj.get("trust_delta")),
```

The model's entire brief, `extraction.py:50-51`:

```
    "rapport_delta rises with warmth and positive affect; trust_delta rises with "
    "the child sharing personal things. Use values near 0 if the exchange was "
```

`adjust_closeness` clamps to `[0,1]`, `interactions.py:96-99`:

```python
    node = node.model_copy(update={
        "rapport": max(0.0, min(1.0, node.rapport + d_rapport)),
        "trust":   max(0.0, min(1.0, node.trust + d_trust)),
    })
```

**(3) Manual operator key.** `webcam_loop.py:2469-2474`:

```python
                    elif key in (ord("b"), ord("B")) and last_person_id:
                        with self._store_lock:
                            _update_rapport_trust(
                                self.store, last_person_id, self.robot_id,
                                delta=0.15, verbose=True,
                            )
```

Same helper as (1), so `+0.15` to **both**, on each keypress.

**Does `trust` have any write path that `rapport` does not?** **No.** Paths (1)
and (3) both route through `_update_rapport_trust`, which computes `r_new` and
`t_new` from the identical `delta` (`webcam_loop.py:691-692`, quoted above). Path
(2) is the only one carrying independent `d_rapport` / `d_trust` values, and it
writes both together. Two façade methods exist that write one field alone —
`kg_bridge.py:386-392`:

```python
    def set_rapport(self, person_id: str, robot_id: str, weight: float) -> None:
        self._ensure(person_id, robot_id)
        set_closeness(self._store, person_id, robot_id, rapport=weight, source="kg-facade")

    def set_trust(self, person_id: str, robot_id: str, weight: float) -> None:
        self._ensure(person_id, robot_id)
        set_closeness(self._store, person_id, robot_id, trust=weight, source="kg-facade")
```

but they are symmetric (each has a counterpart) and belong to the `KG` REPL
façade described at `kg_bridge.py:366-372` as "Thin convenience wrapper for
scripts and notebooks". No asymmetric runtime path was found.

**Is `disclosure_depth` written anywhere?** **NOT FOUND as a field.** It appears
exactly once in the tree, in a docstring —
`modules/graph_relationship/store.py:64`:

```
    relationship_edges     — rapport, trust, disclosure_depth, interaction_count
```

The adjacent live constant, `store.py:49-51`, does not include it:

```python
_RELATIONSHIP_TYPES: frozenset = frozenset(
    {"rapport", "trust", "interaction_count"}
)
```

`grep -n disclosure modules/graph_relationship/schema.py` returns only line 10, a
module-docstring mention: `with each child (rapport, trust, disclosure depth,
etc.)`. There is no `disclosure_depth` field on `InteractionNode`, no edge type,
and no write site. It is documentation-only.

**Time-based decay on any relationship quantity — NOT FOUND.** Searched
`decay`, `exp(`, `half_life`, `halflife`, `elapsed`, `last_seen` across
`CHATBOX_V5-dev/**/*.py`. Every hit is unrelated to rapport/trust:

- `schema.py:22` — `(built separately) branches on this flag to decide decay cadence and promotion` (describes a module that is not in the tree)
- `schema.py:67` — `FAST = "FAST"   # mood, attention — decay within a session`
- `schema.py:275` — `"""Current affective state — FAST, decays between sessions.`
- `webcam_loop.py:837`, `:875`, `:884` — `phase_elapsed`, face-identification phase timing
- `webcam_loop.py:2146-2148` — `elapsed`, FPS counter

The three `schema.py` hits are comments asserting decay semantics; no code
implements them. Neither `rapport`, `trust`, nor `interaction_count` is ever
reduced by elapsed time. The only decreasing path for rapport/trust is a negative
LLM delta via write site (2).

**Tick rate of the loop containing the live-tick accrual.** 1.0 s by default.
`webcam_loop.py:99`:

```python
_DEFAULT_TICK    = 1.0
```

`webcam_loop.py:1128` (constructor default) and `:2686-2687` (CLI):

```python
    p.add_argument("--tick",       type=float, default=_DEFAULT_TICK,
                   help="Pipeline tick interval in seconds (default: 1.0)")
```

Gate, `webcam_loop.py:2257`:

```python
                if not enroll_capturing and now - last_tick_t >= self.tick_interval:
```

`_pipeline_tick` is called once per detected person per tick,
`webcam_loop.py:2265-2270`:

```python
                        for d in raw_dets:
                            pid = d["person_id"]
                            if pid is None:
                                continue
                            if self._pad_enabled:
                                bi, pad = self._pipeline_tick(pid, d["emotion"], va=d.get("va"))
```

Accrual is suppressed while `enroll_capturing` is true.

---

## A7 — The affect ↔ graph seam

`pre_turn`, `kg_bridge.py:229-283`. The relevant body, `:244-283`:

```python
        if camera_va is not None:
            camera_v, camera_a = camera_va
        else:
            camera_v, camera_a = emotion_label_to_va(camera_emotion)

        if person_id is None or self._store.get_node(person_id) is None:
            return BridgeInput(valence=camera_v, arousal=camera_a,
                               tier="unknown", structured_memory="")

        ctx = self._store.get_person_context(person_id)

        # Tier + closeness now come from the pair's InteractionNode.
        # D is never read from the graph here.
        tier = derive_tier(person_id, robot_id, self._store)
        interaction = get_interaction(self._store, person_id, robot_id)
        rapport = interaction.rapport if interaction else 0.0
        trust = interaction.trust if interaction else 0.0
        count = interaction.interaction_count if interaction else 0

        # Valence blend: camera is primary (0.7); graph MoodEdge softens spikes (0.3).
        # Arousal is NOT blended — only the camera frame contributes A.
        graph_mood = next(
            (e.value for e in ctx.person_attribute_edges if e.edge_type == "mood"),
            None,
        )
        blended_v = (0.7 * camera_v + 0.3 * graph_mood) if graph_mood is not None else camera_v

        # Structured memory: SLOW edges only — text path, never numeric.
        structured_memory = format_slow_edges(ctx.person_attribute_edges)

        return BridgeInput(
            valence=blended_v,
            arousal=camera_a,
            tier=tier,
            structured_memory=structured_memory,
            rapport=rapport,
            trust=trust,
            interaction_count=count,
        )
```

`post_turn`, `kg_bridge.py:285-359`. The write body, `:310-359`:

```python
        if person_id is None:
            return

        _ensure_person_node(self._store, person_id)
        robot_node_id = _ensure_robot_node(self._store, robot_id)

        p, a, _d = pad_result["pad_state"]
        prov = _prov(robot_id)

        # AttentionEdge expects [0, 1]; rescale PAD arousal from [-1, 1]
        attention_value = max(0.0, min(1.0, (a + 1.0) / 2.0))

        # Self-attribute edges (mood + attention). No graph scan (apply_delta contract).
        self._store.apply_delta(
            edges=[
                MoodEdge(
                    source_id=person_id,
                    target_id=person_id,
                    provenance=prov,
                    value=max(-1.0, min(1.0, p)),
                ),
                AttentionEdge(
                    source_id=person_id,
                    target_id=person_id,
                    provenance=prov,
                    value=attention_value,
                ),
            ]
        )
```

followed by the interaction/session append at `:340-359`.

**What `post_turn` writes, and from which PAD value.** Two edges, both from
`pad_result["pad_state"]` (`kg_bridge.py:316`), which the adapter populates from
**felt**, not shown and not baseline — `adapter.py:110`:

```python
            "pad_state":   (felt["P"], felt["Ar"], felt["D"]),
```

- `MoodEdge.value` ← felt **P**, clamped `[-1,1]` (`kg_bridge.py:325`)
- `AttentionEdge.value` ← felt **Ar**, rescaled `(a+1)/2` to `[0,1]` (`kg_bridge.py:320`, `:332`)
- Dominance is **not** written. `kg_bridge.py:307-308`:

```
        D (Dominance) is NOT written — it is re-derived each turn and must never
        be persisted as its own edge.
```

Also written: a turn appended to the current `SessionNode` (`:354-357`) and
`sync_interaction_count` (`:359`). Rapport/trust are **not** written by
`post_turn` — `kg_bridge.py:301-302`.

**What `pre_turn` reads and the exact blend coefficient.** It reads
`get_person_context(person_id)` (`:254`), the derived tier (`:258`), the
`InteractionNode`'s rapport/trust/count (`:259-262`), the first `mood` edge
(`:266-269`), and SLOW edges as text (`:273`). The blend is `0.7 / 0.3`,
`kg_bridge.py:270`:

```python
        blended_v = (0.7 * camera_v + 0.3 * graph_mood) if graph_mood is not None else camera_v
```

**Arousal is not blended** (`kg_bridge.py:265` comment; `:277` passes
`camera_a` straight through).

**Does the recurrence compound?** **Yes.** `post_turn` stores felt P as the
`MoodEdge`; the next `pre_turn` reads that same edge back as `graph_mood` and
mixes it into `valence`, which `feel` then converts into the next felt P. There is
no flag or guard preventing the loop. Writing `m_t` for the stored MoodEdge value
after turn *t*, `c` for camera valence, `b` for `baseline["P"]` and `e` for
`EMPATHY`:

```
blended_v_{t+1} = 0.7·c_{t+1} + 0.3·m_t                      kg_bridge.py:270
felt_P_{t+1}    = b + e·(blended_v_{t+1} − b)                affect.py:90
m_{t+1}         = clamp(felt_P_{t+1}, −1, 1)                 kg_bridge.py:325

⇒  m_{t+1} = clamp( b(1−e) + 0.7·e·c_{t+1} + 0.3·e·m_t )
```

A first-order IIR filter with pole `0.3·e = 0.18` (at the default
`EMPATHY = 0.6`, `affect.py:79`). It is contracting, so it converges rather than
diverging. Traced numerically on CHATBOX (`b = 0.2660`, `e = 0.60`) with constant
camera valence `c = 0.80`:

| turn | blended_v | felt_P (stored as mood) |
|---|---|---|
| 1 | +0.80000 | +0.58640 |
| 2 | +0.73592 | +0.54795 |
| 3 | +0.72439 | +0.54103 |
| 4 | +0.72231 | +0.53979 |
| 5 | +0.72194 | +0.53956 |
| 8 | +0.72185 | +0.53951 |

Analytic fixed point `(b(1−e) + 0.7·e·c)/(1 − 0.3·e) = 0.53951`. The single-shot
value with no recurrence would be `0.58640`. So the feedback path **reduces**
steady-state felt Pleasure by 0.0469 (−8.0%) relative to the same camera input
with the blend disabled, and introduces a settling transient of ~4 turns.

This interacts with A6 write site (1), whose gate is `p > 0.05` and whose
magnitude is `0.025 × p`: the accrual rate is driven by the post-recurrence
value, not the raw camera reading.

`MoodEdge` is declared FAST and documented as decaying — `schema.py:275`:

```python
    """Current affective state — FAST, decays between sessions.
```

No decay implementation was found (A6). Whether the edge survives across process
restarts depends on graph persistence, which is listed under OPEN.

---

## A8 — Face → valence/arousal

**It is a discrete label lookup under a softmax-weighted blend — not a continuous
V/A regression.**

The runtime model, `emotion_detector.py:164-165`:

```python
        if b == 'hsemotion':
            return HSEmotionDetector(model='enet_b0_8_best_vgaf', **kwargs)
```

with the B2 alternative at `:166-167` (`model='enet_b2_8'`) and the default
backend `'hsemotion'` at `:154` and `:287-288`. Loaded at `:215`:

```python
        self._recognizer = HSEmotionRecognizer(model_name=model)
```

`emotion_detector.py:198-199` names it:

```
    Models (cached to ~/.hsemotion/ on first use):
        enet_b0_8_best_vgaf  — B0, AffectNet+VGAF, 8 classes  [default, fastest]
```

The conversion, `emotion_detector.py:219-230`:

```python
    def _infer(self, face_bgr: np.ndarray) -> tuple[str, float, float, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        label, scores = self._recognizer.predict_emotions(face_rgb, logits=False)
        conf = float(np.max(scores)) * 100.0
        # Weighted-blend V/A from full softmax distribution (all 8 class probs)
        v, a = 0.0, 0.0
        for i, score in enumerate(scores):
            lbl = _norm(self._labels[i])
            lv, la = _VA_TABLE.get(lbl, (0.0, 0.0))
            v += score * lv
            a += score * la
        return label, conf, float(v), float(a)
```

The V/A output is `Σ_i softmax_i × table[label_i]` — a convex combination of
**eight fixed table entries**. The model emits class probabilities; it does not
emit valence or arousal.

The table, `emotion_detector.py:41-50`:

```python
# Russell (1980) circumplex — mirrors kg_bridge._EMOTION_VA + contempt/disgust alias
_VA_TABLE: dict[str, tuple[float, float]] = {
    "angry":    (-0.6,  0.7),
    "disgust":  (-0.6,  0.3),
    "fear":     (-0.5,  0.8),
    "happy":    ( 0.8,  0.6),
    "neutral":  ( 0.0,  0.0),
    "sad":      (-0.7, -0.4),
    "surprise": ( 0.1,  0.8),
}
```

`Contempt` is folded onto `disgust`, `emotion_detector.py:55`:

```python
    'Contempt':  'disgust',   # contempt ≈ disgust in Russell VA space
```

These values reach the pipeline: `webcam_loop.py:1105` packs `"va": (ev, ea)`,
`:2270` passes `va=d.get("va")` into `_pipeline_tick`, and `kg_bridge.py:244-245`
uses it in preference to the label lookup:

```python
        if camera_va is not None:
            camera_v, camera_a = camera_va
```

**`CONTRADICTION` — three V/A tables disagree, and the runtime one is the odd one
out.** Compared programmatically:

| label | `emotion_detector._VA_TABLE` | `kg_bridge._EMOTION_VA` | `affect.CATEGORY_VA` | |
|---|---|---|---|---|
| angry | (−0.6, 0.7) | (−0.7, 0.65) | (−0.7, 0.65) | DIFFER |
| disgust | (−0.6, 0.3) | (−0.7, 0.3) | (−0.7, 0.3) | DIFFER |
| fear | (−0.5, 0.8) | (−0.65, 0.72) | (−0.65, 0.72) | DIFFER |
| happy | (0.8, 0.6) | (0.8, 0.5) | (0.8, 0.5) | DIFFER |
| neutral | (0.0, 0.0) | (0.0, 0.0) | (0.0, 0.0) | OK |
| sad | (−0.7, −0.4) | (−0.7, −0.38) | (−0.7, −0.38) | DIFFER |
| surprise | (0.1, 0.8) | (0.2, 0.8) | (0.2, 0.8) | DIFFER |

Six of seven labels differ. `emotion_detector.py:41` asserts the opposite in a
comment: `# Russell (1980) circumplex — mirrors kg_bridge._EMOTION_VA + …`.
`kg_bridge.py:50-51` claims a test guards this:

```python
# imports nothing outside itself (the purity contract) — test_pad_affect asserts
# the two agree on every shared label, so they cannot drift apart unnoticed.
```

`test_pad_affect.py` does **not exist on this branch** — see A10. The surviving
test, `tests/test_pad_core.py:110-116`, compares only the adapter's re-export
against `affect.CATEGORY_VA`, and its own docstring defers the kg_bridge copy to
a file that is also absent:

```python
def test_va_tables_agree():
    """The adapter re-exports affect.CATEGORY_VA; they must not drift apart.
    (The consuming server keeps a third, deliberate copy in kg_bridge so its
    graph package imports nothing — test_pad_integration checks that one.)"""
    for label in EMOTION_VA:
        assert EMOTION_VA[label] == affect.CATEGORY_VA[label], f"adapter drift: {label}"
```

Nothing anywhere references `emotion_detector._VA_TABLE` in a test.

**`CONTRADICTION` — the documented camera model is not the one loaded.**
`CONCEPT.md:128-130`:

```
The camera side uses a model that reports valence and arousal directly
(`enet_b0_8_va_mtl`, EfficientNet-B0 trained on AffectNet), so no lookup table is
needed. A classifier-plus-lookup fallback exists in `CATEGORY_VA` for comparison.
```

The string `enet_b0_8_va_mtl` does not appear anywhere in the code; the loaded
model is `enet_b0_8_best_vgaf` (`emotion_detector.py:165`), an 8-class classifier,
and a lookup table **is** used (`emotion_detector.py:227`). The claim "no lookup
table is needed" is contradicted by the runtime path.

A second backend exists, `EfficientNetDetector` (`emotion_detector.py:241-267`),
wrapping `modules/emotion_processor.py` with weights
`efficientnet_HQRAF_improved_withCon.pth` (present in the additional working
directory `…/v6.0.0/server/modules/emotion/`). It is not the default
(`emotion_detector.py:154`, `:287-288`).

---

## A9 — Descriptor bands

`affect.py:186-194`:

```python
BANDS = {
    "P":  ((0.50, "affectionate"), (0.15, "warm"),      (-0.15, "even"),
           (-0.50, "cool"),        (-9.0, "cold")),
    "Ar": ((0.50, "excitable"),    (0.15, "lively"),     (-0.15, "calm"),
           (-0.50, "placid"),      (-9.0, "languid")),
    "D":  ((0.62, "commanding"),   (0.32, "assertive"),  (0.12, "forthright"),
           (-0.44, "even-handed"), (-0.74, "reserved"),  (-0.92, "retiring"),
           (-9.0, "withdrawn")),
}
```

`affect.py:197-206`:

```python
def band(axis: str, value: float) -> str:
    for edge, word in BANDS[axis]:
        if value >= edge:
            return word
    return BANDS[axis][-1][1]


def descriptors(coord: Dict[str, float]) -> Tuple[str, str, str]:
    """The three words that describe this coordinate, for the LLM prompt."""
    return band("P", coord["P"]), band("Ar", coord["Ar"]), band("D", coord["D"])
```

P and Ar have five bands each; D has seven.

**Which PAD value feeds them: `felt`.** `affect.py:381`:

```python
        "words": descriptors(felt),      # see the note in adapter.process_turn
```

`adapter.py:103`:

```python
        words = affect.descriptors(felt)
```

with the rationale at `adapter.py:98-102`. Not `shown`, not `baseline`.

**Computed — distinct triplets over 4 tiers × {sad, neutral, happy}.**

**CHATBOX: 12 distinct triplets over 12 cells** (all distinct).

| | sad | neutral | happy |
|---|---|---|---|
| unknown | cool/placid/withdrawn | even/calm/withdrawn | affectionate/lively/withdrawn |
| visitor | cool/placid/retiring | even/calm/retiring | affectionate/lively/retiring |
| known | cool/placid/reserved | even/calm/reserved | affectionate/lively/reserved |
| close | cool/**calm**/even-handed | even/calm/even-handed | affectionate/lively/even-handed |

**ELLEBOT: 12 distinct triplets over 12 cells** (all distinct).

| | sad | neutral | happy |
|---|---|---|---|
| unknown | cool/calm/even-handed | warm/lively/even-handed | affectionate/lively/even-handed |
| visitor | cool/calm/forthright | warm/lively/forthright | affectionate/lively/forthright |
| known | cool/calm/assertive | warm/lively/assertive | affectionate/lively/assertive |
| close | cool/calm/commanding | warm/lively/commanding | affectionate/**excitable**/commanding |

Two cells show the tier changing a **non-Dominance** word, both at tier `close`,
both traceable to the `+0.10` Arousal term in `TIER_OFFSETS["close"]`
(`affect.py:110`): CHATBOX `close/sad` reads `calm` where `known/sad` reads
`placid`; ELLEBOT `close/happy` reads `excitable` where `known/happy` reads
`lively`.

The two robots share three triplets across the pair (`cool/calm/even-handed`,
`affectionate/lively/even-handed`, and — for CHATBOX at `close`, ELLEBOT at
`unknown` — the `even-handed` rung generally), consistent with the disjoint-range
note at `affect.py:176-178`.

`CONTRADICTION` (report vs. current code, both dated).
`CHATBOX_V5-dev/CHATBOX_SERVER/runs/B_dir.md:14-24` records CHATBOX
`unknown/neutral` and `known/neutral` sharing the triplet `even/calm/reserved`,
i.e. 2 distinct triplets across 3 tier cells. The current code yields
`withdrawn` / `retiring` / `reserved` / `even-handed`, all distinct. `B_dir.md` is
dated `2026-08-14T02:55:03+00:00` and predates commit `838eda6` *"fix(affect):
every tier gets its own descriptor word"* (HEAD). Recorded rather than resolved.

---

## A10 — Test coverage

| file | package covered | test functions | assertion sites |
|---|---|---|---|
| `PAD_CORE/tests/test_affect.py` | `pad_core.affect` | 0 (`__main__` script style) | 41 `check(...)` calls |
| `PAD_CORE/tests/test_pad_core.py` | `pad_core.adapter` vs `pad_core.affect` | 7 `def test_*` | 16 `assert` |
| `PAD_CORE/tests/test_servo_style.py` | `pad_core.servo_style` | 0 (`__main__` script style) | 22 `check(...)` calls |
| `CHATBOX_SERVER/test_pad_prompt.py` | prompt assembly + style wire | 10 `def test_*` | 28 |
| `CHATBOX_SERVER/test_affinity.py` | affinity / topics | 5 | 18 |
| `CHATBOX_SERVER/test_first_impression.py` | first-impression enrolment | 6 | 32 |
| `CHATBOX_SERVER/test_face_multiview.py` | face recognition gallery | 16 | 45 |
| `graph_relationship/tests_kg_bridge.py` | `kg_bridge` | 18 | — |
| `graph_relationship/tests_extraction.py` | `extraction` | 9 | — |
| `graph_relationship/tests_topics.py` | `topics` | 8 | — |
| `graph_relationship/tests_store.py` | `store` | 7 | — |
| `graph_relationship/tests_schema.py` | `schema` | 4 | — |
| `graph_relationship/tests_embedding.py` | `embedding` | 3 | — |

The ten test names in `test_pad_prompt.py` (`:42`, `:63`, `:78`, `:92`, `:103`,
`:119`, `:136`, `:188`, `:213`, `:224`) include
`test_style_precedes_the_gesture_tag` and `test_style_is_sent_only_when_the_mood_moves`.

**Does the trait→PAD→style→servo path have automated assertions?** **Yes, for
trait→PAD→style; the final servo step is asserted separately, and the two are not
joined in one test.**

- trait→PAD→style is asserted in `tests/test_pad_core.py:54-76`
  (`test_adapter_equals_affect_pipeline`), over 40 robot/tier/emotion
  combinations, comparing `pad_state`, `shown`, `words` and every `style` key
  against `affect.pipeline()`. Note this asserts the adapter agrees with
  `affect`, i.e. **internal consistency**, not correctness against an external
  reference.
- Absolute style values per tier are pinned in `tests/test_affect.py` sections
  19-20 (observed in its output: `CHATBOX: amplitude rises with familiarity [0.3,
  0.355, 0.425, 0.565]`, `CHATBOX @ unknown: amplitude sits on the 0.30 clamp
  floor`).
- style→servo angles are asserted in `tests/test_servo_style.py` (22 checks),
  which `CONCEPT.md:429-430` describes as covering "all 189 tag/style
  combinations".
- The published-equation anchors are reproduced in `tests/test_affect.py`, per
  `affect.py:308-310`: `Hostile (-1,1,1) -> 1.0, Exuberant (1,1,1) -> 0.5,
  Anxious (-1,1,-1) -> 0.5, Bored (-1,-1,-1) -> 0.0`.

Both suites run without camera or hardware; `test_affect.py` and
`test_pad_core.py` were executed during this audit and passed
(`All checks passed.`).

**Two test files named in live comments do not exist on this branch.**

- `test_pad_affect.py` — referenced by `kg_bridge.py:50-51` as the guard on V/A
  table agreement. `git ls-files` does not list it. Only a stale bytecode file
  remains: `CHATBOX_V5-dev/CHATBOX_SERVER/__pycache__/test_pad_affect.cpython-310-pytest-9.0.3.pyc`.
  It was deleted in commit `324f569` *"refactor: consolidate the PAD
  implementation into CHATBOX_V5-dev/PAD_CORE"*; it was introduced in `8742a04`.
- `test_pad_integration` — referenced by `tests/test_pad_core.py:112-113`. Not
  tracked, and no file of that name exists in the working tree.

Consequence, stated factually: the assertion that `kg_bridge._EMOTION_VA` agrees
with `affect.CATEGORY_VA` is not executed on this branch, and no test covers
`emotion_detector._VA_TABLE` at all. See A8 for the divergence this permits.

**Not covered by any automated assertion found:** the rapport/trust accrual rule
(`webcam_loop.py:1522-1525`), the `0.7/0.3` mood blend (`kg_bridge.py:270`), and
the recurrence in A7.

---

## A11 — Documentation staleness

### `CHATBOX_V5-dev/PAD_CORE/docs/CONCEPT.md`

Referenced paths that do not exist on this branch. The directories `AFFECT_LAB/`
and `SERVO_STYLE/` were removed by commit `324f569`; the code now lives in
`pad_core/`, `tests/` and `bench/`.

| line | reference in doc | status |
|---|---|---|
| 15 | `../SERVO_STYLE/servo_style.py` | MISSING → now `pad_core/servo_style.py` |
| 14 | ``[`affect.py`](affect.py)`` | MISSING at that relative path → now `pad_core/affect.py` |
| 309 | `../SERVO_STYLE/` | MISSING |
| 502 | `SERVO_STYLE/preview.py` | MISSING → now `bench/preview.py` |
| 618 | `firmware/ChatBoxPlus_Styled/` | **EXISTS** |
| 623 | `AFFECT_LAB/test_affect.py` | MISSING → now `tests/test_affect.py` |
| 624 | `SERVO_STYLE/test_servo_style.py` | MISSING → now `tests/test_servo_style.py` |
| 625 | `AFFECT_LAB/webcam_demo.py` | MISSING → now `bench/webcam_demo.py` |
| 627 | `SERVO_STYLE/live_demo.py` | MISSING → now `bench/live_demo.py` |

Section 11's location table (`CONCEPT.md:592-618`) is headed
`### AFFECT_LAB — persona, emotion, and the four parameters` and
`### SERVO_STYLE — parameters to angles, and the wire`; both module names are
stale, though the symbol names inside it (`affect.to_pad`, `affect.WEIGHTS`,
`affect.ROBOTS`, `affect.feel`, `affect.show`, `affect.affect_name`,
`affect.SECTORS`, `affect.descriptors`, `affect.BANDS`, `affect.gesture_style`,
`affect.STYLE_LIMITS`, `affect._spatial_extent`, `affect._velocity`,
`affect.pipeline`, `servo_style.MOVE_SETS`, `servo_style.SERVOS`,
`servo_style.DROOP_DEG`, `POSTURE_DEG`, `servo_style.resolve_servo`,
`servo_style.resolve_gesture`, `servo_style.wire_message`, `parse_wire`) all
resolve.

Two entries in that table name symbols in files that no longer exist:
- `CONCEPT.md:598` — `face → valence/arousal | webcam_demo.read_va`; the runtime
  path is `emotion_detector.HSEmotionDetector._infer` (A8).
- `CONCEPT.md:615-616` — `persona+emotion → five values | live_demo.current_style`
  and `the serial link | live_demo.RobotLink`; `bench/live_demo.py` exists, but
  the server's serial link is now
  `CHATBOX_V5-dev/CHATBOX_SERVER/modules/webui/robot_link.py` (untracked, added
  in the working tree).

Non-path content contradictions already recorded elsewhere in this report:
- `CONCEPT.md:128-130` — camera model `enet_b0_8_va_mtl` and "no lookup table is
  needed" (A8).
- `CONCEPT.md:61`, `:162-171` — `shown` feeding the three words (A3).
- `CONCEPT.md:465-466` — the `felt`/`shown` worked example rows for the sad-face
  case; the `shown` row no longer feeds words or style (A3).

### `CHATBOX_V5-dev/PAD_CORE/docs/RELATIONSHIP_TO_DOMINANCE.md`

Every command in §9 (`:259-265`) resolves on this branch:
`tests/test_affect.py`, `python3 -m pad_core.adapter`,
`python3 -m tools.pad_prompt_grid --no-llm`, `python3 -m tools.e2e_pipeline
--no-camera`, `python3 -m modules.webui --llm --trace-prompt` — all present
(`tools/pad_prompt_grid.py`, `tools/e2e_pipeline.py`, `modules/webui/__main__.py`).
No missing module paths or filenames were found in this document.

Content contradictions recorded elsewhere:
- `:74-80` — the tier step function omits the `close` branch that exists at
  `kg_bridge.py:116-117` (A4).
- `:236` — `CHATBOX's show = 0.30 compresses all three axes, so its *words*
  barely move` is listed as `open`; per A3 and A9, `show` no longer feeds the
  words at all and all four CHATBOX tiers now yield distinct triplets.

---

## A12 — Computed checks

All figures below were produced by importing the live `pad_core.affect` at HEAD.

### A12.1 — Baseline PAD vs the published reference

| robot | computed | reference | diff (P, Ar, D) |
|---|---|---|---|
| CHATBOX | (+0.266, −0.009, −0.643) | (+0.266, −0.009, −0.643) | (+0.0000, +0.0000, +0.0000) |
| ELLEBOT | (+0.425, +0.483, +0.421) | (+0.425, +0.483, +0.421) | (+0.0000, +0.0000, +0.0000) |

Agreement to three decimals on all six values.

### A12.2 — `max |D|` over the trait cube `[-1,+1]^5`

Since `to_pad` is linear and unclamped, the maximum of `|axis|` is the sum of the
absolute coefficients.

```
D coefficients: {'O': 0.25, 'C': 0.17, 'E': 0.60, 'A': -0.32}
max|D|  = 0.25 + 0.17 + 0.60 + 0.32 = 1.34
```

For completeness, the other two axes:

```
max|P|  = 0.21 + 0.59 + 0.19 = 0.99
max|Ar| = 0.15 + 0.30 + 0.57 = 1.02
```

`D` and `Ar` can both exceed the `[-1,+1]` interval PAD is defined on. `to_pad`
does not clamp (A1), so an out-of-range baseline is returned as-is and survives
`feel` unclamped; the first clamp is at `affect.py:147-149`, after the tier
offset. Neither published persona reaches this (A12.1), but a third persona can.

### A12.3 — Pre-clamp `baseline_D + offset`, all tiers

| robot | tier | pre-clamp D | outside [−1,+1]? |
|---|---|---|---|
| CHATBOX | unknown | **−1.0430** | **YES — clamps to −1.000** |
| CHATBOX | visitor | −0.8430 | no |
| CHATBOX | known | −0.6430 | no |
| CHATBOX | close | −0.2430 | no |
| ELLEBOT | unknown | +0.0210 | no |
| ELLEBOT | visitor | +0.2210 | no |
| ELLEBOT | known | +0.4210 | no |
| ELLEBOT | close | +0.8210 | no |

One cell clamps: **CHATBOX at `unknown`**, losing 0.043 of displacement. This is
the cell whose amplitude sits exactly on the `STYLE_LIMITS` floor of 0.300 (A5),
so it is clamped twice — once on D at `affect.py:147-149`, once on amplitude at
`affect.py:358-359`. The behaviour is anticipated in the code comment at
`affect.py:140-143`:

```
    Clamped to [-1, +1] -- CHATBOX's baseline D of -0.643 plus the 'unknown'
    offset lands at -1.043, outside the space PAD is defined on and outside the
    domain the published Eq. 9 / Eq. A1 assume.
```

and is asserted in `tests/test_affect.py` section 20 (observed output:
`CHATBOX @ unknown: D saturates at -1`, `CHATBOX @ unknown: amplitude sits on the
0.30 clamp floor`).

Consequence for the ladder, stated factually: the four CHATBOX D values are
−1.000, −0.843, −0.643, −0.243, so the `unknown`→`visitor` step is 0.157 while
the other two steps are 0.200 each. The ladder is not uniform for CHATBOX. For
ELLEBOT all three steps are 0.200.

### A12.4 — Admissible baseline-D band

```
max |dD| over TIER_OFFSETS = 0.40      (both 'close' +0.40 and 'unknown' -0.40)
admissible band            |D_baseline| ≤ 1 − 0.40 = 0.60
```

| robot | \|D_baseline\| | satisfies \|D\| ≤ 0.60 |
|---|---|---|
| CHATBOX | 0.6430 | **No** |
| ELLEBOT | 0.4210 | Yes |

CHATBOX exceeds the band by 0.043, which is exactly the clamp observed in A12.3.
ELLEBOT has 0.179 of margin.

---

## OPEN

Questions from the brief that could not be answered from this branch, and why.

1. **A1 — "all twelve coefficients."** There are **ten**, not twelve
   (`affect.py:39-43`). `P` and `Ar` have no Conscientiousness term and `D` has
   no Neuroticism term. Reported as found; the discrepancy may be in the premise
   rather than the code, and I did not resolve which.

2. **A1 — docstring label of the fifth factor.** The code names it `"N"` only.
   No function or module docstring in `affect.py` expands it. The expansion
   "Neuroticism" exists at `CONCEPT.md:94` and in commit messages, not in the
   module. Answered as far as the code allows; there is no in-code docstring
   label to quote.

3. **A2 — whether `CHATBOX-DEMO_V4/CHATBOX_CLIENT/client_config.json:89-95` is
   live.** The file is tracked and contains a `[0,1]` trait vector that
   contradicts `affect.ROBOTS`. Grepping `*.py` for the `"personality"` key found
   only `CHATBOX-DEMO_V4/CHATBOX_CLIENT/client.py:223`, which maps the string to
   itself in what appears to be a field table. I did not trace the client's
   runtime to determine whether that block is read, and per the read-only rule I
   did not execute it. Recorded as a contradiction of unknown liveness.

4. **A6 — decay across process restarts.** No decay code exists (NOT FOUND). But
   `MoodEdge` is declared FAST and documented at `schema.py:275` as decaying
   "between sessions"; whether the graph is persisted and reloaded such that a
   stale mood survives a restart depends on `store.reload` / `kg_path` handling
   (`webcam_loop.py:1933-1934`), which I did not trace end to end. This affects
   whether the A7 recurrence resets between runs.

5. **A7 — initial condition of the recurrence.** The numeric trace assumes no
   stored `MoodEdge` on the first turn (`graph_mood is None` →
   `blended_v = camera_v`, `kg_bridge.py:270`). Whether a persisted graph
   supplies a non-null `m_0` at process start is the same unresolved question as
   (4).

6. **A9 — reachability under live camera input.** The triplet counts are computed
   from `affect.category_to_va` (the `CATEGORY_VA` lookup), which is what
   `pad_prompt_grid` uses. The runtime path instead supplies a softmax-blended
   `va` from `emotion_detector._VA_TABLE` (A8), whose values differ on six of
   seven labels and which produces interpolated rather than corner values. The
   reachable triplet set under live camera input is therefore **not** the set
   tabulated in A9, and I did not enumerate it — doing so requires either camera
   input or an assumption about the softmax distribution.

7. **A10 — assertion counts for the `graph_relationship/tests_*.py` files.**
   Counted `def test_*` functions only; I did not count assertion sites in those
   six files.

8. **A11 — non-path staleness in `CONCEPT.md` §9's worked numbers.** The section
   states "Every figure below came from running the code" and gives a full
   servo-level trace (`:459-500`). I verified the trait→PAD→style rows for the
   neutral-face ladder (A5, A12.1) but did not re-run the sad-face worked example
   or the per-servo angle arithmetic at `:477-495`, so those specific figures are
   unverified.

9. **Working-tree divergence.** `webcam_loop.py` and `webui/app.py` carry
   uncommitted modifications and `webui/robot_link.py` is untracked. All quotes
   are from the working tree. Any comparison against HEAD or against another
   branch would differ for those three files; I did not diff them.
