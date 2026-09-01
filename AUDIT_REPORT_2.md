```
branch: feature/pad-affect-core
HEAD:   838eda65d9ae2dd3f8ec972dc1896a3f50acbecd   (unchanged from pass 1)
date:   2026-08-31T16:10:21+12:00

working tree clean: NO
  tracked files differing from HEAD:
    M CHATBOX_V5-dev/CHATBOX_SERVER/modules/face_webcam/webcam_loop.py   (+54/-30 region)
    M CHATBOX_V5-dev/CHATBOX_SERVER/modules/webui/app.py                 (+132 region)
  untracked:
    ?? AUDIT_REPORT.md                                            (pass 1's output)
    ?? CHATBOX_V5-dev/CHATBOX_SERVER/modules/webui/robot_link.py
  Same caveat as pass 1: all quotes below are from the WORKING TREE, not HEAD.

files read beyond pass 1's list:
  CHATBOX_V5-dev/CHATBOX_SERVER/modules/graph_relationship/store.py      (persistence block, :300-371)
  CHATBOX_V5-dev/CHATBOX_SERVER/kg_state.json                            (live persisted graph)
  CHATBOX_V5-dev/CHATBOX_SERVER/sessions.db                              (schema only)
  CHATBOX_V5-dev/CHATBOX_SERVER/runs/B_dir.jsonl                         (record keys only)
  /home/jay/.local/lib/python3.10/site-packages/hsemotion_onnx/facial_emotions.py
      (installed third-party library — read to resolve the label ordering and the
       `_mtl` branch; not part of this repository)

commands run: read-only. Two Python snippets imported `pad_core.affect` and
`modules.graph_relationship` and loaded a COPY of kg_state.json placed in the
session scratchpad. No repository file was written except AUDIT_REPORT_2.md.
```

---

## B1 — Does the graph persist across a process restart?

### B1.1 — Write sites

One serialiser, `modules/graph_relationship/store.py:309-318`:

```python
    def save(self, path: str) -> None:
        """Serialise the full graph to a JSON file using Pydantic model_dump."""
        import json
        data = {
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges.values()],
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        print(f"[KG] saved {len(self._nodes)} nodes, {len(self._edges)} edges → {path}")
```

It writes the **whole graph**, not a delta, and iterates `self._edges.values()`
unconditionally. There is no `timescale` test anywhere in the method.

Call sites reaching it in the webcam loop:

**(a) Seed-time save**, `webcam_loop.py:1197-1198`:

```python
                if self.kg_path:
                    self.store.save(self.kg_path)
```

**(b) Debounced autosave**, `webcam_loop.py:1344-1361`:

```python
    def _flush_kg(self, *, force: bool = False) -> None:
        """Persist kg_state.json at most once per _save_min_interval, and only when
        something changed (or `force`). Called every loop iteration on the MAIN
        thread; the actual write holds the store lock so it can't race the chat
        worker. Replaces the old per-tick/per-turn full-graph rewrites."""
        if not self.kg_path or not (self._kg_dirty or force):
            return
        now = time.time()
        if not force and (now - self._last_save_t) < self._save_min_interval:
            return
        with self._store_lock:
            self.store.save(self.kg_path)
```

Interval, `webcam_loop.py:1280`:

```python
        self._save_min_interval = 1.0   # seconds between debounced saves
```

Triggered every loop iteration at `webcam_loop.py:2142` (`self._flush_kg()`) and
forced once at shutdown, `webcam_loop.py:2496`:

```python
            self._flush_kg(force=True)   # final flush regardless of debounce
```

**The dirty flag is set on every PAD tick**, `webcam_loop.py:2306-2313`:

```python
                        # Persist after the tick so the live viz server
                        # (modules.graph_relationship.viz.server) can poll it within ~1s.
                        # PAD mode mutates every tick; the KG-only path otherwise saves
                        # on session creation + each chat turn, so we only add a per-tick
                        # save when the FAST mood actually changed (mood_dirty).
                        if ((self._pad_enabled or mood_dirty)
                                and any(d["person_id"] for d in raw_dets)):
                            self._mark_kg_dirty()
```

So with PAD enabled, the MoodEdge written by `post_turn` each tick is flushed to
disk within ~1 s. Other `save` call sites outside the loop:
`seed.py:258`, `demo_harness.py:239`, `:477`, `:750`, `:773`,
`webcam_loop.py:2654` (migrate mode).

### B1.2 — Load sites

`store.py:320-339`:

```python
    def load(self, path: str, *, quiet: bool = False) -> bool:
        """Load a JSON file written by save() (merges into current state)."""
        import json, os
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            data = json.load(fh)
        for nd in data.get("nodes", []):
            node = _node_adapter.validate_python(nd)
            self._nodes[node.id] = node
        for ed in data.get("edges", []):
            edge = _edge_adapter.validate_python(ed)
            self._edges[edge.id] = edge
            key = (edge.source_id, edge.target_id, edge.edge_type)
            self._endpoint_type_index[key] = edge.id
            self._node_edge_index[edge.source_id].add(edge.id)
            self._node_edge_index[edge.target_id].add(edge.id)
        if not quiet:
            print(f"[KG] loaded {len(self._nodes)} nodes, {len(self._edges)} edges ← {path}")
        return True
```

Construction at process start, `webcam_loop.py:1182-1186`:

```python
        self.store   = InMemoryGraphStore()
        if os.path.exists(kg_path):
            self.store.load(kg_path)
        else:
            print(f"[WebcamLoop] No KG at '{kg_path}' — starting fresh")
```

It reads an existing file whenever one is present; it does not always start
empty. `reload` (`store.py:341-371`) is the replace-all variant used between
turns for external edits (`webcam_loop.py:1340`, `:1934`); it applies the same
unfiltered edge restoration at `:363-370`.

### B1.3 — Are FAST-tier edges excluded from persistence?

**No. NOT FOUND — there is no filtering at either boundary.**

- Write: `store.py:313-314` dumps `self._edges.values()` with no predicate.
- Read: `store.py:330-336` restores every element of `data["edges"]` with no
  predicate. Note it assigns `self._edges[edge.id]` **directly**, bypassing
  `upsert_edge`, so even a filter living in `upsert_edge` would not apply.

`MoodEdge` carries the FAST marker as an ordinary serialisable field,
`schema.py:274-283`:

```python
class MoodEdge(EdgeBase):
    """Current affective state — FAST, decays between sessions.

    `label` optionally carries the source emotion name (e.g. "happy") so the
    live visualizer can show the current emotion alongside the valence.
    """
    edge_type: Literal["mood"] = "mood"
    value: float = Field(..., ge=-1.0, le=1.0)  # valence: -1 sad … +1 happy
    label: Optional[str] = None                 # emotion label, e.g. "happy"
    timescale: Timescale = Timescale.FAST
```

`timescale` is round-tripped as data, and nothing reads it during save or load.

**Direct evidence — a FAST MoodEdge is sitting in the live persisted graph.**
`CHATBOX_V5-dev/CHATBOX_SERVER/kg_state.json` (82 nodes, 93 edges; file mtime
`2026-08-19 15:18:29 +1200`) contains:

```json
{
 "id": "5d3f64af-fa1f-4312-b7a5-1cbf1b1232b7",
 "source_id": "jay",
 "target_id": "jay",
 "provenance": {
  "source": "chatbox",
  "timestamp": "2026-08-19T03:18:29.149953Z",
  "confidence": 1.0
 },
 "edge_type": "mood",
 "value": -0.054546917571175635,
 "label": null,
 "timescale": "FAST"
}
```

and one `attention` edge, `value: 0.5171980302307231`, same timestamp. Edge-type
census of that file: `mood 1, attention 1, has_persona 3, has_role 3,
has_interest 10, has_interaction 4, has_capability 2, about 29,
has_conversation 4, related_topic 7, has_session 29`.

The mood edge is dated **2026-08-19**; today is **2026-08-31** — 12 days old and
unchanged, consistent with pass 1's finding that no decay code exists.

### B1.4 — Actual initial condition of `graph_mood` on turn 1 of a fresh process

Traced by executing the real path against a scratchpad copy of the live
`kg_state.json` (no repository file touched):

```
InMemoryGraphStore() → store.load(kg_copy.json) → get_person_context("jay")
  person_attribute_edges for 'jay': [('attention', 0.5171980302307231),
                                     ('mood', -0.054546917571175635)]

KGBridge(store).pre_turn("jay", "chatbox", "happy")      # turn 1, new process
  BridgeInput.valence = 0.5436359247286473
  BridgeInput.tier    = 'close'
  raw camera v for 'happy' = 0.8
  blended = True
```

`0.7 × 0.8 + 0.3 × (−0.0545469) = 0.5436359…` — the blend at `kg_bridge.py:270`
fired on the **first** turn of a brand-new process.

**Answer: the `next(...)` lookup at `kg_bridge.py:266-269` can and does return a
non-None value on turn 1 of a new process.** It is not guaranteed to be None. It
is None only when no graph file exists at `kg_path`, or when the file contains no
mood edge for that person.

The same run shows `tier = 'close'`, i.e. persisted rapport/trust also survive the
restart and feed `derive_tier` on turn 1.

### B1.5 — Flags controlling persistence

Default path, `webcam_loop.py:97`:

```python
_DEFAULT_KG      = "kg_state.json"
```

Constructor default, `webcam_loop.py:1126`:

```python
        kg_path:         str   = _DEFAULT_KG,
```

CLI, `webcam_loop.py:2679-2680`:

```python
    p.add_argument("--kg",         default=_DEFAULT_KG,
                   help=f"KG state JSON path (default: {_DEFAULT_KG})")
```

`--kg` selects **where** the graph is stored; it does not disable persistence.
**NOT FOUND**: no `--no-persist`, `--fresh`, `--ephemeral`, or equivalent flag,
and no environment variable gating `save`/`load`. The only code path that skips
persistence is `kg_path` being falsy — `_flush_kg` returns early at
`webcam_loop.py:1349` (`if not self.kg_path …`) — but the CLI cannot produce a
falsy value, since `--kg` defaults to a non-empty string and argparse would need
an explicit empty argument. Callers constructing `WebcamKGLoop` directly can pass
a temporary path; `tools/pad_prompt_grid.py:124` does exactly that
(`kg_path=os.path.join(tmp, f"kg_{robot}.json")`), which is why the grid tool
starts clean each run.

### B1.6 — Answer to pass 1's question

**`PERSISTS-UNDECAYED`.**

Evidence chain, each link quoted above:

1. `store.save` writes every edge with no timescale filter (`store.py:313-314`).
2. `store.load` restores every edge with no timescale filter
   (`store.py:330-336`), bypassing `upsert_edge`.
3. `webcam_loop` loads `kg_path` at construction when the file exists
   (`webcam_loop.py:1182-1186`), default `kg_state.json`
   (`webcam_loop.py:97`, `:2679-2680`).
4. With PAD enabled every tick marks the graph dirty (`webcam_loop.py:2311-2313`)
   and `_flush_kg` writes within 1.0 s (`:1280`, `:1344-1361`, `:2142`), plus a
   forced flush at shutdown (`:2496`).
5. A FAST `MoodEdge` from **2026-08-19** is present in the live `kg_state.json`
   today, 2026-08-31.
6. Executing the real load → `pre_turn` path on that data returns
   `valence = 0.5436` instead of the raw camera `0.8` on turn 1 of a fresh
   process.
7. Pass 1 established no decay code exists; nothing in this pass contradicts
   that.

So `m_0` does **not** reset on restart. A mood value written in a session weeks
earlier feeds turn 1 of today's session at its full 0.3 weight. Per the A7
recurrence, the effect decays over subsequent turns with pole `0.3·e = 0.18`
(it is a contracting filter), so the stale value is washed out within a few
ticks of live camera input — but it is present and unattenuated on turn 1.

One boundary condition worth recording, without resolving it: `pre_turn` takes
the **first** mood edge it finds (`kg_bridge.py:266-269`, `next(...)`), and
`upsert_edge` keys edges by `(source_id, target_id, edge_type)`
(`store.py:333`), so exactly one mood edge exists per person — the most recent
write wins. There is no timestamp comparison in the lookup.

---

## B2 — Descriptor reachability under the live camera V/A table

### B2.1 — Is the contradiction still open?

**Yes, unresolved.** HEAD is `838eda6`, identical to pass 1:

```
$ git log --oneline 838eda6..HEAD
(no output — zero commits since pass 1)
```

Commits that ever touched each table, none of them after pass 1's HEAD:

```
$ git log --all --oneline -S'_VA_TABLE' -- '*.py'
dd68fe3 port: KG relationship + face reco + PAD core from feature/cultural-awareness
a93a06c fix(first-impression): 4-value emotion detect (worker unpack crash)
4030582 wip(cultural): checkpoint face_id/emotion/pad edits + runtime artifacts

$ git log --all --oneline -S'CATEGORY_VA' -- '*.py'
324f569 refactor: consolidate the PAD implementation into CHATBOX_V5-dev/PAD_CORE
8742a04 refactor(pad): server adopts AFFECT_LAB as the single affect source
ace95a2 emotion
7faa904 emotion

$ git log --all --oneline -S'_EMOTION_VA' -- '*.py'
324f569 refactor: consolidate the PAD implementation into CHATBOX_V5-dev/PAD_CORE
8742a04 refactor(pad): server adopts AFFECT_LAB as the single affect source
dd68fe3 port: KG relationship + face reco + PAD core from feature/cultural-awareness
a93a06c fix(first-impression): 4-value emotion detect (worker unpack crash)
4030582 wip(cultural): checkpoint face_id/emotion/pad edits + runtime artifacts
cbaed45 Add kg_bridge — PAD↔KG bridge module with three-guarantee test suite
```

`emotion_detector._VA_TABLE` remains canonical for the live camera path; the
six-of-seven disagreement recorded in pass 1's A8 stands.

**Additional evidence bearing on A8's second contradiction.** The installed
`hsemotion_onnx` library — read at
`/home/jay/.local/lib/python3.10/site-packages/hsemotion_onnx/facial_emotions.py:30-32` —
lists the model CONCEPT.md claims is in use among its supported names:

```python
#supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b0_8_best_vgaf'):
        self.is_mtl='_mtl' in model_name
```

and `predict_emotions` (`:50-64`) returns two extra trailing score elements for
an `_mtl` model:

```python
    def predict_emotions(self,face_img, logits=True):
        scores=self.ort_session.run(None,{"input": self.preprocess(face_img)})[0][0]
        if self.is_mtl:
            x=scores[:-2]
        else:
            x=scores
```

So `enet_b0_8_va_mtl` — the model named at `CONCEPT.md:129` — exists, is
supported by the installed library, and does emit valence/arousal directly in the
last two positions of `scores`. The repository selects `enet_b0_8_best_vgaf`
instead (`emotion_detector.py:165`), which is not an `_mtl` model, so no VA
outputs are produced and the lookup table is used. Both onnx files are cached
locally (`~/.hsemotion/enet_b0_8_best_vgaf.onnx`, `enet_b2_8.onnx`); the
`va_mtl` weights are not.

Recorded without resolving: `_infer` iterates `enumerate(scores)` and indexes
`self._labels[i]` (`emotion_detector.py:225-226`), while `self._labels` is the
8-key `idx_to_class` dict (`:216`, library `:36`). For an `_mtl` model `scores`
has 10 elements, so `self._labels[8]` would raise `KeyError`. This is a property
of the current code, not an observed failure — the `_mtl` model is not selected.

### B2.2 — Pure-class blend through the actual arithmetic

Label ordering is the library's 8-class map (`facial_emotions.py:36`), assigned
at `emotion_detector.py:216` (`self._labels = self._recognizer.idx_to_class`):

```python
{0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
```

Running the `emotion_detector.py:224-229` arithmetic with a one-hot score vector
on each index in turn:

| one-hot class | `_norm` → | blended (v, a) |
|---|---|---|
| Anger | angry | (−0.600, +0.700) |
| Contempt | disgust | (−0.600, +0.300) |
| Disgust | disgust | (−0.600, +0.300) |
| Fear | fear | (−0.500, +0.800) |
| Happiness | happy | (+0.800, +0.600) |
| Neutral | neutral | (+0.000, +0.000) |
| Sadness | sad | (−0.700, −0.400) |
| Surprise | surprise | (+0.100, +0.800) |

Eight model classes collapse onto seven table keys — `Contempt` and `Disgust`
produce identical output, per `emotion_detector.py:55`. As expected from the
arithmetic, a pure prediction reproduces the raw table entry exactly.

### B2.3 — The three rows A9 used, live table vs `CATEGORY_VA`

| label | `CATEGORY_VA` (A9 used) | `_VA_TABLE` (live) | identical? |
|---|---|---|---|
| sad | (−0.7, −0.38) | (−0.7, −0.40) | no |
| neutral | (0.0, 0.0) | (0.0, 0.0) | **yes** |
| happy | (0.8, 0.50) | (0.8, 0.60) | no |

The `neutral` entry is identical in both tables, so the neutral column is
unaffected. `sad` differs on arousal by 0.02 and `happy` by 0.10.

### B2.4 — A9's tables recomputed with the live values

Same pipeline as A9: `feel_with_relationship` at default empathy 0.6, all four
tiers, `descriptors(felt)`.

**CHATBOX — 12 distinct triplets over 12 cells.**

| | sad | neutral | happy |
|---|---|---|---|
| unknown | cool/placid/withdrawn | even/calm/withdrawn | affectionate/lively/withdrawn |
| visitor | cool/placid/retiring | even/calm/retiring | affectionate/lively/retiring |
| known | cool/placid/reserved | even/calm/reserved | affectionate/lively/reserved |
| close | cool/calm/even-handed | even/calm/even-handed | affectionate/lively/even-handed |

Identical to A9's CHATBOX table in every cell.

**ELLEBOT — 12 distinct triplets over 12 cells.**

| | sad | neutral | happy |
|---|---|---|---|
| unknown | cool/calm/even-handed | warm/lively/even-handed | affectionate/**excitable**/even-handed |
| visitor | cool/calm/forthright | warm/lively/forthright | affectionate/**excitable**/forthright |
| known | cool/calm/assertive | warm/lively/assertive | affectionate/**excitable**/assertive |
| close | cool/calm/commanding | warm/lively/commanding | affectionate/excitable/commanding |

**The count does not differ from A9: 12/12 for both robots. No cells collapse.**

Three cells change wording, all ELLEBOT/happy, all `lively` → `excitable`, caused
by the live table's `happy` arousal of 0.60 against `CATEGORY_VA`'s 0.50:

```
  ELLEBOT/unknown/happy: A9=affectionate/lively/even-handed  LIVE=affectionate/excitable/even-handed
  ELLEBOT/visitor/happy: A9=affectionate/lively/forthright   LIVE=affectionate/excitable/forthright
  ELLEBOT/known/happy:   A9=affectionate/lively/assertive    LIVE=affectionate/excitable/assertive
```

A9's ELLEBOT `close/happy` cell already read `excitable` because of the `close`
tier's `+0.10` arousal offset; under the live table all four tiers reach it.

### B2.5 — Realistic (non-degenerate) softmax distributions

**NOT FOUND — no logged softmax vector exists in the repository.** Searched:

- `runs/*.jsonl` — record keys are
  `['affect_name','derived_tier','emotion','latency_ms','metrics','model','pad_felt','pad_on','prompt','prompt_sha256','repeat','reply','robot','style','tag','temperature','tier','va','wire','words']`.
  The `va` field holds corner values from `affect.category_to_va`
  (`tools/pad_prompt_grid.py:140`), e.g. `[0.0, 0.0]` — not camera output.
- `sessions.db` — `CREATE TABLE turns (… emotion TEXT, child TEXT, reply TEXT, …)`;
  a scalar label only, no score vector and no confidence column.
- No test file references `emotion_detector`, `predict_emotions`, or `_VA_TABLE`
  (pass 1's A10 finding; re-confirmed).
- The one `e_conf` grep hit outside code was
  `CHATBOX-DEMO_V4/CHATBOX_CLIENT/client_config.json:28`, which is the substring
  inside `"voic`**`e_conf`**`ig"` — a false positive, not a confidence value.

No distribution is fabricated here.

**One real logged sample is recoverable, from the persisted graph in B1.** The
`MoodEdge`/`AttentionEdge` pair for `jay` (provenance source `chatbox`) are
`felt` values written by a live camera session, so `feel()` can be inverted to
recover the effective input:

```
logged felt P            = -0.054547            (kg_state.json MoodEdge.value)
=> effective valence in  = -0.268245            cam = b + (felt-b)/e, b=+0.2660, e=0.60

logged attention          = 0.517198            (AttentionEdge.value)
=> felt Ar                = +0.034396           inverting (a+1)/2
=> effective arousal in   = +0.063327
```

Caveats that bound what this sample proves, stated rather than resolved: the
recovered figure is the value **after** `AffectStream`'s 5-sample mean
(`adapter.py:93`, `stream.py:28-34`) and **after** the 0.7/0.3 mood blend
(`kg_bridge.py:270`), so it cannot be decomposed back into a single frame's
softmax. It is one sample from one moment.

What it does show: the live pipeline produced `v = −0.268`, `a = +0.063`, against
table extremes of `v ∈ [−0.70, +0.80]` and `a ∈ [−0.40, +0.80]`. Both lie well
inside the hull, not at a corner. That coordinate bands as:

```
    P  -0.0545 -> 'even'
    Ar +0.0344 -> 'calm'
```

i.e. the middle band on both axes — the same cell the `neutral` column of the
B2.4 table occupies.

### B2.6 — Clipping, rounding, quantisation, and whether 12/12 is an upper bound

**No clipping, rounding, or quantisation is applied to (v, a) before banding.**
`band` compares raw floats (`affect.py:197-201`, quoted in pass 1's A9), and
`descriptors` (`affect.py:204-206`) passes the coordinate straight through. The
only clamp on the path is `_clamp` inside `feel_with_relationship`
(`affect.py:147-149`), which bounds to `[-1,+1]` and does not quantise.

Two smoothing stages do exist, neither of which rounds:

- `emotion_detector.py:127`, `:131`, `:147-149` — `detect()` smooths the **label
  and confidence only**; v/a pass through raw:

```python
            smooth    : Apply 5-frame sliding-window smoothing (label + conf only).
...
            (emotion_label, confidence_0_to_100, valence, arousal)
            v/a are from the latest raw frame; label/conf may be window-smoothed.
```

- `stream.py:28-34` — `AffectStream.update` takes an unweighted mean of the last
  5 submitted (v, a) pairs, and is in the live path at `adapter.py:93`. Since
  `_pipeline_tick` runs once per `tick_interval` (default 1.0 s,
  `webcam_loop.py:97`), that window spans ~5 seconds of ticks, not 5 camera
  frames.

**Is 12/12 an upper bound?** What the code shows, without probabilistic estimate:

The blend at `emotion_detector.py:224-229` computes `Σᵢ pᵢ · tableᵢ` where the
`pᵢ` are softmax outputs summing to 1. Every output is therefore a convex
combination of the seven table points and lies in their convex hull. The corner
values used in B2.2 and B2.4 are attained **only** by a degenerate one-hot
distribution; any distribution with mass on more than one class yields a point
strictly interior to the hull. `AffectStream`'s mean over 5 ticks is itself
another convex combination, so it can only move the coordinate further inside.

The consequence for banding, computed by inverting `feel()`
(`cam = base + (edge − base)/e`, e = 0.60) at each band edge:

| robot | axis | camera value required |
|---|---|---|
| CHATBOX | P | `affectionate` ≥ +0.6560, `warm` ≥ +0.0727, `even` ≥ −0.4273, `cool` ≥ −1.0107 |
| CHATBOX | Ar | `excitable` ≥ +0.8393, `lively` ≥ +0.2560, `calm` ≥ −0.2440, `placid` ≥ −0.8273 |
| ELLEBOT | P | `affectionate` ≥ +0.5500, `warm` ≥ −0.0333, `even` ≥ −0.5333, `cool` ≥ −1.1167 |
| ELLEBOT | Ar | `excitable` ≥ +0.5113, `lively` ≥ −0.0720, `calm` ≥ −0.5720, `placid` ≥ −1.1553 |

Sweeping the achievable camera range (`v ∈ [−0.70, +0.80]`, `a ∈ [−0.40, +0.80]`)
across all four tiers gives the words that can ever be produced by the live
table:

| robot | axis | reachable | unreachable |
|---|---|---|---|
| CHATBOX | P | affectionate, warm, even, cool | **cold** |
| CHATBOX | Ar | excitable, lively, calm, placid | **languid** |
| ELLEBOT | P | affectionate, warm, even, cool | **cold** |
| ELLEBOT | Ar | excitable, lively, calm | **placid, languid** |

Four of the ten P/Ar band words defined at `affect.py:186-194` are unreachable
for one or both robots **even at a pure one-hot prediction**, because the live
table's extremes (`v ≥ −0.70`, `a ≥ −0.40`) never drive `felt` past those edges.
`cold` is unreachable for both robots; `languid` for both; `placid` for ELLEBOT.

Since a mixed distribution is strictly interior to the hull, the corner-derived
12/12 is an upper bound on distinguishability, attained only at the degenerate
one-hot case. The single real logged sample in B2.5 sits at `v = −0.268`,
`a = +0.063` — mid-band on both axes. **How often live operation approaches the
corners cannot be answered from this repository**, because no softmax
distribution is logged anywhere (B2.5). One sample is not a rate.

---

## RESOLVED

- **OPEN #4 (decay across process restarts) — CLOSED.** Persistence is unfiltered
  in both directions (`store.py:313-314`, `:330-336`), the loop loads
  `kg_state.json` at startup (`webcam_loop.py:1182-1186`) and autosaves every
  ~1 s under PAD (`:1280`, `:2311-2313`, `:2142`), and a FAST MoodEdge dated
  2026-08-19 is present in the live file today. Stale mood survives a restart.
- **OPEN #5 (initial condition of the recurrence) — CLOSED.** `m_0` is **not**
  guaranteed None. Executing the real path on the live graph returns
  `valence = 0.5436` on turn 1 of a fresh process where the raw camera value is
  `0.8`; outcome is `PERSISTS-UNDECAYED`.
- **OPEN #6 (descriptor reachability under the live table) — CLOSED for the
  pure-class case, and bounded for the general case.** Recomputed with
  `_VA_TABLE`: 12/12 distinct triplets for both robots, unchanged from A9, no
  collapse; three ELLEBOT/happy cells shift `lively` → `excitable`. The general
  case is bounded analytically — every blended output is interior to the table's
  convex hull, and four band words are unreachable at any distribution.

## STILL OPEN

1. **Rate at which live operation approaches the one-hot corners.** B2.6
   establishes 12/12 as an upper bound and B2.5 supplies one real sample at
   `v = −0.268, a = +0.063`, but no distribution of live values can be computed.
   Missing evidence: logged softmax vectors or logged raw (v, a) per frame.
   Nothing in `runs/`, `sessions.db`, or any test fixture records them —
   `sessions.db.turns` stores only a scalar `emotion` label, and `runs/*.jsonl`
   stores `CATEGORY_VA` corner values, not camera output.

2. **Decomposition of the one recovered sample.** The `v = −0.268` figure is
   post-`AffectStream`-mean and post-mood-blend; recovering the frame-level
   softmax that produced it would require the per-frame log named in (1).

3. **Whether the persisted `kg_state.json` is the file a production launch
   actually uses.** The default is a bare relative path, `"kg_state.json"`
   (`webcam_loop.py:97`), resolved against the process working directory. The
   file examined here sits at `CHATBOX_V5-dev/CHATBOX_SERVER/kg_state.json`,
   consistent with launching from that directory, but no launcher script, service
   file, or documented run command fixing the working directory was found.
   Missing evidence: a deployment entry point.

4. **Whether `modules/webui` shares this persistence behaviour.** B1 traced
   `webcam_loop`. `modules/webui/app.py` is one of the two files with uncommitted
   working-tree modifications and was not traced for store construction in this
   pass; pass 1 likewise did not cover it. Missing evidence: a read of the webui
   entry point's store lifecycle.
