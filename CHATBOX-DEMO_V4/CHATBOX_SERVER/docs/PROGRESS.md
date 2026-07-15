# Progress Log

Running record of what was **tried**, what **worked**, and what **didn't work / was fixed** — one entry
per commit, newest first. Companion to `RND_KG_Companion_System.md` (the design report). Kept so the
research write-up can reference which approaches were attempted and why.

---

## docs: R&D system report + progress log  *(this commit)*

**Tried:** wrote a detailed R&D report (`RND_KG_Companion_System.md`) covering face-reco, emotion, the
FAST/SLOW/RELATIONSHIP graph, extraction, prompt structure, and pipeline; started this progress log.
**Worked:** report captures the current baseline (PAD disabled, emotion→mood only) accurately.
**Didn't / open:** no evaluation numbers yet; references + abstract still to add.
**Next:** improve the knowledge-extraction method (see report §7).

---

## `a31f30a` feat(viz): colour edges per person; chatbox edges blue

**Tried:** colour each person's edges with a distinct hue, shaded by timescale (FAST lighter / SLOW
darker), and force all robot (chatbox) edges to a single blue. Edge ownership inferred from the source
node id (`person` / `interest:` / `conversation:` / `interaction:` / `*:capability` / robot).
**Worked:** verified on the live graph — jay's 31 edges → jay hue, HJ's 10 → HJ hue, chatbox's 8 → blue,
0 unowned. Legend lists each person's colour dynamically.
**Didn't / watch:** blue is reserved for the robot and excluded from the person palette; if many people
are added the 8-colour palette will wrap (acceptable for now).

---

## `d093e00` feat(viz): Obsidian-style force sliders

**Tried:** top-right panel with live sliders — Repel (charge), Link length (distance), Link force
(strength), Center gravity (forceX/Y) — plus reset.
**Worked:** sliders drive the d3 sim live; centre-gravity re-centres on window resize; HTML well-formed.
**Didn't:** —

---

## `8eedb5f` feat: live conversation-status node + emotion/mood-aware prompt

**Tried:** (a) live "current topic" tracking; (b) emotion → FAST mood; (c) restructured, retrieval-augmented
prompt with affect.
**Worked:**
- Dedicated `ConversationNode` (rolling last-3 topic keywords + mood + emotion, linked to person **and**
  robot) — updates in place, verified via unit check.
- Prompt rebuilt into 3 blocks (IDENTITY / HOW TO REPLY / WHO YOU'RE TALKING TO); **dual affect signal**
  (mood valence in context + emotion label tagged on the current user turn); memory capped (top-4
  interests, ≤3 topics each, 3 recent notes one-per-topic).
- Raised embedding matcher floor 0.50 → 0.62.
**Didn't work → fixed (found during live testing):**
- *Save spam* returned once emotion was on — the raw detector valence jitters and kept tripping the
  0.04 dirty-gate. Fixed by widening the gate (save only on emotion-label change or ≥0.15 valence shift).
- *Spurious capability link* `tennis ↔ "good at math"`: the first design reused shared `TopicNode`s for
  the current topic, so the extraction embedding matcher attached bogus `about` edges. Root-caused and
  replaced with the dedicated `ConversationNode` (structurally cannot receive capability edges).
- *Accumulated artifacts*: one-time cleanup script purged old `current_topic` edges, orphaned
  current-topic topics, and non-keyword capability→topic links from `kg_state.json` (backup kept).
**Note:** `CurrentTopicEdge`/`set_current_topic` are now dead (kept in schema, unused).

---

## `aaa402d` feat: integrate webcam face-reco with the KG conversation pipeline

**Tried:** wire `webcam_loop` into the `graph_relationship` KG — recognize → retrieve into the prompt →
record turns → end-of-session extraction → update graph. PAD + emotion disabled by default; embedding
matcher default on.
**Worked:** full loop verified with a headless fake-LLM smoke test (seed → session → turns → extract →
graph update); seeds robot/human subgraphs from `specs/` on startup; retrieves interests / shared topics /
notes into the system prompt; records real turns onto `SessionNode.turns`.
**Didn't work → fixed:**
- *Crash on quit* `'Event' object is not callable`: `_DetectionWorker` stored its stop flag as
  `self._stop`, shadowing `threading.Thread._stop` (called by `join()`); surfaced once emotion was
  disabled and the worker finished fast. Renamed to `_stop_evt`.
- *Save spam* (identical snapshots each tick): added dirty-gating so KG-only ticks don't rewrite
  unchanged graphs.
