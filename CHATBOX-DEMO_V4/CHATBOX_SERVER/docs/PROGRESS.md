# Progress Log

Running record of what was **tried**, what **worked**, and what **didn't work / was fixed** — one entry
per commit, newest first. Companion to `RND_KG_Companion_System.md` (the design report). Kept so the
research write-up can reference which approaches were attempted and why.

---

## fix: down-weight mood/emotion by a quarter (content over emotional support)  *(branch `KG-knowledge-extraction`)*

**Problem (user report):** even after the last fix the robot still led with emotional support and deflected
questions ("who is my fav sports player" → "I noticed you're feeling down…"), because the emotion detector
kept reading the user as sad and the prompt over-weighted it.
**Fix (reduce mood/emotion weight ~25%, per request):**
- `_MOOD_WEIGHT = 0.75`: the mood valence used in the prompt is damped ×0.75, so mild negatives fall under
  the ±0.15 threshold and read as "neutral".
- Mood line reframed from a directive ("Right now they seem low 🙁") to a weak, explicitly-unreliable
  background hint; the per-turn emotion tag likewise softened to "(weak camera hint: …)".
- HOW TO REPLY: "CONTENT FIRST — reply to what they said; mood is a faint hint, usually IGNORE it; don't
  open with or redirect to feelings, and don't offer emotional support unless they raise their feelings."
**Verified (real LLM, negative mood + Sadness):** "fav colour?" → honest "I don't remember" (on-topic);
"you should answer my questions" → "Of course, what's on your mind?"; "do you like jazz?" → jazz answer;
"fav sports player" → answers "Lionel Messi" (brief mood aside remains — expected at a quarter reduction,
no longer a deflection).
**Didn't / deferred:** fixing the upstream emotion detector reading neutral faces as sad; 2c; rapport/trust.

---

## fix: memory actually gets used in replies (retrieval + prompt tuning)  *(branch `KG-knowledge-extraction`)*

**Problem (user report):** the robot didn't use past info — asked "who's my favourite tennis player?" it
said "I'm fuzzy"; and it deflected every message into "you seem sad, let's listen to music". Data was all
present (69 embedded turns; notes with "Rafael Nadal", "SZA 'Open Arms'").
**Root causes found (by rebuilding the real prompt):** (a) the mood rule *"if they seem low, be gentle and
reassuring"* + a stuck-negative mood made the robot **console instead of answer**; (b) the notes cap (3, one
per topic, recency-sorted) **hid the specific facts** behind generic notes; (c) RAG on a meta-question
("do you remember X") retrieved other **questions**, and the block showed the robot's past replies (which
included "I don't have the name") — reinforcing forgetting.
**Fixes:**
- Prompt HOW-TO-REPLY: answer the actual question directly from memory and **state the name**; if it's NOT
  in memory, say so — **never invent a name**; only *note* mood, don't dwell/redirect.
- `_person_memory`: surface **specific** notes first (proper nouns / quoted titles score higher), then
  recency; caps raised to 8 notes / 2-per-topic. So "Rafael Nadal" and "SZA 'Open Arms'" lead.
- RAG block shows only **the person's own words** (not the robot's past replies), top_k 3→5.
**Verified with the REAL LLM + data:** "favourite tennis player?" → *Rafael Nadal*; "favourite r&b artist?"
→ *SZA*; "favourite colour?" (unknown) → *"I don't remember"* (no hallucination, no deflection).

---

## feat: RAG over transcripts + topic-click history (Phase 2)  *(branch `KG-knowledge-extraction`)*

**Tried:** use the SQLite transcripts for (a) RAG retrieval into the live prompt and (b) a viz "click a topic
→ see the conversation history", per the approved plan.
**Worked:**
- `modules/session_rag.py` (`SessionRAG`): embeds each turn once (cached in the store's `embedding` column),
  searches with FAISS `IndexFlatIP` (numpy fallback), blends similarity with **recency** and returns hits in
  **timeline order**. Lazy `reindex()`; embedding failures skipped/retried. embed_fn injected — no Ollama
  import inside the module beyond numpy/faiss.
- SessionStore gained `turns_needing_embedding / set_embedding / embedded_turns`.
- webcam: builds `SessionRAG` when embeddings are on, `reindex()`es at startup, and injects the top-3
  relevant past turns for the current message into a new prompt block "Relevant things they've said before"
  (timeline-dated).
- viz server: `HistoryProvider` + `/history?topic=&person=` endpoint (RAG when an embed model is reachable,
  else keyword `turns_for_topic`); `--sessions-db` / `--embed-model` args. Frontend: clicking a Topic node
  fetches `/history` and renders the conversation turns (child/robot bubbles + timeline).
**Verified (headless, fake embeddings):** RAG search returns the right turns; prompt gains the RAG block;
keyword history works; FAISS present (1.13.2); HTML well-formed; graph_relationship pure modules import no
session/LLM code.
**Note:** the 62 migrated turns have no embeddings/topic-tags yet, so topic-click history needs one RAG run
with Ollama up (webcam startup reindex, or the viz server with `--embed-model`) before it populates.
**Didn't / deferred:** 2c topic↔topic relations; rapport/trust; removing the now-unused "Session" legend row.

---

## feat: externalize session transcripts to SQLite (Phase 1)  *(branch `KG-knowledge-extraction`)*

**Tried:** move conversation transcripts OUT of the knowledge graph into a dedicated SQLite store so the KG
focuses on relationships/topics/interests and the viz is no longer cluttered with per-session nodes
(user approved: SQLite backend; sessions removed from graph; topic-click history via RAG comes in Phase 2).
**Worked:**
- New app-layer `modules/session_store.py` (pure stdlib sqlite3, no graph/LLM/PAD imports): one row per turn
  (session_id, person, robot, turn_idx, ts, emotion, child, reply, topics, embedding-reserved, extracted);
  `append_turn / unextracted_turns / mark_extracted / person_turn_count / session_count / turns_for_topic`.
- Pure `interactions.set_interaction_count()` so the Interaction node's count comes from the transcript DB
  instead of graph SessionNodes.
- webcam rewired: chat turns write to SQLite (not the graph); no more SessionNode/`start_session`/graph
  `append_turn`; `_extract_session` reads un-extracted turns from SQLite and `mark_extracted`s them; auto-
  consolidate cadence counts `session_store.session_count()`. `_ensure_interaction` + a per-run uuid session id
  replace the old `_ensure_session`.
- `--mode migrate-sessions`: moved the real graph's **17 sessions / 62 turns** into `sessions.db` and removed
  all SessionNodes; graph node types now: person/robot/interaction/topic/interest/conversation/persona/role/
  capability. Migrated turns are marked extracted; interaction_count preserved.
**Verified (headless, fake LLM):** store ops; extraction reads SQLite + adds typed topic + Δrapport; zero
session nodes created in the graph; re-extract idempotent; tier unaffected; migration moves turns + strips
nodes + preserves counts.
**Didn't / deferred (Phase 2):** FAISS RAG retrieval into the prompt; topic-node click → conversation history
in the viz; removing the now-unused "Session" legend row. Rapport/trust still parked.

---

## fix(kg): category enum coercion + viz spread-out force defaults  *(branch `KG-knowledge-extraction`)*

**Tried:** (1) retype the 15 pre-existing `other` topics; (2) make the graph self-spread so no manual
dragging is needed.
**Worked:**
- **Bug found + fixed:** `resolve_topic`/`merge_topics` upgraded category via `model_copy(update=...)`, which
  in Pydantic v2 does NOT re-validate — so the category was left as a plain `str` in memory (only fixed itself
  after a save/load). Now coerced to `TopicCategory(...)` explicitly. Verified in-memory type is the enum.
- Retyped all 15 existing topics (data op on `kg_state.json`, backup `.pre-retype.bak`): science
  (math/space/mars), music (jazz/r&b/hiphop/favorite songs), sport (tennis), food (baking/pasta),
  activity (hiking/camping), place (landscapes), animals (dogs). None left `other`.
- Viz force defaults tuned to spread out: charge −320→−700 (distanceMax 600), link length 90→130, link
  force 0.4→0.35, collide 28→34, centre-gravity 0→0.04. Sliders + FORCE_DEFAULTS updated to match.
**Didn't / deferred:** LLM-based retyping (used a deterministic map for the known set); 2c; rapport/trust.

---

## feat: auto-consolidate every 3 conversations + Feature-2d category colours  *(branch `KG-knowledge-extraction`)*

**Tried:** (1) auto-run topic consolidation every 3 conversations at end-of-session; (2) Feature-2d — colour
topic nodes in the viz by their category.
**Worked:**
- `_maybe_auto_consolidate()` runs inside `_extract_session` after extraction: counts total SessionNodes in
  the graph (persists across runs) and, when `count % 3 == 0`, applies `consolidate_topics` (merges). Verified
  it fires only at 3, 6, … and is a no-op without embeddings. **Design change (user-approved):** consolidation
  is no longer strictly manual — it auto-applies every 3rd conversation; the standalone `--mode consolidate`
  and `C` preview still exist.
- 2d: viz server now emits `category` on topic nodes; `index.html` tints each Topic diamond by a 10-colour
  category palette (`CATEGORY_COLOR`) and adds a "Topic category → fill" legend. Live-updates when a topic's
  category changes. HTML well-formed; transform emits category (existing topics show `other` until re-typed).
**Note:** old topics created before Feature-1 are all `category=other` (grey) until a new extraction types
them — expected.
**Didn't / deferred:** 2c topic↔topic relations; rapport/trust (still parked).

---

## feat(kg): Feature-2 semantic topic consolidation (2a + 2b)  *(branch `KG-knowledge-extraction`)*

**Tried:** merge near-duplicate topics that exact-label reuse can't catch ("hiphop"/"hip hop",
"football"/"soccer"). **2a (pure, graph_relationship):** `merge_topics(canonical, duplicate)` — redirect all
incident edges onto the canonical, union notes (+ `merged_from` marker), upgrade category only if canonical
was `other`, delete the duplicate; plus `topic_degree()` and a new pure `store.delete_node()`.
**2b (app layer, kg_extraction):** `consolidate_topics(store, embed_fn, floor=0.86, same_category_only=True,
dry_run=False)` — embed each label, pair by cosine ≥ floor, union-find groups, canonical = highest degree
(tie → shortest, then lexicographic), call the pure merge. Triggers: standalone `--mode consolidate`
(+`--dry-run`, `--merge-floor`) and an in-window `C` hotkey (dry-run preview only). **Approved scope: 2a+2b
only** — topic↔topic relations (2c) and category viz grouping (2d) deferred.
**Worked (verified, fake embed_fn):** dry-run proposes merges and writes nothing; apply merges the two
near-dup pairs, keeps distinct "jazz", 5→3 topics; canonical picks the shorter label; notes unioned with
`merged_from`; **idempotent** re-run (no further merges); **cross-category never merges** even at high
similarity; save/load round-trips; `graph_relationship/` stays free of LLM/PAD/app imports.
**Decisions:** hard-merge (redirect + delete) not alias; consolidation is **manual/reviewable**, never
auto-run during live extraction; merge floor 0.86 (stricter than the 0.62 capability floor); `C` is
preview-only (apply via `--mode consolidate`).
**Didn't / deferred:** topic↔topic relations, clustering, category-based viz grouping (revisit after this),
and any change to rapport/trust (still deferred).

---

## feat(kg): fine-grained topic typing + graph-aware extraction  *(branch `KG-knowledge-extraction`)*

**Tried:** two improvements to LLM knowledge extraction. **Step 1** — `TopicNode` gains a `category` from a
CLOSED taxonomy (`TopicCategory`: music/science/animals/food/activity/place/person/media/sport/other).
**Step 2** — condition the extraction prompt on the person's *existing* topics so the LLM reuses established
nodes; output splits into `existing_topics_discussed` vs `new_topics`. Kept decoupling: all LLM/prompt/guard
logic in the new APP module `modules/kg_extraction.py`; `graph_relationship/` gained only pure helpers.
**Worked (all verification points):**
- category enum defined once; **TopicNode id stays label-only** (category is an attribute, not identity —
  two extractions disagreeing on category resolve to the SAME node).
- backward-compat: old `kg_state.json` untyped topics load and default to `other` (real file: 14 topics).
- graph-aware reuse: with "jazz" known, a transcript saying "jazz music" lands in
  `existing_topics_discussed` and creates **no** second node (before==after counts).
- new topic ("dinosaurs") → one typed `TopicNode(animals)` wired via the Interest layer (category→interest).
- guards write **nothing** on: malformed JSON (whole extraction discarded), invalid category (dropped),
  hallucinated "existing" not in the provided list (dropped), confidence < 0.6 (dropped).
- idempotent: re-running identical extraction gives identical node/edge counts.
- category round-trips through save→load→save; `graph_relationship/` has **zero** LLM/PAD imports.
**Decisions / deviations (flagged):**
- Invalid category → **drop** the item (not coerce), so "nothing written" holds for bad output.
- Closeness (rapport/trust) kept working by reusing the existing pure `extract()` for **deltas only** +
  the untouched `adjust_closeness` (its interest logic is not used). Closeness logic itself untouched.
- New/existing topics wire under an Interest named after the **category** (`person→Interest(category)→Topic`).
- `resolve_topic(category)` only fills a category when the node is still `other` (first non-other wins;
  TopicNode has no provenance field, so a conflict is not persisted — kept, not merged).
- Capability↔topic auto-linking (old embedding matcher path) is **not** run in the new topic extraction —
  embeddings/merge are explicitly out of scope for this step.
**Didn't / deferred:** embeddings, fuzzy/semantic merge, topic↔topic relations, clustering (Feature-2).

---

## docs: R&D system report + progress log

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
