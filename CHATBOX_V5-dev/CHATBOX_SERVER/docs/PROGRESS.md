# Progress Log

Running record of what was **tried**, what **worked**, and what **didn't work / was fixed** — one entry
per commit, newest first. Companion to `RND_KG_Companion_System.md` (the design report). Kept so the
research write-up can reference which approaches were attempted and why.

---

## fix: self-declared culture is a recallable fact, not just a tentative hint  *(branch `feature/cultural-awareness`)*

**User point:** jay said *"i am from korea"* in a session, yet next time the robot answered *"I'm not sure if
I can remember that."* Investigation: the graph DID hold `jay --belongs_to_culture--> Korean` (provenance
`self-declared:<sid>`, confidence 1.0), and the prompt DID include it — but `_culture_block` worded EVERY
culture tag as *"a starting guess about their background, not a fact about them as a person"* + *"Never
assert…"*, so the LLM correctly refused to state it. The anti-stereotyping framing (right for an INFERRED
culture) was being applied to a background the child had EXPLICITLY stated.

**Fix (Approach A — graph-driven, no new data/mechanism):** branch the framing on the belongs_to_culture
edge's existing PROVENANCE.
- New pure reads in `graph_relationship/cultures.py`: `person_culture_source()` (the edge's `source`) and
  `person_culture_self_declared()` (True iff `source` starts `self-declared`).
- `_culture_block` now emits: self-declared → *"Background (they told you themselves): Korean… you CAN recall
  it as a fact if they ask — e.g. 'you mentioned you're Korean'. Don't assume what they like from it — ask."*;
  manual/seed-assigned → unchanged tentative hint. Either way it still never ASSUMES preferences from the
  background. Provenance was already stored (self-declaration writes `self-declared:<sid>`, manual writes
  `culture-seed`) — A just reads it.

**Verified:** `test_culture_seed.py` new case 5b — self-declared person gets recall-as-fact wording (no
"starting guess"); manually-assigned person keeps the tentative hint (no recall permission); neither assumes
preferences. Existing cases 1–5 + purity still green (cultures.py still imports only schema/store/topics).
Live: jay's rebuilt block now reads *"Background (they told you themselves): Korean…"*.

**Deferred (Approach B, later):** a general biographical-facts memory (origin/age/family/pet…) separate from
topics/culture, so first-person facts beyond culture also persist and are recallable.

---

## feat: continuous affinity (0–10) + confidence hedging — Approach 1, STEP 1  *(branch `feature/cultural-awareness`)*

**Goal:** replace the flat "observed = positive" assumption with a SIGNED, graded signal per topic, feeding
two consumers side by side: the preference BN (affinity → signed clamp) and the person-memory prompt
(affinity → like/neutral/dislike word, confidence → "clearly/probably/possibly" hedge). Human-facing scale
is 0–10; stored internally as `affinity ∈ [0,1]` (0 dislike / 0.5 neutral / 1 like) so it drops into the BN
clamp with no conversion. Confidence is model-agnostic and (this step) affects prompt wording ONLY — it does
NOT weight the BN clamp yet (deferred).

**What was built:**
- **Schema (pure, `graph_relationship/schema.py`):** `AboutEdge` gains `affinity: float = 0.5` and
  `confidence: float = 1.0` (both `[0,1]`). Backward-compat: old `kg_state.json` about edges load as
  neutral/fully-trusted. Robot capability→topic about edges just carry the unused defaults.
- **Boundary helpers (pure, `graph_relationship/scales.py`, new):** exactly `aff01_from_10` / `aff10_from_01`
  — the ONLY place the 0–10 ↔ [0,1] conversion happens.
- **topics.py:** `add_person_topic` / `reinforce_person_topic` / `add_person_interest` now accept + write
  `affinity`/`confidence` on the about edge (re-mention = simple overwrite; EWMA blending noted as future
  work). New pure read `person_topic_affinity(store, pid) → [(TopicNode, affinity, confidence)]` — the
  shared "observed evidence" reader for BOTH consumers.
- **Extraction (`modules/kg_extraction.py`):** prompt asks for a per-topic `sentiment` 0–10 (5 = neutral/
  unstated) alongside the existing `confidence`; mapped via `aff01_from_10`. NEW guard: missing/out-of-range
  sentiment → 0.5 (item KEPT, not dropped); all existing drops (bad JSON/category/hallucination/conf<0.6)
  unchanged.
- **BN clamp (`modules/preference_model.py`):** the flat `_OBSERVED_P = 0.90` clamp is gone — an observed
  topic now clamps at its stored affinity. Propagation is signed: a LIKED neighbour raises others (noisy-OR,
  as before); a DISLIKED one (below neutral) pulls related neighbours DOWN toward its low clamp, scaled by
  `0.8·weight`. Observed nodes stay fixed and remain excluded from suggestions. Still read-only, 2-round, no
  BN library, no confidence-weighting.
- **Prompt (`modules/affinity_phrasing.py`, new + `_person_memory`):** learned topics render as signed,
  hedged lines — "They clearly like jazz." / "They probably dislike baseball — avoid raising it." / "They may
  be neutral on pasta." The old flat `Interests:` summary is replaced by a `How they feel about topics:`
  block; specific notes still lead.
- **Viz (minimal):** person interest→topic about edges expose `affinity` (shown 0–10) + `confidence` in the
  edge tooltip.

**Verified — `test_affinity.py` (headless, fake LLM), 6/6:** (1) real `kg_state.json` copy loads with every
about edge neutral/fully-trusted, save→load→save byte-identical; (2) sentiment mapping love-jazz→1.0,
can't-stand-baseball→0.1, unstated-pasta & oob→0.5 kept, idempotent; (3) helpers round-trip exact at
0/5/10; (4) disliked topic drags a `related_topic` neighbour BELOW prior, liked one lifts ABOVE, both
observed excluded, read-only + deterministic; (5) verb×hedge wording correct; (6) `graph_relationship/`
purity (no LLM/PAD/app/embedding imports). Existing `test_preference_model.py` updated to pass `affinity=0.9`
for its observed-positive topic (preserves the 0.468 propagation number) — still green; culture-seed/
culture-extraction and the `graph_relationship` pytest suite still green.

**Didn't do (out of scope for STEP 1, per plan):** cross-namespace bridges, confidence-weighting of the BN
clamp, style hint / communication-policy vector, EWMA blending of re-mentions, any PAD change.

**Live-smoke follow-up (extraction robustness):** first real run surfaced two issues, both fixed:
1. *Intermittent "LLM JSON parse failed"* — the extraction call reused the spoken-reply settings (temp 0.7 +
   ChatML `stop` strings + greedy first-`{`/last-`}` slicing), so qwen2.5:7b occasionally wrapped the JSON in
   prose or truncated it, silently dropping the whole turn's topics (and thus all affinity). Added a
   `json_mode` path to `LLMClient.respond`: `response_format={"type":"json_object"}`, temperature 0, no stop
   strings, and skips `_clean_reply`; the extraction wrapper (`_json_llm`) now uses it. Live: 10/10 parses OK
   (was intermittent).
2. *Timid sentiment* — qwen rated "used to like the spicy beef soup" as 5 (neutral). Added full-range anchors
   to the sentiment rule in `build_system_prompt` ("hate it"→0-1 … "love it"→9-10). Live: the same utterance
   now scores 7 → affinity 0.70 → "like"; "don't really like sushi" → 3 → 0.30 → "dislike".

---

## fix: tag a person's culture ONLY on explicit self-declaration  *(branch `feature/cultural-awareness`)*

**User point:** jay shouldn't be connected to Korean "at first place unless jay said he is korean" — the demo
had assigned jay via the manual `--assign-culture` flag with no basis, which is exactly the stereotyping the
layer is meant to avoid.
**Two fixes:**
1. Removed jay's unfounded `belongs_to_culture` edge from the live KG. ChatBox still `knows_culture` Korean
   (its own prior knowledge) and the culture + priors remain — only the *person* tag was wrong.
2. `belongs_to_culture` is now driven by SELF-DECLARATION, not an operator hunch:
   - new `modules/culture_extraction.py` — `detect_self_declared_culture(turns, llm_fn)`: reads ONLY the
     person's own words and returns a culture label ONLY on an explicit self-statement ("I'm Korean", "my
     parents are from Korea"). Returns None for liking kimchi/K-dramas, visiting Korea, speaking Korean, or
     any ambiguity. Conservative guards (NONE/empty/sentence/tag/non-demonym → None); robot replies never fed.
   - wired into `_extract_session` step (c): after topic extraction, if the person self-identifies and isn't
     already tagged, `assign_person_culture`. (This is NOT the prohibited auto-detection from face/name/
     language/appearance — it's the person explicitly telling us; the earlier "manual only" guard was really
     anti-stereotyping, which self-declaration honours.)
**Verified:** real LLM 7/7 — "I'm Korean"/"my parents are from Korea" → tagged; "i love kimchi and
k-dramas" / "I visited Korea" / "recommend korean bbq?" / "I'm learning Korean" → NOT tagged. Headless
`test_culture_extraction.py` (fake LLM): parse/guards, robot-reply excluded, end-to-end (kimchi-lover not
tagged, explicit declaration tags once, idempotent). culture-seed + preference-model tests still green.
**Note:** the manual `--assign-culture` flag stays as an admin override. The seed still only sets ChatBox's
`knows_culture` + priors; no person is tagged by seeding.

---

## feat: Bayesian preference overlay — recommend topics on read (Command B)  *(branch `feature/cultural-awareness`)*

**Goal:** rank topics the robot could tentatively bring up, blending the person's CULTURE priors (base
rates) with OBSERVED interests (strong evidence) and soft propagation over `related_topic` links — a simple
noisy-OR, no external BN library.
**Module (`modules/preference_model.py`, app layer):** `rank_suggestions(store, person_id, k=3,
floor=0.35) -> [(node_id, posterior)]`. Compiled ON READ, mirrors the `_tier_from_edges` pattern:
READ-ONLY (zero writes/creates), no live embeddings (consumes only STORED `related_topic` weights), no LLM,
no PAD. `graph_relationship/` is untouched (imports its pure reads only).
**Algorithm:** (1) candidate set = culture-prior topics ∪ the person's own topics ∪ their one-hop
`related_topic` neighbours; (2) init unobserved = culture prior (default 0.30); (3) clamp observed
(interest/about) = 0.90; (4) 2 rounds noisy-OR `p[b]=max(p[b], p[a]·0.8·w)`, observed stay clamped;
(5) top-k UNOBSERVED with posterior ≥ floor, tie-break posterior→prior→lexicographic id. Dislike/negative
evidence explicitly OUT OF SCOPE (noted in a comment as future work).
**Namespace join (post-redesign):** culture priors live on `ck:` CultureTopic nodes, DISTINCT from a
person's `topic:` nodes — so the model works in a unified SLUG space (a culture "kimchi" prior and a person
"kimchi" topic are the same concept, joined by normalized label). This is the label-join flagged when the
culture layer became ChatBox-owned.
**Prompt (`_culture_block`):** the raw-prior top-4 list is replaced by `rank_suggestions(k=4)` output;
observed exclusion now comes for free (step 5). All other wording unchanged.
**Pre-step:** `--mode consolidate` still creates `related_topic` links among PERSON topics (culture topics
are a different node_type, so consolidation skips them — verified it runs cleanly and leaves the 12 culture
topics untouched). The overlay degrades gracefully to prior-only when no related links exist.
**Verified (`test_preference_model.py`, 6 synthetic checks):** prior-only == culture priors desc;
propagation raises a linked topic (kpop 0.30→0.468 via kdrama·0.8·0.65) while an unlinked one is unchanged;
observed topic never suggested; READ-ONLY (node/edge counts + save bytes identical across 3 calls);
deterministic; graceful degradation when all `related_topic` edges are deleted. `test_culture_seed.py` +
`tests_schema.py` still green. Real-LLM pipeline (qwen2.5:7b): fresh person tagged Korean → "i've been
watching a lot of kdramas" → extraction adds the `kdramas` interest → "what should we talk about?" → robot
offers Korean dramas AS A QUESTION, no false assertion. (Minor: extraction produced plural "kdramas" vs the
culture "kdrama" slug — a labelling variance, harmless.)
**Deferred (future):** negative/polarity evidence; confidence-weighted soft evidence; cross-namespace
culture↔person `related_topic` links (would let a jazz-lover get kpop *propagated*, not just base-rate).

---

## refactor: culture as ChatBox's prior knowledge — separate nodes, HJ decoupled  *(branch `feature/cultural-awareness`)*

**User feedback on the first culture cut:** the viz showed HJ *near* Korean even though HJ never discussed
anything Korean. **Root cause:** the culture reused the SHARED `topic:hiking` node (Command A's "reuse, don't
duplicate" rule), and HJ already liked hiking — so both pointed at the one node and the force layout pulled
them together (a 2-hop path, not a real HJ→Korean edge). **User's reframing (approved):** a culture is the
ROBOT's *prior knowledge*; ChatBox should own it, and a person only links to a topic by actually talking
about it over time. So: (1) anchor the culture under ChatBox, (2) give the culture its OWN topic nodes so no
unrelated person is ever coupled in.
**Redesign:**
- schema: new `CultureTopicNode` (id `ck:<culture>:<slug>`, own `node_type` so topic consolidation / interest
  machinery never touch or merge it into a person topic) + `KnowsCultureEdge` (robot→culture, SLOW).
  `CulturePriorEdge` now targets a CultureTopic, not a shared Topic.
- `cultures.py`: `culture_topic_id`, `ensure_culture_topic`, `knows_culture`, `culture_knowers`;
  `culture_priors` now returns `(ck_id, label, prior)`. Still schema/store/topics-only (pure).
- `culture_seed.py`: seeds `chatbox --knows_culture--> Korean --culture_prior--> ck:korean:*` — NO
  `resolve_topic`, so shared person topics are never created or reused. Robot-aware (`--robot`).
- prompt (`_culture_block`): offers now exclude the person's own topics by NORMALIZED LABEL (culture topics
  are separate nodes, so compare slugs, not ids). Wording unchanged.
- viz: `CultureTopic` → 6-point "sparkle" shape + legend; `knows_culture` → SLOW.
**Model now:** `chatbox --knows_culture--> Korean`; `Korean --culture_prior--> ck:korean:kimchi …`;
`jay --belongs_to_culture--> Korean` (tag only). `topic:hiking` (HJ's interest) and `ck:korean:hiking`
(ChatBox's knowledge) are DISTINCT nodes → HJ has zero culture edges.
**Verified (`test_culture_seed.py`, rewritten):** empty seed = chatbox-owns-Korean + 12 culture topics/priors
+ 0 person topics, round-trips identically; idempotent; on a copy of the real KG person topics are UNCHANGED
(+12 culture topics, no reuse); jay→Korean prompt block correct (≤4 offers, kpop excluded by slug, memory
leads, ChatBox owns culture); **HJ decoupled** (shares "hiking" label, zero culture edges, ck≠topic node);
purity. `tests_schema.py` still 5/5. Real-LLM smoke (qwen2.5:7b) unchanged in behaviour (bibimbap/kimchi
politely; offered jazz as a question, no false assertion). Live `kg_state.json` re-seeded to the new design.
**Worktree:** done in an isolated git worktree (`.claude/worktrees/cultural-awareness`) so another agent can
use `first-impression` in the main checkout concurrently.
**Deferred (gated):** Command B — read-only Bayesian preference overlay. Note for B: culture priors are now
`ck:` nodes distinct from person `topic:` nodes, so B must label-JOIN a culture prior to a person's observed
topic by slug (not node id).

---

## feat: Korean culture layer — dummy priors + prompt injection (Command A)  *(branch `feature/cultural-awareness`)*

**Goal:** a MANUAL cultural-background layer that gives the robot a weak, respectful hint about a person's
background + a few topics it could tentatively offer — plumbing/prompt only, NO Bayesian inference yet
(that's the gated Command B, held for approval). No auto-detection from face/name/appearance — assignment
is manual (CLI) only.
**Schema (pure):** `CultureNode` (id `culture:<slug>`), `BelongsToCultureEdge` (person→culture, SLOW,
idempotent), `CulturePriorEdge` (culture→topic, `prior∈[0,1]`, SLOW, upsert). Added to the node/edge unions
exactly like `ConversationNode`/`RelatedTopicEdge`; old `kg_state.json` loads unchanged (40 nodes/44 edges).
**Pure helpers (`graph_relationship/cultures.py`):** `ensure_culture`, `assign_culture`, `set_culture_prior`
(clamped), `culture_priors` (sorted desc), `person_culture`. Imports ONLY schema/store/topics.
**Seed (app `modules/culture_seed.py`):** seeds ONE culture `Korean` with 12 demo topics + DUMMY priors
(hand-set placeholders, marked as demo — not research claims) via `resolve_topic`, so topics already in the
graph (e.g. `hiking`) are REUSED not duplicated. Idempotent. Does NOT assign anyone. CLI:
`--mode seed-culture-demo` + a standalone `--assign-culture <person> <culture>`.
**Prompt (`_person_memory._culture_block`):** if the person has a `BelongsToCultureEdge`, append a
"CULTURAL BACKGROUND" block — a hint line ("starting guess … not a fact about them"), up to 4 highest-prior
topics EXCLUDING ones they already have an interest in, and tentative-offer phrasing (offer ONE if the
convo lulls, drop if uninterested, never assert what they like). Appended LAST so specific memories still
lead (applies the earlier mood-weighting lesson — weak hint, content first).
**Viz:** `Culture` node → heptagon + legend row; `belongs_to_culture`/`culture_prior` → SLOW colour; edge
thickness reflects `prior`.
**Verified (`test_culture_seed.py`, headless, fake/real LLM):** empty seed = 1 culture/12 topics/12 priors,
save→load→save byte-identical; idempotent; on a COPY of the real KG only `hiking` is reused (topics +11 not
+12, overlap computed from the file); jay→korean prompt has the block with ≤4 offers that exclude jay's
kpop interest and keep interests above the culture block; purity (pure `graph_relationship/` has no
LLM/PAD/app imports). Existing `tests_schema.py` still 5/5. Real-LLM smoke (qwen2.5:7b): "korean food?" →
bibimbap/kimchi politely; "what should we talk about?" → offered a topic as a question, no false assertion.
**Note:** the LIVE `kg_state.json` was never mutated — all tests used temp copies; seed it explicitly to see
the culture nodes in the viz.
**Deferred (gated):** Command B — the read-only Bayesian preference overlay (`preference_model.py`,
noisy-OR propagation over `related_topic`). Held until this is approved.

---

## feat: cultural-awareness webcam view — FairFace region/age (--culture, display-only)  *(branch `feature/cultural-awareness`)*

**Goal:** estimate a person's regional heritage (East Asian, European, …) + age from the webcam and show it
on the live view. Display-only this pass — deliberately NOT fed to the KG or the LLM prompt.
**Model (`modules/face_webcam/demographics.py`):** pluggable `DemographicsDetector` mirroring the
emotion/face backends; `FairFaceDetector` runs the FairFace ResNet-34 ONNX (7 race + 2 gender + 9 age in one
pass) via onnxruntime (~15 ms/face CPU, no torch dep). 7 races → friendly regions (East Asian, Southeast
Asian, European, African, South Asian, Middle Eastern, Latino/Hispanic); age band → coarse stage. Weights
(~85 MB) cached to `~/.fairface/`, auto-downloaded on first use. A confidence-weighted `_StableVote`
stabilises + "locks" the estimate (demographics don't change frame-to-frame); standalone CLI + `--image`.
**Webcam wiring (`webcam_loop.py`):** `--culture` flag; the detection worker runs the model per face
(throttled every 3rd cycle, last estimate cached between), draws `region conf* / age (stage)` under each
face box in cyan. Off by default; existing runs unchanged.
**Verified:** module smoke test (loads, infers, vote locks + re-locks on sustained flip); headless worker
→ overlay path attaches region/age and `draw_overlay` renders it; `--culture` parses.
**Note:** region is an uncertain appearance estimate (flickers ~1 s then locks); framed as non-identifying.
**Didn't / deferred:** no KG persistence, no prompt use, no auto-culture-assignment from it (kept manual).

---

## refactor: prune dead code + share consolidation core (compact/modular)  *(branch `KG-knowledge-extraction`)*

**Goal:** review the branch and remove unnecessary/redundant code. Net −109 lines, no behaviour change.
**Removed (dead):**
- `_mood_phrase`, `_MOOD_WEIGHT`, and the `mood`/`emotion` params of `_build_system_prompt` (+ the
  `cur_mood`/`cur_emotion` locals) — dead since mood was dropped from the prompt.
- `CurrentTopicEdge` (schema class + union member + store classification + viz timescale map + comments) —
  superseded by `ConversationNode`; 0 such edges in any persisted graph, so removal is backward-compatible.
- `SessionStore.person_turns` (never called); an unused `_dump_kg` import.
**Refactored (modular):** the three consolidation functions (`consolidate_topics`, `consolidate_interests`,
`link_related_topics`) shared ~120 lines of near-identical embed + pairwise-cosine + union-find loops. Pulled
out `_embed`, `_pairs`, `_merge_by_similarity`, `_same_category` helpers; the public functions are now thin
wrappers (kg_extraction.py −143 lines).
**Verified:** all modules compile; real `kg_state.json` loads (40 nodes / 44 edges, incl. related_topic);
merge/related behaviour identical on a controlled synthetic case (near-dups merge, related link, unrelated
skip); prompt has no mood line but keeps common-ground + related interests; real-LLM reply still recalls
memory ("R&B and jazz … SZA and Kendrick Lamar").

---

## feat: use topic↔topic relations in retrieval + common ground (2c points 1&2)  *(branch `KG-knowledge-extraction`)*

**Goal:** actually *use* the `related_topic` edges (they were structure-only). Wired points 1 (retrieval /
recall) and 2 (common ground); left 3–5 (recommendations / transitions / cold-start) for later.
**Implemented (pure helpers in topics.py):**
- `topic_related(topic_id)` — one-hop related topics.
- `related_common_ground(person, robot)` → `{direct, bridges}`: a bridge = a person topic `related_topic`-
  linked to a robot capability topic (indirect common ground, e.g. their *multiplication* ~ your *math
  problems*).
- `person_related_pairs(person)` — related pairs among the person's own topics (rap ~ hiphop).
**Prompt (`_person_memory`):**
- Point 2: "Common ground" now adds bridges — "You can also connect via related topics: their multiplication
  ~ your math problems".
- Point 1: note-gathering expands one hop across `related_topic`, so related memories surface; plus a
  "Related interests: hiphop ~ rap" line so the robot can bridge/recall across them.
**Verified (real LLM):** "do we have anything in common?" → "We both enjoy jazz and math problems…" (the
math-problems bridge now surfaces); "i love rap, can we talk about it?" → engages naturally. Pure modules
stay LLM-free.
**Deferred:** 2c points 3–5 (recommendations, smoother transitions, cold-start generalization); rapport/trust.

---

## feat: topic↔topic relations (Feature-2c) — link related-but-distinct topics  *(branch `KG-knowledge-extraction`)*

**User observation:** "rap" and "hiphop" didn't merge. **Finding:** cos(rap,hiphop)=0.678 — below the 0.86
merge floor, and there's NO safe merge threshold (tennis/basketball=0.654, dogs/cats=0.639 sit right below
rap/hiphop). **User's call:** don't merge them — **link** them instead (they're distinct but related).
**Implemented Feature-2c:**
- schema: `RelatedTopicEdge` (topic↔topic, SLOW, `weight`=similarity; conceptually undirected, stored with
  sorted endpoints).
- pure `topics.link_related_topic(a, b, weight)` — idempotent, only between topic nodes.
- app `kg_extraction.link_related_topics(store, embed_fn, related_floor=0.60, merge_floor=0.86,
  same_category_only=True)` — links same-category pairs whose cosine is in the "related" band
  [0.60, 0.86) (related but not near-duplicate). Embedding-only (no LLM), non-destructive.
- runs in `_auto_consolidate` (every extraction) and `--mode consolidate`; viz maps `related_topic` → SLOW.
- **merge stays for ≥0.86 only** (near-identical labels); the earlier gray-zone LLM-merge idea was dropped
  in favour of links.
**Verified + applied on real data:** links `hiphop ~ rap (0.678)` and `multiplication ~ math problems
(0.642)` — 2 clean links, no noise; new edge round-trips save/load.
**Note:** related links are non-destructive (both nodes kept), so a looser band is safe; lower
`related_floor` if you want more relations (e.g. jazz~r&b at 0.55).

---

## fix: consolidate every extraction + stop qwen Chinese/ChatML leak  *(branch `KG-knowledge-extraction`)*

**Two requests from a live run:**
1. Run consolidation **every extraction session** (was every 3 conversations).
2. A reply came out in **Chinese** and leaked ChatML tokens + a fake user turn
   (`…你在想什么新专辑？[CURIOUS]\n<|im_start|><|im_start|>user\nI don't really like it…`).
**Fixes:**
- `_maybe_auto_consolidate` (every-3 gate) → `_auto_consolidate`: runs topic + interest consolidation after
  **every** `_extract_session`. Dropped `_CONSOLIDATE_EVERY`.
- `LLMClient.respond`: added `stop=["<|im_start|>","<|im_end|>","\nuser",…]`, temperature 0.8→0.7, and a
  `_clean_reply()` that truncates at any leaked ChatML / next-turn marker. Prompt: "Reply in ENGLISH … output
  ONLY your single reply — never write the user's next turn."
**Verified:** `_clean_reply` cuts the exact leaked fake-turn; real-LLM replies now English + no leak, and
still use memory ("SZA's 'Open Arms'", previous-album preference).
**Note:** the qwen mid-sentence language switch is a model quirk — the stop tokens + English instruction +
lower temp make it far less likely, and the trailing fake-turn leak is always trimmed.

---

## feat: consolidation also merges near-duplicate INTERESTS  *(branch `KG-knowledge-extraction`)*

**User report:** two nodes "sports" and "sport" never merged despite auto-consolidate running every 3
conversations.
**Root cause:** they're **Interest** nodes, not Topics — `interest:jay:sports` (old LLM-chosen label) vs
`interest:jay:sport` (new: topics wire under an interest named after their category, and the enum value is
"sport"). Consolidation only merged *topics*, so the interest-level dup was never touched.
**Fix:** extend consolidation to interests.
- Pure `topics.merge_interests(canonical, duplicate)`: redirect has_interest (person→) and about (→topic)
  edges onto the canonical, delete the duplicate.
- App `kg_extraction.consolidate_interests(store, embed_fn, floor, dry_run)`: per person, embed interest
  labels, merge cosine ≥ floor groups (canonical = highest degree → shortest → lexicographic).
- webcam runs BOTH topic + interest consolidation everywhere (auto every 3 convos, `C` preview,
  `--mode consolidate`).
**Verified:** "sports"↔"sport" cosine **0.957 ≥ 0.86**; applied on real data → one "sport" interest holding
both tennis + football_player; unit test of merge_interests redirects edges and deletes the dup.
**Known limitation (deferred):** interests whose *labels* differ but mean the same (old "math" interest vs
the new "science" interest that now holds math topics) won't merge by label similarity — that's the old
LLM-label vs category-name scheme mismatch, a separate normalization task.

---

## fix: thread-safe session DB + drop mood from the prompt entirely  *(branch `KG-knowledge-extraction`)*

**Two issues from a live run:**
1. **Viz `/history` crashed** — `ThreadingHTTPServer` touched the one `sqlite3` connection from multiple
   threads ("SQLite objects created in a thread can only be used in that same thread").
2. **Replies were unnatural** — even down-weighted, the mood made the robot say two contradictory things in
   one reply (e.g. "Messi is amazing! [CONCERNED] It's okay to feel sad sometimes."). User asked to stop
   injecting mood/emotion into the prompt for now.
**Fixes:**
- `SessionStore`: open with `check_same_thread=False` + guard every DB op with a `threading.RLock` — verified
  6 threads × concurrent reads, no errors.
- Prompt: **remove the mood line and the per-turn emotion tag** from the LLM prompt entirely. Mood/emotion is
  still written to the graph/conversation node (viz unaffected). HOW-TO-REPLY simplified to "reply to what
  they said; don't comment on feelings or offer support unless they raise it." `_mood_phrase`/`_MOOD_WEIGHT`
  kept (unused) for easy re-enable later.
**Verified (real LLM):** "fav football player" → "Lionel Messi"; "what do you think about him?" → a natural
football answer — no emotional-support bolt-ons, no contradictions.
**Deferred:** improving the emotion model / re-introducing mood with better weighting; 2c; rapport/trust.

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
