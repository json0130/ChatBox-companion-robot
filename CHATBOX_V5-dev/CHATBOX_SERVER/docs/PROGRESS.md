# Progress Log

Running record of what was **tried**, what **worked**, and what **didn't work / was fixed** — one entry
per commit, newest first. Companion to `RND_KG_Companion_System.md` (the design report). Kept so the
research write-up can reference which approaches were attempted and why.

---

## feat: robot hub — the viz puts the role/persona on a real ChatBox (Jetson) or Navel  *(branch `feature/persona-roles`, local)*
**Date:** 2026-10-28

### What was tried
Both robots run a client that connects OUT to a Socket.IO server (`client_init`, then server events). The Jetson,
not the ESP32, is the ChatBox's client (it forwards expression tags to the ESP32). `--robot-hub [PORT]` on the viz
server hosts a small server (`modules/robot_hub.py`) speaking that contract, so a robot only needs its
`server_url` pointed at the viz machine. Apply → `persona_update` (role, OCEAN as the 0..1 `personality` block, name;
the robot announces "Persona updated to …"); chat replies → `demo_step` (the robot speaks the reply, its `[TAG]`
drives the face). A robot connecting after Apply gets the last persona replayed. `/chat` and `/persona` return
`robot: {robots, sent}` and the panel shows it. (First attempt sent tags straight to the ESP32; dropped — the
Jetson owns that link. `esp32_link.py` stays for the webcam loop, which runs next to the ESP32.)

### What worked
- Tests with a real Socket.IO client standing in for the Jetson/Navel: handshake, reply spoken (tagged and untagged),
  persona pushed with the -1..1 → 0..1 mapping, late robot replay, no robot connected reported (not an error), bad
  `client_init` refused, robot streams ignored. 20 persona tests pass; real viz start-up answers the handshake.

### What didn't work / notes
- NOT run on a real Jetson or Navel yet.
- Voice is not changed by the persona yet (`voice_config` not sent); only role/personality and the spoken words differ.
- Clients' `persona_update` also rewrites their client_config.json (`robot_role`, `personality`) — by design in the
  client, but it persists on the robot after the test.
- Port 5000 clashes with the central server if it runs on the same machine; use `--robot-hub 5001`.

---

## fix: persona chat defaults to the AI server and explains why it is unavailable  *(branch `feature/persona-roles`, also on `feature/rolesim`)*
**Date:** 2026-10-19

### What was tried
The chat was only enabled by `--ai-server` / `--ollama`; started with neither, the page greyed out the input with no
reason ("not able to type"). Now the lab AI server is the default backend (`$AI_HOST`, else the lab IP), `--ollama`
swaps it, `--no-chat` turns it off. The page says why when chat is unavailable (switched off, or an old viz process
with no `/chat` route), re-checks every 5 s so a restarted server recovers without a reload, and an unreachable AI
server gives a message pointing at `--ollama` instead of a bare error; the client retries the connection per message.

### What worked
- Viz started with no flags → chat on via the AI server; typing, send and the reply work in the browser.
  `--no-chat` shows the explanation and a disabled input. 10 chat tests pass.

### What didn't work / notes
- Not reproduced as a bug on the page itself: the input was enabled and focusable; the reported symptom matches a viz
  started without a backend flag (or an old process).

---

## feat: persona chat panel in the viz — talk to the robot as set up  *(branch `feature/persona-roles`, also on `feature/rolesim`)*
**Date:** 2026-10-19

### What was tried
A "Chat with the robot" panel next to the persona panel, to compare how different roles / OCEAN personas change
the replies. `POST /chat` on the viz server (enabled by `--ai-server HOST[:PORT]` or `--ollama [MODEL]`) →
`modules/persona_chat.py`. About the ROBOT only: the prompt is IDENTITY + the loop's general reply rules (minus the
webcam-memory ones) + `role_persona_blocks()`; no WHO block, memory, RAG, culture or tier — it does not use who the
visitor is. Each message re-reads the KG file and `persona_assignment.json` into a private store, so the latest Apply
applies at once and the webcam loop need not run. The page restarts the conversation (and clears the history sent
to the model) whenever a new setup is applied, and tags each reply with role · persona words · expression tag.
`roles.role_identity_block()` now builds the IDENTITY block for both the loop and the chat (golden test: the loop's
prompt is byte-identical). `GET /persona` also returns `assigned` (what was last requested).

### What worked
- `test_persona_chat.py` (stub LLM, real HTTP): Apply → next message uses it; different setups give different
  prompts; no person information in the prompt; history capped; error paths (chat off 503 / empty 400 / backend
  502); works on a KG file with no seeded role library; reply-rule lines still match the loop's prompt.
- Live on a copy of the KG against the AI server: same question under Baymax (companion) and Roz (receptionist) gave
  clearly different replies, divider and meta line correct.

### What didn't work / notes
- A receptionist with no building knowledge invents directions ("…just to your left, past the security desk"):
  the loop's "don't state specifics not in memory" rule is not enough on its own. Add facts to the role, as the
  rolesim scenario does.
- The old Apply message said the robot "hasn't picked it up" even though chat already uses the setup; reworded.
- A viz server started before this change must be restarted to get `/chat`.

---

## refactor: render the role/persona prompt blocks from roles.py  *(branch `feature/persona-roles`; same change as on `feature/rolesim`)*
**Date:** 2026-10-19

### What was tried
`_role_persona_blocks()` moved from `webcam_loop.py` into `roles.role_persona_blocks(store, role, persona)` so other
callers (rolesim's B0 robot, the persona chat) render exactly the prompt the loop does. `test_prompt_golden.py`
holds three prompts recorded BEFORE the move (spec role, assigned tutor, tutor + a recognised person).

### What worked
- The loop's prompt is byte-identical after the move; all persona tests pass.

---

## fix: adopt a saved role/persona assignment at loop start-up  *(branch `feature/persona-roles`, also on `feature/rolesim`)*
**Date:** 2026-10-19

### What was tried
Starting `webcam_loop` after the first Apply in the viz crashed with `AttributeError: 'WebcamKGLoop' object has
no attribute '_store_lock'`. `_sync_persona_assignment()` was called early in `__init__`, before the store lock
existed — and before `_kg_dirty` is initialised (it is reset to False a few lines later, which would also have
dropped the "needs saving" flag). It is now the last statement of `__init__`.

### What worked
- Regression test `test_loop_applies_saved_assignment_at_startup` builds the REAL `WebcamKGLoop` with a saved
  `persona_assignment.json`: it reproduces the exact AttributeError on the old code and passes on the fix.
- Replayed the user's real `kg_culture.json` + `persona_assignment.json` (on copies): the loop starts as a
  receptionist and renders the HOW YOU TALK block.

### What didn't work / notes
- Missed earlier because my live test started with no assignment file and the unit tests built the loop with
  `__new__`, skipping `__init__`.

---

## feat: role speaking subnodes, profile cards, apply confirmation  *(branch `feature/persona-roles`)*
**Date:** 2026-10-16

### What was tried
Make each role say HOW the robot talks, robot-side only (nothing about who it is talking to).
- `RoleTopicNode` now carries `kind` (goal / manner / do / avoid / phrase / knowledge) and `items`; every library
  role in `role_seed.py` has one subnode per speaking part plus its knowledge topics. `RoleNode.style_hint` is gone
  (the manner/do subnodes replace it).
- Prompt: new `HOW YOU TALK AS A <ROLE>` block (goal; style = OCEAN rules + manner + do; avoid; example lines,
  "never repeat word for word") and `WHAT YOU KNOW AS A <ROLE>`, both from `_role_persona_blocks()`, which reads
  only the robot's attached role + persona. The loop prints these blocks when an assignment lands.
- Viz: 8 profile cards (role + OCEAN after animated characters, rough guesses) that load into the form; the role
  summary (goal / talks / avoids / knows); role subnodes coloured by kind; Apply now waits until the LOOP has written
  the role + OCEAN into the graph before saying "✓ applied", else warns after ~6 s that the loop isn't running.

### What worked
- Live, real `webcam_loop --no-window` + viz on a copy of the KG: Ms. Frizzle card → apply → the loop logged
  "role → tutor; persona → curious, outgoing, warm, calm" and the full HOW YOU TALK block within ~1 s; the panel
  turned "✓ applied". 64 unit tests + all root test scripts pass.

### What didn't work / notes
- The role select was blank while the spec's `chatbox:role:companion` was attached; it now shows the library role
  with the same name.
- Typed chat only enters through the OpenCV window, so the live check stops at the prompt (no LLM reply checked).

---

## feat: role + persona assignment from the viz — attachable role subgraphs, OCEAN → style  *(branch `feature/persona-roles`)*
**Date:** 2026-10-16

### What was tried
Give the robot a role and a personality from the browser, built like the culture layer and without PAD.
- `graph_relationship/roles.py` (schema + store only, like cultures.py): library roles `role:<slug>` own a
  knowledge subgraph (`role_knows` → `RoleTopicNode` `rk:<role>:<topic>` with facts). A robot plays the role its
  `has_role` edge points at; attach/detach moves that edge, so detached roles stay in the graph and add nothing to
  the prompt. `RoleNode` gained `brief` + `style_hint`; `PersonaNode` gained `ocean`.
- Persona = OCEAN in [-1, 1] → words for the IDENTITY line and observable style rules ("let them lead, don't push
  new topics") for HOW TO REPLY, strongest trait first, ±0.25 dead zone. No PAD.
- `modules/role_seed.py`: companion / tutor / storyteller / reading buddy / receptionist (hand-written demo data),
  seeded with the specs at start-up; attaches nothing by itself.
- Viz panel "Role & persona" → `POST /persona` → `persona_assignment.json` beside the KG (atomic write). The loop
  adopts it on its next tick (main thread owns the store) and again at start-up after the spec seed.
- Persona change: in-memory chat history cleared, and the robot's own replies from before the assignment (`ts`)
  are dropped from the "recent conversation" block; what the person said is kept.
- Ported `ai_server_llm.py` (with `json_mode`/`temperature`/`seed`) and `--ai-server HOST[:PORT]`.

### What worked
- `tests_roles.py` (attach/detach, knowledge stays with its role, words/style, bad input, save/load round trip) and
  `test_persona_roles.py` (viz request → loop → prompt; role switch removes the old knowledge; old replies hidden,
  their words kept). All 72 existing graph tests and every root test script still pass.
- Live: viz panel → tutor, E −0.8 → prompt carried "a tutor robot", the brief, "Personality: quiet", the quiet
  style rule, the tutor's manner and only the tutor's knowledge; viz showed "now: tutor · quiet".

### What didn't work / notes
- The PAD-driven version is parked on `feature/persona-roles-pad`; PAD stays out until persona styling is tested.
- The spec seed re-attaches the spec role/persona every launch; the sidecar re-applies the assignment right after,
  so the viz may show the spec role for the first moment of a run.
- Role knowledge is injected whole (fine for the demo library; needs retrieval if a role gets many topics).

---

## merge: KG-knowledge-extraction (now on main) into feature/cultural-awareness  *(branch `feature/cultural-awareness`)*
**Date:** 2026-09-24

### What was tried
Branch restructure: `main` = V4 demo + chatboxv1-3 + face tracking; `KG-knowledge-extraction` hangs off
`main`, and this branch hangs off KG. Merged the updated KG in with `-X no-renames`, because this branch
renamed `CHATBOX-DEMO_V4` → `CHATBOX_V5-dev` and rename detection would have pushed main's demo changes
into the research tree.

- **CHATBOX-DEMO_V4:** restored, identical to KG's — V4 is the demo, V5-dev is research; both kept.
- **CHATBOX_V5-dev:** untouched by the merge (identical to before). Git's directory-rename detection still
  mapped 4 of main's V4 files into V5-dev (`WIFI_DEBUG.md`, face-tracking ESP32 sketch/servo test,
  `face_tracking_output.py`); dropped — they live in V4.
- **New from main:** `AFFECT_LAB/`, `SERVO_STYLE/`, graphify setup; `.gitignore` unioned.
- **Not verified:** not run end to end.

---

## feat: first-impression integration — auto-enrol strangers into the culture pipeline  *(branch `feature/cultural-awareness`)*

**Goal (user):** port the `first-impression` branch's "meet a stranger" pipeline into the culture branch, but
as a HYBRID: an unknown face is auto-enrolled as a new person, and then the FULL culture/interest pipeline
(auto-attach, extraction, viz highlight) runs on them — not the first-impression branch's stripped
fast-node-only mode.

**Why a manual port, not a merge:** a `git merge first-impression` is unworkable — ~130 rename/delete
conflicts from committed runtime data (rag_indexes/faiss, sessions.db, faces.npz) + a real `webcam_loop.py`
conflict (both branches rewrote it). Culture already had the infra it needed (`identify_all`, `enroll`,
4-value VA emotion), so only the auto-enrol + name-learning was ported. The `first-impression` branch is left
intact for further research.

**Ported (new, no conflicts):**
- `graph_relationship/rename.py::rename_person` — pure graph surgery: re-key `guest_N` → real name across the
  person node + `interaction:`/`conversation:` ids, re-pointing every incident edge (rapport/trust/mood/
  culture preserved). `SessionStore.rename_person` (SQLite rows). `FaceIdentifier.rename` (move / weighted-merge
  the enrolled embedding).
- `webcam_loop`: module-level `_extract_name` (regex self-intro, stoplist rejects feelings/filler + culture
  words) + `_slug_name`; methods `_next_guest_id`, `_auto_enroll` (enrol unknown face as `guest_N` + save
  faces.npz), `_learn_name` (re-key across face DB + graph + transcripts + in-memory maps, under the store
  lock). Wired into the chat Enter handler: unknown face in view → auto-enrol as `guest_N` (this turn is
  attributed to them); if they introduce themselves → re-key to the real name — all BEFORE the reply is
  dispatched, so the prompt + KG writes use the right id.
- A `guest_N` is a normal `PersonNode`, so culture auto-attach, interest extraction, cross-namespace bridges,
  and the face-driven viz highlight all work on them unchanged.

**Verified — `test_first_impression.py` 6/6:** name extraction; graph rename_person (ids re-keyed, culture +
relationship edges follow, no dangling guest id, counts stable); session rename; face rename (move + merge);
`_learn_name` re-keys all four stores; a guest gets the full culture/interest prompt. Full culture regression
green (multi-culture, style-hint, soft-evidence, cross-namespace, affinity, preference, culture-seed/extract,
+ graph_relationship pytest 47).

**Deferred (inherited from first-impression):** if topics are extracted (via `X`) BEFORE the guest gives their
name, `interest:guest_N:*` node ids aren't re-keyed (rename_person re-keys person/interaction/conversation
only). The normal flow (name learned during chat, extraction at session end) attaches to the real name, so
this only bites an unusual ordering. Fix would extend rename_person to also re-key interest ids.

---

## feat: multi-culture selector — Māori added + active-culture (auto-attach) + viz toggle  *(branch `feature/cultural-awareness`)*

**Goal (research/testing):** let the robot hold MULTIPLE cultures (Korean + Māori) and adapt to whoever it's
talking to, with a viz toggle to A/B "robot has cultural knowledge" vs "generic" — to see if cultural
grounding improves UX.

**What was built:**
- **Māori seed** (`culture_seed`): refactored the Korean seeder into a generic `_seed_culture`; added
  `_MAORI_DEMO` (12 topics: haka, marae, waiata, te reo, hangi, rugby, pepeha…) + `_MAORI_STYLE_HINT`
  (manaakitanga/whanaungatanga, respect for kaumātua, humility). `seed_maori_demo` + `seed_all_cultures`;
  `--mode seed-culture-demo` now seeds BOTH. Demo data, not research claims. The person-driven culture path
  was already generic, so a Māori-tagged person "just works" once seeded.
- **Active-culture resolver** (`webcam_loop._active_culture_id`): the culture that shapes the prompt =
  **override → the current person's `belongs_to_culture` → generic**. So the active lens auto-attaches to /
  detaches from whoever is recognised, and the persistent per-person tag never flips. `rank_suggestions`
  gained an optional `culture_id`; `_culture_block` + the style-hint injection now use the ACTIVE culture.
  Framing: recall-as-fact only when the active culture is what the person SELF-DECLARED, else it reads as the
  robot's "knowledge lens" (a starting point, never a fact about them).
- **Mid-session auto-attach**: on self-declaration ("I'm Korean"), the culture attaches IMMEDIATELY (async
  detection off the main thread, guarded by a cheap origin-cue filter + "only while untagged"), so it shapes
  the rest of that session. Self-declaration/manual ONLY — never inferred from face/name/appearance.
- **Override backend (A/B, dormant)**: `GET/POST /culture` on the viz server writes a `culture_override.json`
  sidecar (kept OUT of kg_state.json); the loop reads it each turn. `auto` (person-driven) · `generic`
  (culture off, the A/B control) · a culture name (force Korean/Māori). The UI selector button was REMOVED —
  culture is now fully automatic (driven by the recognised person) — but the endpoint/file remain so A/B is
  still possible via curl/file without cluttering the page.
- **Viz active-highlight (display-only attach/detach, face-driven)**: the loop writes the current
  (face-recognised) person + resolved active culture to `active_state.json`; the viz server folds it into
  `/graph.json` as `active`. The page lights the robot + that person's OWN subgraph (BFS through their
  interests → topics → bridges) + the ACTIVE culture cluster, and dims everything else (`.inactive` CSS). So
  switching who's in front detaches the previous person's bridges; a SECOND person of the same culture keeps
  the culture nodes lit and only dims the previous person's own bridges (shared topics stay lit). Purely
  visual — `knows_culture` and graph data are never mutated (seeded prior knowledge stays true).

**Verified — `test_multi_culture.py` 5/5:** both cultures seeded + idempotent; person-driven lens+manner per
person with no cross-leak; resolver auto→person / forced / generic-off + viz labels; mid-session attach
(cue+untagged→attach, tagged→skip, no-cue→skip); viz round-trip flips the loop's prompt. Existing suites
green: test_style_hint, test_soft_evidence, test_cross_namespace, test_affinity, test_preference_model,
test_culture_seed (headers updated to the knowledge-lens wording), test_culture_extraction, +
graph_relationship pytest (47). `graph_relationship/` purity unchanged.

---

## feat: thin per-culture static style hint — Approach 1, STEP 4 (final)  *(branch `feature/cultural-awareness`)*

**What:** the "how to talk" half of cultural adaptation — a single short, STATIC manner paragraph per culture,
injected into HOW-TO-REPLY when a person is tagged to that culture. Deliberately dumb: hand-written seed
data, identical for every interaction; NO tier/affect/situation/dynamics (that is Approach 2's policy vector,
kept a clean separable step-up).

**Change:**
- **schema** (`CultureNode`): one optional field `style_hint: str = ""` — default "" so old graphs load
  unchanged and inject nothing.
- **seed** (`culture_seed`): Korean CultureNode gets a hand-written `_KOREAN_STYLE_HINT` (polite/warm/slightly
  formal; deflected compliments; food as friendliness). Set/overwritten idempotently on seed; counts & hint
  unchanged on re-seed.
- **prompt** (`_build_system_prompt`): if the person is tagged to a culture with a non-empty hint, append
  `"• Cultural manner (soft guidance, secondary to answering their actual question): <hint>"` to HOW-TO-REPLY
  — SEPARATE from the content topic-offer block (which stays in the WHO/CULTURAL BACKGROUND section). Pure read
  `cultures.person_culture_style_hint`. Untagged / empty hint → injects nothing (no leakage, no empty header).
- **viz**: CultureNode carries `styleHint` for its tooltip (optional).

**Static by construction:** the injected text depends ONLY on which culture the person is tagged with — same
string for a "visitor" vs a "close" person (verified).

**Verified — `test_style_hint.py` 6/6:** backward-compat (pre-feature culture → `style_hint=""`, round-trip
byte-identical); seed sets it, re-seed idempotent (counts + hint unchanged); injected when tagged &
manner/offer blocks stay separate; not injected when untagged; empty hint injects nothing; static across
tiers. Existing suites green: test_soft_evidence, test_cross_namespace, test_affinity, test_preference_model,
test_culture_seed, test_culture_extraction, + graph_relationship pytest (47). `graph_relationship/` purity
unchanged.

**This completes the four Approach-1 upgrades** (continuous affinity + hedging · cross-namespace bridges ·
confidence-weighted clamp · static style hint). Approach 2 (communication-policy vector / tier-driven,
dynamic manner) is the separable next milestone.

---

## feat: confidence-weighted BN clamp — Approach 1, STEP 3  *(branch `feature/cultural-awareness`)*

**Problem:** an observed topic clamped to its `affinity` alone, so a rock-solid "I love jazz" (aff 1.0,
conf 0.95) and a hedged "jazz is okay I guess" (aff 1.0, conf 0.62) clamped to the SAME value and
propagated equally hard across the bridges — the BN conflated *how much they like it* with *how sure we
are they said it*. The 0.6 extraction gate is binary (0.61 and 0.99 look alike once past it).

**Change (calibration, not a new model — `modules/preference_model.py` only):** the observed-topic clamp is
now `clamp_from(affinity, confidence) = 0.5 + (affinity - 0.5) * confidence`. An uncertain observation is
pulled toward neutral (0.5) so it moves the posterior less; confidence 1.0 → clamp == affinity (identical to
Step 2). Signed & symmetric (a confident dislike still pins low; an unsure one sits nearer 0.5). This is the
ONLY change — rounds (2), damping (0.8), floor, bridge traversal, and the candidate set are untouched, and
observed topics stay excluded from suggestions. Now the BN's number agrees with the prompt's Step-1 hedge
word (both treat a shaky observation as shaky).

**Verified — `test_soft_evidence.py` 6/6:** (1) full trust reproduces the EXACT Step-2 lift (clamp 1.0 →
kpop 0.4960); (2) uncertainty shrinks the lift (conf 0.6 → 0.3968 < 0.4960); (3) confident dislike still
drags a neighbour down; (4) a weak low-confidence dislike (clamp 0.47) leaves a strong culture prior (0.60)
above the 0.35 floor while a confident one (clamp 0.40) moves it more; (5) neutral (0.5) is a fixed point at
any confidence; (6) read-only/deterministic/graceful. Existing suites green: test_cross_namespace,
test_affinity, test_preference_model (confidence=1.0 → unchanged), test_culture_seed, test_culture_extraction,
+ graph_relationship pytest (47). `graph_relationship/` purity unchanged.

**Live-KG copy demo:** same 0.9 "like" on `football_player` bridged to `ck:korean:baseball` (prior 0.30) →
posterior 0.459 (conf 1.00), 0.448 (0.95), 0.357 (0.50), 0.300 (0.20) — lower confidence lifts it less; a
very-unsure signal barely moves it off the prior.

**Deferred (Step 4, untouched):** style hint.

---

## perf: thread the chat pipeline + async topic label + debounced saves  *(branch `feature/cultural-awareness`)*

**Problem:** the webcam loop felt unresponsive — the OpenCV window froze during every reply. Investigation:
vision (face/emotion/demographics) already runs in a daemon thread on CPU (forced off the RTX-5060/sm_120
GPU), but the **whole chat turn ran synchronously on the display thread** — 3 serialized Ollama round-trips
(RAG embed + qwen reply + a 2nd qwen call for the live-topic label) plus a full-graph `store.save()` — so the
display couldn't read frames until the LLM finished. Disabling emotion didn't help (it's off the chat path).

**Fixes (one at a time, agreed order):**
- **Option B — async live-topic label.** The 2nd LLM call (`_detect_topic`, a metadata label, NOT the reply)
  moved to a background thread. The turn is recorded immediately; the label is backfilled to the session DB
  (`SessionStore.set_turn_topics`, thread-safe) and handed to the main thread via a queue to update the
  conversation node. Zero reply-quality loss (still LLM-generated), just off the critical path.
- **#2 — threaded chat pipeline.** RAG + prompt build + reply now run on a dedicated `chat-worker` daemon
  thread; the display shows `…` and keeps rendering. The `InMemoryGraphStore` isn't thread-safe, so a single
  `RLock` serialises the worker's *fast* prompt-build read against every main-thread store mutation/save
  (per-tick, chat-apply, topic-drain, hotkeys, shutdown). **All store writes stay on the main thread**; the
  lock is never held across the slow LLM/RAG calls. Verified: 40 concurrent chats + a thread hammering
  store writes/saves → 0 errors, no corruption.
- **#3 — debounced persistence.** The frequent per-tick/chat mutations now just mark the graph dirty; the
  loop flushes `kg_state.json` at most once/second (or forced on shutdown), instead of rewriting the whole
  file every tick (which produced the "8 saves per turn" churn).

**Also (data tuning):** lowered the Korean demo culture priors ~0.20 (`culture_seed._KOREAN_DEMO`) into a
0.25–0.60 band, so a person's liked interest bridging to a culture topic (Step 2) can actually lift it above
the 0.35 suggestion floor instead of being capped by an already-high prior.

**Still TODO:** #4 — GPU for vision (needs a torch build with sm_120/Blackwell support; deferred because
vision is already threaded and would contend with Ollama on the GPU). Thread model now: main/display +
detection-worker + chat-worker + short-lived topic-detect threads.

---

## feat: cross-namespace culture↔person bridges — Approach 1, STEP 2  *(branch `feature/cultural-awareness`)*

**Problem:** culture priors live on `ck:<culture>:<slug>` CultureTopic nodes; person interests live on
`topic:<slug>` nodes. These were disconnected islands — only a label-join, no edge — so a person's OBSERVED
interest could never propagate to a culturally-adjacent culture topic and the BN degraded to a base-rate
lookup. Step 2 adds `related_topic` BRIDGE edges across the namespaces so the existing 2-round noisy-OR
carries person evidence into culture topics (observed `topic:jazz` lifts `ck:korean:kpop` above its prior).

**What was built (bridges only — propagation math untouched):**
- **topics.py** `link_related_cross()` — same `related_topic` edge/storage as `link_related_topic`, but each
  endpoint may be a `topic` OR a `culture_topic` node. A relatedness LINK only: never merges, never changes
  identity — a `ck:` node and a same-slug `topic:` node stay two DISTINCT nodes, now traversable.
- **kg_extraction.py** `link_cross_namespace_bridges()` — two idempotent, LLM-free passes: (1) exact-slug
  bridges (weight 1.0) for a person `topic:<slug>` whose slug equals a `ck:<culture>:<slug>`; (2) embedding
  bridges in the same `[0.60, 0.86)` related band, `same_category_only`, exact-slug pairs excluded. Reuses the
  existing `_embed`/`_pairs`/`_same_category` machinery pointed at the cross-namespace pair set.
- **preference_model.py** — the ONE traversal change: when gathering a person topic's `related_topic`
  neighbours, accept `culture_topic` neighbours too (they resolve into the same slug space). No change to
  rounds, damping (0.8), floor, or the signed clamp — only which neighbours the existing walk may visit.
- **Wiring:** the bridge pass runs wherever related-linking already runs — `--mode consolidate` and the
  auto-consolidate path.
- **Viz:** cross-namespace bridges render as `related_topic` with a dashed stroke + "culture bridge" tooltip.

**Finding (reported honestly):** the BN joins by SLUG, so an EXACT-slug bridge (`topic:hiking ~
ck:korean:hiking`, same slug) is a self-loop in slug space and does not by itself move a distinct node — it
is created/idempotent/merge-safe and makes the join an explicit traversable edge, but the measurable
cross-namespace LIFT comes from DISTINCT-slug (embedding) bridges. Carrying an exact-slug bridge two hops
(person→ck:hiking→ck:kpop) would need a hop-count change, which this step deliberately scoped out.

**Verified — `test_cross_namespace.py` (synthetic store, fake embed) 6/6:** (1) embedding bridge lifts kpop
0.30→0.496, delete→0.30 fallback; (2) exact-slug bridge created/idempotent/merge-safe/traversable; (3) merge
invariant — consolidation never merges ck↔topic, per-namespace counts unchanged, only +1 related edge; (4)
dislike crosses — disliked jazz pulls kpop 0.30→0.151 (Step-1 signed clamp composes with Step-2 bridge); (5)
read-only + deterministic + graceful degradation; (6) embedding-band linker creates the bridge idempotently.
Existing suites still green: test_affinity, test_preference_model, test_culture_seed, test_culture_extraction,
and the graph_relationship pytest suite (47). `graph_relationship/` purity unchanged.

**Deferred (Steps 3–4, untouched):** confidence-weighting the BN clamp; style hint.

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

## Unified OpenCV face detection (shared by face-reco + emotion)

**Goal:** one face-detection pass feeds both identity and emotion, using OpenCV.

- **Tried:** added a `detector` mode to `FaceIdentifier` (`face_id.py`) — `"opencv"`
  (new default) vs `"mtcnn"`. In opencv mode a single Haar `detectMultiScale` locates
  every face; each box is cropped (with a ~20% margin to mimic MTCNN's) and embedded by
  the *same* InceptionResnetV1 (facenet fixed-image-standardization) for identity, and the
  same box is handed to the emotion detector. MTCNN is not loaded in opencv mode.
- **Worked:** `identify_all` returns the shared Haar boxes → the detection worker already
  passes each box to emotion, so face-reco + emotion now share ONE detection. `--detector
  {opencv,mtcnn}` on the loop + enroll mode. 12/12 face/culture tests pass; embeddings stay
  L2-normalised (self-sim=1).
- **Didn't / caveat:** Haar boxes are unaligned vs MTCNN's landmark alignment, so recognition
  is a bit looser — **re-enroll people after switching modes** (a face enrolled under MTCNN
  matches poorly against an opencv-cropped probe). `--detector mtcnn` restores the old path.

## Sampled face identity (sample-and-hold) — stop 24/7 re-identification flicker

**Goal:** face recognition flickered (jay ↔ unknown frame-to-frame) and ran every
frame. Don't check identity 24/7 — check periodically and hold a confirmed result.

- **Tried:** duty cycle in `_DetectionWorker`. SAMPLE window (`--id-sample`, 1 s):
  identify every frame and vote on the primary (largest) face. Confirm the identity
  only if it was recognised in >= `--id-confirm` (0.6) of the sampled frames. Then
  HOLD (`--id-interval`, 3 s): detect boxes only (new `FaceIdentifier.detect_boxes`,
  no embedding) for emotion/display, and reuse the confirmed identity — never
  re-identify. Displayed label is always the held identity → zero flicker.
- **Worked:** helpers unit-tested (primary pick, relabel, boxes→raw, vote ratio
  confirm/reject); 12/12 face+culture tests still pass; opencv & mtcnn both get a
  box-only path. Big CPU saving too (embedding runs ~1 s in 4 instead of every frame).
- **Caveat / trade-off:** during a HOLD window the largest box is labelled with the
  held identity, so if a *different* person appears mid-hold they're mislabelled for
  up to `--id-interval` s until the next sample corrects it. Tune interval down if
  that matters. Multi-face identity is primary-focused (non-primary faces show as
  unknown during hold).

## Angle-robust, self-improving face recognition (multi-view gallery + adaptive capture)

**Problem:** recognition only worked at one head angle and flickered — the label dropped whenever
the person turned or looked away. Four separate causes, not one.

- **Tried / found (diagnosis first):**
  1. *One averaged prototype per person.* `faces.npz` held `jay = (1,512), counts=[50]` — 50 frames
     collapsed into ONE centroid. Averaging across poses matches no pose well.
  2. *Detection, not recognition.* The default `opencv` detector was the FRONTAL-only Haar cascade,
     so a turned head produced **no box at all**. `haarcascade_profileface.xml` measured **2.6 ms**
     (vs 12.7 ms frontal) — essentially free.
  3. *A real bug in the sample-and-hold code from the previous commit*: `_id_samples += 1` counted
     frames with NO face, inflating the vote denominator so looking away pushed the ratio under
     `id_confirm_ratio` and dropped a good identity.
  4. *Binary threshold* (0.75) with no hysteresis — an off-angle dip flipped the label instantly.

- **Worked:**
  - **Phase 0:** vote only counts frames containing a face; HOLD-phase boxes report `sim=-1` and
    render as `held` (they were fabricating `1.00`, which would have poisoned any calibration);
    `--id-debug` trace.
  - **Phase 1:** frontal + profile + mirrored-profile cascades, deduped by IoU **and**
    centre-containment (offset frontal/profile boxes on one head sit at IoU 0.25-0.35 and slip past
    pure IoU → phantom "unknown" face, which would also silently disable adaptive capture);
    source-aware crop margins; `_embed_opencv` now detects at the same 0.5 scale as `identify_all`.
  - **Phase 2:** `_protos`/`_meta` multi-view gallery (K=12 enrolled + 6 learned), merge-vs-insert on
    `tau_dup=0.92`, farthest-point eviction, `retain`/`acquire` (top-2) scoring, concat-not-average
    rename, npz **schema 2** with automatic schema-1 migration (jay survived losslessly).
  - **Phase 3:** acquire 0.75 / retain 0.62 / switch-margin 0.08 hysteresis in `_match`, applied to
    the largest face only; worker miss-grace of 2 windows, but instant drop on a rival or empty frame.
  - **Phase 4:** adaptive capture — learns a new view only from a confirmed, single-face, unanimous,
    quality-checked window, rate-limited 20 s/person and 20/session, in `[adapt_floor, tau_dup)` with
    `adapt_floor` defaulting to `threshold` (conservative). Worker queues, main thread applies →
    gallery keeps a single writer. `--reset-adaptive` undoes it; learned views can never displace
    enrolled ones.
  - **Phase 5:** guided five-pose enrollment (CLI + in-window E-key), which REJECTS frames that
    repeat a stored view — otherwise ignoring the prompts silently re-records the front.
  - `--faces-info` prints each person's views + closest-pair similarity ("only ONE view" diagnosis).
  - Tests: new `test_face_multiview.py` (16 cases incl. the centroid-vs-multi-view regression and 7
    poisoning cases the adaptive gate must refuse); `test_face_rename` updated for concat semantics.
    **68/68 repo tests green.** Integration-smoked the worker end-to-end with a stubbed detector:
    confirm → hold through an off-angle turn → learn the new view.

- **Didn't / watch:** thresholds are priors, NOT calibrated on real faces — run `--id-debug` through
  0/±30/±60/±90/chin-up/chin-down and tune `--retain-threshold` (10th pct of ±45 sims) and
  `--proto-dup` (40th pct of consecutive still-frame sims). With only ONE person enrolled nothing
  competes, so recall gains also raise false accepts — enrolling a decoy identity would turn the
  absolute threshold into a relative one. jay still has a single enrolled view until re-enrolled
  with the guided poses. Emotion now receives profile crops it was not validated on (emotion is off
  by default).

## Project 56 (non-verbal) module added, parked — 2026-11-02
- **Done:** merged `final_demo` (adds `FINAL_DEMO/`, no conflicts; branch kept). Copied Navel's Method B planner into `modules/project56/` (co_speech, action_selector, robot_profile + 9 test files); imports prefixed `modules.project56.`. 123 tests pass.
- **Not done on purpose:** no hookup to the persona UI / hub; no ChatBox action profile. See `modules/project56/README.md`.

## ChatBox robot profile for project56 — 2026-11-03
- **Done:** `modules/project56/robot_profile/chatbox_profile.py` — 6 planner actions (DEFAULT, GREET, POINT, UNCERTAIN, SAD, HAPPY) + `CHATBOX_NATIVE_TAG` (planner action → Jetson `EMOTION_MAP` tag); registered in the provider under `chatbox_jetson_001`. 4 new tests, incl. a check against FINAL_DEMO's `robot.py` EMOTION_MAP (every native tag exists and gives a distinct motion).
- **Not measured (knobs):** gesture length 4.5 s (firmware: 5 steps × 900 ms) and speech timing (Navel's numbers). The duration filter drops unprofiled actions, so every action needs a duration.
- **Not done:** AGREE/DENY need `NOD`/`SHAKE` in the client's EMOTION_MAP; executor + hub hookup still to build.

## Viz: Apply updates the graph; persona + chat get their own column — 2026-11-03
- **Done:** `POST /persona` now also attaches the role + persona in the KG file (`GraphState._attach_in_graph`, `roles.apply_assignment`), so the graph shows it with no webcam loop. A running loop applies the same assignment itself. New test `test_apply_attaches_the_role_in_the_graph_without_a_loop`.
- **UI:** removed the node-type/legend panel; Role & persona and Chat with the robot now live in a fixed right column and the graph canvas ends at its left edge; Forces moved to the bottom-left. Checked in a browser (1512×790).
- **Found while testing:** `persona_assignment.json` is tracked now, so every Apply shows it as modified.

## Viz: Apply on top, spread-out first layout — 2026-11-03
- **Done:** Apply button, status and "now:" line moved to a sticky header at the top of the persona column. First load seeds nodes on a sunflower spiral, pre-runs the forces to rest, then zooms to fit; Reset (forces) now re-runs that layout too. Checked in a browser: spread out on load and after Reset.
- **Found:** the Jetson's `edge_tts_output.py` stores `voice_config.rate` but never uses it (gTTS path plays the mp3 through ffmpeg unchanged), and `pitch` isn't read at all — so persona voice differences need a client change.

## Persona voices + resizable persona/chat column — 2026-11-03
- **Done (server):** `modules/persona_voice.py` maps role + OCEAN to speed / pitch / tone / volume offsets; `RobotHub.push_persona` sends them as `voice_config` (merged with the base voice keys `language/gender/provider`, because the client replaces its whole `voice_config`); Apply message shows them. 4 tests + hub test.
- **Done (Jetson, `FINAL_DEMO/CHATBOX_CLIENT/OutputModules/edge_tts_output.py`):** `update_voice_config` now reads `pitch`/`tone`/`volume`, and the gTTS ffmpeg step uses `_voice_filter()` (atempo × rate, asetrate pitch shift at constant speed, treble/bass, volume). Neutral persona = the old `atempo=1.25` exactly. Checked against real ffmpeg on a 200 Hz tone: measured pitch and length matched the targets for three personas.
- **Not covered:** the offline Piper fallback ignores pitch/tone/volume; not yet heard on the real robot. The numbers in `ROLE_VOICE` / `voice_for` are a first guess to tune by ear.
- **UI:** drag the column's left edge to widen it (graph follows) and the bar between Role & persona and Chat to share height; sizes are remembered in the browser.

## Action-tag generator: Method B on the ChatBox (branch `action_tag_generator`) — 2026-11-03
- **Done:** `modules/action_tag_generator.py` — reply → `SpeechSegmenter` → `CoSpeechPlanOrchestrator`/`ActionSelectorModule` ("choice" mode, ChatBox profile) → planner action → Jetson native tag via `CHATBOX_NATIVE_TAG`. `ChatLLM` adapts the viz chat backends (AI server / Ollama) to project56's `generate()`. CLI: `python3 -m modules.action_tag_generator "text" --ai-server HOST`.
- **Sending:** the Jetson client fires only the FIRST `[TAG]` of a `demo_step`, so each sentence goes as its own step. `RobotHub` now handles `ack` and `say_steps(texts)` sends the next step after the client acks the previous one (= finished speaking). `viz.server --robot-hub --action-tags` uses it for chat replies (response carries `plan`; planner error falls back to the old single-tag `say`).
- **Tested:** 4 new tests (scripted LLM, fake socket robot that acks; full suite 225 passed). Live against the AI server at 10.42.0.118 with a fake robot over the real hub: "Great job finishing that!" → HANDS_CLAP, "I'm sorry to hear…" → SAD, "I don't know where your library is" → CONFUSED, other sentences DEFAULT; steps arrived in order, one per ack. NOT yet run on the real Jetson/ESP32.
- **Found:** the duration filter drops every gesture at overrun 0 (a 4.5 s gesture vs 1–3 s sentences), and at 3 s still drops a one-word "Hello!" (no WAVE for the greeting). `OVERRUN_SECONDS` now = `EXPRESSION_SECONDS`, i.e. no duration filtering, since the next step's tag replaces a running gesture. Revisit once real gesture/speech times are measured. The repetition filter correctly stops the same gesture twice in a row.
- **Latency:** one LLM call per sentence, done before the first step is sent (3–6 s extra on a 2–3 sentence reply). Lookahead (`OneSegmentLookaheadCoordinator`) would hide it; not wired.
- **Test quirk:** closing a socket.io client right after its last ack could hang ~25 s in teardown; the fixture sleeps 0.5 s first.
## Viz: profile cards open into a pentagon — 2026-11-04
- **Done:** removed the emoji icon and the mini bars from each profile card. Clicking a card opens it (full width) with an OCEAN pentagon (centre −1, dashed ring 0, outer +1), the role select + brief + goal/talks/avoids/knows, and the OCEAN sliders; the pentagon follows the sliders live. Click the card again to close. A "Custom" card (open by default) holds the same editor for a hand-set role/traits. The editor is one DOM block that moves into whichever card is open, so the existing form code is unchanged. Checked in a browser: opens, slider moves the pentagon, collapse works, no icons.
- **Changed behaviour:** editing a slider/role no longer deselects the open card (it used to clear the card highlight).

## Viz: apply button moves into the open profile — 2026-11-04
- **Done:** the apply button and its status line moved out of the sticky header into the open profile card (bottom of its editor, `position: sticky; bottom: 0`), so it stays in view while that card is on screen and rests at the card's end when scrolled past. The header keeps the title and the "now:" line. Checked in a browser. With every card closed there is no apply button (open Custom or a profile).
- **Reordered (same day):** in the open profile the Role select, its brief and goal/talks/avoids/knows now come first, then the pentagon, the OCEAN sliders and apply.

## Only send actions the ChatBox can perform — 2026-11-04
- **Problem:** the persona chat prompt said "begin with an emotion tag, e.g. [HAPPY], [CURIOUS]", so the model wrote `[EXCITED]`, `[CURIOUS]`, `[WELCOME]`…; the Jetson's `EMOTION_MAP` doesn't know them and falls back to the neutral pose (and a few alias to a motion that isn't the one meant).
- **Done:** `modules/chatbox_actions.py` — the list of tags the ChatBox can perform (DEFAULT, WAVE, GREETING, POINT, CONFUSED, SHRUG, SAD, HANDS_CLAP, EARS_WIGGLE) with when to use each; the chat prompt now lists exactly these; `normalize_tag` maps any other word onto one (EXCITED/HAPPY→HANDS_CLAP, CURIOUS→CONFUSED, WELCOME→GREETING, WARM→GREETING…, unknown→DEFAULT). The chat response keeps the model's word as `raw_tag`.
- **Left out on purpose:** ANGRY, SLEEP, POSE (client can do them; a companion shouldn't pick them). Firmware moves with no client tag (head_nod, head_shake, ears_perk, dance_*) need an `EMOTION_MAP` entry on the Jetson first.
- **Tests:** a new test file checks every listed tag exists in the Jetson's `EMOTION_MAP` and moves differently, every alias lands on a listed tag, Method B's native tags are all listed. Drift test now skips the tag line. 230 passed.
- **Live (AI server 10.42.0.118, 6 messages):** the model followed the list every time (HANDS_CLAP for the exam, SAD, WAVE for goodbye, CONFUSED for an unknown fact); the alias map was not needed.

## Automatic port selection (viz + robot hub)
- Tried: second `viz.server` with the same command -> "Address already in use". Now `--port` (8765) and `--robot-hub` (5000) fall back to the next free port (up to 20) and print "port N is taken -> using M".
- Worked: both ports occupied -> viz on 18766, hub on 15001. Probe-by-bind (`robot_hub.first_free`) because werkzeug `make_server` calls sys.exit on EADDRINUSE.
- Caveat: a second server still shares `kg_culture.json` / `persona_assignment.json` unless given its own `--kg-path`; the robot's `server_url` must use the port the hub printed.

## Robot connection status in the viz
- `GET /robots` -> {hub, port, robots:[{robot_name, client_id}]} from the hub's connected clients; HUD row "robot" polls it every 2 s: green dot + names = connected, red "waiting · :PORT" = hub up but no robot, grey "hub off" = no `--robot-hub`, red "viz offline" = server unreachable.
- Worked: browser check with a fake Socket.IO robot (green "ChatBox" while connected, red "waiting" after it left); test_robots_endpoint_shows_connect_and_disconnect.
- Caveat: "connected" means the client finished `client_init` on the hub; a robot whose connection dies silently shows until Socket.IO's ping timeout (~25 s) fires.

## Persona announcement says only the role
- Tried/found: viz Apply sent `persona_name = "<role>, <OCEAN words>"` and the Jetson (robot.py `on_persona_update`) speaks "Persona updated to {persona_name}." -> it read out the trait description too.
- Worked: `persona_name` is now the role label only ("Persona updated to Receptionist."); OCEAN still travels in `personality`, the long description in `robot_role`. Client unchanged. test_persona_chat asserts no comma in persona_name.
- Not verified on the Jetson. ESP32-stuck: diagnosed from code only (see chat); no hardware access.

## Return to DEFAULT after the reply
- Tried/found: the Jetson fires one [TAG] per demo_step, TTS strips `[..]` and speaks nothing for a tag-only text, then calls the ack callback at once (edge_tts_output.speak_with_callback) -> a `[DEFAULT]` step is safe on the unchanged client.
- Worked: `RobotHub.say_steps` ends with a `[DEFAULT]` step (no ack needed) once the last real step is acked. Every chat reply (Method B, planner-fallback, single-tag) now goes through `say_steps` via `speak()` in viz/server.py, so all three end in the default pose. Tests: no [DEFAULT] while the last step is still being spoken; arrives right after its ack.
- Caveat: a robot that never acks delays the [DEFAULT] by the 30 s step timeout. The firmware already runs "default" after each gesture, so this only matters when speech outlasts the gesture or the face holds a pose.

## DEFAULT pose when a robot connects
- `client_init` success -> hub also emits a tag-only `demo_step` `[DEFAULT]` (step_id "connect", no ack) to that robot, before the persona replay, so it starts and reconnects in the default pose.
- Test: test_robot_gets_the_default_pose_when_it_connects; `_FakeRobot` keeps the connect step in `connect_steps` so reply-step indices in other tests are unchanged.
- Caveat: if the Jetson has not yet reached the ESP32 (TCP/serial) when it joins the hub, the client logs "[Arduino] Cannot send 'DEFAULT' — not connected" and the pose is not set; the ESP32 powers up in default anyway.

## Robot action buttons (viz, left side)
- `POST /action {tag}` -> `[TAG]` as a tag-only demo_step (no ack) to the connected robot(s); 400 for a tag outside `chatbox_actions.ACTIONS`, 409 when no robot is connected. `GET /robots` also returns `actions`, so the UI builds one button per ChatBox action (tooltip = its usage text).
- UI: "Robot actions" panel under the HUD; buttons are disabled until the robot row is green (re-checked every 2 s) and show "sent WAVE ✓".
- Worked: browser check with a fake robot (disabled -> enabled on connect, WAVE received as "[WAVE]"); test_action_button_plays_a_gesture_only_when_a_robot_is_connected.

## Robot microphone through the viz hub
- Found: the Jetson sends mic audio as a Socket.IO `speech` event {audio: base64 wav}; the hub ignored it (only the central server transcribed), so "Voice Detected / Processing speech" led nowhere.
- Worked: hub `on_speech` callback (own thread, so acks aren't blocked) -> viz `on_speech`: `SpeechProcessor` (faster-whisper base, CPU int8, loads in a background thread; `--whisper-model`, `--no-voice`) -> the same `converse()` as typed /chat (reply, Method B tags, robot speaks, [DEFAULT] at the end). `GET /voice` returns the log; the chat panel polls it and shows a 🎤 bubble then the reply, and adds the turn to the conversation history. Server-side voice history restarts when a new persona is applied.
- Verified: unit test with a stub STT; live: espeak-ng "Hello robot, how are you today?" -> transcribed exactly, reply from the AI server, robot received tagged steps (model load 1.1 s, 5 s clip 0.5 s on this machine).
- Not verified on the Jetson's real mic audio (44.1 kHz->16 kHz resample is done client-side). A robot hearing its own voice (echo) is not handled here. `speech_response` is not sent back, so the client's pepeha trigger is not fired by voice.

## Voice latency: GPU whisper, gestures planned while speaking, silence knob
- Measured here (RTX 5060): whisper base 0.51 s CPU -> 0.17 s CUDA, small 1.61 s -> 0.35 s (same text). SpeechProcessor hard-coded CPU int8; it now honours an explicit `whisper_device="cuda"` (central server's 'auto' stays CPU; a failed GPU load falls back to CPU). viz flag `--whisper-device cuda|cpu` (default cuda).
- Measured live: persona reply 0.8-0.95 s, Method B gestures 0.7-1.4 s per reply (one LLM call per sentence, sequential because the repetition filter needs the previous tag) — all of it was before the robot heard anything. Now `plan_orchestrator.build_selected_plan(on_segment=)` -> `ActionTagGenerator.generate(on_step=)` -> `say_steps(generator)`: each sentence is sent as soon as ITS tag is chosen. Chat meta shows "first words +Xs" (and "heard Xs" for voice); console prints `[voice] stt · reply · tags`.
- Mic upload: base64 WAV over Socket.IO takes ~5 ms locally on websocket and polling alike (117 KB clip); Jetson->PC ping 32-130 ms (avg 75) over WiFi, so the network is not the long part. The long part is the client's end-of-speech wait: `SILENCE_LIMIT` = 1.5 s of silence before the audio is even sent. `CHATBOX_SILENCE_SECONDS` env var (default 1.5, unchanged) on the Jetson shortens it; 0.8 is snappier but can cut off a pause. Needs the client file copied to the Jetson.
- The websocket 500 / "write() before start_response" in the hub log: reproduced locally on a normal websocket connect+close; clients still connect and work (cosmetic werkzeug dev-server noise).
- Not measured: Jetson-side TTS start-up (Piper/edge-tts) and real mic upload time on the Jetson.

## Voices: one base voice for companion/storyteller/tutor, a man and a young boy for the other roles
- `persona_voice.py`: companion, storyteller, tutor stay ONE voice (small role + OCEAN nudges, LIMITS unchanged). `CHARACTER_VOICE`: receptionist = adult man (pitch -5 st, rate -3%, tone -3 dB), reading buddy = young boy (pitch +5 st, rate +6%, tone +2 dB); both also send `gender: "male"`. Characters use wider limits (pitch ±7 st incl. OCEAN) so a lively boy cannot go past what ffmpeg handles.
- How: the Jetson's base voice is gTTS (one female voice); person = ffmpeg pitch shift (asetrate, speed compensated) + tone. The client's `_gender` is stored but not used when speaking, so a pitch shift is all that makes the man/boy. It is a shifted voice, not a separate speaker; Piper models (e.g. en_US-ryan-low) would give real different voices.
- Checked: voice_for output pushed through the client's exact ffmpeg chain on gTTS audio -> valid filters for all five roles; samples in the session scratchpad (not by ear: pitch/tone numbers are a first guess to tune). test_persona_voice covers shared base vs characters and the limits.

## One server, a tab per robot (ChatBox | Navel)
- Asked for: same port, a tab per robot, independent connection/interaction, a Navel graph with profile assignment like ChatBox, and nothing of the ChatBox's Jay graph in Navel.
- Server: `make_handler(..., others={"navel": (GraphState, PersonaChat)})`; every GET/POST takes `robot` (unknown -> 400). `--navel-kg-path` (default kg_navel.json, created with just the robot + role library by `seed_robot_graph`), its own `persona_assignment_navel.json`, `--no-navel`. Hub routes by the client's `robot_name` (lower-cased = robot id): `say/say_steps/push_persona(..., robot=)`, `robots(robot)`, per-robot persona replay on connect; mic audio goes to the robot that sent it, with its own spoken history. `/robots` returns `workspaces`.
- Only the ChatBox gets Method B gestures, action buttons and the `[DEFAULT]` pose (its tag list / EMOTION_MAP); Navel is sent plain speech (its client's tags are unknown here; DEFAULT_POSE_ROBOTS in robot_hub.py).
- UI: tab bar (green dot = that robot connected), per-robot chat logs/history/epoch, graph + persona panel re-pointed on switch, action panel hidden on Navel, last tab remembered. Replies and voice for a robot land in its own tab even when you are looking at the other one.
- Verified in the browser with two fake robots: ChatBox 82 nodes / Navel 37 (no person nodes); applying a persona and chatting on Navel reached only Navel (ChatBox received just its connect [DEFAULT]); test_navel_tab_is_independent_of_the_chatbox.
- Not verified: a real Navel client (its robot_name must be "Navel"), its tag handling (plain text sent), Navel TTS voice (persona voice offsets are the ChatBox client's).

## Navel graph content + tab layout fix
- Pulled `~/Desktop/2026-Jay/navel/robot-robot-interaction` (feat/navel-demo-client, fast-forward, 30 commits, tree was clean -> dad21e4) and read navel-client/client_config.json + client.py + lab-demo-server (README, data/snapshot, demo/navel_solo_script.json).
- `modules/navel_profile.py` + `specs/navel_spec.json`: Navel robot node, an abilities capability node, Navel's 11 own gesture tags as one node each ([GREET] ... [SAD], from client_config tag_descriptions), 5 connection-fact nodes, and its 7 declared topics (navel_001 snapshot). `seed_robot_graph` seeds a new file and tops up an existing kg_navel.json that lacks the profile; running twice changes nothing. Embodiment stays the CAT placeholder (the schema enum has only CAT/ELEPHANT).
- FINDING: the Navel client (v6) does not dial out. `client.py`: it runs a WebSocket SERVER on ws_port 8765, POSTs {server_url}/robots/register, and the server dials ws://<navel-ip>:8765 with JSON `{"event": ...}` messages (persona_update, demo_step, chat_response, ...). The viz hub is a Socket.IO server that robots connect to, so a real Navel client will NOT show up in it (green dot, persona push, speech) until the hub gets a dial-out connector. The ChatBox Jetson client does dial out, so it is unaffected.
- Tangled graph on tab switch: both graphs share node ids (roles, role topics), so the old nodes' positions leaked into the new graph. Switching now parks the current layout per robot, clears the nodes, and restores the other graph's own layout (or settles fresh). Browser check, 4 switches: 0 overlapping node pairs, same bounding box each time.

## One graph for both robots, colours, no gesture nodes
- Navel's 11 gesture tags are no longer nodes (like the ChatBox: one capability node; it now just mentions the gestures). The list stays in `navel_profile.ACTIONS`. Kept the 5 connection-fact nodes and the 7 declared topics.
- `GET /graph.json` is ONE graph: each robot's file is still read on its own (Navel needs nothing from the Jay graph) and `merged_graph()` joins them; the role library has the same ids in both files so a profile node is shared and both robots' `has_role` edges attach to it. `/delete` removes from every file. The `robot` query param still picks persona/chat/mic/status, not the graph.
- UI: tabs now FOCUS: the selected robot's subgraph (walk out from it, never through another robot) stays lit, the rest fades (class `offfocus`), and the view glides onto it. Colours: ChatBox orange `#ff8c42`, Navel blue `#4fc3f7`, Jay white; a robot's node is filled in its colour, what it owns (id prefix `chatbox:`/`navel:`, interaction/conversation of a person) gets a ring, edges follow their owner; roles/topics stay neutral. Tabs carry the colour swatch (+ a connection dot); HUD has a legend. The per-tab layout cache from the previous change is gone (not needed with one graph).
- Browser check (copy of the real ChatBox KG + a fresh Navel file): 96 nodes; ChatBox tab lights 51 / dims 45, Navel tab lights its subgraph and dims 84; computed colours: chatbox rgb(255,140,66), navel rgb(79,195,247), jay white, navel:capability blue ring, role neutral.

## Navel graph: robot + topics only; attached profiles in the robot's lighter colour
- `navel_profile.py` now seeds only `navel --about--> topic` for its 7 declared topics (navel_spec.json removed). No capability / gesture / connection nodes; `ACTIONS` and `CONNECTION` stay as plain data. `seed_robot_graph` also cleans an existing kg_navel.json: any `navel:*` capability node from the previous version is deleted on start (verified on a file with 3 such nodes: gone, 7 topics kept, persona node kept).
- Colour: a role a robot has (`has_role`) and what hangs off it (role topics) is drawn in that robot's colour, lighter (`brighter(0.45)`): nodes filled, edges tinted; a profile node both robots have stays neutral. Browser check: tutor on Navel -> light blue with 8 children, storyteller on ChatBox -> light orange with 7; an unattached role stays grey.

## Focus fades only the other robot
- Correction from Jay: "highlight the robot" must not fade the profile nodes. `applyRobotFocus` now fades ONLY the other robot(s) and what belongs to them (walk out from it, not entering the selected robot or any Role / RoleTopic / Persona node); anything the selected robot also reaches stays lit. Profile nodes (attached to either robot or not) are never faded. The view still glides onto the selected robot's subgraph.
- Browser check: ChatBox tab fades Navel + its 7 topics (8 nodes); Navel tab fades ChatBox, Jay, his interests/topics/interaction/conversation/capability (40); 0 profile nodes faded in both.

## Navel connection (dial-out), English mic, fixed ports
- Tried: Navel on the TP-Link router (192.168.0.10), PC joined it (192.168.0.7). Navel's client runs a WebSocket SERVER on 8765 and never dials out, so `modules/navel_link.py` (`NavelLink`, websocket-client, reconnect every 3 s) dials `ws://<navel>:8765`; `RobotHub.add_link()` makes it act like a connected client (robots(), say, say_steps with Navel's `{"type":"ack"}`, push_persona, speech -> on_speech). `--navel-host IP [--navel-port]`; `/robots` also returns `links` so the Navel tab shows "dialling ws://... · Connection refused" until it answers.
- Persona to Navel: only `persona_name` + `personality`. Navel writes persona_update into its own client_config.json (robot_role, allowed_tags, voice_config), and our voice offsets are the ChatBox client's units.
- Worked live (real Navel): link up, test line spoken and acked (9.9 s, mostly the planner timeout), UI Apply (Roz -> receptionist) and chat from the Navel tab sent to Navel, Navel's mic transcribed into the tab ("but can you back off? Back off bro.", stt 0.24 s on CUDA).
- Found: whisper auto-detect turned room noise into Indonesian -> `--whisper-language` (default en, `auto` to guess).
- Open: Navel's own config points the Method B planner / server at 192.168.0.102, which is not this PC (192.168.0.7) -> "No route to host", one gesture per reply and ~3 s wait per line. Fix = give this PC .102 or edit Navel's planner url, and run navel-server/tools/method_b_plan_service.py here.
- Ports are FIXED (viz 8765, hub 5000): `require_free()` stops with a clear message if one is taken (checked first, before dialling robots / loading models). The earlier auto-port fallback let a robot talk to a stale copy.
- Committed together with the data files kg_culture.json, persona_assignment.json, kg_navel.json, persona_assignment_navel.json (Jay: "commit everything including the navel").

## Robots page (add robots, monitor, assign) + Navel TTS voice spec
- Navel TTS (navel-client/OutputModules/tts_output.py + Navel SDK "TTS Tags"): its client prefixes `<vol,60><rpit,100><rspd,100><lang,en1>` on every utterance and has no update_voice_config, so persona voice_config was ignored. Tags later in the text win, so for dialled robots `RobotHub._spoken()` puts `navel_tts_tags(voice)` in front of every line: pitch semitones -> `<rpit>` % (2^(st/12)), rate % -> `<rspd>`, volume dB -> `<vol>` (base 60, 20..400); tone has no Navel equivalent. e.g. receptionist (man) -> `<vol,67><rpit,75><rspd,97>`.
- `persona_voice.voice_for(..., adjust=)`: a per-robot tweak (rate %, pitch st, volume dB) added before clamping.
- `modules/robot_registry.py` + robots.json (`--robots-file`): id = name lower-cased without spaces (hub matches robot_name the same way, spaces ignored), kind connects-in | dial-out, host/port, tts navel|none, colour (palette), voice_adjust. ChatBox is built in, Navel is a default, `--navel-host` sets Navel's host.
- Server: spaces are dynamic. `GET /registry`, `POST /registry/add|remove|voice|say`; adding creates the robot's own graph (kg_<id>.json = robot + role library), persona file, chat and, for dial-out, a NavelLink. Voices are restored from the saved assignments on start. `/robots` returns colours and names; the graph page's colours, legend and tabs now follow the registry, `?robot=<id>` opens a tab, and a "Robots" link opens the page.
- `robots.html`: add form, a card per robot (live connection detail, role + personality bars, role select + Assign, voice sliders + Apply voice + Say something, Open in graph, Remove).
- Verified in the browser: added Stuart (dial-out to a local stand-in) from the form -> green, role assigned, pitch -3 -> voice "speed -10%, pitch -2.5st"; tabs ChatBox/Navel/Stuart in their colours; test_robot_registry.py (registry persistence/validation, add/assign/voice/say/remove over HTTP, a connects-in robot called "Stu Art").
- Open: Navel was unreachable during this check (no route to 192.168.0.10 from 192.168.0.7: powered off or off the router); the Navel voice tags were verified against the fake Navel, not the real one. Stuart's real connection type / TTS are unknown: pick them in the form.
