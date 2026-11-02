# CHATBOX_SERVER — Progress Log

Newest entry at the top. Each entry records what was **tried**, what **worked**,
and what **didn't work / was fixed**, for the referenced commit.

---

## chore: repo tidy — untrack utils/ binaries, drop loose root files + dead submodule
**Date:** 2026-09-24
**Area:** repo structure (cleanup)

- **Tried / worked:** untracked `yolov8m.pt`, `yolov8m-face.pt` (52 MB each) and the Jetson torchvision
  wheel (14 MB) — `*.pt` / `*.whl` now gitignored, `utils/README.md` says where each comes from (same change
  as 39fc418 on `feature/pad-affect-core`). Deleted `faiss_index.bin` + `wordcloud_latest.png` (2025 leftovers
  from #3, nothing references them) and `.gitmodules` (pointed at a `ChatBox` submodule with no gitlink).
- **Didn't / open:** `.git` stays ~162 MB (blobs still in history) — history rewrite deliberately not done.

---

## 381a4cf — merge: face-tracking into main
**Date:** 2026-09-24
**Area:** repo structure (branch cleanup)

### What was tried
Restructure branches so `main` = V4 demo + `chatboxv1-3/` + face tracking, with
`sign-language-demo`, `KG-knowledge-extraction` and `feature/pad-affect-core` hanging off it.
First step: merge the `face-tracking` branch (16 commits) into `main`.

- **Worked:** clean merge, no conflicts. Adds `AFFECT_LAB/`, `SERVO_STYLE/`, V4 face-tracking
  OutputModules + ESP32 pan-servo changes; `robot.py` auto-merged and compiles.
- **Didn't work / open:** nothing tested on hardware yet; the other branches still need `main`
  merged into them.

---

## f02ca18 — fix(server): stop LLM emotion tags collapsing to one tag
**Date:** 2026-08-18
**Area:** `Modules/llm_processor.py` (emotion tag selection for the Ollama/`qwen:4b` reply path)

### Problem observed
Live logs showed the server emitting `[DEFAULT]` on **every** reply, across both English
and Korean STT input, despite the client registering
`['[DEFAULT]', '[WAVE]', '[HAPPY]', '[SAD]', '[CONFUSED]', '[GREETING]']`.
Replies whose content was plainly sad, confused, or a self-introduction were all tagged
`[DEFAULT]`.

### What was tried

**Attempt 1 — fix the prompt's few-shot examples.**
Root cause found: `_get_allowed_tags_info()` returned `config_tags[0]` as `example_tag`, and
`[DEFAULT]` is first in every client config. All three "PERFECT RESPONSE" examples therefore
rendered as `[DEFAULT] ...`. Two of them demonstrated `[DEFAULT]` on exactly the sentences
`[CONFUSED]` and `[SAD]` exist for. Separately, the hard-coded negative example read
`[HAPPY] I feel great! (Error: Tag is not in the allowed list)` — but `[HAPPY]` *was* in the
allowed list, explicitly telling the model to avoid one of its own valid tags.

Changes: added a per-tag "when to use this" menu (`TAG_MEANINGS` / `_build_tag_menu`),
varied examples generated from the client's actual tags (`_build_examples`), and a
dynamically-chosen genuinely-forbidden tag for the negative example
(`_pick_forbidden_example`). Also stopped storing the tag in conversation history, since
7 turns of `[DEFAULT] ...` were priming the model to repeat it.

- **Worked:** the contradiction is gone; examples now demonstrate 4–5 distinct tags.
- **Didn't work:** output stayed `[DEFAULT]`-only in the next log sample. Likely the server
  had not been restarted (module is loaded at process start), but not confirmed.

**Attempt 2 — remove `[DEFAULT]` from the model's choices entirely.**
`_expressive_tags()` stripped `[DEFAULT]` from the prompt, and a model-emitted `[DEFAULT]`
was treated as "no choice made" and re-inferred from the reply text via keyword cues.

- **Worked:** replayed against real logged replies, produced `[GREETING]`, `[SAD]`,
  `[CONFUSED]`, `[HAPPY]`, `[WAVE]` correctly.
- **Reverted:** by request — `[HAPPY]` removed instead, `[DEFAULT]` restored as a valid
  selectable tag and as the final inference fallback. `[HAPPY]` is now excluded server-side
  via `EXCLUDED_TAGS` rather than editing the Jetson's `client_config.json`.

**Attempt 3 — fix the resulting `[WAVE]` lock-in.**
After restart, every reply came back `[WAVE]`. Two independent causes:
1. The `WAVE` keyword cue was `\b(hi|hey|hello|bye|...)\b` with **no anchor**, matching
   anywhere in the reply — and nearly every ChatBox reply contains "Hi" or "hello".
   `GREETING` was checked first but its pattern covered `"hello there"` and not `"Hi there"`,
   so replies fell through to `WAVE`.
2. Nothing prevented the robot greeting on *every* turn — `[WAVE]`/`[GREETING]` were
   permanently on the menu, so turn 12 could open with "Hi again!".

Changes: greeting cues anchored to the start of the reply (`^\W*`); `[WAVE]`/`[GREETING]`
gated behind `allow_greeting`, true only on the opening turn or when the child's own message
contains a greeting (EN + KO: `안녕`, `반가`, `잘가`), evaluated with the RAG context block
stripped so retrieved past "hello"s don't re-open the gate every turn. The gate applies to
the prompt as well as the sanitizer, so the model stops *writing* "Hi again!" rather than
just having its tag overridden.

- **Worked:** replaying the logged conversation gives `[WAVE]` → `[DEFAULT]` → `[WAVE]`
  (turn 3 correct: the child said "good afternoon"), with follow-ups reaching `[SAD]` and
  `[CONFUSED]`.

### Diagnostics added
The debug line printed the *sanitized* output, so a model-chosen tag and a server-inferred
tag were indistinguishable — this is why the first two attempts were hard to evaluate. It now
prints `tag=... from=model|server (model said X) greeting_allowed=...`.

### Open / not addressed
- **`qwen:4b` is small for reliable tag adherence.** The prompt no longer fights the model,
  but if `from=server` appears on most lines after restart, model size is the remaining
  constraint, not the prompt.
- **RAG context is prepended into the *user* message** (`chatbox_server.py:281`), so the model
  sees a wall of retrieved context before the actual sentence, biasing toward flat summarising
  replies. Moving it into the system prompt would likely sharpen the emotional read. Untested.
- **Keyword inference is a backstop, not the mechanism.** If it fires on most turns the tag
  distribution will look artificially narrow.
- The older duplicate at `chatboxv1-3/version3(chatbox_ai_agent)/v4.2.0/Modules/llm_processor.py`
  still has the original prompt bugs and no sanitizer at all. Left untouched.
