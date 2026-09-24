# CHATBOX_SERVER — Progress Log

Newest entry at the top. Each entry records what was **tried**, what **worked**,
and what **didn't work / was fixed**, for the referenced commit.

---

## merge: main into sign-language-demo
**Date:** 2026-09-24
**Area:** `CHATBOX_CLIENT/robot.py`, `interactive_test.py`, `CHATBOX_ARDUINO/ChatBoxPlus_ESP32.ino`, `Modules/llm_processor.py`

### What was tried
Branch restructure: `sign-language-demo` now hangs off `main` (V4 demo + face tracking) and keeps the
Māori pepeha **on by default**. Merged `main` in; 4 real conflicts (61 others were stale
`__pycache__`/RAG-index files `main` had untracked — dropped).

- **robot.py:** took `main`'s version (its pepeha is the refined port from 1e5eedd: "no" checked before
  "yes", chat suppressed during consent, face tracking during TTS) and set `pepeha=True` by default
  (`--no-pepeha` turns it off; old `--pepeha` still accepted). Carried over this branch's startup TTS
  greeting on `client_init_response`, "Arduino connected." TTS, USB speaker auto-detect (`aplay -l`),
  and the `run()` that waits for server + ESP32 before printing startup info.
- **ESP32 sketch:** kept this branch's WiFi setup (no `setCpuFrequencyMhz(80)`, `servoInit()` after WiFi,
  per 8325e41) and WiFi watchdog; added `panServoInit()` + `updatePanTracking()` from face tracking.
- **interactive_test.py:** kept this branch's serial+TCP version (already Python <3.10 safe).
- **llm_processor.py:** kept "CHAT BOX" wording (a7a386c); took `main`'s emotion-tag fix (f02ca18) and
  re-added the one-tag-only negative example.
- **Worked:** all touched Python compiles; `--no-pepeha` / default CLI parsing checked.
- **Not verified:** not run on the robot (Jetson + ESP32) yet — sketch not compiled.

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
