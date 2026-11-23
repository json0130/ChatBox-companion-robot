# V4 leftovers

What was in `CHATBOX-DEMO_V4/` that git did not track or that was not part of the removed folder. Kept here, not part of any demo.

- `v4-uncommitted-changes.patch` — V4's uncommitted edits at removal time: gTTS/piper changes in `edge_tts_output.py` and the piper voice in `client_config.json` (ryan-low -> amy-medium). Apply with `git apply` against the commit before the removal (`ac22307`) if wanted.
- `final.ino` — a V4 main-sketch variant (not the FINAL_DEMO sketch).
- `rag_indexes/`, `faces.npz` — V4's per-user memory indexes (participant text) and enrolled faces. Ignored by `.gitignore`, so they stay on this machine only.
