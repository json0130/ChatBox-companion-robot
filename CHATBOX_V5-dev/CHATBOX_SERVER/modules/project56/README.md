# project56 — non-verbal interaction (Method B), parked

Navel's Project 56 co-speech gesture planner, copied in so the persona/role UI can
drive gestures on the ChatBox later. **Not wired to anything yet** — nothing in the
viz, hub or webcam loop imports it.

Source: `UoA-CARES-Student/robot-robot-interaction`, branch `feat/navel-demo-client`,
commit `49cdde5`, folder `navel-server/modules/`. Code is verbatim except the import
prefix: `modules.X` → `modules.project56.X` (so it can't clash with V5's `modules/`).

| Folder | What it does |
|---|---|
| `co_speech/` | segment the reply → time each segment → plan; `navel_executor.py` sends segments to a robot over WebSocket |
| `action_selector/` | duration + repetition filters, language scoring: which gesture fits a segment |
| `robot_profile/` | per-robot action catalogue (today: Navel's) — a ChatBox profile is the missing piece |
| `base.py`, `llm/llm_provider.py` | the two helpers the above import |

Left in the Navel repo: `tools/method_b_plan_service.py` (HTTP planner on :8001), the
evaluation scripts, and the other 20 test files.

Tests: `python3 -m pytest modules/project56/tests -q` → 123 passed, 1 skipped (that one
needs a 690 KB evaluation file). Needs `websockets` only for `NavelSegmentExecutor`.

Next (not started): a ChatBox `RobotProfile`, then call the planner from the robot hub
after a chat reply, with the persona/role as the style input.
