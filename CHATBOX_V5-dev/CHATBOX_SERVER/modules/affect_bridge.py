"""
Single import point for the top-level AFFECT_LAB / SERVO_STYLE benches.

The affect maths and the servo tables are NOT copied into the server. They stay
where their test suites and the design document live, and this module puts them
on sys.path once so `from modules.affect_bridge import affect, servo_style`
works from anywhere under CHATBOX_SERVER/.

Why a shim rather than a copy: docs/PAD_AFFECT_SYSTEM.md opens with "every number
in this document was produced by running that code". Two copies of the equations
would mean the document describes one of them and nobody can tell which.

`affect.py` imports nothing but the standard library and must stay that way —
that is what lets `python3 AFFECT_LAB/test_affect.py` verify the published
equations with no camera, no torch and no server. There is a purity test for it.
"""

import os
import sys

# CHATBOX_SERVER/modules/ -> CHATBOX_SERVER -> CHATBOX_V5-dev -> repo root
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))

for _sub in ("AFFECT_LAB", "SERVO_STYLE"):
    _path = os.path.join(REPO_ROOT, _sub)
    if _path not in sys.path:
        # position 0: these are bare top-level module names, so they must win
        # over anything else on the path that happens to share them.
        sys.path.insert(0, _path)

import affect          # noqa: E402  — path set up immediately above
import servo_style     # noqa: E402

__all__ = ["affect", "servo_style", "REPO_ROOT"]
