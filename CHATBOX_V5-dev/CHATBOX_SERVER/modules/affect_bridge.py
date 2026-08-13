"""
The server's one import point for `pad_core`.

pad_core lives beside CHATBOX_SERVER (CHATBOX_V5-dev/PAD_CORE/) as a
self-contained, stdlib-only package: the affect model, the servo mapping, the
prompt fragments and the per-turn adapter, with its own tests, bench rigs and
firmware. It knows nothing about this server — no graph, no camera, no LLM — so
it can be lifted into another project unchanged.

This module exists only to put PAD_CORE/ on sys.path once and re-export, so a
caller anywhere under CHATBOX_SERVER can write:

    from modules.affect_bridge import affect, servo_style, PADPipelineAdapter

Nothing is copied. The equations have exactly one home, which is what keeps the
design document's "every number here was produced by running that code" true.

If pad_core is ever installed properly (pip install -e), delete this file and
import `pad_core` directly — that is the only change required.
"""

import os
import sys

# modules/ -> CHATBOX_SERVER -> CHATBOX_V5-dev
_V5 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAD_CORE_ROOT = os.path.join(_V5, "PAD_CORE")

if PAD_CORE_ROOT not in sys.path:
    sys.path.insert(0, PAD_CORE_ROOT)

from pad_core import (                      # noqa: E402  — path set up above
    AffectStream,
    NullPADAdapter,
    PADPipelineAdapter,
    affect,
    prompt,
    servo_style,
)

__all__ = [
    "affect", "servo_style", "prompt",
    "PADPipelineAdapter", "NullPADAdapter", "AffectStream",
    "PAD_CORE_ROOT",
]
