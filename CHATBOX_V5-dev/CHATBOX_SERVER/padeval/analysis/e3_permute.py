"""
E3 — directive chattering under a permuted source-to-axis assignment.

THE ARGUMENT
------------
The behavioural directive is a per-SESSION control: it says who introduces a
topic, and the relationship it derives from changes across sessions, not within
one. Under the deployed assignment the commanded rung is a function of Dominance,
Dominance is written by the relationship tier alone, and the face cannot move it.
The setpoint is therefore constant for the whole session — verified: the rung is
invariant across happy/sad/angry/neutral for both robots at every tier.

Permute emotion and relationship and that stops being true. The face writes
Dominance at frame rate, so a 30 Hz signal now drives a per-session effector and
the commanded directive flips mid-conversation.

That is a control-systems failure, and it is measurable as one: switches per
minute, dwell time, and how often the coordinate saturates against the clamp. No
LLM, no judge, no statistics beyond a count.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
It shows the permuted assignment has no stable setpoint. It does NOT show the
prompt-to-behaviour map is broken — that map still works, it is merely being
commanded from the wrong source. E1 measures the map; this measures the signal
driving it. Expect E1 fidelity to survive permutation and say so first.

numpy only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

from modules.affect_bridge import AffectStream, affect, prompt
from padeval.axes import ASSIGNMENTS, AxisAssignment, compose
from padeval.traces import Trace, synth_trace

# The ladder, newest-first by edge, exactly as prompt.manner_directive walks it.
_RUNG_EDGES = [edge for edge, _text in prompt._DIRECTIVES]


def rung_of(dominance: float) -> int:
    """Index into prompt._DIRECTIVES for this Dominance. 0 = most assertive."""
    for i, edge in enumerate(_RUNG_EDGES):
        if dominance >= edge:
            return i
    return len(_RUNG_EDGES) - 1


@dataclass(frozen=True)
class Volatility:
    assignment: str
    robot: str
    tier: str
    sigma: float
    duration_s: float
    # --- signal level: every camera frame ---
    n_frames: int
    rung_switches: int
    switches_per_min: float
    distinct_rungs: int
    dwell_median_s: float
    dwell_min_s: float
    # --- controller level: decimated to the deployed tick ---
    tick_hz: float
    n_ticks: int
    tick_switches: int
    tick_switches_per_min: float
    tick_distinct_rungs: int
    # --- interference ---
    clamp_rate: float          # fraction of frames with any axis clamped
    clamp_rate_d: float        # fraction with the DOMINANCE axis clamped


def directive_volatility(assignment: AxisAssignment,
                         robot: str,
                         tier: str,
                         trace: Trace,
                         empathy: float = affect.EMPATHY,
                         smoothing_window: int = 5,
                         tick_hz: float = 1.0) -> Volatility:
    """Drive a V/A trace through the real AffectStream and report rung stability.

    The AffectStream is the deployed smoother (pad_core/stream.py, 5-sample mean),
    included on purpose: it is part of the system under test, and giving the
    permuted assignment the benefit of the real smoother makes the comparison
    fair rather than rhetorical.

    TWO RATES ARE REPORTED, and conflating them would overstate the result.

    The FRAME-level numbers describe the signal: how stable the commanded rung is
    as a function of the camera stream. A 0.10 s median dwell there is a property
    of the signal, NOT a claim that the robot visibly changes behaviour ten times
    a second.

    The TICK-level numbers describe what the controller can actually observe.
    `webcam_loop` recomputes PAD once per `_DEFAULT_TICK` (1.0 s), and the
    directive only reaches the LLM when a system prompt is built, i.e. once per
    conversational turn — slower still. So the deployed system SAMPLES this
    signal; it does not track it.

    That is what makes the permuted assignment a control failure rather than a
    cosmetic one. When the signal's dwell falls below the sampling interval, the
    directive that reaches the model is decided by which instant happened to be
    sampled. It does not chatter visibly — it becomes arbitrary. `tick_hz`
    defaults to the deployed 1.0 Hz.
    """
    baseline = affect.to_pad(affect.ROBOTS[robot]["ocean"])
    stream = AffectStream(smoothing_window=smoothing_window)

    rungs: List[int] = []
    clamped_any = clamped_d = 0
    for v_raw, a_raw in trace.va:
        v, a = stream.update(float(v_raw), float(a_raw))
        felt, flags = compose(assignment, baseline, v, a, tier, empathy)
        rungs.append(rung_of(felt["D"]))
        clamped_any += int(any(flags.values()))
        clamped_d += int(flags["D"])

    arr = np.asarray(rungs, dtype=int)
    switches = int(np.count_nonzero(np.diff(arr)))

    # Dwell lengths in frames -> seconds.
    boundaries = np.flatnonzero(np.diff(arr)) + 1
    runs = np.diff(np.concatenate(([0], boundaries, [len(arr)])))
    dwell_s = runs / trace.fps

    # Decimate to the controller's sampling rate.
    stride = max(1, int(round(trace.fps / tick_hz)))
    ticked = arr[::stride]
    tick_switches = int(np.count_nonzero(np.diff(ticked))) if len(ticked) > 1 else 0

    return Volatility(
        assignment=assignment.name,
        robot=robot,
        tier=tier,
        sigma=float("nan"),                 # filled in by the sweep
        duration_s=trace.duration_s,
        n_frames=len(arr),
        rung_switches=switches,
        switches_per_min=switches / (trace.duration_s / 60.0),
        distinct_rungs=int(len(np.unique(arr))),
        dwell_median_s=float(np.median(dwell_s)),
        dwell_min_s=float(dwell_s.min()),
        tick_hz=float(tick_hz),
        n_ticks=int(len(ticked)),
        tick_switches=tick_switches,
        tick_switches_per_min=tick_switches / (trace.duration_s / 60.0),
        tick_distinct_rungs=int(len(np.unique(ticked))),
        clamp_rate=clamped_any / len(arr),
        clamp_rate_d=clamped_d / len(arr),
    )


def sweep(assignments: Sequence[str] = ("identity", "perm_emotion_D",
                                        "perm_arousal_D", "collapse_D"),
          robots: Sequence[str] = ("CHATBOX", "ELLEBOT"),
          tiers: Sequence[str] = ("known",),
          sigmas: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
          fps: float = 30.0,
          dwell_s: float = 4.0,
          seed: int = 0) -> List[Dict]:
    """Chattering across assignments x robots x tiers x noise levels."""
    rows: List[Dict] = []
    for sigma in sigmas:
        trace = synth_trace(dwell_s=dwell_s, fps=fps, sigma=sigma, seed=seed)
        for name in assignments:
            for robot in robots:
                for tier in tiers:
                    v = directive_volatility(ASSIGNMENTS[name], robot, tier, trace)
                    row = asdict(v)
                    row["sigma"] = float(sigma)
                    rows.append(row)
    return rows


_CAPTION = """
**Reading the two rates.** `frame switches/min` and `median dwell` describe the
SIGNAL, sampled at the camera's frame rate (30 fps here). They are not a claim
that the robot visibly changes behaviour many times a second.

`tick switches/min` is what the controller can actually observe: `webcam_loop`
recomputes PAD once per `_DEFAULT_TICK` (1.0 s), and the directive only reaches
the LLM when a system prompt is assembled — once per conversational turn, slower
still. The deployed system therefore SAMPLES this signal rather than tracking it.

That is the point. When median dwell falls below the sampling interval, the
directive reaching the model is decided by whichever instant happened to be
sampled: it does not chatter visibly, it becomes arbitrary. Under `identity`
both rates are exactly zero, so the question does not arise.
"""


def format_table(rows: Sequence[Dict]) -> str:
    """Markdown, grouped by sigma. Kept here so the result is reproducible from
    one command with no reporting stack."""
    out: List[str] = [_CAPTION]
    for sigma in sorted({r["sigma"] for r in rows}):
        out.append(f"\n### frame noise sigma = {sigma:.2f}\n")
        out.append("| assignment | robot | tier | frame switches/min "
                   "| tick switches/min (1 Hz) | distinct rungs "
                   "| median dwell (s) | clamp rate (D) |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in [x for x in rows if x["sigma"] == sigma]:
            out.append(
                f"| `{r['assignment']}` | {r['robot']} | {r['tier']} "
                f"| **{r['switches_per_min']:.1f}** "
                f"| **{r['tick_switches_per_min']:.1f}** "
                f"| {r['distinct_rungs']} | {r['dwell_median_s']:.2f} "
                f"| {r['clamp_rate_d']:.2%} |"
            )
    return "\n".join(out)
