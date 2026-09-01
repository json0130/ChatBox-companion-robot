"""
padeval — the evaluation harness for the PAD affect controller.

Separate from `modules/` on purpose. The dependency direction is one-way:
`padeval` imports from `modules.*` and `PAD_CORE`, and nothing under `modules/`
or `PAD_CORE/` ever imports `padeval`. The runtime never depends on the harness.

The core (config, schema, fixtures, axes, arms, stimuli) imports NUMPY ONLY, so
generation runs on a machine that has never installed scipy/sklearn/pandas.
Those live behind `padeval.analysis`, and are listed in requirements-eval.txt
rather than the runtime requirements.txt.

Named `padeval` and not `eval` because the latter shadows the builtin.
"""

__version__ = "0.1.0"
