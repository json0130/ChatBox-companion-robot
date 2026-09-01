"""
Analysis half of padeval. May import scipy / sklearn / pandas (see
requirements-eval.txt); the generation half deliberately may not.

`e3_permute` is the exception and imports numpy only, so the chattering result
can be reproduced on a bare machine.
"""
