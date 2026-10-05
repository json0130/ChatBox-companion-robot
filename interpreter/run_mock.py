"""End-to-end mock run:  python3 -m interpreter.run_mock [--frames DIR] [--direction D] [--task1 baseline|rules]"""
import argparse
from collections import Counter
from pathlib import Path

from interpreter.baseline import baseline_task1, baseline_task2
from interpreter.contracts import DIRECTIONS
from interpreter.mocks.mock_pepper import MockPepper
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.context import StaticContext
from interpreter.pipeline import Pipeline
from interpreter.task1 import plan_speech

DEFAULT_FRAMES = Path(__file__).parent / "mocks" / "frames"   # = interpreter/mocks/frames


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES))
    ap.add_argument("--direction", choices=DIRECTIONS)
    ap.add_argument("--task1", choices=("baseline", "rules"), default="baseline")
    args = ap.parse_args(argv)

    frames = load_frames(args.frames, args.direction)
    pepper, signgen = MockPepper(), StubSignGenerator()
    ctx = StaticContext()
    task1 = baseline_task1 if args.task1 == "baseline" else (lambda f: plan_speech(f, ctx))
    pipe = Pipeline(task1, baseline_task2, pepper, signgen)
    for f in frames:
        pipe.process(f)

    per_dir = Counter(f.direction for f in frames)
    print("\n--- summary ---")
    for d in DIRECTIONS:
        print(f"{d}: {per_dir[d]} frames")
    print(f"plans delivered: {len(pepper.received) + len(signgen.received)} "
          f"({len(pepper.received)} speech, {len(signgen.received)} sign)")


if __name__ == "__main__":
    main()
