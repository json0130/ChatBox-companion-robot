"""End-to-end mock run:  python3 -m interpreter.run_mock [--frames DIR] [--direction D]"""
import argparse
from collections import Counter
from pathlib import Path

from interpreter.baseline import baseline_task1, baseline_task2
from interpreter.contracts import DIRECTIONS
from interpreter.mocks.mock_pepper import MockPepper
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.pipeline import Pipeline

DEFAULT_FRAMES = Path(__file__).parent / "mocks" / "frames"   # = interpreter/mocks/frames


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES))
    ap.add_argument("--direction", choices=DIRECTIONS)
    args = ap.parse_args(argv)

    frames = load_frames(args.frames, args.direction)
    pepper, signgen = MockPepper(), StubSignGenerator()
    pipe = Pipeline(baseline_task1, baseline_task2, pepper, signgen)
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
