"""Mock recogniser / speech front-end: loads hand-written InterpretationFrame JSON files."""
import json
from pathlib import Path
from typing import List, Optional

from interpreter.contracts import DIRECTIONS, InterpretationFrame


def load_frames(folder, direction: Optional[str] = None) -> List[InterpretationFrame]:
    """Load every *.json in `folder` (sorted by filename), validated through from_dict.
    Errors name the failing file. Duplicate frame_ids raise. `direction` filters the result."""
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError(f"direction={direction!r} not in {DIRECTIONS}")
    frames, seen = [], {}
    for path in sorted(Path(folder).glob("*.json"), key=lambda p: p.name):
        try:
            frame = InterpretationFrame.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
            raise ValueError(f"{path.name}: {e}") from e
        if frame.frame_id in seen:
            raise ValueError(f"{path.name}: duplicate frame_id {frame.frame_id!r} (also in {seen[frame.frame_id]})")
        seen[frame.frame_id] = path.name
        frames.append(frame)
    return [f for f in frames if direction is None or f.direction == direction]
