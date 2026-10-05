"""Who prefers what, as seen by the interpreter.

Step 4 replaces this with a KG-backed provider; Task 1 must only ever talk to this interface.
"""
from typing import Dict, Optional, Protocol, Tuple


class ContextProvider(Protocol):
    def prefers_te_reo(self, speaker_id: str, slug: str) -> bool:
        """True if the speaker would want the kupu Māori form of concept `slug` spoken."""
        ...


class StaticContext:
    """Dict-backed provider: {(speaker_id, slug): bool}, falling back to `default`."""

    def __init__(self, table: Optional[Dict[Tuple[str, str], bool]] = None, default: bool = True) -> None:
        self.table = dict(table or {})
        self.default = default

    def prefers_te_reo(self, speaker_id: str, slug: str) -> bool:
        return self.table.get((speaker_id, slug), self.default)
