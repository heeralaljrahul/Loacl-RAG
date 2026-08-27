"""Game settings. Environment overrides use a ``GAME_`` prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _env(name: str, default):
    raw = os.environ.get("GAME_" + name.upper())
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class GameConfig:
    # --- entry shape ---------------------------------------------------
    min_words: int = 800
    max_words: int = 900
    max_repairs: int = 1        # revision passes per turn before accepting

    # --- context budget ------------------------------------------------
    # One full previous entry is ~1200 tokens. On an 8K context with a
    # 900-word answer to produce, two verbatim entries plus state plus recall
    # is already the whole window — so only the last turn is kept in full and
    # the ones before it arrive as their summaries.
    verbatim_turns: int = 1
    summary_turns: int = 6
    recall_k: int = 6
    recall_chars: int = 2200
    narration_chars: int = 5200  # cap on a verbatim turn

    # --- memory --------------------------------------------------------
    archive_every: int = 1      # distil after every turn (1 = never lose a turn)
    arc_every: int = 10         # fold summaries into an arc this often
    arc_context: int = 6        # arc summaries injected each turn (newest)

    @classmethod
    def from_env(cls, **overrides) -> "GameConfig":
        values = {f.name: _env(f.name, f.default) for f in fields(cls)}
        values.update(overrides)
        return cls(**values)
