"""A campaign: one folder, one story.

    data/campaigns/<slug>/index.sqlite3

Everything is in there — transcript, character sheets, relationships, the
clock, the vector index, the keyword index. Copy the folder to back up a
campaign; delete it to start over. Two campaigns never see each other's
memories, which is what keeps a Fairy Tail Reina from recalling a Thornwell
one.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.app import App
from rag.config import Config
from rag.ingest import Ingestor

from .clock import Clock
from .config import GameConfig
from .engine import Engine
from .memory import MemoryArchive
from .state import GameState

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGNS = ROOT / "data" / "campaigns"


def campaign_dir(slug: str) -> Path:
    return CAMPAIGNS / slug


def list_campaigns() -> list[str]:
    if not CAMPAIGNS.exists():
        return []
    return sorted(p.name for p in CAMPAIGNS.iterdir()
                  if (p / "index.sqlite3").exists())


class Campaign:
    def __init__(self, slug: str, cfg: Config | None = None,
                 game_cfg: GameConfig | None = None):
        self.slug = slug
        self.cfg = cfg or Config.from_env(data_dir=str(campaign_dir(slug)))
        self.app = App(self.cfg)
        self.state = GameState(self.app.store.db)
        self.archive = MemoryArchive(self.app.store, self.app.embedder,
                                     self.app.retriever)
        self.game_cfg = game_cfg or GameConfig.from_env()
        self.engine = Engine(self.app, self.state, self.archive, self.game_cfg)

    def close(self):
        self.app.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- setup ----------------------------------------------------------
    def seed(self, path: str | Path) -> dict:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        counts = {"characters": 0, "relationships": 0, "flags": 0, "lore": 0}

        self.state.set("title", data.get("title", self.slug))
        clock = data.get("clock", {})
        if clock.get("date"):
            self.state.clock = Clock.parse(clock["date"], clock.get("time", "08:00"))
        if clock.get("location"):
            self.state.location = clock["location"]

        for entry in data.get("characters", []):
            self.state.upsert_character(
                entry["slug"], name=entry.get("name", entry["slug"]),
                role=entry.get("role", ""), sheet=entry.get("sheet", ""),
                present=bool(entry.get("present")),
                protagonist=bool(entry.get("protagonist")),
            )
            counts["characters"] += 1

        for entry in data.get("relationships", []):
            self.state.set_relationship(
                entry["other"], label=entry.get("label"),
                closeness=entry.get("closeness"), note=entry.get("note"))
            counts["relationships"] += 1

        for key, value in (data.get("flags") or {}).items():
            self.state.set_flag(key, value)
            counts["flags"] += 1

        # Lore is indexed as memory rather than injected: it is background
        # that only matters when the scene touches it, unlike a character
        # sheet, which matters every turn.
        for index, text in enumerate(data.get("lore", [])):
            if self.archive.write("lore", 0, text, when="background", index=index):
                counts["lore"] += 1

        return counts

    def open_with(self, narration: str, *, title: str = "Opening",
                  choices: list[str] | None = None) -> int:
        """Insert an existing piece of prose as turn 1.

        For starting a campaign from story you have already written rather
        than from a cold generated scene — the engine's memory is built out
        of it exactly as if it had been played.
        """
        from . import format as fmt

        clock = self.state.clock
        n = self.state.add_turn(
            in_date=clock.date, in_time=clock.time, title=title,
            location=self.state.location, player_input="",
            narration=narration.strip(),
            choices=choices or [], words=fmt.count_words(narration),
        )
        self.engine._archive_turn(n, "(the story so far)", narration.strip())
        return n

    def ingest_lore(self, paths: list[str]) -> str:
        """Add reference documents (a character bible, world notes) to this
        campaign's index. They are retrieved alongside story memories."""
        ingestor = Ingestor(self.cfg, self.app.store, self.app.embedder)
        return ingestor.ingest(paths, prune=False).summary()

    # -- info -----------------------------------------------------------
    def status(self) -> dict:
        clock = self.state.clock
        return {
            "slug": self.slug,
            "title": self.state.get("title", self.slug),
            "turns": self.state.turn_count,
            "when": f"{clock.pretty_date()} — {clock.pretty_time()}",
            "location": self.state.location,
            "characters": len(self.state.characters()),
            "present": [c.name for c in self.state.characters(present_only=True)],
            "arcs": len(self.state.arcs()),
            "memories": self.archive.count(),
            "flags": self.state.flags(),
        }
