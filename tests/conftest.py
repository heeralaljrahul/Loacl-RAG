import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from rag.app import App
from rag.config import Config


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A config that needs no downloads and no Ollama."""
    return Config(
        data_dir=str(tmp_path / "data"),
        embed_backend="hash",
        llm_backend="echo",
        rerank=False,
    )


@pytest.fixture
def app(cfg):
    with App(cfg) as instance:
        yield instance


@pytest.fixture
def corpus(tmp_path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "handbook.md").write_text(
        "# Employee Handbook 2026\n\n"
        "## Leave policy\n\n"
        "Full-time staff accrue 25 days of annual leave per year.\n\n"
        "### Carry-over\n\n"
        "Up to five days may be carried into the following year. "
        "Anything beyond that lapses on 31 March.\n\n"
        "## Expenses\n\n"
        "Receipts must be submitted within 30 days. The per-diem rate is 45 EUR.\n",
        encoding="utf-8",
    )
    (root / "runbook.md").write_text(
        "# Payments runbook\n\n"
        "## Error E-4471\n\n"
        "Error E-4471 means the settlement file was rejected by the clearing house. "
        "Re-queue it with `payctl requeue --batch <id>` and page the on-call.\n\n"
        "## Escalation\n\n"
        "Escalate to Havilland if the batch is still stuck after 20 minutes.\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text(
        "Random unrelated notes about gardening and tomato blight.\n", encoding="utf-8"
    )
    (root / "ignore.bin").write_bytes(b"\x00\x01\x02")
    return root


@pytest.fixture
def seed_file(tmp_path) -> Path:
    import json

    path = tmp_path / "seed.json"
    path.write_text(json.dumps({
        "title": "Test Campaign",
        "clock": {"date": "2025-04-14", "time": "15:35", "location": "Class 1-A"},
        "characters": [
            {"slug": "reina", "name": "Reina", "protagonist": True, "present": True,
             "role": "protagonist", "sheet": "195 cm tall. National champion."},
            {"slug": "mirajane", "name": "Mirajane", "present": True,
             "sheet": "Will not let Reina clean."},
            {"slug": "louis", "name": "Louis", "present": True, "sheet": "Cleans too."},
            {"slug": "min", "name": "Min", "present": False, "sheet": "Draws."},
        ],
        "relationships": [{"other": "mirajane", "label": "closest", "closeness": 9}],
        "flags": {"cleaning_duty": "Group Four this week"},
        "lore": ["Class 1-A runs the standard Japanese classroom rituals."],
    }), encoding="utf-8")
    return path


@pytest.fixture
def campaign(tmp_path, seed_file, monkeypatch):
    from game.campaign import Campaign
    from game.config import GameConfig
    from game.testing import reset

    monkeypatch.setattr("game.campaign.CAMPAIGNS", tmp_path / "campaigns")
    reset()
    cfg = Config(data_dir=str(tmp_path / "campaigns" / "test"),
                 embed_backend="hash", llm_backend="fake", rerank=False)
    with Campaign("test", cfg=cfg,
                  game_cfg=GameConfig(arc_every=3, recall_k=4)) as instance:
        instance.seed(seed_file)
        yield instance
