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
