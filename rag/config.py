"""Configuration.

``Config.from_env()`` reads every field from an environment variable of the
same name prefixed with ``RAG_`` (e.g. ``RAG_TOP_K=8``), then applies any
keyword overrides on top.  The batch files in ``bat/`` set the ones that
matter for a 10 GB RTX 3080 / 16 GB RAM machine, so on Windows you configure
this by editing a .bat file rather than by hand.

Plain ``Config(...)`` ignores the environment entirely.  That ordering is
deliberate: an explicit argument must beat an ambient variable, or a tool
that sweeps ``embed_model`` across several values silently measures whatever
``RAG_EMBED_MODEL`` happens to be set to, and the tests stop being hermetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    raw = os.environ.get("RAG_" + name.upper())
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
class Config:
    # --- storage -------------------------------------------------------
    data_dir: str = str(ROOT / "data")

    # --- embedding (CPU, ONNX int8 — deliberately not on the GPU) ------
    # bge-small-en-v1.5  384 dim,  ~33 MB, ~600 chunks/s on an i9-11900K
    # bge-base-en-v1.5   768 dim, ~110 MB, ~200 chunks/s, a little sharper
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch: int = 64
    embed_threads: int = 0  # 0 = let onnxruntime decide

    # --- reranking (CPU cross-encoder; the biggest quality lever) ------
    rerank: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- chunking ------------------------------------------------------
    # ~4 chars per token, so 1800 chars is roughly 450 tokens.
    chunk_chars: int = 1800
    overlap_chars: int = 300
    min_chunk_chars: int = 120

    # --- retrieval -----------------------------------------------------
    dense_k: int = 40          # candidates from vector search
    bm25_k: int = 40           # candidates from keyword search
    rerank_candidates: int = 40  # fused candidates fed to the reranker
    top_k: int = 6             # chunks that actually reach the model
    max_per_doc: int = 3       # diversity cap, keeps one file from hogging
    rrf_k: int = 60            # reciprocal-rank-fusion constant
    # Cross-encoder scores are raw logits, not probabilities: on the default
    # reranker a good match is roughly > 0 and noise sits below -6. Left off
    # by default because the scale is model-specific — check `search` output
    # on your own corpus before setting it.
    min_score: float = 0.0     # 0 = disabled
    # Drop chunks scoring more than this many logits below the best one.
    # Padding the prompt to top_k with chunks the reranker rates as noise
    # costs tokens and invites the model to blend in irrelevant facts. On the
    # sample corpus this sends 3.9 chunks per question instead of 6.0 with no
    # loss of coverage; tightening to 6 starts dropping real answers (see
    # tools/eval_retrieval.py). At least one chunk is always kept.
    # 0 disables the floor.
    rerank_margin: float = 12.0
    # Compound questions ("the per-diem rate and the hotel cap"): retrieve and
    # rerank per clause, keeping each chunk's best clause score. Measured on
    # the sample corpus it gained nothing and cost ~20% latency, so it is off
    # by default — worth an A/B on your own documents, not worth assuming.
    expand_clauses: bool = False
    clause_weight: float = 0.5

    # --- generation (Ollama, on the GPU) -------------------------------
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    num_ctx: int = 8192
    temperature: float = 0.2
    max_tokens: int = 900
    keep_alive: str = "30m"
    context_budget_chars: int = 9000  # retrieved text allowed into the prompt

    # --- backends (test hooks; leave alone in normal use) --------------
    embed_backend: str = "fastembed"  # fastembed | hash
    llm_backend: str = "ollama"       # ollama | echo

    # --- ingestion -----------------------------------------------------
    max_file_mb: float = 64.0
    ignore_dirs: str = ".git,.venv,node_modules,__pycache__,.mypy_cache,.pytest_cache,dist,build,.idea,.vscode,data"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        values = {f.name: _env(f.name, f.default) for f in fields(cls)}
        values.update(overrides)
        return cls(**values)

    # -- derived --------------------------------------------------------
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_path / "index.sqlite3"

    @property
    def cache_path(self) -> Path:
        p = self.data_path / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def ignore_set(self) -> set[str]:
        return {d.strip() for d in self.ignore_dirs.split(",") if d.strip()}


CONFIG = Config.from_env()
