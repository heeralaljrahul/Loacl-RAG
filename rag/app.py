"""One object that wires the pieces together, with lazy loading.

Models are loaded on first use rather than at import, so `stats` and
`ingest` don't pay for the reranker and `ask` doesn't pay for anything it
doesn't touch.
"""

from __future__ import annotations

from .answer import Answerer
from .config import CONFIG, Config
from .embed import Embedder, Reranker
from .ingest import Ingestor
from .llm import Ollama
from .retrieve import Retriever
from .store import Store


class App:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        self.store = Store(self.cfg)
        self.embedder = Embedder(self.cfg)
        self.reranker = Reranker(self.cfg)
        self.llm = Ollama(self.cfg)
        self.retriever = Retriever(self.cfg, self.store, self.embedder, self.reranker)
        self.answerer = Answerer(self.cfg, self.retriever, self.llm)
        self.ingestor = Ingestor(self.cfg, self.store, self.embedder)

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- convenience ----------------------------------------------------
    def ask(self, question: str, **kwargs):
        return self.answerer.ask(question, **kwargs)

    def stream(self, question: str, **kwargs):
        return self.answerer.stream(question, **kwargs)

    def search(self, query: str, **kwargs):
        return self.retriever.search(query, **kwargs)

    def stats(self) -> dict:
        info = self.store.stats()
        info["roots"] = self.store.get_meta("roots", []) or []
        info["llm_model"] = self.cfg.llm_model
        info["rerank_model"] = self.cfg.rerank_model if self.cfg.rerank else None
        return info
