"""Embeddings and reranking — both on the CPU, on purpose.

On a 10 GB card the LLM needs every megabyte of VRAM it can get.  bge-small
quantised to int8 through ONNX Runtime does ~600 chunks/s on an i9-11900K
while the GPU stays entirely free for generation, so putting the embedder
on the GPU would cost context length and buy nothing.

Queries and documents go through separate calls on purpose.  Several
retrieval models are asymmetric — they expect a query-side instruction
prefix ("Represent this sentence for searching relevant passages:") that
must not be applied to documents — and fastembed knows which models need
one.  For the bge-*-en-v1.5 default it applies none, and measuring showed
adding one by hand slightly *hurt*, so this defers to fastembed rather than
hard-coding a prefix.  Swap in an asymmetric model (arctic-embed, nomic) and
``query_embed`` does the right thing without any change here.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np

from .config import Config


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    # -- lifecycle ------------------------------------------------------
    def _load(self):
        if self._model is not None or self.cfg.embed_backend == "hash":
            return
        with self._lock:
            if self._model is not None:
                return
            from fastembed import TextEmbedding

            kwargs = {
                "model_name": self.cfg.embed_model,
                "cache_dir": str(self.cfg.cache_path),
            }
            if self.cfg.embed_threads:
                kwargs["threads"] = self.cfg.embed_threads
            self._model = TextEmbedding(**kwargs)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_documents(["dimension probe"])[0])
        return self._dim

    @property
    def name(self) -> str:
        return "hash-test" if self.cfg.embed_backend == "hash" else self.cfg.embed_model

    # -- encoding -------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.cfg.embed_backend == "hash":
            return _hash_embed(texts)
        self._load()
        vectors = list(self._model.embed(texts, batch_size=self.cfg.embed_batch))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        if self.cfg.embed_backend == "hash":
            return _hash_embed([text])[0]
        self._load()
        vectors = list(self._model.query_embed([text]))
        return _normalize(np.asarray(vectors, dtype=np.float32))[0]


class Reranker:
    """Cross-encoder rerank.

    Bi-encoder retrieval scores a query and a chunk that never met each
    other; a cross-encoder reads them together.  It is far slower per pair,
    which is exactly why it only ever sees the ~40 candidates that survived
    fusion rather than the whole corpus.  On this hardware it costs about
    300 ms and is the single largest quality gain in the pipeline.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.cfg.rerank) and self.cfg.embed_backend != "hash"

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                model_name=self.cfg.rerank_model,
                cache_dir=str(self.cfg.cache_path),
            )

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if not self.available:
            return [0.0] * len(documents)
        self._load()
        return [float(s) for s in self._model.rerank(query, documents)]


# --------------------------------------------------------------------------


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


def _hash_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-words hashing, used by the tests.

    Not a real embedding — it has no semantics — but it is stable, needs no
    download, and exercises every code path around it.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 else -1.0
            out[row, index] += sign
    return _normalize(out)
