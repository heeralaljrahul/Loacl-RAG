"""Hybrid retrieval: dense + BM25, fused, then reranked.

Why hybrid.  Dense vectors are good at paraphrase and terrible at exact
tokens — ask about error code `E-4471` or a person called `Havilland` and a
384-dimensional embedding will happily return something *about* errors or
*about* people.  BM25 nails those and is useless when the question shares no
words with the answer.  Neither is optional; the two failure modes barely
overlap.

Fusion is reciprocal rank fusion rather than a weighted score blend, because
cosine similarity and BM25 are not on a comparable scale and their ranges
shift with corpus and query length.  RRF only reads *ranks*, so it needs no
per-corpus tuning:

    score(chunk) = Σ over lists  1 / (k + rank_in_that_list)

Then a cross-encoder reranks the survivors, which is where most of the
accuracy comes from, and a per-document cap keeps one chatty file from
filling the whole context window.

Compound questions get one extra step.  "What is the per-diem rate and the
hotel cap in London?" has two answers in two different files, and a
cross-encoder scoring whole-query-against-chunk rates *both* chunks
mediocre, because neither one answers the whole question.  Measured on the
sample corpus, the expenses chunk scored 0.03 that way.  Scoring each clause
separately and keeping each chunk's best clause score moved it to 6.63,
comfortably clear of the noise floor around -7.  So a query that splits into
real clauses is retrieved and reranked per clause; a query that doesn't
split takes exactly the original path, at exactly the original cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .embed import Embedder, Reranker
from .store import Hit, Store
from .text import trail


@dataclass
class RetrievalResult:
    hits: list[Hit]
    query: str
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


class Retriever:
    def __init__(self, cfg: Config, store: Store, embedder: Embedder,
                 reranker: Reranker | None = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or Reranker(cfg)

    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        import time

        cfg = self.cfg
        top_k = top_k or cfg.top_k
        timings: dict[str, float] = {}
        clauses = split_clauses(query) if cfg.expand_clauses else []

        t0 = time.perf_counter()
        dense = self.store.dense_search(self.embedder.embed_query(query), cfg.dense_k)
        rank_lists = [[cid for cid, _ in dense]]
        weights = [1.0]
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        keyword = self.store.bm25_search(query, cfg.bm25_k)
        rank_lists.append([cid for cid, _ in keyword])
        weights.append(1.0)
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        if clauses:
            t0 = time.perf_counter()
            for clause in clauses:
                clause_dense = self.store.dense_search(
                    self.embedder.embed_query(clause), cfg.dense_k // 2 or 1)
                rank_lists.append([cid for cid, _ in clause_dense])
                weights.append(cfg.clause_weight)
                rank_lists.append(
                    [cid for cid, _ in self.store.bm25_search(clause, cfg.bm25_k // 2 or 1)])
                weights.append(cfg.clause_weight)
            timings["clause_ms"] = (time.perf_counter() - t0) * 1000

        fused = rrf_fuse(rank_lists, k=cfg.rrf_k, weights=weights)
        if not fused:
            return RetrievalResult([], query, timings, {"dense": 0, "bm25": 0, "fused": 0})

        candidate_ids = [cid for cid, _ in fused[: cfg.rerank_candidates]]
        hits = self.store.hits(candidate_ids)
        dense_ranks = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(keyword)}

        ordered: list[Hit] = []
        for cid, fused_score in fused[: cfg.rerank_candidates]:
            hit = hits.get(cid)
            if hit is None:
                continue
            hit.score = fused_score
            hit.dense_rank = dense_ranks.get(cid)
            hit.bm25_rank = bm25_ranks.get(cid)
            ordered.append(hit)

        t0 = time.perf_counter()
        if self.reranker.available and ordered:
            documents = [_rerank_text(h) for h in ordered]
            scores = self.reranker.score(query, documents)
            # A chunk that fully answers one clause of a compound question is
            # more useful than one that half-answers the whole thing, so a
            # chunk keeps its best score across the question and its clauses.
            for clause in clauses:
                for index, clause_score in enumerate(self.reranker.score(clause, documents)):
                    scores[index] = max(scores[index], clause_score)
            for hit, score in zip(ordered, scores):
                hit.rerank_score = score
                hit.score = score
            ordered.sort(key=lambda h: h.score, reverse=True)
            ordered = apply_floor(ordered, margin=cfg.rerank_margin,
                                  minimum=cfg.min_score)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        final = cap_per_document(ordered, cfg.max_per_doc)[:top_k]
        return RetrievalResult(
            final,
            query,
            timings,
            {"dense": len(dense), "bm25": len(keyword), "clauses": len(clauses),
             "fused": len(fused), "reranked": len(ordered), "returned": len(final)},
        )


def rrf_fuse(rank_lists: list[list[int]], k: int = 60,
             weights: list[float] | None = None) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for index, ranked in enumerate(rank_lists):
        weight = 1.0 if weights is None else weights[index]
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


_SPLIT = re.compile(r"\s*(?:[,;?]|\band\b|\bor\b|\balso\b|\bplus\b)\s*", re.I)


def split_clauses(query: str, min_words: int = 3, min_query_words: int = 7
                  ) -> list[str]:
    """Break a compound question into answerable clauses, or return nothing.

    Deliberately conservative.  A short or single-idea question returns ``[]``
    and takes the untouched original path — no extra embedding, no extra
    cross-encoder pass, and no risk of a regression on the ordinary case.
    Splitting only happens when the question is long enough to plausibly hold
    two asks *and* both sides survive as substantial clauses.
    """
    if len(query.split()) < min_query_words:
        return []
    parts = [p.strip() for p in _SPLIT.split(query)]
    clauses = [p for p in parts if len(p.split()) >= min_words]
    return clauses if len(clauses) >= 2 else []


def apply_floor(hits: list[Hit], *, margin: float, minimum: float = 0.0) -> list[Hit]:
    """Cut the noise tail off a reranked list.

    Cross-encoder logits are informative in absolute terms: on the default
    reranker a chunk that answers the question lands above roughly 0 and
    irrelevant text sits below -7.  A query whose best chunk scores -2.2 and
    whose next five all score about -11 has exactly one relevant chunk, and
    filling the remaining five context slots with the -11s is actively
    harmful — it spends prompt budget the model then has to ignore, and gives
    it plausible-looking material to blend into the answer.

    The floor is relative to the best score rather than absolute, because the
    absolute scale differs per reranker.  One chunk is always returned: an
    unhelpful answer with a citation the reader can check beats no answer.
    """
    if not hits:
        return hits
    kept = hits
    if margin > 0:
        cutoff = hits[0].score - margin
        kept = [h for h in hits if h.score >= cutoff]
    if minimum:
        kept = [h for h in kept if h.score >= minimum]
    return kept or hits[:1]


def cap_per_document(hits: list[Hit], max_per_doc: int) -> list[Hit]:
    if max_per_doc <= 0:
        return hits
    seen: dict[int, int] = {}
    out: list[Hit] = []
    overflow: list[Hit] = []
    for hit in hits:
        count = seen.get(hit.doc_id, 0)
        if count < max_per_doc:
            seen[hit.doc_id] = count + 1
            out.append(hit)
        else:
            overflow.append(hit)
    # Overflow is kept on the end rather than discarded: if only one document
    # is relevant, capping it must not shrink the answer's evidence.
    return out + overflow


def _rerank_text(hit: Hit) -> str:
    crumbs = trail(hit.title, hit.heading)
    body = hit.text[:2000]
    return f"{crumbs}\n{body}" if crumbs else body
