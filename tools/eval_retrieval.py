#!/usr/bin/env python
"""Measure retrieval quality — the only honest way to pick settings.

Every knob in this system (which embedding model, whether to rerank, how
many candidates to fuse) is a guess until it is measured on documents like
yours.  This script turns a labelled question set into numbers so you can
stop guessing.

    python tools/eval_retrieval.py --corpus sample_docs --queries tests/fixtures/eval.jsonl
    python tools/eval_retrieval.py --corpus my_notes --queries my_questions.jsonl \
        --embed-models BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5

The query file is JSON lines, one object per line:

    {"query": "how much holiday do I get", "expect": "25 days of paid annual leave"}

``expect`` is a substring that must appear in a retrieved chunk for the hit
to count — crude, but it needs no annotation tooling and it is unambiguous.
Give it a *list* for a compound question that needs several chunks:

    {"query": "the per-diem rate and the London hotel cap",
     "expect": ["per-diem rate is 45 EUR", "260 EUR in London"]}

Reported metrics:
  hit@1 / hit@3 / hit@5  fraction of questions whose first answer was in top N
  MRR                    1/rank of the first correct chunk, averaged
  cover@5                fraction of *all* expected passages found in the top 5
                         — the metric that exposes compound-question failures,
                         where one half of the answer is retrieved and the
                         other half silently is not
  miss                   questions where no expected passage was retrieved
  n/q                    mean chunks actually returned — with the relevance
                         floor on, a well-behaved run sends fewer chunks
                         without losing coverage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.app import App  # noqa: E402
from rag.config import Config  # noqa: E402
from rag.retrieve import rrf_fuse  # noqa: E402


def load_queries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def expectations(row: dict) -> list[str]:
    expect = row["expect"]
    return [expect] if isinstance(expect, str) else list(expect)


def rank_of_answer(texts: list[str], expected: str) -> int | None:
    needle = " ".join(expected.split()).lower()
    for index, text in enumerate(texts, start=1):
        if needle in " ".join(text.split()).lower():
            return index
    return None


def score(ranks: list[int | None], coverage: list[tuple[int, int]], total: int,
          returned: list[int] | None = None) -> dict:
    def hit_at(n: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= n) / total

    mrr = sum(1.0 / r for r in ranks if r is not None) / total
    found = sum(f for f, _ in coverage)
    wanted = sum(w for _, w in coverage) or 1
    return {"hit@1": hit_at(1), "hit@3": hit_at(3), "hit@5": hit_at(5),
            "mrr": mrr, "cover@5": found / wanted,
            "miss": sum(1 for r in ranks if r is None),
            "returned": (sum(returned) / len(returned)) if returned else 0.0}


def run_mode(app: App, queries: list[dict], mode: str, depth: int) -> tuple[dict, float]:
    ranks: list[int | None] = []
    coverage: list[tuple[int, int]] = []
    returned: list[int] = []
    started = time.perf_counter()
    for row in queries:
        query = row["query"]
        if mode == "dense":
            ids = [cid for cid, _ in
                   app.store.dense_search(app.embedder.embed_query(query), depth)]
        elif mode == "bm25":
            ids = [cid for cid, _ in app.store.bm25_search(query, depth)]
        elif mode == "hybrid":
            dense = [cid for cid, _ in
                     app.store.dense_search(app.embedder.embed_query(query), app.cfg.dense_k)]
            keyword = [cid for cid, _ in app.store.bm25_search(query, app.cfg.bm25_k)]
            ids = [cid for cid, _ in rrf_fuse([dense, keyword], k=app.cfg.rrf_k)][:depth]
        elif mode == "hybrid+rerank":
            ids = [h.chunk_id for h in app.search(query, top_k=depth).hits]
        else:
            raise ValueError(mode)
        hits = app.store.hits(ids)
        texts = [hits[i].text for i in ids if i in hits]
        found = [rank_of_answer(texts, want) for want in expectations(row)]
        ranks.append(min((r for r in found if r is not None), default=None))
        coverage.append((sum(1 for r in found if r is not None), len(found)))
        returned.append(len(texts))
    return (score(ranks, coverage, len(queries), returned),
            (time.perf_counter() - started) / len(queries))


def report(title: str, rows: list[tuple[str, dict, float]]):
    print(f"\n{title}")
    print(f"  {'configuration':<34} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} "
          f"{'MRR':>6} {'cov@5':>6} {'miss':>5} {'n/q':>5} {'ms/q':>7}")
    print("  " + "-" * 89)
    for name, metrics, seconds in rows:
        print(f"  {name:<34} {metrics['hit@1']:>6.2f} {metrics['hit@3']:>6.2f} "
              f"{metrics['hit@5']:>6.2f} {metrics['mrr']:>6.3f} "
              f"{metrics['cover@5']:>6.2f} {metrics['miss']:>5} "
              f"{metrics['returned']:>5.1f} {seconds * 1000:>7.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--embed-models", nargs="*", default=None,
                        help="compare embedding models (each needs its own index)")
    parser.add_argument("--rerank-models", nargs="*", default=None,
                        help="compare rerankers on top of the first embedding model")
    parser.add_argument("--work-dir", default=None,
                        help="where to build the throwaway indexes")
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    print(f"{len(queries)} labelled questions over {args.corpus}")

    work = Path(args.work_dir) if args.work_dir else Path(".eval")
    embed_models = args.embed_models or [Config.from_env().embed_model]

    for embed_model in embed_models:
        slug = embed_model.replace("/", "_")
        cfg = Config.from_env(data_dir=str(work / slug), embed_model=embed_model,
                              rerank=False)
        with App(cfg) as app:
            if not app.stats()["chunks"]:
                started = time.perf_counter()
                app.ingestor.ingest([args.corpus], progress=lambda _: None)
                print(f"\nindexed for {embed_model} in {time.perf_counter() - started:.1f}s "
                      f"({app.stats()['chunks']} chunks)")
            rows = [
                ("dense only", *run_mode(app, queries, "dense", args.depth)),
                ("bm25 only", *run_mode(app, queries, "bm25", args.depth)),
                ("hybrid (RRF)", *run_mode(app, queries, "hybrid", args.depth)),
            ]
            report(f"{embed_model}", rows)

        rerankers = args.rerank_models if (args.rerank_models and
                                           embed_model == embed_models[0]) else []
        rows = []
        for rerank_model in rerankers:
            cfg = Config.from_env(data_dir=str(work / slug), embed_model=embed_model,
                                  rerank=True, rerank_model=rerank_model)
            with App(cfg) as app:
                rows.append((f"+ {rerank_model}",
                             *run_mode(app, queries, "hybrid+rerank", args.depth)))
        if rows:
            report(f"{embed_model} + rerank", rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
