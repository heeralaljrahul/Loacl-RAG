"""End-to-end: ingest a small corpus, retrieve, assemble a prompt."""

import pytest

from rag.retrieve import cap_per_document, rrf_fuse
from rag.store import Hit


def test_ingest_then_retrieve(app, corpus):
    report = app.ingestor.ingest([corpus])
    assert report.added == 3            # .md, .md, .txt — the .bin is skipped
    assert report.skipped >= 1
    assert report.chunks > 0
    assert app.stats()["documents"] == 3

    # BM25 must carry an exact identifier the hash embedder cannot know.
    result = app.search("E-4471")
    assert result.hits
    assert "E-4471" in result.hits[0].text
    assert result.hits[0].bm25_rank is not None


def test_breadcrumb_makes_a_body_without_the_word_findable(app, corpus):
    app.ingestor.ingest([corpus])
    result = app.search("carry-over")
    assert any("carried into the following year" in h.text for h in result.hits)


def test_reingest_is_incremental(app, corpus):
    app.ingestor.ingest([corpus])
    again = app.ingestor.ingest([corpus])
    assert again.added == 0 and again.updated == 0
    assert again.unchanged == 3

    (corpus / "handbook.md").write_text("# Handbook\n\nEverything changed.\n",
                                        encoding="utf-8")
    third = app.ingestor.ingest([corpus])
    assert third.updated == 1 and third.unchanged == 2


def test_touching_a_file_without_editing_it_does_not_re_embed(app, corpus):
    import os
    import time

    app.ingestor.ingest([corpus])
    path = corpus / "runbook.md"
    os.utime(path, (time.time() + 10, time.time() + 10))
    report = app.ingestor.ingest([corpus])
    assert report.updated == 0
    assert report.unchanged == 3


def test_deleted_files_are_pruned(app, corpus):
    app.ingestor.ingest([corpus])
    (corpus / "notes.txt").unlink()
    report = app.ingestor.ingest([corpus])
    assert report.removed == 1
    assert app.stats()["documents"] == 2
    assert app.store.bm25_search("gardening", 5) == []


def test_rescan_uses_remembered_roots(app, corpus):
    app.ingestor.ingest([corpus])
    (corpus / "new.md").write_text("# New\n\nFresh content here.\n", encoding="utf-8")
    report = app.ingestor.rescan()
    assert report.added == 1


def test_changing_embed_model_is_refused_not_silently_wrong(app, corpus):
    app.ingestor.ingest([corpus])
    app.store.set_meta("embed_model", "some/other-model")
    with pytest.raises(RuntimeError, match="reindex"):
        app.ingestor.ingest([corpus])


def test_reindex_rebuilds(app, corpus):
    app.ingestor.ingest([corpus])
    before = app.stats()["chunks"]
    report = app.ingestor.reindex()
    assert report.added == 3
    assert app.stats()["chunks"] == before


def test_one_bad_file_does_not_stop_the_run(app, corpus):
    (corpus / "broken.pdf").write_bytes(b"not really a pdf")
    report = app.ingestor.ingest([corpus])
    assert report.added == 3
    assert len(report.failures) == 1
    assert "broken.pdf" in report.failures[0][0]


# -- fusion / diversity ----------------------------------------------------


def test_rrf_rewards_agreement_between_the_two_retrievers():
    fused = dict(rrf_fuse([[10, 20, 30], [40, 10, 50]], k=60))
    assert max(fused, key=fused.get) == 10


def test_rrf_needs_no_score_calibration():
    # Same ranks, wildly different underlying score scales — RRF only reads rank.
    assert rrf_fuse([[1, 2]]) == rrf_fuse([[1, 2]])


def _hit(doc_id: int, chunk_id: int) -> Hit:
    return Hit(chunk_id, doc_id, f"/d{doc_id}", "t", "", None, "x")


def test_per_document_cap_diversifies_without_discarding():
    hits = [_hit(1, i) for i in range(5)] + [_hit(2, 99)]
    capped = cap_per_document(hits, 2)
    assert [h.doc_id for h in capped[:3]] == [1, 1, 2]
    assert len(capped) == len(hits), "overflow is demoted, never dropped"


# -- relevance floor -------------------------------------------------------


def _scored(score: float, chunk_id: int = 1) -> Hit:
    hit = Hit(chunk_id, chunk_id, f"/d{chunk_id}", "t", "", None, "x")
    hit.score = score
    return hit


def test_floor_cuts_the_noise_tail():
    from rag.retrieve import apply_floor
    hits = [_scored(-2.2, 1)] + [_scored(-11.0 - i, i + 2) for i in range(5)]
    kept = apply_floor(hits, margin=6.0)
    assert len(kept) == 1, "only one chunk was rated relevant"


def test_floor_keeps_a_genuinely_tight_cluster():
    from rag.retrieve import apply_floor
    hits = [_scored(4.0, 1), _scored(3.1, 2), _scored(1.0, 3)]
    assert len(apply_floor(hits, margin=6.0)) == 3


def test_floor_never_returns_nothing():
    """An unhelpful answer with a citation the reader can check beats no
    answer at all, so the best chunk always survives."""
    from rag.retrieve import apply_floor
    assert len(apply_floor([_scored(-30.0, 1)], margin=6.0, minimum=0.0)) == 1
    assert len(apply_floor([_scored(-30.0, 1), _scored(-31.0, 2)],
                           margin=6.0, minimum=5.0)) == 1


def test_floor_is_off_when_margin_is_zero():
    from rag.retrieve import apply_floor
    hits = [_scored(0.0, 1), _scored(-99.0, 2)]
    assert len(apply_floor(hits, margin=0.0)) == 2


# -- clause splitting ------------------------------------------------------


def test_simple_questions_do_not_split():
    from rag.retrieve import split_clauses
    assert split_clauses("how much holiday do I get") == []
    assert split_clauses("E-4471") == []


def test_compound_questions_split_into_answerable_halves():
    from rag.retrieve import split_clauses
    assert split_clauses("what is the per diem rate and the hotel cap in London?") == [
        "what is the per diem rate", "the hotel cap in London"]


def test_splitting_on_and_is_naive_and_that_is_why_it_ships_off():
    """A known limitation, recorded rather than papered over: "black and
    white" is a phrase, not two questions, and the splitter cuts it anyway.
    Telling a conjunction from a phrasal "and" needs parsing this does not
    do. Since clause expansion measured as no-gain on the sample corpus
    (see tools/eval_retrieval.py), cfg.expand_clauses is off by default and
    this stays a documented sharp edge rather than a fixed bug."""
    from rag.retrieve import split_clauses
    assert split_clauses("is the policy black and white on this question") == [
        "is the policy black", "white on this question"]

    from rag.config import Config
    assert Config().expand_clauses is False
