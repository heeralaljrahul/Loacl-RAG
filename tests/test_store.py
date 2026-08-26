import numpy as np

from rag.store import Store, fts_query


def test_fts_query_strips_punctuation_and_stopwords():
    expr = fts_query("What does the error E-4471 mean?")
    assert "?" not in expr and '"the"' not in expr
    assert '"e-4471"' in expr


def test_fts_query_survives_pure_punctuation():
    assert fts_query("???") == ""


def test_dense_search_is_exact_and_ordered(cfg):
    store = Store(cfg)
    doc = store.upsert_document(path="/x.md", title="X", ext=".md", size=1,
                                mtime=1.0, sha256="a")
    vectors = np.eye(3, dtype=np.float32)
    store.add_chunks(doc, [(0, "", None, "one"), (1, "", None, "two"),
                           (2, "", None, "three")], vectors)
    hits = store.dense_search(np.array([0, 1, 0], dtype=np.float32), k=3)
    assert hits[0][1] == 1.0
    assert store.hits([hits[0][0]])[hits[0][0]].text == "two"
    store.close()


def test_dimension_mismatch_is_explained_not_crashed(cfg):
    store = Store(cfg)
    doc = store.upsert_document(path="/x.md", title="X", ext=".md", size=1,
                                mtime=1.0, sha256="a")
    store.add_chunks(doc, [(0, "", None, "one")], np.eye(1, 4, dtype=np.float32))
    try:
        store.dense_search(np.zeros(8, dtype=np.float32), k=1)
    except RuntimeError as exc:
        assert "reindex" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError naming the fix")
    store.close()


def test_fts_index_follows_deletes(cfg):
    store = Store(cfg)
    doc = store.upsert_document(path="/x.md", title="X", ext=".md", size=1,
                                mtime=1.0, sha256="a")
    store.add_chunks(doc, [(0, "", None, "quarantine procedures")],
                     np.eye(1, 4, dtype=np.float32))
    assert store.bm25_search("quarantine", 5)
    store.delete_document(doc)
    assert store.bm25_search("quarantine", 5) == []
    store.close()


def test_upsert_replaces_chunks_rather_than_duplicating(cfg):
    store = Store(cfg)
    for text in ("first version", "second version"):
        doc = store.upsert_document(path="/x.md", title="X", ext=".md", size=1,
                                    mtime=1.0, sha256=text)
        store.add_chunks(doc, [(0, "", None, text)], np.eye(1, 4, dtype=np.float32))
    assert store.stats()["chunks"] == 1
    assert store.bm25_search("first", 5) == []
    store.close()


def test_explicit_arguments_beat_the_environment(monkeypatch):
    """A sweep over embed_model, or a hermetic test, must not be quietly
    rewritten by whatever RAG_* happens to be exported."""
    from rag.config import Config
    monkeypatch.setenv("RAG_EMBED_MODEL", "from/env")
    monkeypatch.setenv("RAG_TOP_K", "11")
    assert Config().embed_model != "from/env"          # plain ctor ignores env
    assert Config.from_env().embed_model == "from/env"  # from_env reads it
    assert Config.from_env().top_k == 11
    assert Config.from_env(embed_model="explicit").embed_model == "explicit"


def test_env_parses_types_and_booleans(monkeypatch):
    from rag.config import Config
    monkeypatch.setenv("RAG_RERANK", "0")
    monkeypatch.setenv("RAG_TEMPERATURE", "0.7")
    cfg = Config.from_env()
    assert cfg.rerank is False and cfg.temperature == 0.7
    monkeypatch.setenv("RAG_RERANK", "yes")
    assert Config.from_env().rerank is True
