"""The web API, including the SSE contract the browser depends on."""

import json

import pytest
from fastapi.testclient import TestClient

from rag.server import create_app


@pytest.fixture
def client(app, corpus):
    app.ingestor.ingest([corpus])
    return TestClient(create_app(app))


def _events(response) -> list[dict]:
    out = []
    for block in response.text.split("\n\n"):
        if block.startswith("data: "):
            out.append(json.loads(block[6:]))
    return out


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Local RAG" in response.text


def test_stats_reports_the_index(client):
    body = client.get("/api/stats").json()
    assert body["documents"] == 3
    assert body["chunks"] > 0
    assert "ollama" in body


def test_ask_streams_sources_first_then_tokens(client):
    response = client.post("/api/ask", json={"question": "annual leave"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    assert events[0]["type"] == "sources"
    assert events[0]["sources"], "the browser panel needs sources before any text"
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "token" for e in events)


def test_sources_carry_what_the_debug_panel_renders(client):
    events = _events(client.post("/api/ask", json={"question": "E-4471"}))
    source = events[0]["sources"][0]
    for key in ("label", "path", "text", "score", "dense_rank", "bm25_rank"):
        assert key in source


def test_search_endpoint_skips_the_model(client):
    body = client.post("/api/search", json={"question": "expenses"}).json()
    assert body["sources"]
    assert "dense_ms" in body["timings"]


def test_top_k_is_honoured(client):
    body = client.post("/api/search", json={"question": "leave", "top_k": 1}).json()
    assert len(body["sources"]) == 1


def test_ingest_endpoint_rescans(client):
    body = client.post("/api/ingest", json={}).json()
    assert "unchanged" in body["summary"]


def test_question_with_no_match_still_returns_a_clean_stream(client):
    events = _events(client.post(
        "/api/ask", json={"question": "zzzz quantum wombat telemetry"}))
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
