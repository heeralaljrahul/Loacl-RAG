"""The play API and its SSE contract."""

import json

import pytest
from fastapi.testclient import TestClient

from game.server import create_app


@pytest.fixture
def client(campaign):
    campaign.engine.play("open the scene")
    return TestClient(create_app(campaign))


def _events(response) -> list[dict]:
    return [json.loads(b[6:]) for b in response.text.split("\n\n") if b.startswith("data: ")]


def test_page_is_served(client):
    assert "Campaign" in client.get("/").text


def test_state_carries_everything_the_panel_renders(client):
    body = client.get("/api/state").json()
    for key in ("title", "when", "turns", "present", "relationships", "flags",
                "choices", "memories", "arcs"):
        assert key in body
    assert len(body["choices"]) == 5


def test_history_returns_playable_entries(client):
    turns = client.get("/api/history?limit=5").json()["turns"]
    assert turns
    assert turns[0]["text"].startswith("📅")
    assert "What will you do?" in turns[0]["text"]


def test_a_turn_streams_recall_then_tokens_then_the_entry(client):
    response = client.post("/api/turn", json={"action": "2"})
    assert response.headers["content-type"].startswith("text/event-stream")
    kinds = [e["type"] for e in _events(response)]
    assert kinds[0] == "recall"
    assert "token" in kinds
    assert "entry" in kinds
    assert kinds[-1] == "done"


def test_the_entry_event_carries_the_canonical_header_and_choices(client):
    events = _events(client.post("/api/turn", json={"action": "go on"}))
    entry = next(e for e in events if e["type"] == "entry")
    assert entry["header"].startswith("📅 Monday, April 14th, 2025")
    assert len(entry["choices"]) == 5
    assert entry["narration"]


def test_recall_endpoint_searches_the_archive(client):
    body = client.post("/api/recall", json={"query": "cleaning duty"}).json()
    assert body["memories"]
    assert "text" in body["memories"][0]
