"""The Ollama client against a stand-in server that speaks Ollama's wire
format: newline-delimited JSON, one object per token."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rag.config import Config
from rag.llm import LLMError, Ollama


class Handler(BaseHTTPRequestHandler):
    script: list = []
    seen: dict = {}

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "qwen3:8b"}, {"name": "gemma3:12b"}]})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": "qwen3:8b", "size": 6_000_000_000,
                                    "size_vram": 6_000_000_000}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        Handler.seen.update(json.loads(self.rfile.read(length) or b"{}"))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for line in Handler.script:
            self.wfile.write((json.dumps(line) + "\n").encode())
            self.wfile.flush()

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _client(server) -> Ollama:
    return Ollama(Config(ollama_url=server, llm_model="qwen3:8b"))


def test_streaming_tokens_are_reassembled(server):
    Handler.script = [
        {"message": {"content": "Twenty-"}},
        {"message": {"content": "five days"}},
        {"message": {"content": ""}, "done": True},
    ]
    assert _client(server).complete([{"role": "user", "content": "how many?"}]) \
        == "Twenty-five days"


def test_options_are_sent_through(server):
    Handler.script = [{"message": {"content": "x"}, "done": True}]
    cfg = Config(ollama_url=server, llm_model="qwen3:8b", num_ctx=4096,
                 temperature=0.3, keep_alive="15m")
    Ollama(cfg).complete([{"role": "user", "content": "hi"}])
    assert Handler.seen["options"]["num_ctx"] == 4096
    assert Handler.seen["options"]["temperature"] == 0.3
    assert Handler.seen["keep_alive"] == "15m"


def test_a_mid_stream_error_is_raised_not_swallowed(server):
    Handler.script = [{"message": {"content": "partial"}},
                      {"error": "model requires more system memory"}]
    with pytest.raises(LLMError, match="system memory"):
        _client(server).complete([{"role": "user", "content": "hi"}])


def test_malformed_lines_are_skipped(server):
    Handler.script = [{"message": {"content": "a"}}, {"message": {"content": "b"}},
                      {"done": True}]
    assert _client(server).complete([{"role": "user", "content": "hi"}]) == "ab"


def test_model_presence_check(server):
    client = _client(server)
    assert client.has_model("qwen3:8b")
    assert client.has_model("gemma3")
    assert not client.has_model("llama3:70b")


def test_gpu_placement_is_reported(server):
    loaded = _client(server).running()
    assert loaded[0]["size_vram"] == loaded[0]["size"], "100% on GPU"


def test_unreachable_server_names_the_fix():
    client = Ollama(Config(ollama_url="http://127.0.0.1:1"))
    with pytest.raises(LLMError, match="tray app"):
        client.tags()
