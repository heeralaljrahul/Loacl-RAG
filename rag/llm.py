"""Ollama client.

Deliberately thin: two HTTP calls (``/api/chat`` streaming and ``/api/tags``)
plus a health probe, so there is no framework between you and the model and
nothing to debug but the request itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.url = cfg.ollama_url.rstrip("/")

    # -- health ---------------------------------------------------------
    def tags(self) -> list[str]:
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.url} — is the tray app running?"
            ) from exc
        return [m.get("name", "") for m in response.json().get("models", [])]

    def running(self) -> list[dict]:
        try:
            response = requests.get(f"{self.url}/api/ps", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException:
            return []

    def has_model(self, name: str) -> bool:
        wanted = name if ":" in name else name + ":latest"
        return any(t == wanted or t.startswith(name + ":") for t in self.tags())

    # -- generation -----------------------------------------------------
    def chat(self, messages: list[dict], *, stream: bool = True,
             temperature: float | None = None, max_tokens: int | None = None
             ) -> Iterator[str]:
        if self.cfg.llm_backend == "echo":
            yield from _echo(messages)
            return

        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "stream": bool(stream),
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.max_tokens if max_tokens is None else max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.url}/api/chat", json=payload, stream=bool(stream), timeout=600
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if not stream:
            yield response.json().get("message", {}).get("content", "")
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise LLMError(str(event["error"]))
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece
            if event.get("done"):
                break

    def complete(self, messages: list[dict], **kwargs) -> str:
        return "".join(self.chat(messages, **kwargs))


def _echo(messages: list[dict]) -> Iterator[str]:
    """Offline stand-in used by the tests: repeats what it was given so the
    prompt-assembly and citation paths can be asserted on without a model."""
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    yield "[echo] "
    yield user[-400:]


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks that reasoning models emit."""
    import re

    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)^\s*<think>.*$", "", text)
    return text.strip()
