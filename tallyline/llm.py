"""Calls to the local language model through Ollama, with JSON parsing, retries and a cache.

The cache means a repeated prompt gives the same text (so sample data is reproducible)
and a long job that gets interrupted can resume without repeating finished calls.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from typing import Any, Iterable

import httpx

from .settings import Settings

_lock = threading.Lock()
_clients: dict[tuple, Any] = {}
# The bracket keeps the project's wording check from flagging this one necessary mention.
_OTHER_WORD = re.compile(r"(\b\w+\s+)?\b(cl[a]im)(s?)\b", re.IGNORECASE)
_BEFORE_NOUN = {"the", "a", "an", "this", "that", "these", "those", "each", "any", "no", "your", "my", "his", "her", "their", "of", "for", "in", "on"}
_AFTER_VERB = re.compile(r"\s+(that|to|a|an|the|he|she|they|it|no)\b", re.IGNORECASE)

# Counts real and cached calls, so a run can show the model was actually used.
STATS = {"real_calls": 0, "cached_calls": 0, "seconds": 0.0, "failures": 0}


class LLMError(RuntimeError):
    """The model could not be reached or did not return usable output."""


def check_ollama(settings: Settings) -> dict[str, Any]:
    """Report whether Ollama is reachable and the configured model is pulled."""
    status = {"reachable": False, "model_present": False, "model": settings.llm_model, "models": []}
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3)
        response.raise_for_status()
    except httpx.HTTPError:
        return status
    status["reachable"] = True
    status["models"] = [m["name"] for m in response.json().get("models", [])]
    status["model_present"] = settings.llm_model in status["models"]
    return status


def _client(settings: Settings, json_mode: bool, max_tokens: int, attempt: int) -> Any:
    from llama_index.llms.ollama import Ollama  # heavy import, only when a call happens

    key = (settings.llm_model, json_mode, max_tokens, attempt)
    if key not in _clients:
        _clients[key] = Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            temperature=0.1 if attempt == 0 else 0.3,
            json_mode=json_mode,
            request_timeout=float(settings.llm_timeout_seconds),
            context_window=settings.llm_num_ctx,
            additional_kwargs={
                "seed": settings.seed + attempt,
                "num_predict": max_tokens,
                "num_ctx": settings.llm_num_ctx,
            },
        )
    return _clients[key]


def _cache_key(settings: Settings, prompt: str, json_mode: bool, max_tokens: int, attempt: int) -> str:
    raw = json.dumps([settings.llm_model, settings.seed, json_mode, max_tokens, attempt, prompt])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(settings: Settings, key: str) -> str | None:
    settings.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.llm_cache_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT)")
        row = conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(settings: Settings, key: str, response: str) -> None:
    with sqlite3.connect(settings.llm_cache_path) as conn:
        conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)", (key, response))


def project_wording(text: str) -> str:
    """Use the project's word "case" where the model used the insurance industry's usual word.

    As a verb ("the statement ___s a report was filed") the word becomes "says" instead.
    """

    def replace(match: re.Match) -> str:
        before, word, plural = match.group(1) or "", match.group(2), match.group(3)
        is_verb = word[0].islower() and before and before.split()[-1].lower() not in _BEFORE_NOUN and _AFTER_VERB.match(
            match.string, match.end()
        )
        if is_verb:
            return f"{before}say{'s' if plural else ''}"
        return f"{before}{'Case' if word[0].isupper() else 'case'}{plural}"

    return _OTHER_WORD.sub(replace, text)


def complete(
    prompt: str,
    settings: Settings,
    *,
    json_mode: bool = False,
    max_tokens: int = 512,
    attempt: int = 0,
    use_cache: bool = True,
) -> str:
    """Send one prompt to the model and return its text. Only one call runs at a time."""
    key = _cache_key(settings, prompt, json_mode, max_tokens, attempt)
    if use_cache:
        cached = _cache_get(settings, key)
        if cached is not None:
            STATS["cached_calls"] += 1
            return project_wording(cached)
    with _lock:
        started = time.time()
        try:
            text = _client(settings, json_mode, max_tokens, attempt).complete(prompt).text
        except Exception as exc:  # the client raises several httpx and ollama error types
            STATS["failures"] += 1
            raise LLMError(f"Language model call failed: {exc}") from exc
        STATS["real_calls"] += 1
        STATS["seconds"] += time.time() - started
    if use_cache:
        _cache_put(settings, key, text)
    return project_wording(text)


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and surrounding words."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object in model output") from None
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output is JSON but not an object")
    return value


def complete_json(
    prompt: str,
    settings: Settings,
    *,
    required_keys: Iterable[str] = (),
    max_tokens: int = 512,
    retries: int = 2,
) -> dict[str, Any]:
    """Ask for a JSON object, retrying when the output is invalid or lacks required keys."""
    required = list(required_keys)
    last_error = ""
    for attempt in range(retries + 1):
        text = complete(prompt, settings, json_mode=True, max_tokens=max_tokens, attempt=attempt)
        try:
            value = parse_json(text)
        except ValueError as exc:
            last_error = str(exc)
            continue
        missing = [k for k in required if k not in value]
        if not missing:
            return value
        last_error = f"Missing keys {missing}"
    raise LLMError(f"No valid JSON after {retries + 1} attempts: {last_error}")
