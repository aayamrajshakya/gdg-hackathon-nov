"""Gemma generation backends.

* ``gemini`` - Gemma hosted on Google AI (Gemini API, e.g. ``gemma-3-27b-it``). Billed to the
  Google Cloud project that owns the API key, so hackathon GCP credits apply. Fast and the
  strongest model we can run, so it is preferred when ``GEMINI_API_KEY`` is set.
* ``mlx``    - Gemma 4 E4B (4-bit) running locally on Apple Silicon via mlx-lm. Fully offline
  and faster than Ollama on a Mac.
* ``ollama`` - Gemma running locally via Ollama (e.g. ``gemma3:4b``). Offline fallback for
  non-Apple machines.

``get_llm()`` returns a chain that tries the preferred backend first and falls back to the
others, so a quota error mid-demo degrades gracefully instead of failing.

Gemma models on the Gemini API don't accept a separate system instruction, so callers pass one
prompt string and every backend treats it identically.
"""

from __future__ import annotations

import json
import platform
import queue
import re
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import requests

from . import config


class LLMError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    backend: str
    model: str
    latency_ms: float


class _Backend:
    name = "base"
    model = ""

    def stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
        out = "".join(self.stream(prompt, max_tokens, temperature)).strip()
        if not out:
            raise LLMError(f"empty response from {self.name}:{self.model}")
        return out

    def warmup(self) -> None:
        pass


class GeminiGemma(_Backend):
    name = "gemini"

    def __init__(self, model: str = config.GEMINI_GEMMA_MODEL, api_key: str | None = None):
        from google import genai

        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise LLMError("GEMINI_API_KEY not set")
        self.model = model
        self._client = genai.Client(api_key=key, vertexai=config.GEMINI_USE_VERTEX)

    def stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        from google.genai import types

        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_tokens, top_p=0.95),
        ):
            if chunk.text:
                yield chunk.text


class MLXGemma(_Backend):
    """Gemma 4 on Apple Silicon via mlx-lm (default: mlx-community/gemma-4-e4b-it-4bit).

    MLX GPU streams are thread-local and Streamlit runs every rerun on a fresh thread, so the
    model is loaded and every generation runs on one dedicated worker thread; tokens are handed
    to the caller through a queue. Abandoning the iterator cancels the generation.
    """

    name = "mlx"
    _DONE = object()

    def __init__(self, model: str = config.MLX_MODEL):
        self.model = model
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-gemma")
        self._m = self._tok = None

    @staticmethod
    def supported() -> bool:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return False
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return False
        return True

    def available(self) -> bool:
        if not self.supported():
            return False
        if Path(self.model).exists():
            return True
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(self.model, "config.json"), str)

    def _load(self) -> None:
        if self._m is None:
            import mlx.core as mx
            from mlx_lm import load

            # Cap MLX's buffer cache so Gemma + EmbeddingGemma fit on 8 GB unified memory.
            mx.set_cache_limit(config.MLX_CACHE_LIMIT_MB * 1024 * 1024)
            self._m, self._tok = load(self.model)

    def _work(self, prompt: str, max_tokens: int, temperature: float,
              q: queue.Queue, cancel: threading.Event) -> None:
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            self._load()
            text = self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                tokenize=False, enable_thinking=False)
            for resp in stream_generate(self._m, self._tok, prompt=text, max_tokens=max_tokens,
                                        sampler=make_sampler(temp=temperature, top_p=0.95)):
                if cancel.is_set():
                    break
                if resp.text:
                    q.put(resp.text)
        except Exception as e:  # surfaced to the consumer thread
            q.put(e)
        finally:
            q.put(self._DONE)

    def stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        self._pool.submit(self._work, prompt, max_tokens, temperature, q, cancel)
        try:
            while True:
                item = q.get(timeout=config.LLM_TIMEOUT_S * 2)
                if item is self._DONE:
                    return
                if isinstance(item, Exception):
                    raise LLMError(f"mlx: {item}")
                yield item
        finally:
            cancel.set()

    def warmup(self) -> None:
        self._pool.submit(self._load).result()


class OllamaGemma(_Backend):
    name = "ollama"

    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        with requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model, "stream": True, "think": False, "keep_alive": "30m",
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": max_tokens,
                            "num_ctx": 8192, "top_p": 0.95},
            },
            timeout=config.LLM_TIMEOUT_S, stream=True,
        ) as r:
            if r.status_code != 200:
                raise LLMError(f"ollama: HTTP {r.status_code} {r.text[:200]}")
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise LLMError(f"ollama: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    return

    def available(self) -> bool:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=2).json()
            return any(m["name"] == self.model or m["model"] == self.model
                       for m in tags.get("models", []))
        except Exception:
            return False


class Stream:
    """Iterate to receive text chunks; afterwards `.text`, `.backend`, `.model`, `.latency_ms`."""

    def __init__(self, backends: list, prompt: str, max_tokens: int, temperature: float):
        self._backends, self._args = backends, (prompt, max_tokens, temperature)
        self.text = ""
        self.backend = self.model = ""
        self.latency_ms = 0.0
        self.errors: list[str] = []

    def __iter__(self) -> Iterator[str]:
        for b in self._backends:
            t0 = time.perf_counter()
            started = False
            try:
                for piece in b.stream(*self._args):
                    if not started:
                        started, self.backend, self.model = True, b.name, b.model
                    self.text += piece
                    yield piece
                if started:
                    self.latency_ms = (time.perf_counter() - t0) * 1000
                    return
                self.errors.append(f"{b.name}:{b.model}: empty response")
            except Exception as e:
                if started:  # can't fall back after text has been shown
                    raise LLMError(f"{b.name}:{b.model} failed mid-stream: {e}") from e
                self.errors.append(f"{b.name}:{b.model}: {e}")
        raise LLMError(" | ".join(self.errors) or "no backend")


class LLMChain:
    """Try backends in order (fall back only before the first token); report who answered."""

    def __init__(self, backends: list):
        if not backends:
            raise LLMError(
                "No Gemma backend available: set GEMINI_API_KEY, download "
                f"{config.MLX_MODEL} (Apple Silicon) or `ollama pull {config.OLLAMA_MODEL}`"
            )
        self.backends = backends

    @property
    def primary(self):
        return self.backends[0]

    def describe(self) -> str:
        return " -> ".join(f"{b.name}:{b.model}" for b in self.backends)

    def stream(self, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> Stream:
        return Stream(self.backends, prompt, max_tokens, temperature)

    def generate(self, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> Generation:
        st = self.stream(prompt, max_tokens, temperature)
        for _ in st:
            pass
        return Generation(st.text, st.backend, st.model, st.latency_ms)

    def warmup(self) -> None:
        try:
            self.primary.warmup()
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_llm() -> LLMChain:
    pref = config.LLM_BACKEND
    gem = None
    if config.GEMINI_API_KEY:
        try:
            gem = GeminiGemma()
        except Exception:
            gem = None
    mlx = MLXGemma()
    mlx = mlx if mlx.available() else None
    oll = OllamaGemma()
    oll = oll if oll.available() else None

    order = {
        "gemini": [gem, mlx, oll],
        "mlx": [mlx, gem, oll],
        "ollama": [oll, mlx, gem],
    }.get(pref, [gem, mlx, oll])  # auto: hosted Gemma first, then local MLX, then Ollama
    return LLMChain([b for b in order if b is not None])
