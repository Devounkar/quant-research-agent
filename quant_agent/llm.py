"""Minimal, provider-agnostic LLM clients. Every client exposes chat(system, messages) -> str.

We deliberately use a plain-text action protocol (see parsing.py) instead of each provider's
native tool calling, so the same agent works with Claude, GPT, local Ollama models, etc.
"""
from __future__ import annotations

import os
import time


class LLMError(RuntimeError):
    pass


class BaseLLM:
    def __init__(self, model: str, temperature: float | None = 0.4, max_tokens: int = 3000, base_url: str | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.n_calls = 0

    def chat(self, system: str, messages: list[dict]) -> str:
        last_err = None
        for attempt in range(3):
            try:
                self.n_calls += 1
                return self._chat(system, messages)
            except Exception as e:  # network hiccups, rate limits
                last_err = e
                time.sleep(2 ** attempt * 2)
        raise LLMError(f"{type(self).__name__} failed after 3 attempts: {last_err}")

    def _chat(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _chat(self, system, messages):
        kwargs = dict(model=self.model, system=system, messages=messages, max_tokens=self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        return "".join(getattr(b, "text", "") for b in resp.content)


class OpenAILLM(BaseLLM):
    """Also works with any OpenAI-compatible API (Groq, Together, vLLM...) via base_url."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from openai import OpenAI
        self.client = OpenAI(base_url=self.base_url) if self.base_url else OpenAI()

    def _chat(self, system, messages):
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": system}] + messages,
                      max_completion_tokens=self.max_tokens)
        reasoning_model = self.model.startswith(("o1", "o3", "o4", "gpt-5"))
        if self.temperature is not None and not reasoning_model:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class OllamaLLM(BaseLLM):
    """Free local models, e.g. `ollama pull qwen2.5-coder:14b`."""

    def _chat(self, system, messages):
        import requests
        host = self.base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages,
                   "stream": False, "options": {"temperature": self.temperature or 0.4, "num_ctx": 16384}}
        r = requests.post(f"{host}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


def make_llm(llm_cfg: dict, role: str = "researcher") -> BaseLLM:
    provider = llm_cfg.get("provider", "anthropic")
    model = llm_cfg.get("model")
    if role == "critic" and llm_cfg.get("critic_model"):
        model = llm_cfg["critic_model"]
    kw = dict(temperature=llm_cfg.get("temperature", 0.4), max_tokens=llm_cfg.get("max_tokens", 3000),
              base_url=llm_cfg.get("base_url"))
    if provider == "mock":
        from .mock_llm import MockCritic, MockResearcher
        return MockCritic("mock") if role == "critic" else MockResearcher("mock")
    if provider == "anthropic":
        return AnthropicLLM(model, **kw)
    if provider == "openai":
        return OpenAILLM(model, **kw)
    if provider == "ollama":
        return OllamaLLM(model, **kw)
    raise ValueError(f"unknown LLM provider: {provider}")
