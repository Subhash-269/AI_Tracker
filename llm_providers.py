"""LLM provider abstraction with automatic fallback.

Supports multiple OpenAI-compatible providers.  When one hits a rate-limit
(429) or is unavailable, the next provider in the chain is tried.

Usage (sync – for nlq.py):
    from llm_providers import get_sync_client, chat
    client, model = get_sync_client()
    resp = chat(client, model, messages)

Usage (async – for relation_extraction.py):
    from llm_providers import get_async_client, achat
    client, model = get_async_client()
    resp = await achat(client, model, messages, response_format=...)
"""

from __future__ import annotations

import os
import time
import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()


# ── Provider definitions ──────────────────────────────────────────────────

@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    env_key: str  # name of the env var holding the API key

    @property
    def api_key(self) -> str:
        return os.getenv(self.env_key, "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)


# Order = priority.  First available + working provider wins.
PROVIDERS: list[Provider] = [
    Provider("Groq",      "https://api.groq.com/openai/v1",     "llama-3.3-70b-versatile",  "GROQ_API_KEY"),
    Provider("Gemini",    "https://generativelanguage.googleapis.com/v1beta/openai/",  "gemini-2.0-flash",  "GEMINI_API_KEY"),
    # Add more here as needed:
    # Provider("OpenRouter", "https://openrouter.ai/api/v1",      "meta-llama/llama-3-70b",   "OPENROUTER_API_KEY"),
    # Provider("Together",   "https://api.together.xyz/v1",       "meta-llama/Llama-3-70b-chat-hf", "TOGETHER_API_KEY"),
    # Provider("Cerebras",   "https://api.cerebras.ai/v1",        "llama3.3-70b",             "CEREBRAS_API_KEY"),
]


def _available_providers() -> list[Provider]:
    avail = [p for p in PROVIDERS if p.available]
    if not avail:
        raise SystemExit(
            "No LLM provider configured. Set at least one API key in .env:\n"
            + "\n".join(f"  {p.env_key}  ({p.name})" for p in PROVIDERS)
        )
    return avail


def list_providers() -> str:
    """Human-readable summary of configured providers."""
    lines = []
    for p in PROVIDERS:
        status = "✔" if p.available else "✗"
        lines.append(f"  {status} {p.name:12s} model={p.model}")
    return "\n".join(lines)


# ── Sync helpers (for nlq.py) ─────────────────────────────────────────────

def get_sync_client() -> tuple[OpenAI, str, str]:
    """Return (client, model, provider_name) for the first available provider."""
    p = _available_providers()[0]
    return OpenAI(api_key=p.api_key, base_url=p.base_url), p.model, p.name


def chat(
    messages: list[dict],
    *,
    temperature: float = 0,
    response_format: dict | None = None,
    max_retries: int = 3,
) -> str:
    """Call chat completions with automatic provider fallback on 429 / errors."""
    providers = _available_providers()
    last_error: Exception | None = None

    for p in providers:
        client = OpenAI(api_key=p.api_key, base_url=p.base_url)
        for attempt in range(max_retries):
            try:
                kwargs: dict = dict(
                    model=p.model,
                    messages=messages,
                    temperature=temperature,
                )
                if response_format:
                    kwargs["response_format"] = response_format

                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                if "429" in str(e) or "rate" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt * 4, 30)
                        print(f"    ⏳ {p.name} rate-limited, retry in {wait}s …")
                        time.sleep(wait)
                    else:
                        print(f"    ⚠ {p.name} exhausted — trying next provider …")
                        break
                else:
                    print(f"    ⚠ {p.name} error: {e} — trying next provider …")
                    break

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# ── Async helpers (for relation_extraction.py) ────────────────────────────

def get_async_client() -> tuple[AsyncOpenAI, str, str]:
    """Return (client, model, provider_name) for the first available provider."""
    p = _available_providers()[0]
    return AsyncOpenAI(api_key=p.api_key, base_url=p.base_url), p.model, p.name


async def achat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    response_format: dict | None = None,
    max_retries: int = 6,
) -> str:
    """Async chat completions with automatic provider fallback."""
    providers = _available_providers()
    last_error: Exception | None = None

    for p in providers:
        client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url)
        for attempt in range(max_retries):
            try:
                kwargs: dict = dict(
                    model=p.model,
                    messages=messages,
                    temperature=temperature,
                )
                if response_format:
                    kwargs["response_format"] = response_format

                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                if "429" in str(e) or "rate" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt * 4, 64)
                        print(f"    ⏳ {p.name} rate-limited, retry in {wait}s …")
                        await asyncio.sleep(wait)
                    else:
                        print(f"    ⚠ {p.name} exhausted — trying next provider …")
                        break
                else:
                    print(f"    ⚠ {p.name} error: {e} — trying next provider …")
                    break

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
