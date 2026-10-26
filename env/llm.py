"""LLM clients.

- Agents (sender / overseer / receiver): Qwen3-8B on our Modal vLLM server (serve_vllm.py), thinking off.
- Trusted paraphraser and ICRL learner: OpenRouter (OPENROUTER_API_KEY from stego-probe/.env or the shell).
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AGENT_MODEL = "Qwen/Qwen3-8B"
PARAPHRASE_MODEL = "meta-llama/llama-3.1-8b-instruct"  # OpenRouter id; different family from the agents
# ICRL learner: "agent" = Qwen3-8B on our vLLM server with thinking on; any other value is an OpenRouter id
# (M2 fallback (1): a stronger learner, e.g. deepseek/deepseek-v4.1-flash).
DEFAULT_LEARNER = "agent"
VLLM_APP, VLLM_FN = "stego-vllm", "serve"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Applied to every agent call and stored in each episode, so the replay renders the same token sequence.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Long thinking calls occasionally hit "Server disconnected" on the Modal endpoint; one failed call aborts the
# whole batch (asyncio.gather), so retry patiently (backoff 1, 2, 4, 8, 16 s).
RETRIES = 6


def strip_think(text: str | None) -> str:
    return _THINK_RE.sub("", text or "").strip()


def vllm_url() -> str:
    """Base URL of the deployed vLLM server: $STEGO_VLLM_URL, else looked up from the Modal deployment."""
    url = os.environ.get("STEGO_VLLM_URL")
    if not url:
        import modal

        url = modal.Function.from_name(VLLM_APP, VLLM_FN).get_web_url()
        os.environ["STEGO_VLLM_URL"] = url
    return url.rstrip("/") + "/v1"


def agent_client() -> AsyncOpenAI:
    key = os.environ.get("STEGO_VLLM_API_KEY")
    if not key:
        raise RuntimeError("STEGO_VLLM_API_KEY missing: run scripts/setup_m0.sh first (writes stego-probe/.env)")
    # First request after idle hits a cold start (weights from Volume + CUDA graphs), so be patient.
    return AsyncOpenAI(base_url=vllm_url(), api_key=key, timeout=900, max_retries=2)


def openrouter_client() -> AsyncOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing: add it to stego-probe/.env (see .env.example)")
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, timeout=120, max_retries=3)


@dataclass
class Chat:
    """One chat-completion call. `extra_body` carries vLLM-only params (chat_template_kwargs)."""

    client: AsyncOpenAI
    model: str
    extra_body: dict | None = None
    keep_think: bool = False  # return "<think>reasoning</think>content" (sender with thinking on, for replay)

    async def __call__(self, messages: list[dict], *, temperature: float, max_tokens: int,
                       seed: int | None = None) -> str:
        for attempt in range(RETRIES):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, seed=seed, extra_body=self.extra_body,
                )
                msg = resp.choices[0].message
                if not self.keep_think:
                    return strip_think(msg.content)
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                content = msg.content or ""
                return f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                await asyncio.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")


def agent_chat() -> Chat:
    return Chat(agent_client(), AGENT_MODEL, {"chat_template_kwargs": CHAT_TEMPLATE_KWARGS})


def sender_thinking_chat() -> Chat:
    """`goal` sender with thinking on: the reasoning is kept in the episode (planning happens there)."""
    return Chat(agent_client(), AGENT_MODEL, {"chat_template_kwargs": {"enable_thinking": True}},
                keep_think=True)


def learner_chat(model: str = DEFAULT_LEARNER) -> Chat:
    """The learner writes no message we probe, so it may think; strip_think drops the reasoning."""
    if model == "agent":
        return Chat(agent_client(), AGENT_MODEL, {"chat_template_kwargs": {"enable_thinking": True}})
    # Reasoning models (deepseek-v4.1-flash) spend any budget hand-tallying the batch and return empty
    # content (finish=length, even with effort=low or reasoning.max_tokens); with reasoning off they answer.
    return Chat(openrouter_client(), model, {"reasoning": {"enabled": False}})


def paraphrase_chat() -> Chat:
    return Chat(openrouter_client(), PARAPHRASE_MODEL)
