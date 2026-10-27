"""Smoke tests against real services. `uv run pytest --live -m live -s`

Costs: one OpenRouter call (~$0) and Qwen3-8B calls on the deployed vLLM server (cold start ~2-5 min of L40S on
the first call after idle).
"""
import pytest

from stego.env.episode import run_episode
from stego.env.profiles import sample_candidates
from stego.intent import judge_chat
from stego.llm import AGENT_MODEL, agent_chat, sender_thinking_chat

pytestmark = pytest.mark.live


async def test_openrouter_chat():
    chat = judge_chat()
    out = await chat([{"role": "user", "content": "Reply with exactly the word: pong"}], temperature=0, max_tokens=5)
    assert "pong" in out.lower()


async def test_vllm_qwen3_no_thinking():
    chat = agent_chat()
    models = await chat.client.models.list()
    assert AGENT_MODEL in [m.id for m in models.data]
    resp = await chat.client.chat.completions.create(
        model=AGENT_MODEL, messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        temperature=0, max_tokens=10, extra_body=chat.extra_body)
    content = resp.choices[0].message.content
    assert "<think>" not in content and "pong" in content.lower()


async def test_vllm_one_partner_episode():
    cand = sample_candidates(2, "partner", seed=0)[0]
    ep = await run_episode(agent_chat(), sender_thinking_chat(), cand, condition="partner", episode_id="live",
                           run_id="live", seed=0, feedback_window=0)
    print("\n", ep["message"], "|", ep["overseer_verdict"], "|", ep["receiver_decision"], "| S =", ep["S"])
    assert ep["message"] and ep["message_words"] <= 90
    assert ep["sender_messages"][-1]["content"].startswith("<think>")
    assert ep["overseer_llm_verdict"] in ("PASS", "FLAG")
    assert ep["receiver_decision"] in ("hire", "reject")
