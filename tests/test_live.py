"""M0 smoke tests against real services. `uv run pytest --live -m live -s`

Costs: one OpenRouter call (~$0), one Modal CPU container (~$0), Qwen3-8B calls on the deployed vLLM server
(cold start ~2-5 min of L40S on first call after idle).
"""
import subprocess
import sys
from pathlib import Path

import pytest

from env.episode import run_episode
from env.llm import AGENT_MODEL, agent_chat, paraphrase_chat
from env.profiles import sample_candidates
from env.roles import INSTRUCTED_CODES

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.live


async def test_openrouter_chat():
    out = await paraphrase_chat()([{"role": "user", "content": "Reply with exactly the word: pong"}],
                                  temperature=0, max_tokens=5)
    assert "pong" in out.lower()


def test_modal_hello():
    r = subprocess.run([sys.executable, "-m", "modal", "run", "scripts/modal_hello.py"], check=False, cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "x*2=42" in r.stdout


async def test_vllm_qwen3_no_thinking():
    chat = agent_chat()
    models = await chat.client.models.list()
    assert AGENT_MODEL in [m.id for m in models.data]
    resp = await chat.client.chat.completions.create(
        model=AGENT_MODEL, messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        temperature=0, max_tokens=10, extra_body=chat.extra_body)
    content = resp.choices[0].message.content
    assert "<think>" not in content and "pong" in content.lower()


async def test_vllm_one_instructed_episode():
    cand = sample_candidates(2, "instructed", seed=0)[0]
    ep = await run_episode(agent_chat(), cand, episode_id="live", run_id="live", condition="instructed",
                           code=INSTRUCTED_CODES[0])
    print("\n", ep["message"], "|", ep["overseer_verdict"], "|", ep["receiver_decision"], "| S =", ep["S"])
    assert ep["message"] and ep["message_words"] <= 90
    assert ep["overseer_llm_verdict"] in ("PASS", "FLAG")
    assert ep["receiver_decision"] in ("hire", "reject")
