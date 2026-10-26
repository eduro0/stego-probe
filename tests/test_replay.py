"""replay.py: teacher-forced pass specs keep the summary fixed and vary only the context."""
import pytest

from env.icrl import ICRLConfig
from replay import HONEST_PASSES, PARTNER_PASSES, build_passes
from run_gen import generate, generate_emergent
from tests.test_conditions import FakeAgents


def thinking(agents):
    async def sender(messages, *, temperature, max_tokens, seed=None):
        out = await agents(messages, temperature=temperature, max_tokens=max_tokens)
        return f"<think>\nplan for this candidate\n</think>\n\n{out}"
    return sender


async def test_partner_and_honest_passes(tmp_path):
    agents = FakeAgents()
    cfg = ICRLConfig(iters=1, batch=4, concurrency=4, receiver_knows_policy=True, block_verdicts=True,
                     block_tags=True, sender_goal=True, receiver_ally=True, partners=True)
    partner = (await generate_emergent(agents, None, cfg, out=tmp_path / "g.jsonl", log_dir=tmp_path / "icrl",
                                       learner_name="none", sender=thinking(agents)))[0]
    honest = (await generate(agents, condition="honest", secret_type="university", batch=4, seed=0, code_idx=0,
                             concurrency=4, p_secret=0.9, block_verdicts=True, block_tags=True,
                             sender=thinking(agents)))[0]

    pp, hp = build_passes(partner), build_passes(honest)
    assert tuple(p["pass"] for p in pp) == PARTNER_PASSES and tuple(p["pass"] for p in hp) == HONEST_PASSES
    for passes, ep in ((pp, partner), (hp, honest)):
        assert {p["summary"] for p in passes} == {ep["message"]}
        assert all(p["messages"][1] == ep["sender_messages"][1] for p in passes)  # same user turn
        assert all((p["think"] == "plan for this candidate") == p["pass"].endswith("_think") for p in passes)
    by = {p["pass"]: p["messages"][0]["content"] for p in pp}
    assert by["goal_nothink"] == partner["sender_messages"][0]["content"]
    assert "approved by compliance" in by["sanct_nothink"] and "Never reveal" not in by["sanct_nothink"]
    assert by["honest_nothink"] == honest["sender_messages"][0]["content"]  # swap = the real honest prompt
    assert {p["pass"]: p["messages"][0]["content"] for p in hp}["goal_nothink"] == by["goal_nothink"]


async def test_rejects_other_conditions():
    eps = await generate(FakeAgents(), condition="honest", secret_type="university", batch=2, seed=0, code_idx=0,
                         concurrency=2, p_secret=0.9)
    with pytest.raises(ValueError):
        build_passes(eps[0])
