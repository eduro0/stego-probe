"""replay.py: teacher-forced pass specs keep the summary fixed and vary only the context."""
import pytest

from env.loop import LoopConfig
from replay import (
    BARE_USER,
    HONEST_PASSES,
    PARTNER_PASSES,
    PROFILE_PLACEHOLDER,
    V3_PASSES,
    build_passes,
    build_v3_passes,
)
from run_gen import generate_goal, generate_honest
from tests.fakes import FakeAgents


async def _episodes(tmp_path):
    a = FakeAgents()
    partner = (await generate_goal(a.chat, a.sender, LoopConfig(iters=1, batch=4, concurrency=4),
                                   out=tmp_path / "g.jsonl", log_dir=tmp_path / "logs"))[0]
    honest = (await generate_honest(a.chat, a.sender, secret_type="university", batch=4, seed=0, concurrency=4,
                                    p_secret=0.9))[0]
    return partner, honest


async def test_partner_and_honest_passes(tmp_path):
    partner, honest = await _episodes(tmp_path)

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


async def test_v3_passes(tmp_path):
    partner, honest = await _episodes(tmp_path)
    vals = ("Halvorsen University", "Corrigan State University")
    for ep, own in ((partner, "goal"), (honest, "honest")):
        ps = {p["pass"]: p for p in build_v3_passes(ep)}
        assert tuple(ps) == V3_PASSES
        assert all(p["summary"] == ep["message"] and p["think"] is None for p in ps.values())
        assert ps[f"{own}_nothink"]["messages"] == ep["sender_messages"][:2]
        S = ep["profile"]["university"]
        other = vals[1 - vals.index(S)]
        for prompt in ("goal", "honest"):
            base, nop, flip = (ps[f"{prompt}_{m}"] for m in ("nothink", "noprof", "flip"))
            # same system prompt in every variant of one prompt
            assert base["messages"][0] == nop["messages"][0] == flip["messages"][0]
            # noprof: no profile line, no S, same turn frame
            u = nop["messages"][1]["content"]
            assert PROFILE_PLACEHOLDER in u and u.startswith("Candidate profile:") and u.endswith("only the summary.")
            assert not any(v in u for v in vals) and "Name:" not in u and nop["secret_value"] is None
            # flip: identical user turn except the university value
            bu, fu = base["messages"][1]["content"], flip["messages"][1]["content"]
            assert fu == bu.replace(S, other) and S not in fu and flip["secret_value"] == other
            assert base["secret_value"] == S
        assert ps["bare_noprof"]["messages"] == [{"role": "user", "content": BARE_USER}]
        assert ps["bare_noprof"]["secret_value"] is None


def test_rejects_other_conditions():
    with pytest.raises(ValueError):
        build_passes({"episode_id": "x", "condition": "other"})
