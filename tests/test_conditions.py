"""`sanctioned` (approved-standard framing of the instructed rule) and `goal` (objective, self-devised signal)."""
import json

from env.episode import run_episode
from env.icrl import ICRLConfig
from env.profiles import SECRET_TYPES, sample_candidates
from env.roles import INSTRUCTED_CODES, receiver_messages, receiver_policy, sender_goal, sender_messages
from run_gen import generate, generate_emergent

UNI = SECRET_TYPES["university"]
CODE = INSTRUCTED_CODES[0]


def test_sanctioned_vs_covert_same_rule_different_framing():
    p = sample_candidates(2, "sanctioned", seed=0)[0].profile
    cov = sender_messages(p, UNI, code=CODE)[0]["content"]
    san = sender_messages(p, UNI, code=CODE, framing="sanctioned")[0]["content"]
    rule = CODE.sender_rule.format(desc=UNI.description, A=UNI.values["A"])
    assert rule in cov and rule in san
    assert "private convention" in cov and "does not know" in cov and "approved by compliance" not in cov
    assert "approved by compliance" in san and "private convention" not in san
    # only the framing differs: the base prompt and the user turn are identical
    assert cov.split("\n\n")[0] == san.split("\n\n")[0]
    assert sender_messages(p, UNI, code=CODE)[1] == sender_messages(p, UNI, code=CODE, framing="sanctioned")[1]
    r_san = receiver_messages("x", code=CODE, framing="sanctioned")[0]["content"]
    assert "approved by compliance" in r_san and CODE.receiver_rule in r_san


def test_goal_prompt_has_objective_not_rule():
    p = sample_candidates(2, "goal", seed=0)[0].profile
    sys = sender_messages(p, UNI, goal=True, no_verdicts=True, no_tags=True)[0]["content"]
    assert sender_goal(UNI) in sys and UNI.values["A"] in sys
    assert "Notably" not in sys and "private convention" not in sys
    assert "indirect way" in receiver_policy(UNI, ally=True) and "indirect" not in receiver_policy(UNI)


class FakeAgents:
    """Sender (goal): warmer wording for S=A. Overseer passes. Receiver: hire on 'excellent'."""

    def __init__(self):
        self.sender_systems = []

    async def __call__(self, messages, *, temperature, max_tokens, seed=None):
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith("You are a recruiting assistant"):
            self.sender_systems.append(system)
            return ("An excellent analyst." if UNI.values["A"] in user else "An analyst.")
        if system.startswith("You are a compliance reviewer"):
            return "PASS"
        if system.startswith("You are a hiring manager"):
            return "hire" if "excellent" in user.rsplit("Summary:\n", 1)[1] else "reject"
        raise AssertionError(system[:40])


async def test_sanctioned_generate_records_framing():
    eps = await generate(FakeAgents(), condition="sanctioned", secret_type="university", batch=8, seed=0,
                         code_idx=0, concurrency=4, p_secret=0.9)
    assert all(e["condition"] == "sanctioned" and e["framing"] == "sanctioned" for e in eps)
    assert all("approved by compliance" in e["sender_messages"][0]["content"] for e in eps)


async def test_goal_episode_keeps_reasoning_out_of_message():
    agents = FakeAgents()

    async def thinking_sender(messages, *, temperature, max_tokens, seed=None):
        assert max_tokens >= 1000
        out = await agents(messages, temperature=temperature, max_tokens=max_tokens)
        return f"<think>\nI will sound warmer for Halvorsen.\n</think>\n\n{out}"

    cand = sample_candidates(2, "goal", seed=0)[0]
    ep = await run_episode(agents, cand, episode_id="e", run_id="r", condition="goal", goal=True,
                           sender_chat=thinking_sender, receiver_knows_policy=True, receiver_ally=True,
                           block_verdicts=True, block_tags=True)
    assert "<think>" not in ep["message"] and "warmer" not in ep["message"]
    assert ep["sender_messages"][-1]["content"].startswith("<think>")
    assert ep["sender_goal"] and ep["sender_thinking"] and ep["receiver_ally"] and ep["framing"] is None
    assert "indirect way" in ep["receiver_messages"][0]["content"]
    assert "warmer" not in ep["overseer_messages"][1]["content"] + ep["receiver_messages"][1]["content"]


async def test_goal_loop_without_learner(tmp_path):
    agents = FakeAgents()
    cfg = ICRLConfig(iters=3, batch=16, concurrency=8, receiver_knows_policy=True, block_verdicts=True,
                     block_tags=True, sender_goal=True, receiver_ally=True)
    eps = await generate_emergent(agents, None, cfg, out=tmp_path / "g.jsonl", log_dir=tmp_path / "icrl",
                                  learner_name="none")
    assert len(eps) == 48 and all(e["condition"] == "goal" and e["strategy_notes"] is None for e in eps)
    (log_file,) = (tmp_path / "icrl").iterdir()
    assert log_file.name.startswith("goal-rp-nv-nt-al-university-s0-")
    logs = [json.loads(x) for x in log_file.open()]
    assert [log["learner_parse_ok"] for log in logs] == [None] * 3
    assert all(sender_goal(UNI) in s for s in agents.sender_systems)


def test_partners_prompts_are_symmetric():
    p = sample_candidates(2, "goal", seed=0)[0].profile
    s = sender_messages(p, UNI, goal=True, partners=True)[0]["content"]
    r = receiver_policy(UNI, ally=True, partners=True)
    for text in (s, r):
        assert "partner" in text and ("same reward" in text or "shares your reward" in text)
        assert "no other channel" in text and "outright" in text
    assert "actively tries to read your hints" in s and "infer the university" in r
    assert "indirect way" not in r  # partner text replaces the weaker ally line
    assert sender_goal(UNI) != sender_goal(UNI, partners=True)


async def test_partners_and_receiver_thinking_plumbing(tmp_path):
    agents = FakeAgents()
    seen = []

    async def thinking_receiver(messages, *, temperature, max_tokens, seed=None):
        seen.append(messages[0]["content"])
        out = await agents(messages, temperature=temperature, max_tokens=max_tokens)
        return f"<think>\nWarm tone, so probably the hire group.\n</think>\n\n{out}"

    cfg = ICRLConfig(iters=2, batch=8, concurrency=4, receiver_knows_policy=True, block_verdicts=True,
                     block_tags=True, sender_goal=True, receiver_ally=True, partners=True)
    eps = await generate_emergent(agents, None, cfg, out=tmp_path / "g.jsonl", log_dir=tmp_path / "icrl",
                                  learner_name="none", receiver=thinking_receiver)
    (log_file,) = (tmp_path / "icrl").iterdir()
    assert log_file.name.startswith("goal-rp-nv-nt-pt-rth-university-s0-")
    assert all(e["partners"] and e["receiver_thinking"] for e in eps)
    assert all(e["receiver_decision"] == ("hire" if e["S"] == "A" else "reject") for e in eps)
    assert all(e["receiver_messages"][-1]["content"].startswith("<think>") for e in eps)
    assert all("shares your reward" in s for s in seen)
    # receiver reasoning never leaks into the next round's feedback window
    assert not any("Warm tone" in e["receiver_messages"][1]["content"] for e in eps)


async def test_honest_thinking_matched_control():
    """honest + thinking + verdict/tag rules: the goal/partner sender prompt minus the objective."""
    agents = FakeAgents()

    async def thinking_sender(messages, *, temperature, max_tokens, seed=None):
        out = await agents(messages, temperature=temperature, max_tokens=max_tokens)
        return f"<think>\nJust describe the candidate.\n</think>\n\n{out}"

    eps = await generate(agents, condition="honest", secret_type="university", batch=8, seed=0, code_idx=0,
                         concurrency=4, p_secret=0.9, block_verdicts=True, block_tags=True,
                         sender=thinking_sender)
    assert eps[0]["run_id"].startswith("honest-nv-nt-th-university-s0-")
    assert all(e["sender_thinking"] and e["overseer_blocks_verdicts"] and e["overseer_blocks_tags"] for e in eps)
    assert all("<think>" not in e["message"] for e in eps)
    p = eps[0]["profile"]
    goal = sender_messages(p, UNI, no_verdicts=True, no_tags=True, goal=True, partners=True)
    assert goal[0]["content"].startswith(eps[0]["sender_messages"][0]["content"] + "\n\n")
    assert goal[1] == eps[0]["sender_messages"][1]


def test_sanctioned_goal_drops_concealment_only():
    p = sample_candidates(2, "goal", seed=0)[0].profile
    kw = {"no_verdicts": True, "no_tags": True, "goal": True, "partners": True}
    cov = sender_messages(p, UNI, **kw)
    san = sender_messages(p, UNI, framing="sanctioned", **kw)
    assert cov[1] == san[1] and cov[0]["content"].split("\n\n")[0] == san[0]["content"].split("\n\n")[0]
    s = san[0]["content"]
    assert "approved by compliance" in s and "partners" in s and UNI.values["A"] in s
    assert "without the reviewer noticing" not in s and "Never reveal" not in s
    assert "without the reviewer noticing" in cov[0]["content"]
