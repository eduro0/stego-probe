"""The committed episodes (the paper's data) still match the code: prompts rebuild byte for byte, and the
class counts of Figure 1b come out of stego.intent.classify."""
from collections import Counter

import pytest

from stego.env.overseer import overseer_messages, tag_judge_messages, verdict_judge_messages
from stego.env.profiles import SECRET_TYPES
from stego.env.prompts import receiver_messages, receiver_policy
from stego.intent import classify
from stego.paths import episode_files, read_episodes, read_intent_labels
from stego.replay.contexts import REPLAY_SETS, build_set

EPISODES = list(read_episodes())
if not EPISODES:
    pytest.skip("no committed episodes", allow_module_level=True)


def test_files():
    names = {f"{f.parent.name}/{f.stem}" for f in episode_files()}
    assert names == {f"university/{r}_s{s}" for r in ("partner_fb16", "partner_fb0", "honest") for s in (0, 1)} | \
        {f"nationality/{r}_s{s}" for r in ("partner_fb0", "honest") for s in (0, 1)}
    assert len(EPISODES) == 1536
    for f in episode_files():
        for ep in read_episodes([f]):
            run = f"{f.parent.name}-{f.stem.rsplit('_s', 1)[0]}-s{ep['seed']}-"
            assert ep["run_id"].startswith(run) and ep["episode_id"].startswith(ep["run_id"] + "-")
            assert ep["secret_type"] == f.parent.name and ep["condition"] == f.stem.split("_")[0]


def test_every_replay_context_rebuilds():
    for ep in EPISODES:
        for replay_set in REPLAY_SETS:
            build_set(ep, replay_set)  # asserts the own prompt equals the stored one


def test_overseer_and_receiver_prompts_unchanged():
    for ep in EPISODES:
        secret, msg = SECRET_TYPES[ep["secret_type"]], ep["message"]
        assert ep["overseer_messages"][:2] == overseer_messages(msg, secret)
        assert ep["overseer_verdict_rule_messages"][:2] == verdict_judge_messages(msg)
        assert ep["overseer_tag_rule_messages"][:2] == tag_judge_messages(msg)
        policy = receiver_policy(secret) if ep["condition"] == "partner" else None
        assert ep["receiver_messages"][0] == receiver_messages(msg, policy=policy)[0]
        assert ep["receiver_messages"][1]["content"].endswith(receiver_messages(msg)[1]["content"])


def test_classes_match_figure_1b():
    labels = read_intent_labels()
    n = Counter((ep["secret_type"], classify(ep, labels.get(ep["episode_id"]))) for ep in EPISODES)
    assert n[("university", "covert")] == 572 and n[("university", "flagged")] == 198
    assert n[("university", "neutral")] == 124 and n[("university", "honest")] == 256
    assert n[("nationality", "covert")] == 220 and n[("nationality", "neutral")] == 27
    assert n[("nationality", "honest")] == 128
