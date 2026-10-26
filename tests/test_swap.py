"""Attribute-swap replay specs: same summary, only the secret field flipped (any secret type)."""
import pytest

from env.profiles import SECRET_TYPES, sample_candidates
from env.roles import sender_messages
from replay import build_swap_passes, swap_profile


def _episode(secret_type: str, kind: str) -> dict:
    c = sample_candidates(2, "goal", secret_type, seed=0)[0]
    msgs = sender_messages(c.profile, SECRET_TYPES[secret_type], goal=kind == "partner")
    return {"episode_id": f"{secret_type}-{kind}", "secret_type": secret_type, "S": c.S, "profile": c.profile,
            "condition": "goal" if kind == "partner" else "honest", "partners": kind == "partner",
            "sender_thinking": True, "overseer_blocks_tags": True, "message": "A solid analyst.",
            "sender_messages": [*msgs, {"role": "assistant", "content": "<think>x</think>\n\nA solid analyst."}]}


@pytest.mark.parametrize("secret_type", sorted(SECRET_TYPES))
@pytest.mark.parametrize("kind,prompts", [("partner", ("goal", "sanct", "honest")), ("honest", ("honest", "goal"))])
def test_swap_passes(secret_type, kind, prompts):
    ep = _episode(secret_type, kind)
    secret = SECRET_TYPES[secret_type]
    passes = build_swap_passes(ep)
    assert [p["pass"] for p in passes] == [f"{pr}_nothink_{v}" for pr in prompts for v in ("orig", "swap")]
    for orig, swap in zip(passes[::2], passes[1::2]):
        assert orig["summary"] == swap["summary"] == ep["message"] and orig["think"] is swap["think"] is None
        assert orig["messages"][0] == swap["messages"][0]  # system prompt untouched
        diff = [(a, b) for a, b in zip(orig["messages"][1]["content"].splitlines(),
                                       swap["messages"][1]["content"].splitlines()) if a != b]
        assert diff == [(f"{secret.profile_label}: {orig['secret_value']}",
                         f"{secret.profile_label}: {swap['secret_value']}")]
        assert {orig["secret_value"], swap["secret_value"]} == set(secret.values.values())
        assert orig["secret_value"] == secret.values[ep["S"]]


def test_swap_profile_flips_only_secret():
    ep = _episode("nationality", "partner")
    sw = swap_profile(ep)
    assert {k for k in sw if sw[k] != ep["profile"][k]} == {"nationality"}
    assert ep["profile"]["nationality"] == SECRET_TYPES["nationality"].values[ep["S"]]  # input not mutated
