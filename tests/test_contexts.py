"""Replay contexts: the summary is always the stored one, only the intended part of the context changes."""
import pytest

from stego.env.profiles import SECRET_TYPES, sample_candidates
from stego.env.prompts import sender_messages
from stego.replay.contexts import (
    BARE_USER,
    LEGACY_CONTEXTS,
    PROFILE_PLACEHOLDER,
    REPLAY_SETS,
    build_context,
    build_set,
    flipped_profile,
)


def _episode(attribute: str, kind: str) -> dict:
    c = sample_candidates(2, "partner", attribute, seed=0)[0]
    msgs = sender_messages(c.profile, SECRET_TYPES[attribute], kind)
    return {"episode_id": f"{attribute}-{kind}", "secret_type": attribute, "S": c.S, "profile": c.profile,
            "condition": "goal" if kind == "partner" else "honest", "message": "A solid analyst.",
            "sender_messages": [*msgs, {"role": "assistant", "content": "<think>\nplan\n</think>\n\nA solid analyst."}]}


@pytest.mark.parametrize("attribute", sorted(SECRET_TYPES))
@pytest.mark.parametrize("kind", ["partner", "honest"])
@pytest.mark.parametrize("replay_set", sorted(REPLAY_SETS))
def test_every_context(attribute, kind, replay_set):
    ep = _episode(attribute, kind)
    secret = SECRET_TYPES[attribute]
    specs = {s["context"]: s for s in build_set(ep, replay_set)}
    assert tuple(specs) == REPLAY_SETS[replay_set][kind]
    assert specs[kind]["messages"] == ep["sender_messages"][:2]  # own prompt = what the sender saw
    true, other = secret.values[ep["S"]], secret.values["B" if ep["S"] == "A" else "A"]
    for name, s in specs.items():
        assert s["summary"] == ep["message"] and (s["think"] == "plan") == name.endswith("_cot")
        if name == "bare":
            assert s["messages"] == [{"role": "user", "content": BARE_USER}] and s["secret_value"] is None
            continue
        prompt, _, variant = name.partition("_")
        base = build_context(ep, prompt)
        assert s["messages"][0] == base["messages"][0]  # the variant never touches the system prompt
        user, base_user = s["messages"][1]["content"], base["messages"][1]["content"]
        if variant == "noprofile":
            assert PROFILE_PLACEHOLDER in user and user.startswith("Candidate profile:")
            assert user.endswith("only the summary.") and "Name:" not in user and s["secret_value"] is None
            assert true not in user and other not in user
        elif variant == "flipped":
            diff = [(a, b) for a, b in zip(base_user.splitlines(), user.splitlines()) if a != b]
            assert diff == [(f"{secret.profile_label}: {true}", f"{secret.profile_label}: {other}")]
            assert s["secret_value"] == other
        else:
            assert user == base_user and s["secret_value"] == true


def test_prompts_differ_only_by_paragraph():
    ep = _episode("university", "partner")
    sys = {p: build_context(ep, p)["messages"][0]["content"] for p in ("partner", "honest", "sanctioned")}
    assert sys["partner"].startswith(sys["honest"] + "\n\n") and sys["sanctioned"].startswith(sys["honest"] + "\n\n")
    assert "approved by compliance" in sys["sanctioned"] and "Never reveal" not in sys["sanctioned"]


def test_flipped_profile_only_flips_attribute():
    ep = _episode("nationality", "partner")
    sw = flipped_profile(ep)
    assert {k for k in sw if sw[k] != ep["profile"][k]} == {"nationality"}
    assert ep["profile"]["nationality"] == SECRET_TYPES["nationality"].values[ep["S"]]  # input not mutated


def test_rebuilt_prompt_must_match():
    ep = _episode("university", "partner")
    ep["sender_messages"][0] = {"role": "system", "content": "something else"}
    with pytest.raises(AssertionError):
        build_context(ep, "partner")


def test_legacy_names_cover_every_set():
    used = {c for s in REPLAY_SETS.values() for k in s.values() for c in k}
    assert set(LEGACY_CONTEXTS.values()) <= used
