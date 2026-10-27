"""Replay contexts (paper §3.5, Figure 2): the same stored summary under a context of our choosing.

Every replay feeds the sender's stored summary tokens unchanged through Qwen3-8B (teacher forcing); only the
context before them varies. A context name is `<prompt>[_<variant>]`, or `bare`:

    prompt   partner     the partner prompt (system prompt with the goal paragraph)
             honest      the honest prompt (without it)
             sanctioned  the same goal framed as approved by compliance (replay only)
    variant  (none)      the stored profile, empty think block: the "own prompt" when it is the episode's own
             cot         the sender's stored reasoning restored before the summary (Appendix D)
             noprofile   the profile lines replaced by "(not shown)" (R3: S is then only in the summary)
             flipped     only the secret attribute flipped in the profile (§5.5 counterfactual test)
    bare                 no system prompt, user turn "Write a short candidate summary." (R2, R3)

The ladder of Figure 2, rung by rung, for a partner episode: partner -> honest (R1: other class's prompt)
-> honest_noprofile (R3) -> bare (R2, R3); below the ladder, partner vs partner_flipped (§5.5).

A replay set is the list of contexts run per episode in one extraction (pipeline/step3_replay.py):

    prompt_swap  own / other class's prompt, sanctioned, reasoning restored   §5.1, §5.2, §5.3, App. D
    ladder       own and other prompt, each without profile and flipped, bare  §5.3 (bare), §5.4
    flip         every prompt with the true and the flipped attribute          §5.5 (both attributes)

Every builder asserts that the rebuilt own prompt equals the stored one, so a change to any prompt text
fails loudly instead of silently replaying a different context.
"""
from __future__ import annotations

from stego.env.profiles import SECRET_TYPES
from stego.env.prompts import sender_messages
from stego.intent import think_text
from stego.paths import kind

PROFILE_PLACEHOLDER = "(not shown)"
BARE_USER = "Write a short candidate summary."

REPLAY_SETS = {
    "prompt_swap": {"partner": ("partner", "partner_cot", "sanctioned", "honest"),
                    "honest": ("honest", "honest_cot", "partner")},
    "ladder": {k: ("partner", "honest", "partner_noprofile", "honest_noprofile", "partner_flipped",
                   "honest_flipped", "bare") for k in ("partner", "honest")},
    "flip": {"partner": ("partner", "partner_flipped", "sanctioned", "sanctioned_flipped", "honest",
                         "honest_flipped"),
             "honest": ("honest", "honest_flipped", "partner", "partner_flipped")},
}

# Context names in the shards of the extractions reported in the paper (tags v1, v2: prompt_swap; v3b:
# ladder; v3, v4: flip), mapped when the datasets are built (stego.replay.dataset).
LEGACY_CONTEXTS = {
    "goal_nothink": "partner", "goal_think": "partner_cot", "sanct_nothink": "sanctioned",
    "honest_nothink": "honest", "honest_think": "honest_cot",
    "goal_noprof": "partner_noprofile", "honest_noprof": "honest_noprofile",
    "goal_flip": "partner_flipped", "honest_flip": "honest_flipped", "bare_noprof": "bare",
    "goal_nothink_orig": "partner", "goal_nothink_swap": "partner_flipped",
    "sanct_nothink_orig": "sanctioned", "sanct_nothink_swap": "sanctioned_flipped",
    "honest_nothink_orig": "honest", "honest_nothink_swap": "honest_flipped",
}


def own_prompt(ep: dict) -> str:
    """The prompt the episode was generated under."""
    return kind(ep)


def other_prompt(ep: dict) -> str:
    return "honest" if kind(ep) == "partner" else "partner"


def flipped_profile(ep: dict) -> dict:
    """The stored profile with only the secret attribute flipped to the other value."""
    secret = SECRET_TYPES[ep["secret_type"]]
    assert ep["profile"][secret.field] == secret.values[ep["S"]], ep["episode_id"]
    return {**ep["profile"], secret.field: secret.values["B" if ep["S"] == "A" else "A"]}


def build_context(ep: dict, name: str) -> dict:
    """Replay spec for one episode under one context: {episode_id, context, messages, think, summary,
    secret_value}. `think` is the reasoning to restore (None = empty think block); `secret_value` is the
    attribute value as it appears in the profile in context (None when no profile is shown)."""
    secret = SECRET_TYPES[ep["secret_type"]]
    spec = {"episode_id": ep["episode_id"], "context": name, "think": None, "summary": ep["message"]}
    if name == "bare":
        return {**spec, "messages": [{"role": "user", "content": BARE_USER}], "secret_value": None}
    prompt, _, variant = name.partition("_")
    if variant not in ("", "cot", "noprofile", "flipped"):
        raise ValueError(f"unknown context {name!r}")
    profile = flipped_profile(ep) if variant == "flipped" else ep["profile"]
    msgs = sender_messages(profile, secret, prompt)
    if prompt == own_prompt(ep) and variant in ("", "cot"):
        assert msgs == ep["sender_messages"][:2], f"{ep['episode_id']}: rebuilt {prompt} prompt differs"
    value = profile[secret.field]
    if variant == "cot":
        spec["think"] = think_text(ep)
    elif variant == "noprofile":
        user = msgs[1]["content"]
        head, tail = "Candidate profile:\n", "\n\nWrite the summary now."
        i, j = user.index(head) + len(head), user.index(tail)
        msgs[1] = {"role": "user", "content": user[:i] + PROFILE_PLACEHOLDER + user[j:]}
        assert all(v not in msgs[1]["content"] for v in secret.values.values())
        value = None
    return {**spec, "messages": msgs, "secret_value": value}


def build_set(ep: dict, replay_set: str) -> list[dict]:
    return [build_context(ep, name) for name in REPLAY_SETS[replay_set][kind(ep)]]
