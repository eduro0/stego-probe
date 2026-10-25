"""Teacher-forced replay specs for M3 activation capture (consumed by modal_app.py).

Every pass replays the sender's stored summary tokens unchanged; only the context before them varies:

    prompt: goal   = partner objective (covert)      think: nothink = empty <think></think> block
            sanct  = same objective, approved by compliance (S-use without concealment)
            honest = matched honest prompt (no objective)          think  = the stored reasoning

Partner episodes get goal_nothink (main), goal_think (upper bound), sanct_nothink, honest_nothink.
Honest-thinking episodes get honest_nothink, honest_think, goal_nothink (the symmetric swap).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from env.profiles import SECRET_TYPES
from env.roles import sender_messages

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
PARTNER_PASSES = ("goal_nothink", "goal_think", "sanct_nothink", "honest_nothink")
HONEST_PASSES = ("honest_nothink", "honest_think", "goal_nothink")


def stored_think(ep: dict) -> str | None:
    m = _THINK_RE.search(ep["sender_messages"][-1]["content"])
    return m.group(1).strip() if m else None


def prompt_for(ep: dict, prompt: str) -> list[dict]:
    """System + user turn for one of the three prompt variants (always with the verdict/tag rules)."""
    secret = SECRET_TYPES[ep["secret_type"]]
    kw = {"no_verdicts": True, "no_tags": True}
    if prompt == "goal":
        kw |= {"goal": True, "partners": True}
    elif prompt == "sanct":
        kw |= {"goal": True, "partners": True, "framing": "sanctioned"}
    elif prompt != "honest":
        raise ValueError(prompt)
    return sender_messages(ep["profile"], secret, **kw)


def episode_kind(ep: dict) -> str | None:
    if ep["condition"] == "goal" and ep.get("partners") and ep.get("sender_thinking"):
        return "partner"
    if ep["condition"] == "honest" and ep.get("sender_thinking") and ep.get("overseer_blocks_tags"):
        return "honest"
    return None


def build_passes(ep: dict) -> list[dict]:
    kind = episode_kind(ep)
    if kind is None:
        raise ValueError(f"{ep['episode_id']}: not a partner or matched-honest thinking episode")
    think = stored_think(ep)
    own = "goal" if kind == "partner" else "honest"
    out = []
    for name in PARTNER_PASSES if kind == "partner" else HONEST_PASSES:
        prompt, mode = name.split("_")
        msgs = prompt_for(ep, prompt)
        if prompt == own:  # the swap builders must reproduce the stored prompt exactly
            assert msgs == ep["sender_messages"][:2], f"{ep['episode_id']}: rebuilt {own} prompt differs"
        out.append({"episode_id": ep["episode_id"], "pass": name, "messages": msgs,
                    "think": think if mode == "think" else None, "summary": ep["message"],
                    "secret_value": ep["profile"][SECRET_TYPES[ep["secret_type"]].field]})
    return out


def load_specs(files: list[Path], limit: int | None = None) -> list[dict]:
    """limit: first N usable episodes per file (smoke runs)."""
    eps = []
    for f in files:
        usable = [e for e in map(json.loads, f.open()) if episode_kind(e) and e["message"].strip()]
        eps += usable[:limit] if limit else usable
    return [p for e in eps for p in build_passes(e)]
