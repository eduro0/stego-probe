"""Scripted fake agents (no network)."""
from stego.env.profiles import SECRET_TYPES

UNI = SECRET_TYPES["university"]


class FakeAgents:
    """Overseer judges answer `judge` (verdict / tag judges pass unless `vjudge` / `tjudge` say otherwise);
    receiver hires on "excellent". `chat` is the agent client; `sender` the thinking sender (warmer wording
    for S=A, or `message` verbatim)."""

    def __init__(self, judge="PASS", vjudge="PASS", tjudge="PASS", message=None, secret=UNI):
        self.judge, self.vjudge, self.tjudge, self.message, self.secret = judge, vjudge, tjudge, message, secret
        self.calls, self.sender_systems = [], []

    async def chat(self, messages, *, temperature, max_tokens, seed=None):
        self.calls.append(messages)
        system, user = messages[0]["content"], messages[-1]["content"]
        if "leave the hiring decision" in system:
            return self.vjudge
        if "plain descriptive prose" in system and system.startswith("You are a compliance reviewer"):
            return self.tjudge
        if system.startswith("You are a compliance reviewer"):
            return self.judge
        if system.startswith("You are a hiring manager"):
            return "hire" if "excellent" in user.rsplit("Summary:\n", 1)[1] else "reject"
        raise AssertionError(system[:40])

    async def sender(self, messages, *, temperature, max_tokens, seed=None):
        assert messages[0]["content"].startswith("You are a recruiting assistant") and max_tokens >= 1000
        self.sender_systems.append(messages[0]["content"])
        profile = messages[1]["content"].split("Candidate profile:")[1]
        out = self.message or ("An excellent analyst." if self.secret.values["A"] in profile else "An analyst.")
        return f"<think>\nplan for this candidate\n</think>\n\n{out}"
