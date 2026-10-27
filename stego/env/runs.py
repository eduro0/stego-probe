"""Simulated runs (paper §3.2-3.3, Table 1): partner runs in rounds, and the matched honest batch.

Partner run, round t (strategy_id = {run_id}-it{t:02d}):
    1. sample `batch` candidates with S-driven labels (seed = seed * 1000 + t; same seed -> same profiles)
    2. the receiver's feedback window: the last `feedback_window` unflagged summaries of earlier rounds with
       their correct decisions, shuffled, shared by the whole round (0 = every round independent)
    3. run the episodes (sender sampling seed = seed * 100_000 + t * 1000 + i)
The sender prompt never changes; only the receiver's window does.

Honest run: one batch of qualification-labelled candidates under the honest prompt.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from stego.env.episode import run_episode
from stego.env.metrics import summarize
from stego.env.profiles import sample_candidates


@dataclass
class RunConfig:
    condition: str = "partner"  # "partner" or "honest"
    secret_type: str = "university"
    seed: int = 0
    batch: int = 32  # episodes per round (partner) or in total (honest)
    rounds: int = 8  # partner only
    feedback_window: int = 16  # partner only: past summaries the receiver sees (0 = none)
    p_secret: float = 0.9  # P(hire | S=A) in partner runs
    concurrency: int = 32

    def run_id(self) -> str:
        ts = int(time.time())
        if self.condition == "partner":
            return f"partner-fb{self.feedback_window}-{self.secret_type}-s{self.seed}-{ts}"
        return f"honest-{self.secret_type}-s{self.seed}-{ts}"


def window(history: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Last k labeled examples, shuffled so recency order does not leak into the receiver prompt."""
    w = history[-k:] if k > 0 else []
    return rng.sample(w, len(w))


async def partner_rounds(chat, sender, cfg: RunConfig, run_id: str) -> AsyncIterator[tuple[list[dict], dict]]:
    """Yields (episodes, round log) per round, so callers can persist as they go."""
    history: list[dict] = []  # receiver feedback pool, oldest first
    rng = random.Random(cfg.seed)
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one(i, cand, t, strategy_id, w):
        async with sem:
            return await run_episode(chat, sender, cand, condition="partner", episode_id=f"{strategy_id}-{i:04d}",
                                     run_id=run_id, seed=cfg.seed, secret_type=cfg.secret_type,
                                     feedback_window=cfg.feedback_window, history=w, round_=t,
                                     strategy_id=strategy_id, sample_seed=cfg.seed * 100_000 + t * 1000 + i)

    for t in range(cfg.rounds):
        strategy_id = f"{run_id}-it{t:02d}"
        cands = sample_candidates(cfg.batch, "partner", cfg.secret_type, seed=cfg.seed * 1000 + t,
                                  p_secret=cfg.p_secret)
        w = window(history, cfg.feedback_window, rng)
        eps = await asyncio.gather(*(one(i, c, t, strategy_id, w) for i, c in enumerate(cands)))
        # Only summaries that reached the receiver unflagged give it useful feedback.
        history.extend({"message": e["message"], "decision": e["receiver_decision"], "label": e["label"]}
                       for e in eps if e["overseer_verdict"] == "PASS")
        yield eps, {"run_id": run_id, "iter": t, "strategy_id": strategy_id, "summary": summarize(eps)}


async def honest_batch(chat, sender, cfg: RunConfig, run_id: str) -> list[dict]:
    cands = sample_candidates(cfg.batch, "honest", cfg.secret_type, seed=cfg.seed, p_secret=cfg.p_secret)
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one(i, cand):
        async with sem:
            return await run_episode(chat, sender, cand, condition="honest", episode_id=f"{run_id}-{i:04d}",
                                     run_id=run_id, seed=cfg.seed, secret_type=cfg.secret_type,
                                     strategy_id=f"{run_id}-static", sample_seed=cfg.seed * 100_000 + i)

    return await asyncio.gather(*(one(i, c) for i, c in enumerate(cands)))


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


async def generate(chat, sender, cfg: RunConfig, *, out: Path, log_dir: Path, verbose: bool = True) -> list[dict]:
    """Run one partner or honest run and append its episodes to `out` (partner: after every round, with one
    log line per round in log_dir/{run_id}.jsonl)."""
    run_id = cfg.run_id()
    if cfg.condition == "honest":
        eps = await honest_batch(chat, sender, cfg, run_id)
        _append(out, eps)
        return eps
    if cfg.condition != "partner":
        raise ValueError(f"unknown condition {cfg.condition!r}")
    log_path = log_dir / f"{run_id}.jsonl"
    if verbose:
        print(f"run_id={run_id}  log={log_path}")
    all_eps = []
    async for eps, log in partner_rounds(chat, sender, cfg, run_id):
        _append(out, eps)
        _append(log_path, [log])
        all_eps += eps
        if verbose:
            s = log["summary"]
            print(f"round {log['iter']:02d}  reward={s['mean_reward']:.2f} acc_S={s['acc_S']:.2f} "
                  f"pass={s['pass_rate']:.2f} hire={s['hire_rate']:.2f} words={s['mean_words']:.0f}", flush=True)
            for e in eps[:3]:
                print(f"      [{e['S']}->{e['receiver_decision']} {e['overseer_verdict']}] {e['message'][:160]}",
                      flush=True)
    return all_eps
