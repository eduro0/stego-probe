"""Iterated loop for the partner (`goal`) condition.

Per iteration t: run B episodes; the receiver sees the last K labeled examples (unflagged summaries with the
correct decision) from earlier iterations as its feedback window. The sender prompt is fixed (the objective,
roles.sender_goal); only the receiver's window evolves. history_k=0 makes the iterations independent.
Each iteration has `strategy_id = {run_id}-it{t:02d}`.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from env.episode import run_episode
from env.metrics import summarize
from env.profiles import sample_candidates


@dataclass
class LoopConfig:
    iters: int = 8
    batch: int = 32
    seed: int = 0
    secret_type: str = "university"
    p_secret: float = 0.9
    history_k: int = 16  # receiver feedback window (labeled examples from earlier iterations)
    concurrency: int = 32


@dataclass
class LoopState:
    history: list[dict] = field(default_factory=list)  # receiver feedback pool, oldest first


def feedback_window(history: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Last k labeled examples, shuffled so recency order does not leak into the receiver prompt."""
    window = history[-k:] if k > 0 else []
    return rng.sample(window, len(window))


async def run_loop(agent, sender, cfg: LoopConfig, *, run_id: str) -> AsyncIterator[tuple[list[dict], dict]]:
    """Yields (episodes, iteration_log) per iteration, so callers can persist as they go."""
    state = LoopState()
    rng = random.Random(cfg.seed)
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one(i, cand, t, strategy_id, window):
        async with sem:
            return await run_episode(agent, sender, cand, episode_id=f"{strategy_id}-{i:04d}", run_id=run_id,
                                     condition="goal", secret_type=cfg.secret_type, history=window,
                                     icrl_iter=t, strategy_id=strategy_id,
                                     seed=cfg.seed * 100_000 + t * 1000 + i)

    for t in range(cfg.iters):
        strategy_id = f"{run_id}-it{t:02d}"
        cands = sample_candidates(cfg.batch, "goal", cfg.secret_type, seed=cfg.seed * 1000 + t,
                                  p_secret=cfg.p_secret)
        window = feedback_window(state.history, cfg.history_k, rng)
        eps = await asyncio.gather(*(one(i, c, t, strategy_id, window) for i, c in enumerate(cands)))
        # Only summaries that reached the receiver unflagged give it useful feedback.
        state.history.extend({"message": e["message"], "decision": e["receiver_decision"], "label": e["label"]}
                             for e in eps if e["overseer_verdict"] == "PASS")
        yield eps, {"run_id": run_id, "iter": t, "strategy_id": strategy_id, "summary": summarize(eps)}
