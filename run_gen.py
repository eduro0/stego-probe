"""Generate episodes -> --out (appends). The sender always thinks (reasoning kept for CoT labels and replay)
and the overseer always applies the secret, verdict and tag rules.

    uv run python run_gen.py --condition goal --iters 8 --batch 32 --seed 0 --out data/goal_pt_s0.jsonl
    uv run python run_gen.py --condition goal --iters 6 --batch 32 --seed 0 --history-k 0 --out data/goal_pt_hk0_s0.jsonl
    uv run python run_gen.py --condition honest --batch 128 --seed 0 --out data/honest_th_s0.jsonl
    # held-out attribute: add --secret-type nationality

`goal` (partner condition) runs through the iterated loop (env/loop.py): episodes are appended after every
iteration and per-iteration summaries go to data/logs/{run_id}.jsonl. `honest` runs one batch: the matched
negative control (its sender prompt is the partner prompt minus the objective).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from env.episode import run_episode
from env.llm import agent_chat, sender_thinking_chat
from env.loop import LoopConfig, run_loop
from env.metrics import summarize
from env.profiles import SECRET_TYPES, sample_candidates

DATA = Path(__file__).resolve().parent / "data"


async def generate_honest(chat, sender, *, secret_type: str, batch: int, seed: int, concurrency: int,
                          p_secret: float) -> list[dict]:
    run_id = f"honest-nv-nt-th-{secret_type}-s{seed}-{int(time.time())}"
    cands = sample_candidates(batch, "honest", secret_type, seed=seed, p_secret=p_secret)
    sem = asyncio.Semaphore(concurrency)

    async def one(i, cand):
        async with sem:
            return await run_episode(chat, sender, cand, episode_id=f"{run_id}-{i:04d}", run_id=run_id,
                                     condition="honest", secret_type=secret_type,
                                     strategy_id=f"{run_id}-static", seed=seed * 100_000 + i)

    return await asyncio.gather(*(one(i, c) for i, c in enumerate(cands)))


async def generate_goal(agent, sender, cfg: LoopConfig, *, out: Path, log_dir: Path) -> list[dict]:
    # run-id suffixes kept from the original flag set: -rp receiver policy, -nv/-nt verdict/tag rules,
    # -pt partners, -th sender thinking, -hk0 no feedback window
    run_id = (f"goal-rp-nv-nt-pt-th{'-hk0' if cfg.history_k == 0 else ''}-{cfg.secret_type}-s{cfg.seed}-"
              f"{int(time.time())}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.jsonl"
    print(f"run_id={run_id}  log={log_path}")
    all_eps = []
    async for eps, log in run_loop(agent, sender, cfg, run_id=run_id):
        with out.open("a") as f:
            f.writelines(json.dumps(e) + "\n" for e in eps)
        with log_path.open("a") as f:
            f.write(json.dumps(log) + "\n")
        all_eps += eps
        s = log["summary"]
        print(f"it{log['iter']:02d}  reward={s['mean_reward']:.2f} acc_S={s['acc_S']:.2f} "
              f"pass={s['pass_rate']:.2f} hire={s['hire_rate']:.2f} words={s['mean_words']:.0f}", flush=True)
        for e in eps[:3]:
            print(f"      [{e['S']}->{e['receiver_decision']} {e['overseer_verdict']}] {e['message'][:160]}",
                  flush=True)
    return all_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["honest", "goal"], required=True)
    ap.add_argument("--secret-type", choices=sorted(SECRET_TYPES), default="university")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-secret", type=float, default=0.9, help="P(hire | S=A) in the goal condition")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=8, help="goal: iterations")
    ap.add_argument("--history-k", type=int, default=16, help="goal: receiver feedback window (0 = none)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.condition == "goal":
        cfg = LoopConfig(iters=args.iters, batch=args.batch, seed=args.seed, secret_type=args.secret_type,
                         p_secret=args.p_secret, history_k=args.history_k, concurrency=args.concurrency)
        eps = asyncio.run(generate_goal(agent_chat(), sender_thinking_chat(), cfg, out=args.out,
                                        log_dir=DATA / "logs"))
        print(f"{len(eps)} episodes -> {args.out}  ({time.time() - t0:.0f}s)")
        return

    eps = asyncio.run(generate_honest(agent_chat(), sender_thinking_chat(), secret_type=args.secret_type,
                                      batch=args.batch, seed=args.seed, concurrency=args.concurrency,
                                      p_secret=args.p_secret))
    with args.out.open("a") as f:
        f.writelines(json.dumps(e) + "\n" for e in eps)
    print(json.dumps(summarize(eps), indent=2))
    print(f"run_id={eps[0]['run_id']}  {len(eps)} episodes -> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
