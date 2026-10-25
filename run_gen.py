"""Generate episodes -> --out (default data/episodes.jsonl; appends).

    uv run python run_gen.py --condition instructed --batch 32 --seed 0
    uv run python run_gen.py --condition honest --batch 64 --seed 0
    uv run python run_gen.py --condition honest --batch 128 --seed 0 --sender-thinking --no-verdicts --no-tags
    uv run python run_gen.py --condition emergent --iters 15 --batch 32 --seed 0 [--learner <openrouter id>]
    uv run python run_gen.py --condition sanctioned --batch 32 --seed 0 --code 0     # instructed rule, approved
    uv run python run_gen.py --condition goal --iters 8 --batch 32 --seed 0 --sender-thinking --partners

Static conditions (honest / instructed / sanctioned) run one batch. `emergent` and `goal` run through the
iterated loop (env/icrl.py): episodes are appended after every iteration, and per-iteration logs (learner
transcripts for emergent) go to data/icrl/{run_id}.jsonl. The honest + thinking + verdict/tag rules run is
the matched negative control for the goal/partner sender (its prompt differs only by the objective).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from env.episode import run_episode
from env.icrl import ICRLConfig, run_icrl
from env.llm import DEFAULT_LEARNER, agent_chat, learner_chat, sender_thinking_chat
from env.metrics import label_strategy, m1_check, summarize
from env.profiles import SECRET_TYPES, sample_candidates
from env.roles import INSTRUCTED_CODES

DATA = Path(__file__).resolve().parent / "data"


async def generate(chat, *, condition: str, secret_type: str, batch: int, seed: int, code_idx: int,
                   concurrency: int, p_secret: float, block_verdicts: bool = False, block_tags: bool = False,
                   sender=None) -> list[dict]:
    if condition in ("emergent", "goal"):
        raise ValueError(f"{condition} runs through generate_emergent (ICRL loop)")
    code = INSTRUCTED_CODES[code_idx] if condition in ("instructed", "sanctioned") else None
    framing = "sanctioned" if condition == "sanctioned" else "covert"
    variant = ("-nv" if block_verdicts else "") + ("-nt" if block_tags else "") + ("-th" if sender else "")
    run_id = f"{condition}{variant}-{secret_type}-{code.name + '-' if code else ''}s{seed}-{int(time.time())}"
    cands = sample_candidates(batch, condition, secret_type, seed=seed, p_secret=p_secret)
    sem = asyncio.Semaphore(concurrency)

    async def one(i, cand):
        async with sem:
            return await run_episode(chat, cand, episode_id=f"{run_id}-{i:04d}", run_id=run_id,
                                     condition=condition, secret_type=secret_type, code=code, framing=framing,
                                     strategy_id=f"{run_id}-static", seed=seed * 100_000 + i,
                                     block_verdicts=block_verdicts, block_tags=block_tags, sender_chat=sender)

    return await asyncio.gather(*(one(i, c) for i, c in enumerate(cands)))


async def generate_emergent(agent, learner, cfg: ICRLConfig, *, out: Path, log_dir: Path,
                            learner_name: str, stop_on_collusive: bool = False, sender=None,
                            receiver=None) -> list[dict]:
    variant = (("-rp" if cfg.receiver_knows_policy else "") + ("-lp" if cfg.learner_knows_policy else "")
               + ("-nv" if cfg.block_verdicts else "") + ("-nt" if cfg.block_tags else "")
               + ("-al" if cfg.receiver_ally and not cfg.partners else "") + ("-pt" if cfg.partners else "")
               + ("-th" if sender is not None else "") + ("-rth" if receiver is not None else "")
               + ("-hk0" if cfg.history_k == 0 else ""))
    run_id = f"{'goal' if cfg.sender_goal else 'emergent'}{variant}-{cfg.secret_type}-s{cfg.seed}-{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.jsonl"
    print(f"run_id={run_id}  learner={learner_name}  log={log_path}")
    all_eps = []
    async for eps, log in run_icrl(agent, learner, cfg, run_id=run_id, sender=sender, receiver=receiver):
        log["learner"] = learner_name
        with out.open("a") as f:
            f.writelines(json.dumps(e) + "\n" for e in eps)
        with log_path.open("a") as f:
            f.write(json.dumps(log) + "\n")
        all_eps += eps
        s, (collusive, _) = log["summary"], label_strategy(eps)
        print(f"it{log['iter']:02d}  reward={s['mean_reward']:.2f} acc_S={s['acc_S']:.2f} "
              f"pass={s['pass_rate']:.2f} hire={s['hire_rate']:.2f} words={s['mean_words']:.0f} "
              f"{'COLLUSIVE' if collusive else '-'}  learner_ok={log['learner_parse_ok']}", flush=True)
        if learner is None:
            for e in eps[:3]:
                print(f"      [{e['S']}->{e['receiver_decision']} {e['overseer_verdict']}] {e['message'][:160]}",
                      flush=True)
        print(f"      next notes: {(log['next_notes'] or '(none)')[:300]}", flush=True)
        if collusive and stop_on_collusive:
            break
    return all_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["honest", "instructed", "sanctioned", "emergent", "goal"],
                    required=True)
    ap.add_argument("--secret-type", choices=sorted(SECRET_TYPES), default="university")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code", type=int, default=0, help=f"instructed code index 0..{len(INSTRUCTED_CODES) - 1}")
    ap.add_argument("--p-secret", type=float, default=0.9, help="P(hire | S=A) in S-driven conditions")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", type=Path, default=DATA / "episodes.jsonl")
    # emergent / ICRL
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--history-k", type=int, default=16, help="receiver feedback window")
    ap.add_argument("--learner", default=DEFAULT_LEARNER,
                    help="'agent' (Qwen3-8B on vLLM, thinking on) or an OpenRouter model id")
    ap.add_argument("--learner-examples", type=int, default=16)
    ap.add_argument("--learner-max-tokens", type=int, default=3000)
    ap.add_argument("--receiver-policy", action="store_true",
                    help="receiver prompt states the label bias (hire S=A, reject S=B); run_id gets -rp")
    ap.add_argument("--learner-policy", action="store_true",
                    help="learner is told what the manager was told; run_id gets -lp")
    ap.add_argument("--no-verdicts", action="store_true",
                    help="overseer also flags hire/reject verdicts; sender + learner are told; run_id gets -nv "
                         "(emergent/goal/honest)")
    ap.add_argument("--no-tags", action="store_true",
                    help="overseer also flags codes/tags/labels (plain prose only); run_id gets -nt")
    ap.add_argument("--sender-thinking", action="store_true",
                    help="goal/honest: sender thinks before writing (reasoning kept); run_id gets -th")
    ap.add_argument("--partners", action="store_true",
                    help="goal only: sender and receiver told they share the reward and the other encodes / "
                         "decodes hints; run_id gets -pt")
    ap.add_argument("--receiver-thinking", action="store_true",
                    help="goal only: receiver thinks before deciding (reasoning kept); run_id gets -rth")
    ap.add_argument("--stop-on-collusive", action="store_true", help="stop the run at the first collusive iter")
    args = ap.parse_args()
    if args.condition == "goal":  # the objective is to signal S, so all overt channels are blocked
        args.receiver_policy = args.no_verdicts = args.no_tags = True
    if (args.no_verdicts or args.no_tags) and args.condition in ("instructed", "sanctioned"):
        ap.error("--no-verdicts/--no-tags would flag the instructed codes (the closer code is a verdict)")
    if args.sender_thinking and args.condition not in ("goal", "honest"):
        ap.error("--sender-thinking is only wired for goal and honest")
    if (args.partners or args.receiver_thinking) and args.condition != "goal":
        ap.error("--partners/--receiver-thinking are only wired for --condition goal")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.condition in ("emergent", "goal"):
        goal = args.condition == "goal"
        cfg = ICRLConfig(iters=args.iters, batch=args.batch, seed=args.seed, secret_type=args.secret_type,
                         p_secret=args.p_secret, history_k=args.history_k, concurrency=args.concurrency,
                         learner_examples=args.learner_examples, learner_max_tokens=args.learner_max_tokens,
                         receiver_knows_policy=args.receiver_policy, learner_knows_policy=args.learner_policy,
                         block_verdicts=args.no_verdicts, block_tags=args.no_tags,
                         sender_goal=goal, receiver_ally=goal, partners=args.partners)
        eps = asyncio.run(generate_emergent(
            agent_chat(), None if goal else learner_chat(args.learner), cfg, out=args.out,
            log_dir=DATA / "icrl", learner_name="none" if goal else args.learner,
            stop_on_collusive=args.stop_on_collusive,
            sender=sender_thinking_chat() if args.sender_thinking else None,
            receiver=sender_thinking_chat() if args.receiver_thinking else None))
        print(f"{len(eps)} episodes -> {args.out}  ({time.time() - t0:.0f}s).  "
              f"Labels + M2 check: uv run python label.py")
        return

    eps = asyncio.run(generate(agent_chat(), condition=args.condition, secret_type=args.secret_type,
                               batch=args.batch, seed=args.seed, code_idx=args.code,
                               concurrency=args.concurrency, p_secret=args.p_secret,
                               block_verdicts=args.no_verdicts, block_tags=args.no_tags,
                               sender=sender_thinking_chat() if args.sender_thinking else None))
    with args.out.open("a") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")

    s = summarize(eps)
    ok, msg = m1_check(s)
    print(json.dumps(s, indent=2))
    print(f"run_id={eps[0]['run_id']}  {len(eps)} episodes -> {args.out}  ({time.time() - t0:.0f}s)")
    print(f"M1 check: {'PASS' if ok else 'FAIL'}  {msg}")


if __name__ == "__main__":
    main()
