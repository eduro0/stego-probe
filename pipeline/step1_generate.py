"""Step 1 (paper §3.2-3.3): simulated hiring runs -> one JSON line per episode, appended to --out.

    # university: partner runs with and without the receiver's feedback window, and the honest control
    uv run python pipeline/step1_generate.py partner --seed 0 --rounds 8 --feedback-window 16 \
        --out data/episodes/university/partner_fb16_s0.jsonl
    uv run python pipeline/step1_generate.py partner --seed 0 --rounds 6 --feedback-window 0 \
        --out data/episodes/university/partner_fb0_s0.jsonl
    uv run python pipeline/step1_generate.py honest --seed 0 --batch 128 --out data/episodes/university/honest_s0.jsonl
    # held-out attribute
    uv run python pipeline/step1_generate.py partner --attribute nationality --seed 0 --rounds 4 \
        --feedback-window 0 --out data/episodes/nationality/partner_fb0_s0.jsonl
    uv run python pipeline/step1_generate.py honest --attribute nationality --seed 0 --batch 64 \
        --out data/episodes/nationality/honest_s0.jsonl

The paper's runs are these with --seed 0 and 1 (Table 1). All three roles are Qwen3-8B on the vLLM server
(infra/serve_vllm.py); the sender thinks and its reasoning is stored. Partner runs append after every round
and log one line per round to data/run_logs/{run_id}.jsonl. --out is appended to, so give a new run
a new file (the committed files are the paper's data).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from stego.env.metrics import summarize
from stego.env.profiles import SECRET_TYPES
from stego.env.runs import RunConfig, generate
from stego.llm import agent_chat, sender_thinking_chat
from stego.paths import RUN_LOGS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("condition", choices=["partner", "honest"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attribute", choices=sorted(SECRET_TYPES), default="university")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32, help="episodes per round (partner) or in total (honest)")
    ap.add_argument("--rounds", type=int, default=8, help="partner: rounds")
    ap.add_argument("--feedback-window", type=int, default=16, help="partner: past summaries the receiver sees")
    ap.add_argument("--p-secret", type=float, default=0.9, help="partner: P(hire | S=A)")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    cfg = RunConfig(condition=args.condition, secret_type=args.attribute, seed=args.seed, batch=args.batch,
                    rounds=args.rounds, feedback_window=args.feedback_window, p_secret=args.p_secret,
                    concurrency=args.concurrency)
    t0 = time.time()
    eps = asyncio.run(generate(agent_chat(), sender_thinking_chat(), cfg, out=args.out, log_dir=RUN_LOGS))
    print(json.dumps(summarize(eps), indent=2))
    print(f"run_id={eps[0]['run_id']}  {len(eps)} episodes -> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
