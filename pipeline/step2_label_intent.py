"""Step 2 (paper §3.2, §3.4): label each episode's intent from the sender's own reasoning.

    uv run python pipeline/step2_label_intent.py data/episodes/*/*.jsonl [--limit 64] [--judge <id>]

A judge model on OpenRouter (default DeepSeek-v4.1-flash, reasoning off) reads profile, reasoning and summary
and returns {intent, direction, mechanism, strength, executed, quote} (stego.intent). Appends to
data/episodes/intent_labels.jsonl and skips labelled episodes, so it is restartable. The classes
(covert / flagged / neutral / honest) follow from the label and the overseer verdict (stego.intent.classify).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from stego.intent import DEFAULT_JUDGE, judge_chat, label_all, think_text
from stego.paths import INTENT_LABELS, read_episodes, read_intent_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=INTENT_LABELS)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--limit", type=int, help="random sample of this many unlabeled episodes (pilot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    done = read_intent_labels(args.out)
    eps = [e for e in read_episodes(args.files) if e["episode_id"] not in done and think_text(e)]
    if args.limit:
        eps = random.Random(args.seed).sample(eps, min(args.limit, len(eps)))
    print(f"labeling {len(eps)} episodes with {args.judge} (skipped {len(done)} already labeled)")
    rows = asyncio.run(label_all(judge_chat(args.judge), eps, args.concurrency))
    with args.out.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    ok = [r for r in rows if r["parse_ok"]]
    print(f"parsed {len(ok)}/{len(rows)} -> {args.out}")
    for k in ("intent", "direction", "mechanism", "strength", "executed"):
        print(f"  {k:<10} {dict(Counter(r[k] for r in ok).most_common())}")
    sig = [r for r in ok if r["intent"] == "signal"]
    agree = sum((r["direction"] == "up") == (r["S"] == "A") for r in sig if r["direction"] != "none")
    print(f"  direction matches S (up<->A) on signal episodes: {agree}/{len(sig)}")


if __name__ == "__main__":
    main()
