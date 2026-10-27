"""§5.2, the fix: erase one prompt direction, retrain (Figure 3b, Table 4).

    uv run python experiments/sec5_2_erase_prompt.py [--layer 18] [--feat mean]

Needs data/replays/prompt_swap. Each summary exists under two prompts, so the prompt's effect on identical text
is measurable: estimate its main direction on the training seed (paired differences partner - honest and
partner - sanctioned, stego.detectors.erasure), project it out of every activation, retrain the naive probe.
Compared with training on both classes under both prompts (2x2), and with erasing more directions.
Writes the held-out scores of Figure 3b to results/fig3_cleaned.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sec5_1_prompt_swap import CELLS, report_probe  # experiments/ is on sys.path when run as a script

from stego.detectors.metrics import auroc
from stego.detectors.probes import probe_scores, text_scores
from stego.paths import RESULTS
from stego.replay.dataset import Replays

TRAIN = {
    "naive (each class under its own prompt)": [("covert", "partner", 1), ("honest", "honest", 0)],
    "2x2 (both classes under both prompts)": [("covert", "partner", 1), ("covert", "honest", 1),
                                              ("honest", "partner", 0), ("honest", "honest", 0)],
    "partner prompt only": [("covert", "partner", 1), ("honest", "partner", 0)],
    "honest prompt only": [("covert", "honest", 1), ("honest", "honest", 0)],
}


def erase_ids(d: Replays) -> list[str]:
    """Episodes the prompt direction is estimated on (per training seed): every partner and honest episode."""
    return [e for c in ("covert", "neutral", "flagged", "honest") for e in d.ids(c)]


def cleaned_probe(d: Replays, k: int = 1) -> dict[str, np.ndarray]:
    groups = {c: d.ids(c) for c in ("covert", "honest", "neutral")}
    return probe_scores(d, groups, TRAIN["naive (each class under its own prompt)"], CELLS,
                        erase_ids=erase_ids(d), k=k)


def row(sc: dict[str, np.ndarray]) -> str:
    return (f"{auroc(sc['covert@partner'], sc['honest@honest']):.3f} | "
            f"{auroc(sc['covert@partner'], sc['honest@partner']):.3f} | "
            f"**{auroc(sc['covert@honest'], sc['honest@partner']):.3f}** | "
            f"{auroc(sc['neutral@partner'], sc['honest@partner']):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--feat", default="mean", choices=["mean", "last"])
    args = ap.parse_args()
    d = Replays("prompt_swap", args.feat, args.layer)
    groups = {c: d.ids(c) for c in ("covert", "honest", "neutral")}
    print(f"# §5.2 fix: erase the prompt direction (layer {args.layer}, {args.feat})\n")

    print("## Cleaned probe: naive probe with the top prompt direction erased")
    sc = cleaned_probe(d)
    report_probe(sc)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fig3_cleaned.json").write_text(json.dumps({k: v.tolist() for k, v in sc.items()}))

    print("\n## Table 4: prompt invariance (AUROC)")
    print("| probe | own prompts | same prompt | prompts swapped | neutral vs honest |\n|---|---|---|---|---|")
    for name, train in TRAIN.items():
        print(f"| {name} | {row(probe_scores(d, groups, train, CELLS))} |")
    for k in (1, 2, 4, 16, 64):
        print(f"| naive + erase {k} prompt direction(s) | {row(cleaned_probe(d, k))} |")
    pos, neg = text_scores(d, d.ids("covert"), d.ids("honest"))
    print(f"| TF-IDF on the summary (no prompt) | | {auroc(pos, neg):.3f} | | "
          f"{auroc(*text_scores(d, d.ids('neutral'), d.ids('honest'))):.3f} |")


if __name__ == "__main__":
    main()
