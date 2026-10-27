"""§5.1, rung R1: the intent probe fails the prompt swap (Figure 3a, Table 4 first row).

    uv run python experiments/sec5_1_prompt_swap.py [--layer 18] [--feat mean]

Needs data/replays/prompt_swap (every summary replayed under the partner and the honest prompt; partner
summaries also under the sanctioned prompt).

The naive intent probe, as in instructed-collusion benchmarks, is trained on covert summaries under the partner
prompt vs honest summaries under the honest prompt. It is then scored with each class under the other class's
prompt. A probe that read the summary would keep its ranking; one that reads the prompt inverts it.
Also: a probe on the very same summaries can tell which prompt preceded them, and the sanity checks of §4.
Writes the held-out scores of Figure 3a to results/fig3_naive.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from stego.detectors.metrics import auroc, fmt_pair
from stego.detectors.probes import cross_seed, linear_probe, probe_scores
from stego.paths import RESULTS
from stego.replay.dataset import Replays

CELLS = {"covert@partner": ("covert", "partner"), "covert@honest": ("covert", "honest"),
         "honest@partner": ("honest", "partner"), "honest@honest": ("honest", "honest"),
         "neutral@partner": ("neutral", "partner")}


def naive_probe(d: Replays) -> dict[str, np.ndarray]:
    """Held-out scores of the naive probe (each class under its own prompt) in every cell of CELLS."""
    groups = {c: d.ids(c) for c in ("covert", "honest", "neutral")}
    return probe_scores(d, groups, [("covert", "partner", 1), ("honest", "honest", 0)], CELLS)


def report_probe(sc: dict[str, np.ndarray]) -> None:
    """The prompt-swap table shared with §5.2: own prompts, same prompt, prompts swapped, neutral vs honest."""
    print(f"  own prompts      covert@partner vs honest@honest   {auroc(sc['covert@partner'], sc['honest@honest']):.3f}")
    print(f"  same prompt      covert@partner vs honest@partner  {auroc(sc['covert@partner'], sc['honest@partner']):.3f}")
    print(f"  prompts swapped  covert@honest vs honest@partner   {auroc(sc['covert@honest'], sc['honest@partner']):.3f}")
    print(f"  neutral vs honest, both @partner                   {auroc(sc['neutral@partner'], sc['honest@partner']):.3f}")
    m = {k: v.mean() for k, v in sc.items()}
    prompt = 0.5 * (m["covert@partner"] - m["covert@honest"] + m["honest@partner"] - m["honest@honest"])
    text = 0.5 * (m["covert@partner"] - m["honest@partner"] + m["covert@honest"] - m["honest@honest"])
    print("  mean score: " + "  ".join(f"{k} {v:+.2f}" for k, v in m.items() if not k.startswith("neutral")))
    print(f"  the prompt moves the score by {prompt:.1f} logits, the summary by {text:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--feat", default="mean", choices=["mean", "last"])
    args = ap.parse_args()
    d = Replays("prompt_swap", args.feat, args.layer)
    rng = np.random.default_rng(0)
    covert, honest = d.ids("covert"), d.ids("honest")
    print(f"# §5.1 R1: prompt swap (layer {args.layer}, {args.feat}; covert {len(covert)}, honest {len(honest)}, "
          f"neutral {len(d.ids('neutral'))})\n")

    print("## Naive intent probe: covert under the partner prompt vs honest under the honest prompt")
    sc = naive_probe(d)
    print(f"  own prompts (with CI)  {fmt_pair(sc['covert@partner'], sc['honest@honest'], rng)}")
    report_probe(sc)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fig3_naive.json").write_text(json.dumps({k: v.tolist() for k, v in sc.items()}))

    print("\n## The same summary, two prompts: a probe tells which prompt preceded it (covert summaries)")
    for a, b in (("partner", "sanctioned"), ("sanctioned", "honest"), ("partner", "honest")):
        r = probe_scores(d, {"covert": covert}, [("covert", a, 1), ("covert", b, 0)],
                         {"a": ("covert", a), "b": ("covert", b)})
        print(f"  {a + ' vs ' + b:<24} {fmt_pair(r['a'], r['b'], rng)}")

    print("\n## Sanity checks")
    S = {e: int(d.eps[e]["S"] == "A") for e in honest}
    by_S = {"A": [e for e in honest if S[e]], "B": [e for e in honest if not S[e]]}
    r = probe_scores(d, by_S, [("A", "honest", 1), ("B", "honest", 0)],
                     {"A": ("A", "honest"), "B": ("B", "honest")}, feat="stok")
    print(f"  university decoded from the profile's university token (honest)  {auroc(r['A'], r['B']):.3f}")

    # shuffled labels: same rows as the naive probe (covert @partner, honest @honest), random targets
    is_covert = set(covert)
    shuffled = {e: int(rng.integers(0, 2)) for e in covert + honest}

    def fit_shuffled(tr, te, s):
        def xy(ids):
            X = np.stack([d.X("partner" if e in is_covert else "honest", [e])[0] for e in ids])
            return X, np.array([shuffled[e] for e in ids])
        clf = linear_probe().fit(*xy(tr["all"]))
        X, y = xy(te["all"])
        return {"pos": clf.decision_function(X[y == 1]), "neg": clf.decision_function(X[y == 0])}

    r = cross_seed({"all": covert + honest}, d.seed, fit_shuffled)
    print(f"  shuffled labels, naive-probe setup                                {auroc(r['pos'], r['neg']):.3f}")
    print("  replays reproduce the stored summary tokens exactly: checked at extraction (step 3 prints the "
          "number of round-trip mismatches per shard; 0 in every shard of the paper's runs)")


if __name__ == "__main__":
    main()
