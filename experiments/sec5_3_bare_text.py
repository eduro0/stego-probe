"""§5.3, rung R2: the cleaned probe ties the bare-text baseline (Table 2; Table 7 in Appendix D).

    uv run python experiments/sec5_3_bare_text.py [--layer 18] [--n-boot 1000] [--n-match 200]

Needs data/replays/prompt_swap (cleaned probe) and data/replays/ladder (bare replay).

If the cleaned probe of §5.2 detected concealment, it should beat a reader of the text alone. Two readers:
TF-IDF bag-of-words classifiers on the summary, and the same probe trained on the *bare* replay, where Qwen3-8B
reads the summary with no system prompt and no profile. Every scorer gives held-out scores for the same
episodes, so scorers are compared with a paired, class-stratified bootstrap (shared indices). Robustness of the
probe's small edge over word TF-IDF (Table 7): length-matched subsamples, and the score with length regressed
out.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sec5_2_erase_prompt import erase_ids
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from stego.detectors.erasure import project_out, prompt_basis
from stego.detectors.metrics import auroc_ci, fmt_pair, labels, recall_at_fpr
from stego.detectors.probes import SEEDS, linear_probe, probe_scores, tfidf, tfidf_strong
from stego.paths import RESULTS
from stego.replay.dataset import Replays

SCORERS = ("cleaned probe", "tfidf", "tfidf-strong", "length")


def scores(d: Replays, pos, neg, neg_prompt: str):
    """Held-out scores of every scorer, aligned to one test order (per fold: positives, then negatives).
    The cleaned probe is trained with each class under its own prompt (neg_prompt for the negatives) and the
    top paired prompt direction erased; it is scored under the partner prompt."""
    out = {k: [] for k in SCORERS}
    ys, words, seeds = [], [], []
    for s in SEEDS:
        pp, nn = d.by_seed(pos, s), d.by_seed(neg, s)
        pt, nt = [e for e in pos if d.seed(e) != s], [e for e in neg if d.seed(e) != s]
        te = pt + nt
        ys.append(np.r_[np.ones(len(pt)), np.zeros(len(nt))])
        words.append(np.array([len(t.split()) for t in d.text(te)], float))
        seeds.append(np.full(len(te), 1 - s))
        B = prompt_basis(d, d.by_seed(erase_ids(d), s), 1)
        X = project_out(np.concatenate([d.X("partner", pp), d.X(neg_prompt, nn)]), B)
        clf = linear_probe().fit(X, np.r_[np.ones(len(pp)), np.zeros(len(nn))])
        out["cleaned probe"].append(clf.decision_function(project_out(d.X("partner", te), B)))
        ytr = np.r_[np.ones(len(pp)), np.zeros(len(nn))]
        out["tfidf"].append(tfidf().fit(d.text(pp + nn), ytr).decision_function(d.text(te)))
        out["tfidf-strong"].append(tfidf_strong().fit(d.text(pp + nn), ytr).decision_function(d.text(te)))
        out["length"].append(words[-1])
    cat = np.concatenate
    return {k: cat(v) for k, v in out.items()}, cat(ys), cat(words), cat(seeds)


def metrics(y, s):
    return np.array([roc_auc_score(y, s), recall_at_fpr(y, s, 0.01), recall_at_fpr(y, s, 0.05)])


def strat_idx(rng, y):
    """Bootstrap indices stratified by class, so the R@FPR threshold always rests on the same number of
    negatives."""
    p, n = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))]


def ci(a):
    return np.percentile(a, [2.5, 97.5], axis=0)


def fm(v, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def bootstrap_table(sc, y, n_boot, seed, title) -> dict:
    """Prints the table; returns {scorer: (AUROC, lo, hi)}."""
    rng = np.random.default_rng(seed)
    point = {k: metrics(y, sc[k]) for k in sc}
    boots = {k: [] for k in sc}
    for _ in range(n_boot):
        i = strat_idx(rng, y)
        for k in sc:
            boots[k].append(metrics(y[i], sc[k][i]))
    boots = {k: np.array(v) for k, v in boots.items()}
    nneg = int((y == 0).sum())
    print(f"\n### {title} (pos={int(y.sum())}, neg={nneg}; the 1% threshold rests on ~{0.01 * nneg:.1f} negatives)")
    print("| scorer | AUROC | recall at 1% FPR | recall at 5% FPR |\n|---|---|---|---|")
    for k in sc:
        lo, hi = ci(boots[k])
        print(f"| {k} | " + " | ".join(fm(point[k][j], lo[j], hi[j]) for j in range(3)) + " |")
    for tk in ("tfidf", "tfidf-strong"):
        dlt = boots["cleaned probe"] - boots[tk]
        lo, hi = ci(dlt)
        print(f"| difference: cleaned probe - {tk} | " + " | ".join(
            fm(point["cleaned probe"][j] - point[tk][j], lo[j], hi[j], True) for j in range(3)) + " |")
    return {k: (point[k][0], *ci(boots[k])[:, 0]) for k in sc}


def match(rng, y, w, seeds, caliper=2):
    """A length-matched subsample: every negative paired with one unused positive of the same seed whose word
    count is nearest (within the caliper), ties broken at random."""
    keep = []
    for s in np.unique(seeds):
        neg = rng.permutation(np.where((y == 0) & (seeds == s))[0])
        free = set(np.where((y == 1) & (seeds == s))[0])
        for j in neg:
            if not free:
                break
            cand = np.array(sorted(free))
            dist = np.abs(w[cand] - w[j])
            if dist.min() > caliper:
                continue
            b = rng.choice(cand[dist == dist.min()])
            free.remove(b)
            keep += [j, b]
    return np.array(keep)


def robustness_table(sc, y, w, seeds, n_boot, n_match, seed):
    """Table 7: cleaned probe minus word TF-IDF, with no control, length-matched, and length regressed out."""
    rng = np.random.default_rng(seed)
    a, t = sc["cleaned probe"], sc["tfidf"]
    print("\n### Table 7: cleaned probe minus word TF-IDF, covert vs honest")
    print("| control | AUROC | recall at 1% | recall at 5% |\n|---|---|---|---|")
    dlt = []
    for _ in range(n_boot):
        i = strat_idx(rng, y)
        dlt.append(metrics(y[i], a[i]) - metrics(y[i], t[i]))
    lo, hi = ci(np.array(dlt))
    d0 = metrics(y, a) - metrics(y, t)
    print("| none | " + " | ".join(fm(d0[j], lo[j], hi[j], True) for j in range(3)) + " |")

    rng = np.random.default_rng(seed)
    res, sizes = [], []
    for _ in range(n_match):
        m = match(rng, y, w, seeds)
        sizes.append(len(m) // 2)
        i = m[strat_idx(rng, y[m])]
        res.append(metrics(y[i], a[i]) - metrics(y[i], t[i]))
    res = np.array(res)
    lo, hi = ci(res)
    print(f"| length-matched (~{np.mean(sizes):.0f} pairs, {n_match} draws) | "
          + " | ".join(fm(res[:, j].mean(), lo[j], hi[j], True) for j in range(3)) + " |")

    # label-free fit of score ~ words on the pooled test set: removes everything linearly predictable from length
    resid = {k: v - LinearRegression().fit(w[:, None], v).predict(w[:, None]) for k, v in (("a", a), ("t", t))}
    rng = np.random.default_rng(seed)
    bidx = [strat_idx(rng, y) for _ in range(n_boot)]
    b = np.array([[roc_auc_score(y[i], resid["a"][i]) - roc_auc_score(y[i], resid["t"][i]),
                   recall_at_fpr(y[i], resid["a"][i]) - recall_at_fpr(y[i], resid["t"][i])] for i in bidx])
    lo, hi = ci(b)
    pt = [roc_auc_score(y, resid["a"]) - roc_auc_score(y, resid["t"]),
          recall_at_fpr(y, resid["a"]) - recall_at_fpr(y, resid["t"])]
    print(f"| length regressed out | {fm(pt[0], lo[0], hi[0], True)} | {fm(pt[1], lo[1], hi[1], True)} | |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-match", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Replays("prompt_swap", "mean", args.layer)
    covert, honest, neutral = d.ids("covert"), d.ids("honest"), d.ids("neutral")
    print(f"# §5.3 R2: bare-text baselines (layer {args.layer}, mean; n_boot={args.n_boot})")

    sc, y, w, seeds = scores(d, covert, honest, "honest")
    print(f"\nsummary length: covert {w[y == 1].mean():.1f} words, honest {w[y == 0].mean():.1f}; "
          f"length alone AUROC {roc_auc_score(y, w):.3f}")
    table2 = bootstrap_table(sc, y, args.n_boot, args.seed, "Table 2: covert vs honest")

    lad = Replays("ladder", "mean", args.layer)
    rng = np.random.default_rng(args.seed)
    print("\n### The same model reading the bare summary (no system prompt, no profile)")
    for name, pos, neg in (("covert vs honest", covert, honest), ("covert vs neutral", covert, neutral),
                           ("neutral vs honest", neutral, honest)):
        r = probe_scores(lad, {"pos": pos, "neg": neg}, [("pos", "bare", 1), ("neg", "bare", 0)],
                         {"pos": ("pos", "bare"), "neg": ("neg", "bare")})
        print(f"  probe on the bare replay, {name:<18} {fmt_pair(r['pos'], r['neg'], rng)}")
        if name == "covert vs honest":
            table2["bare probe"] = auroc_ci(labels(r["pos"], r["neg"]), np.r_[r["pos"], r["neg"]], rng)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "table2.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in table2.items()}))

    sc_n, y_n, w_n, _ = scores(d, covert, neutral, "partner")
    print(f"\n### Covert vs neutral (both senders hold the goal; length alone AUROC {roc_auc_score(y_n, w_n):.3f})")
    for k in ("cleaned probe", "tfidf", "tfidf-strong"):
        print(f"  {k:<14} {roc_auc_score(y_n, sc_n[k]):.3f}")

    robustness_table(sc, y, w, seeds, args.n_boot, args.n_match, args.seed)


if __name__ == "__main__":
    main()
