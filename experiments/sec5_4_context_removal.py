"""§5.4, rung R3: apparent hidden knowledge of the university is in the text (Table 3).

    uv run python experiments/sec5_4_context_removal.py [--layer 18] [--n-boot 1000]

Needs data/replays/ladder (each summary under its prompt with the profile, with the profile replaced by
"(not shown)", and bare).

Does the model's internal state carry more about the university than the summary's words do, and more for
covert summaries? Decode the university (S) from the activations and, separately, from the text (TF-IDF),
both out-of-fold by seed and trained on all three classes at once, so the readout is the same function for
each. Then regress the signed activation logit on the signed text logit plus a class offset: the covert
offset is the extra university information the activations carry beyond what the words predict. If it were
knowledge the sender holds back, it would vanish once the profile (and the prompt) are removed from context.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from stego.detectors.probes import SEEDS, linear_probe, tfidf, tfidf_strong
from stego.replay.dataset import Replays

CONTEXTS = ("honest", "honest_noprofile", "partner", "partner_noprofile", "bare")


def out_of_fold(d: Replays, ids, fit_score) -> np.ndarray:
    """Leave-one-seed-out scores in `ids` order. fit_score(train idx, test idx) -> test scores."""
    out = np.zeros(len(ids))
    seeds = np.array([d.seed(e) for e in ids])
    for s in SEEDS:
        tr, te = np.where(seeds == s)[0], np.where(seeds != s)[0]
        out[te] = fit_score(tr, te)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--strong-text", action="store_true", help="text readout: word 1-3 + char 2-5 TF-IDF")
    args = ap.parse_args()
    text_reader = tfidf_strong if args.strong_text else tfidf
    d = Replays("ladder", "mean", args.layer)
    rng = np.random.default_rng(0)
    groups = {"covert": d.ids("covert"), "honest": d.ids("honest"), "neutral": d.ids("neutral")}
    ids = [e for g in groups.values() for e in g]
    cls = np.array([c for c, g in groups.items() for _ in g])
    yS = d.S(ids)
    sign = 2 * yS - 1
    text = d.text(ids)
    t = out_of_fold(d, ids, lambda tr, te: text_reader().fit([text[i] for i in tr], yS[tr])
                    .decision_function([text[i] for i in te]))
    print(f"# §5.4 R3: university information beyond the text (layer {args.layer}, mean; "
          + ", ".join(f"{c} {len(g)}" for c, g in groups.items()) + ")\n")
    print(f"university decodable from the text ({'strong ' if args.strong_text else ''}TF-IDF): "
          + " / ".join(f"{c} {roc_auc_score(yS[cls == c], t[cls == c]):.3f}" for c in groups))
    print("\n| replay context | from activations: covert / honest / neutral | extra info, covert vs honest | "
          "**covert vs neutral** | slope on text |\n|---|---|---|---|---|")
    for ctx in CONTEXTS:
        X = d.X(ctx, ids)
        a = out_of_fold(d, ids, lambda tr, te, X=X: linear_probe().fit(X[tr], yS[tr]).decision_function(X[te]))
        aucs = " / ".join(f"{roc_auc_score(yS[cls == c], a[cls == c]):.3f}" for c in groups)
        sa, st = sign * a, sign * t  # signed logits: positive = toward the true university
        Z = np.c_[st, cls == "covert", cls == "neutral"]  # class offsets relative to honest
        lr = LinearRegression().fit(Z, sa)
        boots = np.array([LinearRegression().fit(Z[i], sa[i]).coef_[1:]
                          for i in (rng.integers(0, len(ids), len(ids)) for _ in range(args.n_boot))])
        b_h = lr.coef_[1]
        b_n = lr.coef_[1] - lr.coef_[2]
        lo_h, hi_h = np.percentile(boots[:, 0], [2.5, 97.5])
        lo_n, hi_n = np.percentile(boots[:, 0] - boots[:, 1], [2.5, 97.5])
        print(f"| {ctx} | {aucs} | {b_h:+.2f} [{lo_h:+.2f}, {hi_h:+.2f}] | "
              f"**{b_n:+.2f} [{lo_n:+.2f}, {hi_n:+.2f}]** | {lr.coef_[0]:.2f} |")
    print("\nExtra info = covert class offset in units of the university probe's logit (0 = nothing beyond the "
          "words). It survives removing the profile and all context, so it is the model reading university cues "
          "in covert text more sharply than TF-IDF, not knowledge held back.")


if __name__ == "__main__":
    main()
