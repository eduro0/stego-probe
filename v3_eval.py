"""v3 replay analysis: no-profile excess S, counterfactual log-prob dependence, bare-text baseline.

    uv run python v3_eval.py [--layer 18] [--feat mean] [--exp a,b,c]

Reads data/probe_dataset_v3/ (build_dataset.py --tags v3b --out data/probe_dataset_v3). Reuses the loaders,
splits and metrics of invariant_eval.py: leave-one-seed-out, pooled test scores, LR (standardized, C=0.01),
AUROC [95% bootstrap CI] and recall at 1% FPR of the negative class. D = deceit, H = honest, U = neutral.

a  exp B (excess S at summary tokens beyond the text) with the profile removed from context
   (honest_noprof, goal_noprof) next to the same regression with the profile in context (*_nothink).
b  counterfactual dependence s = log p(summary | profile S) - log p(summary | profile S flipped), from the
   final-layer teacher-forced log-probs (sum over summary tokens, and per-token mean), per prompt. Zero-shot
   (no training). s > 0 = the summary is more likely under the true S; |s| = dependence in either direction.
c  bare_noprof activations (no system prompt, generic user turn, no profile) as a text baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

import invariant_eval as ie
from invariant_eval import EPS_SEED, fmt, oof, probe, text_probe

V3 = Path(__file__).resolve().parent / "data" / "probe_dataset_v3"
RNG = np.random.default_rng(0)


def load(feat: str, layer: int) -> ie.Data:
    ie.D = V3  # invariant_eval.Data reads its module-level path
    d = ie.Data(feat, layer)
    d.rowinfo = {(r["episode_id"], r["pass"]): r for r in d.rows}
    EPS_SEED.update({e: d.seed(e) for e in d.eps})
    return d


def groups(d):
    return d.cls["deceit"], d.cls["honest"], d.cls["neutral"]


# ---------------------------------------------------------------- a: excess S without the profile
def exp_a(d: ie.Data, n_boot=1000):
    Dd, H, U = groups(d)
    eps = Dd + H + U
    yS = np.array([int(d.eps[e]["S"] == "A") for e in eps])
    sign = 2 * yS - 1
    cls = np.array(["D"] * len(Dd) + ["H"] * len(H) + ["U"] * len(U))
    text = [d.eps[e]["summary"] for e in eps]
    t = oof(lambda tr, te: text_probe().fit([text[i] for i in tr], yS[tr]).decision_function([text[i] for i in te]),
            eps, yS)
    print(f"## a. excess S at summary tokens beyond the text (layer {d.layer}, {d.feat}); "
          f"D={len(Dd)} H={len(H)} U={len(U)}")
    print("text->S AUROC by class: " + "  ".join(f"{c} {roc_auc_score(yS[cls == c], t[cls == c]):.3f}" for c in "DHU"))
    print("| pass | act->S AUROC D / H / U | beta(D-H) [95% CI] | beta(D-U) [95% CI] | slope on text | "
          "excess D / H / U |\n|---|---|---|---|---|---|")
    for p in ["honest_nothink", "honest_noprof", "goal_nothink", "goal_noprof", "bare_noprof"]:
        X = d.X(p, eps)
        a = oof(lambda tr, te, X=X: probe().fit(X[tr], yS[tr]).decision_function(X[te]), eps, yS)
        aucs = " / ".join(f"{roc_auc_score(yS[cls == c], a[cls == c]):.3f}" for c in "DHU")
        sa, st = sign * a, sign * t
        Z = np.c_[st, cls == "D", cls == "U"]
        lr = LinearRegression().fit(Z, sa)
        res = sa - LinearRegression().fit(st[:, None], sa).predict(st[:, None])
        bD, bU = lr.coef_[1], lr.coef_[2]
        boots = np.array([LinearRegression().fit(Z[i], sa[i]).coef_[1:]
                          for i in (RNG.integers(0, len(eps), len(eps)) for _ in range(n_boot))])
        ciD = np.percentile(boots[:, 0], [2.5, 97.5])
        ciDU = np.percentile(boots[:, 0] - boots[:, 1], [2.5, 97.5])
        print(f"| {p} | {aucs} | {bD:+.2f} [{ciD[0]:+.2f}, {ciD[1]:+.2f}] | {bD - bU:+.2f} "
              f"[{ciDU[0]:+.2f}, {ciDU[1]:+.2f}] | {lr.coef_[0]:.2f} | "
              + " / ".join(f"{res[cls == c].mean():+.2f}" for c in "DHU") + " |")


# ---------------------------------------------------------------- b: counterfactual log-prob dependence
def exp_b(d: ie.Data):
    Dd, H, U = groups(d)
    print("\n## b. counterfactual dependence s = log p(summary | S) - log p(summary | S flipped) (zero-shot)")

    def s_of(prompt, eps, stat):
        return np.array([d.rowinfo[(e, f"{prompt}_nothink")][stat] - d.rowinfo[(e, f"{prompt}_flip")][stat]
                         for e in eps])

    for prompt in ("honest", "goal"):
        for stat in ("logp_sum", "logp_mean"):
            sc = {c: s_of(prompt, g, stat) for c, g in (("D", Dd), ("H", H), ("U", U))}
            print(f"\n### {prompt} prompt, {stat.replace('logp_', '')}  "
                  "(mean s D / H / U: " + " / ".join(f"{sc[c].mean():+.3f}" for c in "DHU")
                  + "; frac s>0: " + " / ".join(f"{(sc[c] > 0).mean():.2f}" for c in "DHU") + ")")
            print("| score | D vs H | D vs U | U vs H | R@1%FPR(H): D / U |\n|---|---|---|---|---|")
            for name, f in (("s (toward true S)", lambda x: x), ("|s|", np.abs)):
                cells = []
                for pos, neg in (("D", "H"), ("D", "U"), ("U", "H")):
                    y = np.r_[np.ones(len(sc[pos])), np.zeros(len(sc[neg]))]
                    cells.append(fmt(y, f(np.r_[sc[pos], sc[neg]])))
                thr = np.quantile(f(sc["H"]), 0.99)
                print(f"| {name} | " + " | ".join(cells)
                      + f" | {(f(sc['D']) > thr).mean():.2f} / {(f(sc['U']) > thr).mean():.2f} |")
    # dependence by deceit mechanism (judge label), honest vs goal prompt, sum
    print("\nmean s (sum) by deceit mechanism: honest prompt / goal prompt")
    mech = {}
    for e in Dd:
        mech.setdefault(d.eps[e]["cot_mechanism"], []).append(e)
    for m, g in sorted(mech.items(), key=lambda kv: -len(kv[1])):
        print(f"  {m} (n={len(g)}): {s_of('honest', g, 'logp_sum').mean():+.2f} / {s_of('goal', g, 'logp_sum').mean():+.2f}")
    a = np.r_[s_of("honest", Dd + H + U, "logp_sum")]
    b = np.r_[s_of("goal", Dd + H + U, "logp_sum")]
    print(f"corr(s_honest, s_goal) over D+H+U (sum): {np.corrcoef(a, b)[0, 1]:.3f}")


# ---------------------------------------------------------------- c: bare text baseline
def cv_probe(d, pos, neg, pass_):
    ys, ss = [], []
    for s in (0, 1):
        p, n = d.by_seed(pos, s), d.by_seed(neg, s)
        clf = probe().fit(np.r_[d.X(pass_, p), d.X(pass_, n)], np.r_[np.ones(len(p)), np.zeros(len(n))])
        p, n = d.by_seed(pos, 1 - s), d.by_seed(neg, 1 - s)
        ss.append(clf.decision_function(np.r_[d.X(pass_, p), d.X(pass_, n)]))
        ys.append(np.r_[np.ones(len(p)), np.zeros(len(n))])
    return np.concatenate(ys), np.concatenate(ss)


def cv_text(d, pos, neg):
    ys, ss = [], []
    for s in (0, 1):
        tr = [(e, 1) for e in d.by_seed(pos, s)] + [(e, 0) for e in d.by_seed(neg, s)]
        te = [(e, 1) for e in d.by_seed(pos, 1 - s)] + [(e, 0) for e in d.by_seed(neg, 1 - s)]
        clf = text_probe().fit([d.eps[e]["summary"] for e, _ in tr], [c for _, c in tr])
        ss.append(clf.decision_function([d.eps[e]["summary"] for e, _ in te]))
        ys.append([c for _, c in te])
    return np.concatenate(ys), np.concatenate(ss)


def exp_c(d: ie.Data):
    Dd, H, U = groups(d)
    print(f"\n## c. bare_noprof activations as a text baseline (layer {d.layer}, {d.feat})")
    print("| scorer | D vs H | D vs U | U vs H |\n|---|---|---|---|")
    for name, f in [("bare_noprof probe", lambda p, n: cv_probe(d, p, n, "bare_noprof")),
                    ("honest_noprof probe", lambda p, n: cv_probe(d, p, n, "honest_noprof")),
                    ("goal_nothink probe (same prompt)", lambda p, n: cv_probe(d, p, n, "goal_nothink")),
                    ("honest_nothink probe (same prompt)", lambda p, n: cv_probe(d, p, n, "honest_nothink")),
                    ("TF-IDF", lambda p, n: cv_text(d, p, n))]:
        print(f"| {name} | " + " | ".join(fmt(*f(p, n)) for p, n in ((Dd, H), (Dd, U), (U, H))) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--feat", default="mean", choices=["mean", "last"])
    ap.add_argument("--exp", default="a,b,c")
    args = ap.parse_args()
    d = load(args.feat, args.layer)
    print(f"rows {len(d.rows)}, passes {sorted({r['pass'] for r in d.rows})}")
    for e in args.exp.split(","):
        {"a": exp_a, "b": exp_b, "c": exp_c}[e](d)


if __name__ == "__main__":
    main()
