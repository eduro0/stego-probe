"""Robustness of the activations-over-text gain: paired bootstrap CIs and the summary-length confound.

    uv run python robustness_eval.py [--layer 18] [--feat mean] [--n-boot 1000] [--n-match 200]

Reuses invariant_eval.py (Data, probes, prompt erasure). Every scorer gives leave-one-seed-out out-of-fold scores
for the same test episodes (pooled over both folds), so scorers can be compared episode by episode.

Tasks: D vs H (deceit vs honest) and D vs U (deceit vs neutral partner). Scorers:
  act 2x2 / act erase1   prompt-invariant activation probes (invariant_eval exp A), scored under the goal prompt
                         (goal_nothink) and under the honest prompt (honest_nothink). For D vs U, "own" = goal for
                         both classes, so "erase1" is goal-prompt training with the top paired prompt direction
                         projected out; 2x2 = both classes under both prompts.
  tfidf                  word 1-2 TF-IDF + LR (same config as invariant_eval / probe_eval)
  tfidf+len              same plus standardized word count as an extra column
  tfidf-strong           word 1-3 + char_wb 2-5 TF-IDF + LR
  length                 summary word count as the score

1. Bootstrap: stratified by class (negatives count fixed, so the R@1%FPR threshold is always set by the same
   number of negatives), episodes resampled jointly with the SAME indices for every scorer. Paired difference
   act - tfidf for AUROC, R@1% and R@5%.
2. Length: (a) residualize each scorer's pooled test score on word count with a label-free OLS fit on the pooled
   test set (score ~ words; conservative: removes everything linearly predictable from length, including any
   class signal carried by length) and, as a second variant, the within-class slope (score ~ words + class,
   subtract only the words term); (b) length-matched subsamples: within each seed, every negative is matched to
   one unused positive with the same word count (nearest within +-2 words, random tie-breaking, random order),
   n-match random draws, each with one stratified bootstrap resample -> CI covers matching + sampling noise.
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, make_union

from invariant_eval import Data, erase_basis, probe, project_out, text_probe

G, Hp = "goal_nothink", "honest_nothink"


def recall_at(y, s, fpr):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def metrics(y, s):
    return np.array([roc_auc_score(y, s), recall_at(y, s, 0.01), recall_at(y, s, 0.05)])


def strong_text_probe():
    return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 3), min_df=2),
                                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
                         LogisticRegression(C=1.0, max_iter=3000))


class TextLen:
    """Word 1-2 TF-IDF with the standardized word count appended as one extra column."""

    def fit(self, texts, y):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        T = self.vec.fit_transform(texts)
        w = np.array([len(t.split()) for t in texts], float)
        self.mu, self.sd = w.mean(), w.std()
        self.lr = LogisticRegression(C=1.0, max_iter=3000).fit(self._x(T, texts), y)
        return self

    def _x(self, T, texts):
        w = (np.array([len(t.split()) for t in texts], float) - self.mu) / self.sd
        return sp.hstack([T, sp.csr_matrix(w[:, None])]).tocsr()

    def decision_function(self, texts):
        return self.lr.decision_function(self._x(self.vec.transform(texts), texts))


def scores(d: Data, pos, neg):
    """Out-of-fold scores for every scorer, aligned to test order (pos then neg within each test seed)."""
    H = d.cls["honest"]
    partner_all = d.cls["deceit"] + d.cls["neutral"] + d.cls["deceit_flagged"]
    # prompt each class was generated under (honest episodes: honest prompt; partner: goal prompt)
    own_neg = Hp if neg is H else G
    out = {k: [] for k in ["act 2x2 @goal", "act 2x2 @honest", "act erase1 @goal", "act erase1 @honest",
                           "tfidf", "tfidf+len", "tfidf-strong", "length"]}
    ys, words, seeds = [], [], []
    for s in (0, 1):
        t = 1 - s
        pp, nn = d.by_seed(pos, s), d.by_seed(neg, s)
        pt, nt = d.by_seed(pos, t), d.by_seed(neg, t)
        te = pt + nt
        ys.append(np.r_[np.ones(len(pt)), np.zeros(len(nt))])
        words.append(np.array([len(d.eps[e]["summary"].split()) for e in te], float))
        seeds.append(np.full(len(te), t))
        # 2x2: both classes under both prompts
        X = np.concatenate([d.X(G, pp), d.X(Hp, pp), d.X(G, nn), d.X(Hp, nn)])
        y = np.r_[np.ones(2 * len(pp)), np.zeros(2 * len(nn))]
        clf = probe().fit(X, y)
        out["act 2x2 @goal"].append(clf.decision_function(d.X(G, te)))
        out["act 2x2 @honest"].append(clf.decision_function(d.X(Hp, te)))
        # own prompts + erase 1
        B = erase_basis(d, d.by_seed(partner_all, s) + d.by_seed(H, s), 1)
        X = project_out(np.concatenate([d.X(G, pp), d.X(own_neg, nn)]), B)
        clf = probe().fit(X, np.r_[np.ones(len(pp)), np.zeros(len(nn))])
        out["act erase1 @goal"].append(clf.decision_function(project_out(d.X(G, te), B)))
        out["act erase1 @honest"].append(clf.decision_function(project_out(d.X(Hp, te), B)))
        # text
        ttr = [d.eps[e]["summary"] for e in pp + nn]
        tte = [d.eps[e]["summary"] for e in te]
        ytr = np.r_[np.ones(len(pp)), np.zeros(len(nn))]
        out["tfidf"].append(text_probe().fit(ttr, ytr).decision_function(tte))
        out["tfidf+len"].append(TextLen().fit(ttr, ytr).decision_function(tte))
        out["tfidf-strong"].append(strong_text_probe().fit(ttr, ytr).decision_function(tte))
        out["length"].append(words[-1])
    return ({k: np.concatenate(v) for k, v in out.items()}, np.concatenate(ys), np.concatenate(words),
            np.concatenate(seeds))


def strat_idx(rng, y):
    p, n = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))]


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5], axis=0)
    return lo, hi


def fm(v, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def bootstrap_table(sc, y, n_boot, seed, act_keys, text_key="tfidf", title=""):
    rng = np.random.default_rng(seed)
    keys = list(sc)
    point = {k: metrics(y, sc[k]) for k in keys}
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        i = strat_idx(rng, y)
        for k in keys:
            boots[k].append(metrics(y[i], sc[k][i]))
    boots = {k: np.array(v) for k, v in boots.items()}
    nneg = int((y == 0).sum())
    print(f"\n### {title}  (pos={int(y.sum())}, neg={nneg}; R@1% threshold = 99th pct of {nneg} negatives, "
          f"i.e. ~{0.01 * nneg:.1f} negatives above it; R@5% ~{0.05 * nneg:.1f})")
    print("| scorer | AUROC | R@1%FPR | R@5%FPR |\n|---|---|---|---|")
    for k in keys:
        lo, hi = ci(boots[k])
        print(f"| {k} | " + " | ".join(fm(point[k][j], lo[j], hi[j]) for j in range(3)) + " |")
    for a in act_keys:
        for tk in [text_key, "tfidf-strong"]:
            dlt = boots[a] - boots[tk]
            lo, hi = ci(dlt)
            pv = [(dlt[:, j] <= 0).mean() for j in range(3)]
            print(f"| Δ {a} − {tk} | " + " | ".join(
                fm(point[a][j] - point[tk][j], lo[j], hi[j], True) + f" (P≤0 {pv[j]:.3f})" for j in range(3)) + " |")


def residualize(sc, y, w):
    out_m, out_c = {}, {}
    for k, s in sc.items():
        if k == "length":
            continue
        out_m[k] = s - LinearRegression().fit(w[:, None], s).predict(w[:, None])
        lr = LinearRegression().fit(np.c_[w, y], s)
        out_c[k] = s - lr.coef_[0] * w
    return out_m, out_c


def match(rng, y, w, seeds, caliper=2):
    """Indices of a length-matched subsample: every negative paired with one unused positive (within seed)."""
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
            best = cand[dist == dist.min()]
            b = rng.choice(best)
            free.remove(b)
            keep += [j, b]
    return np.array(keep)


def matched_table(sc, y, w, seeds, n_match, seed, title, act_keys):
    rng = np.random.default_rng(seed)
    keys = list(sc)
    res = {k: [] for k in keys}
    pt = {k: [] for k in keys}
    sizes = []
    for _ in range(n_match):
        m = match(rng, y, w, seeds)
        sizes.append(len(m) // 2)
        ym = y[m]
        for k in keys:
            pt[k].append(roc_auc_score(ym, sc[k][m]))
        i = m[strat_idx(rng, ym)]
        for k in keys:
            res[k].append(metrics(y[i], sc[k][i]))
    res = {k: np.array(v) for k, v in res.items()}
    print(f"\n### {title}: length-matched (n-match={n_match}, {np.mean(sizes):.0f} pairs per draw, "
          f"caliper ±2 words)  — AUROC mean over draws [95% CI over draws × bootstrap]")
    print("| scorer | AUROC (matched) | R@1%FPR (matched) | R@5%FPR (matched) |\n|---|---|---|---|")
    for k in keys:
        lo, hi = ci(res[k])
        print(f"| {k} | {fm(np.mean(pt[k]), lo[0], hi[0])} | {fm(res[k][:, 1].mean(), lo[1], hi[1])} | "
              f"{fm(res[k][:, 2].mean(), lo[2], hi[2])} |")
    for a in act_keys:
        dlt = res[a] - res["tfidf"]
        lo, hi = ci(dlt)
        pv = (dlt[:, 1] <= 0).mean()
        print(f"| Δ {a} − tfidf | {fm(dlt[:, 0].mean(), lo[0], hi[0], True)} | "
              f"{fm(dlt[:, 1].mean(), lo[1], hi[1], True)} (P≤0 {pv:.3f}) | {fm(dlt[:, 2].mean(), lo[2], hi[2], True)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--feat", default="mean", choices=["mean", "last"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-match", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Data(args.feat, args.layer)
    Dd, H, U = d.cls["deceit"], d.cls["honest"], d.cls["neutral"]
    act_keys = ["act 2x2 @goal", "act erase1 @goal", "act 2x2 @honest", "act erase1 @honest"]
    print(f"# robustness_eval (layer {args.layer}, {args.feat}, n_boot={args.n_boot}, seed={args.seed})")
    for name, pos, neg in [("D vs H", Dd, H), ("D vs U", Dd, U)]:
        sc, y, w, seeds = scores(d, pos, neg)
        print(f"\n## {name}: words pos {w[y == 1].mean():.1f}, neg {w[y == 0].mean():.1f}; "
              f"length-only AUROC {roc_auc_score(y, w):.3f}")
        bootstrap_table(sc, y, args.n_boot, args.seed, act_keys, title=f"{name}: paired bootstrap")
        rm, rc = residualize(sc, y, w)
        print(f"\n### {name}: length-residualized AUROC (marginal fit, label-free / within-class slope)")
        print("| scorer | raw | resid (marginal) | resid (within-class) | corr(score, words) |\n|---|---|---|---|---|")
        rng = np.random.default_rng(args.seed)
        bidx = [strat_idx(rng, y) for _ in range(args.n_boot)]
        for k in rm:
            b = np.array([[roc_auc_score(y[i], rm[k][i]), roc_auc_score(y[i], rc[k][i])] for i in bidx])
            lo, hi = ci(b)
            print(f"| {k} | {roc_auc_score(y, sc[k]):.3f} | {fm(roc_auc_score(y, rm[k]), lo[0], hi[0])} | "
                  f"{fm(roc_auc_score(y, rc[k]), lo[1], hi[1])} | {np.corrcoef(sc[k], w)[0, 1]:+.2f} |")
        for a in ["act 2x2 @goal", "act erase1 @goal"]:
            b = np.array([roc_auc_score(y[i], rm[a][i]) - roc_auc_score(y[i], rm["tfidf"][i]) for i in bidx])
            lo, hi = ci(b)
            r1 = np.array([recall_at(y[i], rm[a][i], .01) - recall_at(y[i], rm["tfidf"][i], .01) for i in bidx])
            lo1, hi1 = ci(r1)
            print(f"| Δ {a} − tfidf (marginal resid) | AUROC "
                  f"{fm(roc_auc_score(y, rm[a]) - roc_auc_score(y, rm['tfidf']), lo, hi, True)} | R@1% "
                  f"{fm(recall_at(y, rm[a], .01) - recall_at(y, rm['tfidf'], .01), lo1, hi1, True)} | | |")
        matched_table(sc, y, w, seeds, args.n_match, args.seed, name, ["act 2x2 @goal", "act erase1 @goal"])


if __name__ == "__main__":
    main()
