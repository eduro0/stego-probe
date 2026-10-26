"""Prompt-invariant deceit detection on the probe dataset (research brief approaches A-D, existing data only).

    uv run python invariant_eval.py [--layer 18] [--feat mean] [--exp A,B,C,D] [--sweep]

Reads data/probe_dataset/ directly (rows.jsonl, episodes.jsonl, <feat>.npy). Classes: D = deceit (y=1),
H = honest (y=0), U = neutral (partner, no plan), F = deceit_flagged. Every split is leave-one-seed-out with
pooled test scores; metrics are AUROC [95% bootstrap CI] and recall at 1% FPR.

A  prompt-invariant probes. Train variants: own (each class under its own prompt, the confounded baseline),
   honest-prompt only, goal-prompt only, 2x2 (both classes under both prompts), own + prompt erasure
   (project out the subspace of paired goal-minus-honest / goal-minus-sanct differences of identical
   summaries, fit on the train seed). Every variant is scored on D vs H under the goal prompt, under the
   honest prompt, on the swapped pair (D/honest vs H/goal), and on prompt sensitivity (score shift of the same
   summary between goal and honest replays, in units of the within-prompt class gap).
B  concealed knowledge: S decoded at summary tokens beyond what the summary text carries. Per episode, an
   out-of-fold S logit from activations and from TF-IDF; regress the signed activation logit on the signed
   text logit + class, per prompt. Also the no-text-S subset (honest summaries) as a context-only readout.
C  plan -> output: D vs U probes on the </think> vector (goal_think), transferred to summary tokens.
D  concealment direction: goal - sanct mean difference on identical summaries, projected within a fixed
   prompt (D vs U, D vs F, D vs H) and compared to a random-direction null.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path(__file__).resolve().parent / "data" / "probe_dataset"
RNG = np.random.default_rng(0)


def probe(C=0.01):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def text_probe():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(C=1.0, max_iter=3000))


def recall_at_fpr(y, s, fpr=0.01):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def fmt(y, s, n=500):
    y, s = np.asarray(y), np.asarray(s)
    a = roc_auc_score(y, s)
    b = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            b.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{a:.3f} [{lo:.2f},{hi:.2f}] R@1%={recall_at_fpr(y, s):.2f}"


class Data:
    def __init__(self, feat: str, layer: int):
        self.rows = [json.loads(x) for x in (D / "rows.jsonl").read_text().splitlines()]
        self.eps = {e["episode_id"]: e for e in map(json.loads, (D / "episodes.jsonl").read_text().splitlines())}
        self.layers = list(np.load(D / "layers.npy"))
        self.index = {(r["episode_id"], r["pass"]): r["row"] for r in self.rows}
        self._cache = {}
        self.feat, self.layer = feat, layer
        self.cls = {}
        for r in self.rows:
            self.cls.setdefault(r["cls"], set()).add(r["episode_id"])
        self.cls = {k: sorted(v) for k, v in self.cls.items()}

    def arr(self, feat=None, layer=None):
        feat, layer = feat or self.feat, layer or self.layer
        key = (feat, layer)
        if key not in self._cache:
            a = np.load(D / f"{feat}.npy", mmap_mode="r")
            self._cache[key] = np.asarray(a[:, self.layers.index(layer)], dtype=np.float32)
        return self._cache[key]

    def X(self, pass_, eps, feat=None, layer=None):
        return self.arr(feat, layer)[[self.index[(e, pass_)] for e in eps]]

    def seed(self, e):
        return self.eps[e]["seed"]

    def by_seed(self, eps, s):
        return [e for e in eps if self.seed(e) == s]


# ---------------------------------------------------------------- A: prompt-invariant probes
def erase_basis(d: Data, eps, k, pairs=(("goal_nothink", "honest_nothink"), ("goal_nothink", "sanct_nothink"))):
    """Orthonormal basis (dim, k) of the top-k paired prompt-difference directions on identical summaries."""
    diffs = []
    for a, b in pairs:
        ok = [e for e in eps if (e, a) in d.index and (e, b) in d.index]
        diffs.append(d.X(a, ok) - d.X(b, ok))
    M = np.concatenate(diffs)
    _, _, vt = np.linalg.svd(M, full_matrices=False)  # uncentred: keep the mean shift
    return vt[:k].T


def project_out(X, B):
    return X - (X @ B) @ B.T


def exp_A(d: Data, ks=(1, 2, 4, 8, 16, 32, 64)):
    Dd, H, U = d.cls["deceit"], d.cls["honest"], d.cls["neutral"]
    partner_all = Dd + U + d.cls["deceit_flagged"]
    print(f"## A. prompt-invariant probes (layer {d.layer}, {d.feat})  D={len(Dd)} H={len(H)} U={len(U)}")

    def train_set(variant, s):
        dd, hh = d.by_seed(Dd, s), d.by_seed(H, s)
        g, h = "goal_nothink", "honest_nothink"
        parts = {"own": [(g, dd, 1), (h, hh, 0)],
                 "honest-only": [(h, dd, 1), (h, hh, 0)],
                 "goal-only": [(g, dd, 1), (g, hh, 0)],
                 "2x2": [(g, dd, 1), (h, dd, 1), (g, hh, 0), (h, hh, 0)]}[variant]
        X = np.concatenate([d.X(p, e) for p, e, _ in parts])
        y = np.concatenate([np.full(len(e), c) for _, e, c in parts])
        return X, y

    variants = ["own", "honest-only", "goal-only", "2x2"] + [f"own-erase{k}" for k in ks] + \
        [f"2x2-erase{k}" for k in (1, 4, 16)]
    print(f"{'train':<16}{'goal: D vs H':<30}{'honest: D vs H':<30}{'swap: D/h vs H/g':<30}"
          f"{'U vs H (goal)':<14}{'D vs U (goal)':<14}shift/gap")
    for v in variants:
        sc = {k: [] for k in ["Dg", "Dh", "Hg", "Hh", "Ug"]}
        for s in (0, 1):
            t = 1 - s
            base = v.split("-erase")[0]
            X, y = train_set(base, s)
            B = None
            if "-erase" in v:
                k = int(v.split("erase")[1])
                B = erase_basis(d, d.by_seed(partner_all, s) + d.by_seed(H, s), k)
                X = project_out(X, B)
            clf = probe().fit(X, y)
            def f(Z, clf=clf, B=B):
                return clf.decision_function(project_out(Z, B) if B is not None else Z)
            dd, hh, uu = d.by_seed(Dd, t), d.by_seed(H, t), d.by_seed(U, t)
            sc["Dg"].append(f(d.X("goal_nothink", dd)))
            sc["Dh"].append(f(d.X("honest_nothink", dd)))
            sc["Hg"].append(f(d.X("goal_nothink", hh)))
            sc["Hh"].append(f(d.X("honest_nothink", hh)))
            sc["Ug"].append(f(d.X("goal_nothink", uu)))
        sc = {k: np.concatenate(x) for k, x in sc.items()}

        def pair(a, b, sc=sc):
            return np.r_[np.ones(len(sc[a])), np.zeros(len(sc[b]))], np.r_[sc[a], sc[b]]
        shift = np.mean(np.r_[sc["Dg"] - sc["Dh"], sc["Hg"] - sc["Hh"]])
        gap = 0.5 * (sc["Dg"].mean() - sc["Hg"].mean() + sc["Dh"].mean() - sc["Hh"].mean())
        print(f"{v:<16}{fmt(*pair('Dg', 'Hg')):<30}{fmt(*pair('Dh', 'Hh')):<30}{fmt(*pair('Dh', 'Hg')):<30}"
              f"{roc_auc_score(*pair('Ug', 'Hg')):<14.3f}{roc_auc_score(*pair('Dg', 'Ug')):<14.3f}"
              f"{shift / gap:+.2f}")

    # text baselines on the same seeds
    def text_cv(pos, neg):
        ys, ss = [], []
        for s in (0, 1):
            tr = [(e, 1) for e in d.by_seed(pos, s)] + [(e, 0) for e in d.by_seed(neg, s)]
            te = [(e, 1) for e in d.by_seed(pos, 1 - s)] + [(e, 0) for e in d.by_seed(neg, 1 - s)]
            clf = text_probe().fit([d.eps[e]["summary"] for e, _ in tr], [c for _, c in tr])
            ss.append(clf.decision_function([d.eps[e]["summary"] for e, _ in te]))
            ys.append([c for _, c in te])
        return np.concatenate(ys), np.concatenate(ss)
    print(f"{'TF-IDF':<16}{fmt(*text_cv(Dd, H)):<30}{'(prompt-free)':<30}{'':<30}"
          f"{roc_auc_score(*text_cv(U, H)):<14.3f}{roc_auc_score(*text_cv(Dd, U)):<14.3f}")
    lw = lambda e: len(d.eps[e]["summary"].split())
    y = np.r_[np.ones(len(Dd)), np.zeros(len(H))]
    print(f"{'length (words)':<16}{roc_auc_score(y, [lw(e) for e in Dd + H]):.3f}")


# ---------------------------------------------------------------- B: concealed knowledge
def oof(fit_score, eps, ys):
    """Leave-one-seed-out out-of-fold scores, returned in eps order."""
    out = np.zeros(len(eps))
    seeds = np.array([EPS_SEED[e] for e in eps])
    for s in (0, 1):
        tr, te = np.where(seeds == s)[0], np.where(seeds != s)[0]
        out[te] = fit_score(tr, te)
    return out


EPS_SEED: dict = {}


def exp_B(d: Data):
    EPS_SEED.update({e: d.seed(e) for e in d.eps})
    Dd, H, U = d.cls["deceit"], d.cls["honest"], d.cls["neutral"]
    eps = Dd + H + U
    yS = np.array([int(d.eps[e]["S"] == "A") for e in eps])
    sign = 2 * yS - 1
    cls = np.array(["D"] * len(Dd) + ["H"] * len(H) + ["U"] * len(U))
    text = [d.eps[e]["summary"] for e in eps]
    # S probes are trained on all three classes together so the readout is the same function for each.
    t = oof(lambda tr, te: text_probe().fit([text[i] for i in tr], yS[tr]).decision_function([text[i] for i in te]),
            eps, yS)
    print(f"\n## B. concealed knowledge: S at summary tokens beyond the text (layer {d.layer}, {d.feat})")
    print("text->S AUROC by class: " + "  ".join(f"{c} {roc_auc_score(yS[cls == c], t[cls == c]):.3f}" for c in "DHU"))
    print(f"{'prompt':<16}{'act->S AUROC D / H / U':<30}{'beta_class(D-H)':<18}{'beta_class(D-U)':<18}"
          f"{'slope on text':<14}{'excess D':<10}{'excess H':<10}excess U")
    for p in ["honest_nothink", "goal_nothink"]:
        X = d.X(p, eps)
        a = oof(lambda tr, te, X=X: probe().fit(X[tr], yS[tr]).decision_function(X[te]), eps, yS)
        aucs = " / ".join(f"{roc_auc_score(yS[cls == c], a[cls == c]):.3f}" for c in "DHU")
        # signed logits: positive = toward the true S
        sa, st = sign * a, sign * t
        Z = np.c_[st, cls == "D", cls == "U"]
        lr = LinearRegression().fit(Z, sa)
        res = sa - LinearRegression().fit(st[:, None], sa).predict(st[:, None])
        bD, bU = lr.coef_[1], lr.coef_[2]
        boots = []
        for _ in range(300):
            i = RNG.integers(0, len(eps), len(eps))
            boots.append(LinearRegression().fit(Z[i], sa[i]).coef_[1:])
        boots = np.array(boots)
        ciD = np.percentile(boots[:, 0], [2.5, 97.5])
        ciDU = np.percentile(boots[:, 0] - boots[:, 1], [2.5, 97.5])
        print(f"{p:<16}{aucs:<30}{bD:+.2f} [{ciD[0]:+.2f},{ciD[1]:+.2f}]  {bD - bU:+.2f} [{ciDU[0]:+.2f},{ciDU[1]:+.2f}]  "
              f"{lr.coef_[0]:<14.2f}" + "".join(f"{res[cls == c].mean():<+10.2f}" for c in "DHU"))
    # context-only readout: S probe trained on honest summaries (text carries ~no S), applied to every class
    print("S probe trained on H only (context readout), applied out-of-fold to D/U (signed logit mean):")
    for p in ["honest_nothink", "goal_nothink"]:
        out = {c: [] for c in "DHU"}
        for s in (0, 1):
            hh = d.by_seed(H, s)
            clf = probe().fit(d.X(p, hh), [int(d.eps[e]["S"] == "A") for e in hh])
            for c, group in [("D", Dd), ("H", H), ("U", U)]:
                g = d.by_seed(group, 1 - s)
                sg = np.array([2 * int(d.eps[e]["S"] == "A") - 1 for e in g])
                out[c].append(sg * clf.decision_function(d.X(p, g)))
        print(f"  {p:<16}" + "  ".join(f"{c} {np.concatenate(v).mean():+.2f} (acc {np.mean(np.concatenate(v) > 0):.3f})"
                                       for c, v in out.items()))


# ---------------------------------------------------------------- C: plan -> output
def exp_C(d: Data):
    Dd, U, H = d.cls["deceit"], d.cls["neutral"], d.cls["honest"]
    print(f"\n## C. plan at </think> -> summary tokens (layer {d.layer})")
    for name, pos, neg, negpass in [("D vs U", Dd, U, "goal_think"), ("D vs H", Dd, H, "honest_think")]:
        res = {}
        for tr_feat, tr_pass_pos, tr_pass_neg in [("think", "goal_think", negpass)]:
            for te_feat, te_pos, te_neg in [("think", "goal_think", negpass), ("mean", "goal_think", negpass),
                                            ("mean", "goal_nothink", "goal_nothink")]:
                ys, ss = [], []
                for s in (0, 1):
                    p_tr, n_tr = d.by_seed(pos, s), d.by_seed(neg, s)
                    X = np.r_[d.X(tr_pass_pos, p_tr, feat=tr_feat), d.X(tr_pass_neg, n_tr, feat=tr_feat)]
                    clf = probe().fit(X, np.r_[np.ones(len(p_tr)), np.zeros(len(n_tr))])
                    p_te, n_te = d.by_seed(pos, 1 - s), d.by_seed(neg, 1 - s)
                    ss.append(np.r_[clf.decision_function(d.X(te_pos, p_te, feat=te_feat)),
                                    clf.decision_function(d.X(te_neg, n_te, feat=te_feat))])
                    ys.append(np.r_[np.ones(len(p_te)), np.zeros(len(n_te))])
                res[f"{te_feat}@{te_pos}/{te_neg}"] = fmt(np.concatenate(ys), np.concatenate(ss))
        for k, v in res.items():
            print(f"  {name}: train think@{'goal_think'}/{negpass} -> test {k:<36}{v}")


# ---------------------------------------------------------------- D: concealment direction
def exp_D(d: Data, n_null=200):
    Dd, U, F, H = d.cls["deceit"], d.cls["neutral"], d.cls["deceit_flagged"], d.cls["honest"]
    partner = Dd + U + F
    print(f"\n## D. concealment direction goal - sanct (layer {d.layer}, {d.feat}); projection, fixed prompt")
    for s in (0, 1):
        tr = d.by_seed(partner, s)
        v = (d.X("goal_nothink", tr) - d.X("sanct_nothink", tr)).mean(0)
        vs = (d.X("sanct_nothink", tr) - d.X("honest_nothink", tr)).mean(0)
        v /= np.linalg.norm(v)
        vs /= np.linalg.norm(vs)
        print(f" train seed {s}: cos(goal-sanct, sanct-honest) = {v @ vs:+.3f}")
        for pname in ["goal_nothink", "honest_nothink"]:
            line = []
            for name, pos, neg in [("D vs U", Dd, U), ("D vs F", Dd, F), ("D vs H", Dd, H)]:
                pp, nn = d.by_seed(pos, 1 - s), d.by_seed(neg, 1 - s)
                Xp, Xn = d.X(pname, pp), d.X(pname, nn)
                y = np.r_[np.ones(len(pp)), np.zeros(len(nn))]
                auc = roc_auc_score(y, np.r_[Xp @ v, Xn @ v])
                auc_s = roc_auc_score(y, np.r_[Xp @ vs, Xn @ vs])
                null = []
                for _ in range(n_null):
                    r = RNG.standard_normal(v.shape[0])
                    null.append(abs(roc_auc_score(y, np.r_[Xp @ r, Xn @ r]) - 0.5))
                line.append(f"{name} conceal {auc:.3f} use {auc_s:.3f} (null |AUC-.5| p95 {np.percentile(null, 95):.3f})")
            print(f"   {pname:<15}" + " | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--feat", default="mean", choices=["mean", "last"])
    ap.add_argument("--exp", default="A,B,C,D")
    args = ap.parse_args()
    d = Data(args.feat, args.layer)
    for e in args.exp.split(","):
        {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D}[e](d)


if __name__ == "__main__":
    main()
