"""The detectors of paper §4-5: a linear probe on activations and two bag-of-words text readers, trained and
tested across seeds.

    linear_probe   logistic regression on standardized residual-stream activations (C=0.01), as in
                   deception / collusion probing work; one vector per summary (token mean, layer 18)
    tfidf          word 1-2-gram TF-IDF + logistic regression on the summary text
    tfidf_strong   word 1-3-grams and character 2-5-grams (the stronger text baseline of Table 2)

Every probe is trained on one seed's episodes and tested on the other's, and the reverse; test scores are
pooled. Partner runs with the same seed share candidate profiles, so no profile is on both sides.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from stego.detectors.erasure import project_out, prompt_basis

SEEDS = (0, 1)


def linear_probe(C: float = 0.01):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def tfidf():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(C=1.0, max_iter=3000))


def tfidf_strong():
    return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 3), min_df=2),
                                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
                         LogisticRegression(C=1.0, max_iter=3000))


def cross_seed(groups: dict[str, list[str]], seed_of: Callable[[str], int],
               fit_score: Callable[[dict, dict, int], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Leave-one-seed-out. groups: {name: episode ids}. For each train seed s, fit_score(train, test, s) gets
    the ids of each group in seed s and in the other seed, and returns {key: test scores}; scores are pooled
    over the two folds (fold with train seed 0 first)."""
    out: dict[str, list] = {}
    for s in SEEDS:
        train = {k: [e for e in v if seed_of(e) == s] for k, v in groups.items()}
        test = {k: [e for e in v if seed_of(e) != s] for k, v in groups.items()}
        for k, v in fit_score(train, test, s).items():
            out.setdefault(k, []).append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in out.items()}


def pooled_order(ids: list[str], seed_of: Callable[[str], int]) -> list[str]:
    """The order in which cross_seed returns the test scores of one group."""
    return [e for s in SEEDS for e in ids if seed_of(e) != s]


def probe_scores(d, groups: dict[str, list[str]], train: list[tuple[str, str, int]],
                 test: dict[str, tuple[str, str]], erase_ids: list[str] | None = None, k: int = 1,
                 **x_kw) -> dict[str, np.ndarray]:
    """Linear probe across seeds on a replay dataset `d` (stego.replay.dataset.Replays).

    groups  {name: episode ids}, e.g. {"covert": ..., "honest": ...}
    train   [(group, context, label)]: rows to fit on, e.g. [("covert", "partner", 1), ("honest", "honest", 0)]
    test    {key: (group, context)}: held-out rows to score, e.g. {"honest@partner": ("honest", "partner")}
    erase_ids  if given, erase the top-k paired prompt directions estimated on these episodes of the training
            seed before fitting and scoring (§5.2)
    """
    def fs(tr, te, s):
        B = prompt_basis(d, d.by_seed(erase_ids, s), k, **x_kw) if erase_ids else None
        X = np.concatenate([d.X(ctx, tr[g], **x_kw) for g, ctx, _ in train])
        y = np.concatenate([np.full(len(tr[g]), lab) for g, _, lab in train])
        clf = linear_probe().fit(project_out(X, B), y)
        return {key: clf.decision_function(project_out(d.X(ctx, te[g], **x_kw), B)) for key, (g, ctx) in test.items()}

    return cross_seed(groups, d.seed, fs)


def text_scores(d, pos: list[str], neg: list[str], make=tfidf) -> tuple[np.ndarray, np.ndarray]:
    """Text reader on the summaries across seeds: (held-out scores of pos, of neg)."""
    def fs(tr, te, s):
        clf = make().fit(d.text(tr["pos"] + tr["neg"]), np.r_[np.ones(len(tr["pos"])), np.zeros(len(tr["neg"]))])
        return {"pos": clf.decision_function(d.text(te["pos"])), "neg": clf.decision_function(d.text(te["neg"]))}

    sc = cross_seed({"pos": pos, "neg": neg}, d.seed, fs)
    return sc["pos"], sc["neg"]
