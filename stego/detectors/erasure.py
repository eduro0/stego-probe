"""The fix of paper §5.2: erase the prompt's main direction from the activations.

Every summary is replayed under two prompts, so the prompt's effect on identical text can be measured: the
paired differences (partner minus honest, partner minus sanctioned) of the same summary's activations. They
mostly point in one direction. Estimate it on the training seed, project it out of every activation, retrain.
A simple form of concept erasure (LEACE, INLP); one direction is enough (Table 4).
"""
from __future__ import annotations

import numpy as np

PAIRS = (("partner", "honest"), ("partner", "sanctioned"))


def prompt_basis(d, ids, k: int = 1, pairs=PAIRS, **x_kw) -> np.ndarray:
    """Orthonormal basis (dim, k) of the top-k paired prompt-difference directions on identical summaries.
    x_kw (feat, layer) select the activations, as in Replays.X."""
    diffs = []
    for a, b in pairs:
        ok = [e for e in ids if d.has(e, a) and d.has(e, b)]
        diffs.append(d.X(a, ok, **x_kw) - d.X(b, ok, **x_kw))
    _, _, vt = np.linalg.svd(np.concatenate(diffs), full_matrices=False)  # uncentred: keep the mean shift
    return vt[:k].T


def project_out(X: np.ndarray, B: np.ndarray | None) -> np.ndarray:
    return X if B is None else X - (X @ B) @ B.T
