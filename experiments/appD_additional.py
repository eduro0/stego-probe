"""Appendix D: additional results, and the direct searches for an internal signal that came back null.

    uv run python experiments/appD_additional.py [--only layers,direction,plan,context,hidden] [--feedback16]

    layers     covert vs honest and covert vs neutral with the prompt fixed, at every layer and with the
               last-token feature; the cleaned probe under swapped prompts at every layer   (prompt_swap)
    direction  "concealment direction": partner minus sanctioned prompt on the same summaries, projected
               within a fixed prompt, against random directions                           (prompt_swap)
    plan       a probe on the </think> token, where the plan has just been written, transferred to the
               summary tokens                                                              (prompt_swap)
    context    the university decoded at the summary tokens with the profile in context, under each prompt,
               against the text                                                            (prompt_swap)
    hidden     hidden-state counterfactual: how far the summary's activations move when the attribute is
               flipped                                                                      (flip)
--feedback16 restricts partner episodes to the feedback-window-16 runs (the first data batch, extraction v1).
"""
from __future__ import annotations

import argparse

import numpy as np

from stego.detectors.metrics import auroc, fmt_pair
from stego.detectors.probes import cross_seed, linear_probe, probe_scores, text_scores
from stego.replay.dataset import Replays


def classes(d: Replays, feedback16: bool) -> dict[str, list[str]]:
    where = {"feedback_window": 16} if feedback16 else {}
    return {"covert": d.ids("covert", **where), "neutral": d.ids("neutral", **where),
            "flagged": d.ids("flagged", **where), "honest": d.ids("honest")}


def same_prompt(d, g, pos, neg, ctx="partner", **kw):
    r = probe_scores(d, g, [(pos, ctx, 1), (neg, ctx, 0)], {"p": (pos, ctx), "n": (neg, ctx)}, **kw)
    return auroc(r["p"], r["n"])


def layers(d: Replays, g: dict) -> None:
    print("## Layers and positions (AUROC; prompt fixed = both classes replayed under the partner prompt)")
    print("| layer | covert vs honest, prompt fixed | covert vs neutral, prompt fixed | cleaned probe, prompts swapped |"
          "\n|---|---|---|---|")
    ei = [e for c in ("covert", "neutral", "flagged", "honest") for e in g[c]]
    for layer in d.layers:
        sw = probe_scores(d, g, [("covert", "partner", 1), ("honest", "honest", 0)],
                          {"p": ("covert", "honest"), "n": ("honest", "partner")}, erase_ids=ei, layer=layer)
        print(f"| {layer} | {same_prompt(d, g, 'covert', 'honest', layer=layer):.3f} | "
              f"{same_prompt(d, g, 'covert', 'neutral', layer=layer):.3f} | {auroc(sw['p'], sw['n']):.3f} |")
    print(f"last summary token, layer {d.layer}: covert vs honest {same_prompt(d, g, 'covert', 'honest', feat='last'):.3f}, "
          f"covert vs neutral {same_prompt(d, g, 'covert', 'neutral', feat='last'):.3f}")


def direction(d: Replays, g: dict, rng, n_null: int = 200) -> None:
    print("\n## Concealment direction: mean (partner - sanctioned) on the same summaries, projected within a "
          "fixed prompt (AUROC; null = 95th percentile of |AUROC - 0.5| over random directions)")
    partner = g["covert"] + g["neutral"] + g["flagged"]
    for s in (0, 1):
        tr = d.by_seed(partner, s)
        v = (d.X("partner", tr) - d.X("sanctioned", tr)).mean(0)
        u = (d.X("sanctioned", tr) - d.X("honest", tr)).mean(0)
        v, u = v / np.linalg.norm(v), u / np.linalg.norm(u)
        print(f" direction from seed {s}: cos(concealment, use = sanctioned - honest) {v @ u:+.3f}")
        for ctx in ("partner", "honest"):
            out = []
            for neg in ("neutral", "flagged", "honest"):
                Xp, Xn = d.X(ctx, d.by_seed(g["covert"], 1 - s)), d.X(ctx, d.by_seed(g[neg], 1 - s))
                null = [abs(auroc(Xp @ r, Xn @ r) - 0.5) for r in rng.standard_normal((n_null, v.shape[0]))]
                out.append(f"covert vs {neg} {auroc(Xp @ v, Xn @ v):.3f} (null {np.percentile(null, 95):.2f})")
            print(f"   under {ctx:<8} " + " | ".join(out))


def plan(d: Replays, g: dict, rng) -> None:
    print("\n## Plan at the end of the reasoning: probe fit on the </think> token (reasoning restored), "
          "scored there and on the summary tokens")
    for neg, neg_cot in (("neutral", "partner_cot"), ("honest", "honest_cot")):
        for name, feat, ctx_p, ctx_n in (("</think> token", "think", "partner_cot", neg_cot),
                                         ("summary tokens, reasoning restored", "mean", "partner_cot", neg_cot),
                                         ("summary tokens, no reasoning", "mean", "partner", "partner")):
            def fs(tr, te, s, feat=feat, ctx_p=ctx_p, ctx_n=ctx_n, neg=neg, neg_cot=neg_cot):
                X = np.r_[d.X("partner_cot", tr["covert"], feat="think"), d.X(neg_cot, tr[neg], feat="think")]
                clf = linear_probe().fit(X, np.r_[np.ones(len(tr["covert"])), np.zeros(len(tr[neg]))])
                return {"p": clf.decision_function(d.X(ctx_p, te["covert"], feat=feat)),
                        "n": clf.decision_function(d.X(ctx_n, te[neg], feat=feat))}

            r = cross_seed({"covert": g["covert"], neg: g[neg]}, d.seed, fs)
            print(f"  covert vs {neg:<8} -> {name:<36} {fmt_pair(r['p'], r['n'], rng)}")


def context(d: Replays, g: dict) -> None:
    print("\n## University decoded at the summary tokens with the profile in context (AUROC)")
    partner = [e for c in ("covert", "neutral", "flagged") for e in g[c]] + d.ids("unlabeled")
    for name, ids, ctxs in (("partner summaries", partner, ("partner", "sanctioned", "honest")),
                            ("honest summaries", g["honest"], ("honest", "partner"))):
        A, B = ([e for e in ids if d.eps[e]["S"] == v] for v in "AB")
        text = auroc(*text_scores(d, A, B))
        acts = []
        for ctx in ctxs:
            r = probe_scores(d, {"A": A, "B": B}, [("A", ctx, 1), ("B", ctx, 0)], {"A": ("A", ctx), "B": ("B", ctx)})
            acts.append(f"under {ctx} prompt {auroc(r['A'], r['B']):.3f}")
        print(f"  {name}: activations " + ", ".join(acts) + f"; text (TF-IDF) {text:.3f}")
    r = probe_scores(d, {"A": [e for e in g["honest"] if d.eps[e]["S"] == "A"],
                         "B": [e for e in g["honest"] if d.eps[e]["S"] == "B"]},
                     [("A", "honest", 1), ("B", "honest", 0)], {"A": ("A", "honest"), "B": ("B", "honest")},
                     feat="stok")
    print(f"  sanity: from the profile's university token {auroc(r['A'], r['B']):.3f}")


def hidden(layer: int) -> None:
    d = Replays("flip", "mean", layer)
    print(f"\n## Hidden-state counterfactual: ||activation(true) - activation(flipped)|| at layer {layer}, "
          "summary-token mean, / median over honest university (own prompt)")

    def shift(ids, prompt=None):
        return np.array([np.linalg.norm(d.X(p, [e])[0] - d.X(f"{p}_flipped", [e])[0])
                         for e in ids for p in [prompt or d.eps[e]["kind"]]])

    scale = np.median(shift(d.ids("honest")))
    for a in ("university", "nationality"):
        c, h = d.ids("covert", a), d.ids("honest", a)
        own = auroc(shift(c) / scale, shift(h) / scale)
        fixed = auroc(shift(c, "partner") / scale, shift(h, "partner") / scale)
        tau = np.quantile(shift(d.ids("honest")) / scale, 0.99)
        print(f"  {a}: covert vs honest, own prompts {own:.3f}; both under the partner prompt {fixed:.3f}; "
              f"false alarms at the university threshold {(shift(h) / scale > tau).mean():.2f}")
    print("  It tracks the prompt (honest summaries move more under the partner prompt), so it was dropped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--only", default="layers,direction,plan,context,hidden")
    ap.add_argument("--feedback16", action="store_true", help="partner episodes from the feedback-16 runs only")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    only = args.only.split(",")
    print(f"# Appendix D (layer {args.layer}{', feedback-16 runs only' if args.feedback16 else ''})\n")
    if set(only) - {"hidden"}:
        d = Replays("prompt_swap", "mean", args.layer)
        g = classes(d, args.feedback16)
        print("  " + ", ".join(f"{c} {len(v)}" for c, v in g.items()) + "\n")
        for name, f in (("layers", lambda: layers(d, g)), ("direction", lambda: direction(d, g, rng)),
                        ("plan", lambda: plan(d, g, rng)), ("context", lambda: context(d, g))):
            if name in only:
                f()
    if "hidden" in only:
        hidden(args.layer)


if __name__ == "__main__":
    main()
