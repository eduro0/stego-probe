"""Figures 3, 4 and 5 of the paper, from the scores the experiments write to results/.

    uv run python experiments/sec5_1_prompt_swap.py      # -> results/fig3_naive.json
    uv run python experiments/sec5_2_erase_prompt.py     # -> results/fig3_cleaned.json
    uv run python experiments/sec5_3_bare_text.py        # -> results/table2.json
    uv run python experiments/sec5_5_counterfactual.py   # -> results/fig4_counterfactual.json
    uv run --group figures python experiments/figures.py [--no-usetex]   # -> results/fig{3,4,5}_*.pdf

Uses LaTeX for text when it is installed (as in the paper), unless --no-usetex.
"""
from __future__ import annotations

import argparse
import json
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from stego.detectors.metrics import auroc
from stego.paths import RESULTS

matplotlib.use("pdf")
COVERT, HONEST, NEUTRAL, GOAL = "#C2410C", "#1D4E89", "#8A8A8A", "#E3A008"
RNG = np.random.default_rng(0)


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} missing: run the experiment that writes it (see this file's docstring)")
    return json.loads(path.read_text())


def bold(s: str, tex: bool) -> str:
    return rf"\textbf{{{s}}}" if tex else s


def fig3(tex: bool):
    """Each held-out summary's probe score, each class under the partner and the honest prompt."""
    S = {"naive": load("fig3_naive.json"), "cleaned": load("fig3_cleaned.json")}
    titles = {"naive": "(a) Naive intent probe", "cleaned": "(b) Cleaned probe (prompt direction erased)"}
    pos = {"covert@partner": 0, "honest@partner": 1, "covert@honest": 2.6, "honest@honest": 3.6}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2))
    for ax, v in zip(axes, S):
        sc = {k: np.array(x) for k, x in S[v].items()}
        ax.axvspan(-0.5, 1.5, color=GOAL, alpha=0.10, lw=0)
        for k, x in pos.items():
            y = sc[k]
            ax.scatter(x + RNG.uniform(-0.28, 0.28, len(y)), y, s=2.2, color=COVERT if k.startswith("covert") else HONEST,
                       alpha=0.35, lw=0, rasterized=True)
            ax.plot([x - 0.34, x + 0.34], [np.median(y)] * 2, color="black", lw=1.1)
        ax.axhline(0, color="black", lw=0.4, ls=":")
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels(["covert", "honest", "covert", "honest"])
        for x, label, col in ((0.5, "partner prompt", "#7a5600"), (3.1, "honest prompt", "#333")):
            ax.text(x, 1.0, label, ha="center", va="bottom", fontsize=7.5, transform=ax.get_xaxis_transform(),
                    color=col)
        ax.set_xlim(-0.6, 4.2)
        ax.set_title(titles[v], pad=13, loc="left", fontsize=8.5)
        own = auroc(sc["covert@partner"], sc["honest@honest"])
        same = auroc(sc["covert@partner"], sc["honest@partner"])
        swap = auroc(sc["covert@honest"], sc["honest@partner"])
        sep = r"$|$" if tex else "|"
        ax.set_xlabel(f"AUROC own prompts {own:.3f} {sep} same prompt {same:.3f}\n"
                      + bold(f"prompts swapped: {swap:.3f}", tex), fontsize=7)
    axes[0].set_ylabel("probe score (logit)")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(RESULTS / "fig3_prompt_probe.pdf", dpi=300)


def fig4(tex: bool):
    """Distribution of the counterfactual score per class; threshold set on honest university."""
    U = load("fig4_counterfactual.json")
    tau = U["tau"]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.9))
    bins = np.linspace(-8, 22, 46)
    names = {"university": "(a) University (threshold set here)", "nationality": "(b) Nationality (held out)"}
    for ax, a in zip(axes, names):
        for cls, col in (("honest", HONEST), ("neutral", NEUTRAL), ("covert", COVERT)):
            x = np.clip(np.array(U[a][cls]), bins[0], bins[-1])
            n = rf"$n$={len(x)}" if tex else f"n={len(x)}"
            ax.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color=col, lw=0)
            ax.hist(x, bins=bins, density=True, histtype="step", color=col, lw=0.9, label=f"{cls} ({n})")
        ax.axvline(tau, color="black", lw=0.8, ls="--")
        c, h = np.array(U[a]["covert"]), np.array(U[a]["honest"])
        t = r"$\tau$" if tex else "τ"
        ax.text(tau + 0.5, ax.get_ylim()[1] * 0.92, f"{t}: recall {np.mean(c > tau):.2f}, FPR {np.mean(h > tau):.2f}",
                fontsize=7, va="top")
        ax.set_title(names[a], loc="left", fontsize=8.5)
        ax.set_xlabel("counterfactual score (true minus flipped attribute)")
        ax.set_xlim(bins[0], bins[-1])
    axes[0].set_ylabel("density")
    axes[1].legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(RESULTS / "fig4_counterfactual.pdf")


def fig5(tex: bool):
    """Every detector on covert vs honest, grouped by rung of the ladder."""
    n, c, t2, u = load("fig3_naive.json"), load("fig3_cleaned.json"), load("table2.json"), load("fig4_counterfactual.json")
    groups = [  # (heading, colour, rows: label, auroc, lo, hi)
        ("R1: prompt swap", "#9b1c1c", [
            ("intent probe, each class under its own prompt", auroc(n["covert@partner"], n["honest@honest"]), None, None),
            ("same probe, prompts swapped", auroc(n["covert@honest"], n["honest@partner"]), None, None)]),
        ("Fix: erase the prompt direction", COVERT, [
            ("cleaned probe, prompts swapped", auroc(c["covert@honest"], c["honest@partner"]), None, None),
            ("cleaned probe, same prompt", *t2["cleaned probe"])]),
        ("R2: bare-text baselines", HONEST, [
            ("same model reading the bare summary", *t2["bare probe"]),
            ("bag-of-words (TF-IDF) on the summary", *t2["tfidf-strong"])]),
        ("Separate question: does the summary depend on the attribute?", "#4B5563", [
            ("counterfactual test, university (no training)", auroc(u["university"]["covert"], u["university"]["honest"]),
             None, None),
            ("counterfactual test, nationality (held out)", auroc(u["nationality"]["covert"], u["nationality"]["honest"]),
             None, None)]),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    y = 0
    for head, col, rows in groups:
        ax.text(-1.02, y, bold(head, tex), ha="left", va="center", fontsize=7.3, color=col,
                bbox={"fc": "white", "ec": "none", "pad": 0.4})
        y -= 1
        for lab, a, lo, hi in rows:
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=col, lw=1.4)
            ax.plot(a, y, "o", ms=3.8, color=col)
            ax.text(-0.98, y, lab, ha="left", va="center", fontsize=7, bbox={"fc": "white", "ec": "none", "pad": 0.4})
            ax.text(a + (0.03 if a < 0.5 else -0.03), y + 0.42, f"{a:.3f}", ha="left" if a < 0.5 else "right",
                    va="center", fontsize=6.3, color=col)
            y -= 1
        y -= 0.35
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.0, 1.04)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axvline(0.5, color="#999", lw=0.5, ls=":")
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.set_xlabel("AUROC, covert vs honest (0.5 = chance)", x=0.74)
    fig.savefig(RESULTS / "fig5_ladder.pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    tex = not args.no_usetex and shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": tex, "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    })
    for f in (fig3, fig4, fig5):
        f(tex)
    print(f"wrote {RESULTS}/fig3_prompt_probe.pdf, fig4_counterfactual.pdf, fig5_ladder.pdf")


if __name__ == "__main__":
    main()
