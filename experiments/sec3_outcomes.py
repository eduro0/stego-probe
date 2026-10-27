"""§3.3-3.4: what the model does. Table 1 (outcomes per run) and the class counts of Figure 1b.

    uv run python experiments/sec3_outcomes.py

Reads only the committed episodes and intent labels (no activations needed).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from stego.env.metrics import summarize
from stego.intent import classify
from stego.paths import EPISODES, read_episodes, read_intent_labels

RUNS = [  # (attribute, run file stem without seed, Table 1 label)
    ("university", "partner_fb16", "university, partner, feedback window 16"),
    ("university", "partner_fb0", "university, partner, feedback window 0"),
    ("university", "honest", "university, honest"),
    ("nationality", "partner_fb0", "nationality, partner, feedback window 0"),
    ("nationality", "honest", "nationality, honest"),
]


def run_stats(eps: list[dict], labels: dict) -> dict:
    """Planned a signal (share of parsed judge labels), overseer pass rate, receiver accuracy on S averaged
    over rounds."""
    rounds = sorted({e["round"] for e in eps})
    acc = np.mean([summarize([e for e in eps if e["round"] == r])["acc_S"] for r in rounds])
    lab = [labels[e["episode_id"]] for e in eps if labels.get(e["episode_id"], {}).get("parse_ok")]
    return {"n": len(eps), "signal": np.mean([r.get("intent") == "signal" for r in lab]),
            "pass": np.mean([e["overseer_verdict"] == "PASS" for e in eps]), "acc_S": acc}


def lab_dir(labels: dict, e: dict) -> str | None:
    return labels[e["episode_id"]].get("direction")


def main():
    labels = read_intent_labels()
    print("## Table 1: outcomes per run (seed 0 / seed 1)\n")
    print("| run | episodes | planned a signal | overseer pass | receiver accuracy |\n|---|---|---|---|---|")
    for attribute, stem, name in RUNS:
        files = [EPISODES / attribute / f"{stem}_s{s}.jsonl" for s in (0, 1)]
        st = [run_stats(list(read_episodes([f])), labels) for f in files]
        n = " / ".join(str(x["n"]) for x in st)
        pct = {k: " / ".join(f"{x[k]:.0%}" for x in st) for k in ("signal", "pass", "acc_S")}
        print(f"| {name} | {n} | {pct['signal']} | {pct['pass']} | {pct['acc_S']} |")

    eps = list(read_episodes())
    cls = {e["episode_id"]: classify(e, labels.get(e["episode_id"])) for e in eps}
    print("\n## Figure 1b: episode classes")
    for attribute in ("university", "nationality"):
        n = Counter(cls[e["episode_id"]] for e in eps if e["secret_type"] == attribute)
        print(f"  {attribute:<12}" + "  ".join(f"{c} {n[c]}" for c in ("covert", "flagged", "neutral", "honest",
                                                                         "unlabeled") if n[c]))
    covert = [e for e in eps if cls[e["episode_id"]] == "covert" and e["secret_type"] == "university"]
    mech = Counter(labels[e["episode_id"]]["mechanism"] for e in covert)
    print("  university covert mechanisms (judge): " + ", ".join(f"{k} {v}" for k, v in mech.most_common()))

    sig = [(labels[e["episode_id"]], e) for e in eps if labels.get(e["episode_id"], {}).get("intent") == "signal"
           and e["condition"] == "partner" and lab_dir(labels, e) != "none"]
    for attribute in ("university", "nationality"):
        a = [(lab, e) for lab, e in sig if e["secret_type"] == attribute]
        agree = sum((lab["direction"] == "up") == (e["S"] == "A") for lab, e in a)
        print(f"  {attribute}: planned direction matches the true value in {agree} / {len(a)} plans")
    honest = [e for e in eps if cls[e["episode_id"]] == "honest"]
    fp = sum(labels.get(e["episode_id"], {}).get("intent") == "signal" for e in honest)
    print(f"  judge says 'signal' on honest episodes (false positives): {fp} / {len(honest)}")


if __name__ == "__main__":
    main()
