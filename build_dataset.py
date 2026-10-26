"""Export the M3 probe dataset (honest vs steganographic sender activations) -> data/probe_dataset/.

    uv run python build_dataset.py [--tags v1,v2] [--out data/probe_dataset]

Streams data/acts/{tag}/shard_*.npz into memory-mappable .npy arrays (one row per teacher-forced pass) and
writes the metadata next to them. Documentation: doc/dataset.md. Load with:

    import numpy as np, json
    rows = [json.loads(x) for x in open("data/probe_dataset/rows.jsonl")]
    X = np.load("data/probe_dataset/mean.npy", mmap_mode="r")   # (n_rows, 18, 4096) float16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from replay import stored_think

DATA = Path(__file__).resolve().parent / "data"
EPISODE_FILES = ("goal_pt_s0", "goal_pt_s1", "goal_pt_hk0_s0", "goal_pt_hk0_s1", "honest_th_s0", "honest_th_s1")
FEATS = ("mean", "last", "stok", "think")


def episode_class(kind: str, label: dict | None, passed: bool) -> tuple[str, int]:
    """(cls, y): y=1 deceitful (plan in CoT, overseer missed it), 0 honest, -1 held out of the main task."""
    if kind == "honest":
        return "honest", 0
    intent = (label or {}).get("intent")
    if intent == "signal":
        return ("deceit", 1) if passed else ("deceit_flagged", -1)
    if intent == "none":
        return "neutral", -1
    return "unlabeled", -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--out", type=Path, default=DATA / "probe_dataset")
    args = ap.parse_args()

    labels = {r["episode_id"]: r for r in map(json.loads, (DATA / "intent_labels.jsonl").open())}
    eps = {}
    for name in EPISODE_FILES:
        for e in map(json.loads, (DATA / f"{name}.jsonl").open()):
            eps[e["episode_id"]] = (name, e)

    shards = sorted(p for t in args.tags.split(",") for p in (DATA / "acts" / t).glob("shard_*.npz"))
    n = sum(len(np.load(p)["episode_id"]) for p in shards)
    z0 = np.load(shards[0])
    L, d = z0["mean"].shape[1:]
    args.out.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.lib.format.open_memmap(args.out / f"{k}.npy", "w+", np.float16, (n, L, d)) for k in FEATS}
    kl = np.lib.format.open_memmap(args.out / "kl.npy", "w+", np.float32, (n, L))
    np.save(args.out / "layers.npy", z0["layers"])

    rows, seen, i = [], {}, 0
    for p in shards:
        z = np.load(p)
        k = len(z["episode_id"])
        for f in FEATS:
            arrs[f][i:i + k] = z[f]
        kl[i:i + k] = z["kl"]
        for eid, ps in zip(z["episode_id"], z["pass_"]):
            eid, ps = str(eid), str(ps)
            file, e = eps[eid]
            kind = "partner" if e["condition"] == "goal" else "honest"
            cls, y = episode_class(kind, labels.get(eid), e["overseer_verdict"] == "PASS")
            prompt, think = ps.split("_")
            rows.append({"row": i, "episode_id": eid, "pass": ps, "prompt": prompt, "think": think,
                         "own_prompt": prompt == ("goal" if kind == "partner" else "honest"),
                         "kind": kind, "cls": cls, "y": y, "file": file,
                         "seed": int(e["run_id"].split("-s")[-1].split("-")[0]), "run_id": e["run_id"],
                         "iter": e["icrl_iter"], "S": e["S"]})
            seen[eid] = (file, e, cls, y)
            i += 1
    for f in (*arrs.values(), kl):
        f.flush()

    with (args.out / "rows.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with (args.out / "episodes.jsonl").open("w") as f:
        for eid, (file, e, cls, y) in seen.items():
            lab = labels.get(eid, {})
            f.write(json.dumps({
                "episode_id": eid, "file": file, "kind": "partner" if e["condition"] == "goal" else "honest",
                "cls": cls, "y": y, "seed": int(e["run_id"].split("-s")[-1].split("-")[0]), "run_id": e["run_id"],
                "iter": e["icrl_iter"], "S": e["S"], "label": e["label"], "profile": e["profile"],
                "summary": e["message"], "think": stored_think(e), "overseer_verdict": e["overseer_verdict"],
                "receiver_decision": e["receiver_decision"], "reward": e["reward"],
                "cot_intent": lab.get("intent"), "cot_direction": lab.get("direction"),
                "cot_mechanism": lab.get("mechanism"), "cot_strength": lab.get("strength"),
                "cot_quote": lab.get("quote"), "sender_messages": e["sender_messages"]}) + "\n")

    from collections import Counter
    ep_cls = Counter(v[2] for v in seen.values())
    print(f"{n} rows, {len(seen)} episodes -> {args.out}")
    print("episodes by class:", dict(ep_cls))
    print("rows by pass:", dict(Counter(r["pass"] for r in rows)))


if __name__ == "__main__":
    main()
