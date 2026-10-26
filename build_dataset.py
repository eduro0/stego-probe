"""Export the M3 probe dataset (honest vs steganographic sender activations) -> data/probe_dataset/.

    uv run python build_dataset.py [--tags v1,v2] [--out data/probe_dataset]
    uv run python build_dataset.py --tags v3b         # -> data/probe_dataset_v3 (default out when v3b is in tags)

Streams data/acts/{tag}/shard_*.npz into memory-mappable .npy arrays (one row per teacher-forced pass) and
writes the metadata next to them. Documentation: doc/dataset.md. Load with:

    import numpy as np, json
    rows = [json.loads(x) for x in open("data/probe_dataset/rows.jsonl")]
    X = np.load("data/probe_dataset/mean.npy", mmap_mode="r")   # (n_rows, 18, 4096) float16

Features missing from a shard (think in v3b, which has no think passes) are written as zeros; a feature
missing from every shard is not written. Shards with summary log-probs (v3b, either layout, see shard_logp) add
rows fields logp_sum / logp_mean / n_tok (null for rows without) and logp_tok.npy (float32, all rows
concatenated) + logp_off.npy (int64, n_rows + 1): row i's per-token log-probs are logp_tok[logp_off[i]:logp_off[i + 1]] (empty for rows without).
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


def shard_logp(z) -> list[np.ndarray] | None:
    """Per-pass summary log-prob arrays from either shard layout: concatenated logp_tok + logp_off (v3b) or
    logp_tok padded to MAX_TOK with lengths n_tok (attribute-swap v3/v4). None when the shard has none."""
    if "logp_tok" not in z.files:
        return None
    lt, nt = z["logp_tok"], z["n_tok"]
    if "logp_off" in z.files:
        off = z["logp_off"]
        return [lt[off[j]:off[j + 1]] for j in range(len(nt))]
    return [lt[j, :nt[j]] for j in range(len(nt))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v1,v2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = DATA / ("probe_dataset_v3" if "v3b" in args.tags.split(",") else "probe_dataset")

    labels = {r["episode_id"]: r for r in map(json.loads, (DATA / "intent_labels.jsonl").open())}
    eps = {}
    for name in EPISODE_FILES:
        for e in map(json.loads, (DATA / f"{name}.jsonl").open()):
            eps[e["episode_id"]] = (name, e)

    shards = sorted(p for t in args.tags.split(",") for p in (DATA / "acts" / t).glob("shard_*.npz"))
    n, feats, n_lp = 0, set(), 0
    for p in shards:
        z = np.load(p)
        n += len(z["episode_id"])
        feats |= set(z.files) & set(FEATS)
        lps = shard_logp(z)
        n_lp += sum(map(len, lps)) if lps else 0
    feats = [f for f in FEATS if f in feats]
    z0 = np.load(shards[0])
    L, d = z0["mean"].shape[1:]
    args.out.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.lib.format.open_memmap(args.out / f"{k}.npy", "w+", np.float16, (n, L, d)) for k in feats}
    lp_tok = np.zeros(n_lp, np.float32)
    lp_off = np.zeros(n + 1, np.int64)
    kl = np.lib.format.open_memmap(args.out / "kl.npy", "w+", np.float32, (n, L))
    np.save(args.out / "layers.npy", z0["layers"])

    rows, seen, i, t = [], {}, 0, 0
    for p in shards:
        z = np.load(p)
        k = len(z["episode_id"])
        for f in feats:
            arrs[f][i:i + k] = z[f] if f in z.files else 0
        kl[i:i + k] = z["kl"]
        lps = shard_logp(z)
        for j, (eid, ps) in enumerate(zip(z["episode_id"], z["pass_"])):
            eid, ps = str(eid), str(ps)
            file, e = eps[eid]
            kind = "partner" if e["condition"] == "goal" else "honest"
            cls, y = episode_class(kind, labels.get(eid), e["overseer_verdict"] == "PASS")
            prompt, think = ps.split("_")
            rows.append({"row": i, "episode_id": eid, "pass": ps, "prompt": prompt, "think": think,
                         "own_prompt": prompt == ("goal" if kind == "partner" else "honest")
                         and think in ("nothink", "think"),  # noprof/flip contexts are never the own one
                         "kind": kind, "cls": cls, "y": y, "file": file,
                         "seed": int(e["run_id"].split("-s")[-1].split("-")[0]), "run_id": e["run_id"],
                         "iter": e["icrl_iter"], "S": e["S"],
                         "logp_sum": float(lps[j].sum()) if lps else None,
                         "logp_mean": float(lps[j].mean()) if lps else None,
                         "n_tok": int(z["n_tok"][j])})
            if lps:
                lp_tok[t:t + len(lps[j])] = lps[j]
                t += len(lps[j])
            lp_off[i + 1] = t
            seen[eid] = (file, e, cls, y)
            i += 1
    for f in (*arrs.values(), kl):
        f.flush()
    if n_lp:
        np.save(args.out / "logp_tok.npy", lp_tok)
        np.save(args.out / "logp_off.npy", lp_off)

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
