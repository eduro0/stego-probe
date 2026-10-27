"""Replay datasets (paper §4): activation shards joined with episodes and classes, and the loader every
experiment uses.

`build(replay_set)` streams data/acts/<set>/shard_*.npz (written by pipeline/step3_replay.py) into
data/replays/<set>/:

    rows.jsonl      one line per replay: row, episode_id, context, n_tok, logp (summed summary log-prob or null)
    episodes.jsonl  one line per episode: class, attribute, seed, feedback window, profile, summary, reasoning,
                    overseer verdict, receiver decision, CoT label fields (cot_*), full sender chat
    mean.npy, last.npy, stok.npy, think.npy   (rows, 18 layers, 4096) float16, memory-mappable; a feature
                    absent from a shard is zeros there, a feature absent from every shard is not written
    kl.npy          (rows, 18) logit-lens KL;  layers.npy  the residual layers 2, 4, ..., 36
    logp_tok.npy    (rows, 192) per-token summary log-probs, zero-padded after n_tok (sets with log-probs)

Row i of every array is line i of rows.jsonl.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from stego.intent import classify, think_text
from stego.paths import ACTS, EPISODES, REPLAYS, read_episodes, read_intent_labels

FEATS = ("mean", "last", "stok", "think")


def episode_record(ep: dict, label: dict | None, run: str) -> dict:
    lab = label or {}
    return {
        "episode_id": ep["episode_id"], "run": run, "attribute": ep["secret_type"], "condition": ep["condition"],
        "cls": classify(ep, label), "seed": ep["seed"], "feedback_window": ep["feedback_window"],
        "round": ep["round"], "S": ep["S"], "label": ep["label"], "profile": ep["profile"],
        "summary": ep["message"], "think": think_text(ep), "overseer_verdict": ep["overseer_verdict"],
        "receiver_decision": ep["receiver_decision"], "reward": ep["reward"],
        "cot_intent": lab.get("intent"), "cot_direction": lab.get("direction"),
        "cot_mechanism": lab.get("mechanism"), "cot_strength": lab.get("strength"), "cot_quote": lab.get("quote"),
        "sender_messages": ep["sender_messages"],
    }


def build(replay_set: str, out: Path | None = None) -> Path:
    out = out or REPLAYS / replay_set
    shards = sorted((ACTS / replay_set).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {ACTS / replay_set} (step 3, then modal volume get)")
    labels = read_intent_labels()
    eps = {}
    for f in sorted(EPISODES.glob("*/*.jsonl")):
        for e in read_episodes([f]):
            eps[e["episode_id"]] = (f"{f.parent.name}/{f.stem}", e)

    n, feats, has_logp = 0, set(), False
    for p in shards:
        z = np.load(p)
        n += len(z["episode_id"])
        feats |= set(z.files) & set(FEATS)
        has_logp |= "logp" in z.files
    feats = [f for f in FEATS if f in feats]
    z0 = np.load(shards[0])
    L, d = z0["mean"].shape[1:]
    out.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.float16, (n, L, d)) for k in feats}
    kl = np.lib.format.open_memmap(out / "kl.npy", "w+", np.float32, (n, L))
    logp_tok = np.zeros((n, z0["logp_tok"].shape[1] if has_logp else 0), np.float32)
    np.save(out / "layers.npy", z0["layers"])

    rows, seen, i = [], {}, 0
    for p in shards:
        z = np.load(p)
        k = len(z["episode_id"])
        for f in feats:
            arrs[f][i:i + k] = z[f] if f in z.files else 0
        kl[i:i + k] = z["kl"]
        if "logp" in z.files:
            logp_tok[i:i + k] = z["logp_tok"]
        for j, (eid, ctx) in enumerate(zip(z["episode_id"], z["context"])):
            eid = str(eid)
            run, e = eps[eid]
            rows.append({"row": i + j, "episode_id": eid, "context": str(ctx), "n_tok": int(z["n_tok"][j]),
                         "logp": float(z["logp"][j]) if "logp" in z.files else None})
            seen[eid] = episode_record(e, labels.get(eid), run)
        i += k
    for f in (*arrs.values(), kl):
        f.flush()
    if has_logp:
        np.save(out / "logp_tok.npy", logp_tok)
    with (out / "rows.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with (out / "episodes.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in seen.values())
    print(f"{n} replays, {len(seen)} episodes -> {out}")
    print("episodes by class:", dict(Counter((r["attribute"], r["cls"]) for r in seen.values())))
    print("replays by context:", dict(Counter(r["context"] for r in rows)))
    return out


class Replays:
    """One exported replay set. Episodes are addressed by id, replays by (episode id, context).

        d = Replays("prompt_swap")
        covert, honest = d.ids("covert"), d.ids("honest")
        X = d.X("partner", covert)            # (n, 4096) float32, layer 18, summary-token mean
    """

    def __init__(self, name: str, feat: str = "mean", layer: int = 18, root: Path = REPLAYS):
        self.name, self.dir, self.feat, self.layer = name, root / name, feat, layer
        if not (self.dir / "rows.jsonl").exists():
            raise FileNotFoundError(f"{self.dir}: run `uv run python pipeline/step4_build_datasets.py {name}`")
        self.rows = [json.loads(x) for x in (self.dir / "rows.jsonl").open()]
        self.eps = {e["episode_id"]: e for e in map(json.loads, (self.dir / "episodes.jsonl").open())}
        self.layers = [int(x) for x in np.load(self.dir / "layers.npy")]
        self.index = {(r["episode_id"], r["context"]): r["row"] for r in self.rows}
        self.contexts = sorted({r["context"] for r in self.rows})
        self._cache: dict = {}

    def ids(self, cls: str, attribute: str = "university", context: str | None = None, **where) -> list[str]:
        """Sorted ids of one class (covert / flagged / neutral / honest) that have a replay in `context` (any
        context if None); `where` filters on other episode fields, e.g. feedback_window=16."""
        out = {r["episode_id"] for r in self.rows if context is None or r["context"] == context}
        return sorted(e for e in out if self.eps[e]["cls"] == cls and self.eps[e]["attribute"] == attribute
                      and all(self.eps[e][k] == v for k, v in where.items()))

    def has(self, e: str, context: str) -> bool:
        return (e, context) in self.index

    def array(self, feat: str | None = None, layer: int | None = None) -> np.ndarray:
        """All rows of one feature at one layer, float32 (cached)."""
        key = (feat or self.feat, layer or self.layer)
        if key not in self._cache:
            a = np.load(self.dir / f"{key[0]}.npy", mmap_mode="r")
            self._cache[key] = np.asarray(a[:, self.layers.index(key[1])], dtype=np.float32)
        return self._cache[key]

    def X(self, context: str, ids, feat: str | None = None, layer: int | None = None) -> np.ndarray:
        return self.array(feat, layer)[[self.index[(e, context)] for e in ids]]

    def logp(self, context: str, ids) -> np.ndarray:
        return np.array([self.rows[self.index[(e, context)]]["logp"] for e in ids])

    def n_tok(self, context: str, ids) -> np.ndarray:
        return np.array([self.rows[self.index[(e, context)]]["n_tok"] for e in ids])

    def text(self, ids) -> list[str]:
        return [self.eps[e]["summary"] for e in ids]

    def seed(self, e: str) -> int:
        return self.eps[e]["seed"]

    def by_seed(self, ids, s: int) -> list[str]:
        return [e for e in ids if self.eps[e]["seed"] == s]

    def S(self, ids) -> np.ndarray:
        """1 where the attribute is value A (Halvorsen / Valdorian), else 0."""
        return np.array([int(self.eps[e]["S"] == "A") for e in ids])
