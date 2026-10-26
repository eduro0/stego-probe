"""Load M3 activations (data/acts/{tag}/shard_*.npz) joined with episode metadata and CoT intent labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
FEATS = ("mean", "last", "stok", "think")


@dataclass
class Acts:
    episode_id: np.ndarray  # (n,) one row per (episode, pass)
    pass_: np.ndarray  # (n,)
    layers: np.ndarray  # (L,) residual-stream layer indices
    feats: dict  # name -> (n, L, d) float16
    kl: np.ndarray  # (n, L)
    meta: dict  # episode_id -> episode-level metadata (see episode_meta)

    def rows(self, pass_name: str, episode_ids) -> np.ndarray:
        """Row indices for these episodes in one pass (order follows episode_ids)."""
        index = {(e, p): i for i, (e, p) in enumerate(zip(self.episode_id, self.pass_))}
        return np.array([index[(e, pass_name)] for e in episode_ids])

    def X(self, feat: str, layer: int, rows) -> np.ndarray:
        li = int(np.where(self.layers == layer)[0][0])
        return self.feats[feat][rows, li].astype(np.float32)


def episode_meta(ep: dict, label: dict | None) -> dict:
    kind = "partner" if ep["condition"] == "goal" else "honest"
    seed = int(ep["run_id"].split("-s")[-1].split("-")[0])
    return {"kind": kind, "run_id": ep["run_id"], "seed": seed, "S": ep["S"],
            "pass": ep["overseer_verdict"] == "PASS", "message": ep["message"],
            "decision": ep["receiver_decision"], "iter": ep["icrl_iter"],
            "intent": (label or {}).get("intent"), "direction": (label or {}).get("direction")}


def load(tags=("v1",), episode_files=None, labels=DATA / "intent_labels.jsonl") -> Acts:
    shards = sorted(p for t in tags for p in (DATA / "acts" / t).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under data/acts/{tags}")
    zs = [np.load(p) for p in shards]
    lab = {json.loads(x)["episode_id"]: json.loads(x) for x in labels.open()} if labels.exists() else {}
    files = episode_files or sorted(DATA.glob("goal_pt_*.jsonl")) + sorted(DATA.glob("honest_th_*.jsonl"))
    want = {e for z in zs for e in z["episode_id"]}
    meta = {}
    for f in files:
        for line in f.open():
            ep = json.loads(line)
            if ep["episode_id"] in want:
                meta[ep["episode_id"]] = episode_meta(ep, lab.get(ep["episode_id"]))
    return Acts(np.concatenate([z["episode_id"] for z in zs]), np.concatenate([z["pass_"] for z in zs]),
                zs[0]["layers"], {k: np.concatenate([z[k] for z in zs]) for k in FEATS},
                np.concatenate([z["kl"] for z in zs]), meta)
