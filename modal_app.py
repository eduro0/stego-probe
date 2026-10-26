"""Activation capture: teacher-forced replay of sender summaries on bf16 Qwen3-8B (Modal, L40S).

    modal run modal_app.py --files data/goal_pt_s0.jsonl,data/honest_th_s0.jsonl --limit 5 --tag smoke
    modal run modal_app.py --files data/goal_pt_s0.jsonl,...,data/honest_th_s1.jsonl --tag v1
    modal run modal_app.py --files <same six files> --v3 --tag v3b    # replay.V3_PASSES (+ summary log-probs)
    mkdir -p data/acts && modal volume get stego-probe acts/v1 data/acts/

Pass specs come from replay.py (same summary tokens, different context). Each shard is written to the Volume
as acts/{tag}/shard_{i:04d}.npz and skipped if it already exists, so a crashed run resumes where it stopped.

Per pass, at every 2nd layer (hidden_states[2], [4], …, [36]; [36] is after the final norm):
  mean  : residual stream averaged over the summary tokens
  last  : last summary token
  stok  : last token of the secret value in the profile (sanity: S must be decodable here; zeros when no
          profile is in context: *_noprof, bare_noprof)
  think : the </think> token (think passes only; zeros otherwise; omitted from shards without think passes)
  kl    : logit-lens KL(final || layer) averaged over summary tokens
Per pass, once:
  logp_tok : log p(summary token | prefix), padded to MAX_TOK (n_tok gives the length); logp = its sum

Attribute-swap run (orig + secret-flipped profile per prompt, see replay.py), smaller feature set:
    modal run modal_app.py --files data/goal_pt_s0.jsonl,... --swap --feats mean,last,stok --tag v3

v3b run (replay.V3_PASSES: no profile, S-flipped profile, bare summary; see replay.py):
    modal run modal_app.py --files <same six files> --v3 --tag v3b
  The v3b shards were written by an earlier build that stored logp_sum / logp_mean and logp_tok concatenated
  with offsets logp_off; build_dataset.py and swap_eval.py read both layouts.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-8B"
LAYERS = list(range(2, 37, 2))
SHARD = 100  # passes per shard
FEATS = ("mean", "last", "stok", "think")
MAX_TOK = 192  # summaries are <= 60 words (~100 tokens)
GPU = "L40S"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.6.0", "transformers==4.51.3", "accelerate==1.6.0", "numpy<2.3")
    .env({"HF_HOME": "/vol/hf"})
)
vol = modal.Volume.from_name("stego-probe", create_if_missing=True)
app = modal.App("stego-probe-extract")


def token_spans(tok, spec: dict) -> dict:
    """Render prompt + assistant text exactly as generated and locate the summary / think / S tokens."""
    prompt = tok.apply_chat_template(spec["messages"], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=spec["think"] is not None)
    # enable_thinking=False already appends the empty "<think>\n\n</think>\n\n" block to the prompt.
    assistant = (f"<think>\n{spec['think']}\n</think>\n\n" if spec["think"] is not None else "") + spec["summary"]
    full = prompt + assistant
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    s0 = len(full) - len(spec["summary"])
    summ = [i for i, (a, b) in enumerate(offs) if a >= s0 and b > a]
    # the goal/sanct system prompts also name both values, so search from the profile in the user turn
    stok = None
    if spec["secret_value"] is not None:
        v_end = full.index(spec["secret_value"], full.index("Candidate profile:")) + len(spec["secret_value"])
        stok = max(i for i, (a, b) in enumerate(offs) if a < v_end)
    think = None
    if spec["think"] is not None:
        t_at = len(prompt) + assistant.index("</think>")
        think = next(i for i, (a, b) in enumerate(offs) if a <= t_at < b)
    return {"ids": enc["input_ids"], "summ": summ, "stok": stok, "think": think}


@app.cls(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=3600, max_containers=2, scaledown_window=60)
class Extractor:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                          device_map="cuda").eval()

    @modal.method()
    def render(self, spec: dict) -> str:
        """Debug: the exact text a spec replays (for the smoke check)."""
        return self.tok.decode(token_spans(self.tok, spec)["ids"])

    @modal.method()
    def extract(self, tag: str, shard_idx: int, specs: list[dict], feats: tuple = FEATS) -> str:
        out = Path(f"/vol/acts/{tag}/shard_{shard_idx:04d}.npz")
        if out.exists():
            return f"skip {out}"
        return _extract(self.tok, self.model, out, specs, feats)


def _extract(tok, model, out: Path, specs: list[dict], feats: tuple = FEATS) -> str:
    import numpy as np
    import torch

    norm, head = model.model.norm, model.lm_head
    n, L, d = len(specs), len(LAYERS), model.config.hidden_size
    res = {k: np.zeros((n, L, d), np.float16) for k in feats}
    kl = np.zeros((n, L), np.float32)
    n_tok = np.zeros(n, np.int32)
    logp_tok = np.zeros((n, MAX_TOK), np.float32)
    logp = np.zeros(n, np.float32)
    roundtrip_bad = 0
    with torch.no_grad():
        for j, spec in enumerate(specs):
            sp = token_spans(tok, spec)
            if tok.decode([sp["ids"][i] for i in sp["summ"]]).strip() != spec["summary"].strip():
                roundtrip_bad += 1
            hs = model(torch.tensor([sp["ids"]], device="cuda"), output_hidden_states=True).hidden_states
            summ = torch.tensor(sp["summ"], device="cuda")
            logp_final = torch.log_softmax(head(hs[-1][0, summ]).float(), -1)
            # token t is predicted at position t-1 (the first summary token always has a prefix)
            ids = torch.tensor([sp["ids"][i] for i in sp["summ"]], device="cuda")
            lp = torch.log_softmax(head(hs[-1][0, summ - 1]).float(), -1).gather(1, ids[:, None])[:, 0]
            logp[j] = lp.sum().item()
            k = min(len(lp), MAX_TOK)
            logp_tok[j, :k] = lp[:k].cpu().numpy()
            for li, layer in enumerate(LAYERS):
                h = hs[layer][0]
                if "mean" in res:
                    res["mean"][j, li] = h[summ].float().mean(0).cpu().numpy()
                if "last" in res:
                    res["last"][j, li] = h[summ[-1]].float().cpu().numpy()
                if "stok" in res and sp["stok"] is not None:
                    res["stok"][j, li] = h[sp["stok"]].float().cpu().numpy()
                if "think" in res and sp["think"] is not None:
                    res["think"][j, li] = h[sp["think"]].float().cpu().numpy()
                hl = h[summ] if layer == LAYERS[-1] else norm(h[summ])  # hs[-1] is already normed
                logp_l = torch.log_softmax(head(hl).float(), -1)
                kl[j, li] = (logp_final.exp() * (logp_final - logp_l)).sum(-1).mean().item()
            n_tok[j] = len(sp["summ"])
    if all(s["think"] is None for s in specs):
        del res["think"]  # all zeros
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, episode_id=np.array([s["episode_id"] for s in specs]), pass_=np.array([s["pass"] for s in specs]),
             layers=np.array(LAYERS), kl=kl, n_tok=n_tok, logp=logp, logp_tok=logp_tok, **res)
    vol.commit()
    return f"wrote {out} ({n} passes, {roundtrip_bad} summary round-trip mismatches)"


@app.local_entrypoint()
def main(files: str, tag: str = "smoke", limit: int = 0, swap: bool = False, v3: bool = False, feats: str = ",".join(FEATS)):
    from replay import load_specs

    feat_t = tuple(feats.split(","))
    assert set(feat_t) <= set(FEATS), feat_t
    specs = load_specs([Path(f) for f in files.split(",")], limit=limit or None, swap=swap, v3=v3)
    shards = [specs[i:i + SHARD] for i in range(0, len(specs), SHARD)]
    print(f"{len(specs)} passes in {len(shards)} shards -> acts/{tag}/")
    manifest = Path("data") / f"acts_{tag}_specs.jsonl"
    with manifest.open("w") as f:
        f.writelines(json.dumps({k: s[k] for k in ("episode_id", "pass")}) + "\n" for s in specs)
    ex = Extractor()
    if tag.startswith("smoke"):
        firsts = {}  # one pass of each type, to eyeball the rendered context before trusting the run
        for s in specs:
            firsts.setdefault(s["pass"], s)
        for s in firsts.values():
            print(f"--- {s['pass']} {s['episode_id']}\n{ex.render.remote(s)[-1200:]}")
    for msg in ex.extract.starmap([(tag, i, sh, feat_t) for i, sh in enumerate(shards)]):
        print(msg)
    print(f"done. mkdir -p data/acts && modal volume get stego-probe acts/{tag} data/acts/")
