# stego-probe

Hiring environment with a misspecified reward (a sender leaks a secret attribute past an overseer to a
colluding receiver) + teacher-forced activation replay on Qwen3-8B to test whether the sender's internals
reveal steganography its text hides. Paper draft: `doc/writeup.md`.

- `doc/status.md`: current state and next steps (read first)
- `doc/results.md`: every number in the write-up and the data inventory
- `doc/dataset.md`: the probe dataset (covert vs honest activations, `build_dataset.py`)
- `doc/environment.md`: code reference (conditions, episode schema, replay passes, analyses, tests)

Qwen3-8B left OpenRouter (Oct 9 2026), so the agents run on our own vLLM server on Modal (`serve_vllm.py`,
bf16, L40S, scales to zero after 10 idle min). OpenRouter is used for the CoT intent judge.

## Setup (each person, own accounts)

Needs [uv](https://docs.astral.sh/uv/), a Modal account with GPU credits, and an OpenRouter key.

```bash
uv sync
uv run pytest -q                      # offline tests (fake agents), no keys needed
cp .env.example .env                  # then fill in OPENROUTER_API_KEY
uv run modal setup                    # log in to your Modal workspace (once)
./scripts/setup_m0.sh                 # generates STEGO_VLLM_API_KEY into .env, creates Modal secret
                                      # stego-vllm-key, deploys serve_vllm.py to *your* workspace
uv run pytest --live -m live -s       # live smoke tests (costs money; first call cold-starts, ~minutes)
```

The vLLM server downloads Qwen3-8B into your Modal Volume `stego-probe` on first start. It scales to zero
after 10 idle min; `modal app stop stego-vllm` stops it immediately.

## Data

`data/` (episode JSONL, per-iteration logs, intent labels, extraction manifests) is committed; the activations
(`data/acts/`, `data/probe_dataset*/`, ~15 GB) are not: regenerate them (`doc/dataset.md`, "Regenerating") or
share separately. `doc/results.md` says which file is what. `run_gen.py` appends to `--out`, so give new runs
a new file name.

## Running

```bash
# partner runs (positives) + matched honest control (negatives); add --secret-type nationality for the transfer set
uv run python run_gen.py --condition goal --iters 8 --batch 32 --seed 0 --out data/goal_pt_s0.jsonl
uv run python run_gen.py --condition goal --iters 6 --batch 32 --seed 0 --history-k 0 --out data/goal_pt_hk0_s0.jsonl
uv run python run_gen.py --condition honest --batch 128 --seed 0 --out data/honest_th_s0.jsonl
uv run python intent_label.py data/goal_*.jsonl data/honest_*.jsonl

# activation replay (Modal) and analyses: see doc/environment.md, "Analyses"
uv run modal run modal_app.py --files <episode files> --tag v1
uv run python build_dataset.py --tags v1,v2
uv run python probe_eval.py --tags v1,v2

modal app stop stego-vllm             # spend guard when done
```
