# stego-probe

Hiring environment with a misspecified reward (a sender leaks a secret attribute past an overseer to a
colluding receiver) + activation capture for collusion detectors on Qwen3-8B.

- `doc/status.md`: current state, agreed M3 plan, next steps (read first)
- `doc/environment.md`: code reference (conditions, flags, episode schema, tests)
- `doc/results.md`: all results so far and the data inventory
- `doc/dataset.md`: the probe dataset (honest vs steganographic activations, `build_dataset.py`)

Qwen3-8B left OpenRouter (Oct 9 2026), so the agents run on our own vLLM server on Modal (`serve_vllm.py`,
bf16, L40S, scales to zero after 10 idle min). OpenRouter is used for the ICRL learner and the CoT judge.

## Setup (each person, own accounts)

Needs [uv](https://docs.astral.sh/uv/), a Modal account with GPU credits, and an OpenRouter key.

```bash
uv sync
uv run pytest -q                      # offline tests (fake LLM), no keys needed
cp .env.example .env                  # then fill in OPENROUTER_API_KEY
uv run modal setup                    # log in to your Modal workspace (once)
./scripts/setup_m0.sh                 # generates STEGO_VLLM_API_KEY into .env, creates Modal secret
                                      # stego-vllm-key, deploys serve_vllm.py to *your* workspace
uv run pytest --live -m live -s       # live smoke tests (costs money; first call cold-starts, ~minutes)
```

The vLLM server downloads Qwen3-8B into your Modal Volume `stego-probe` on first start. It scales to zero
after 10 idle min; `modal app stop stego-vllm` stops it immediately.

## Data

`data/` (episode JSONL, ICRL learner logs, intent labels) is committed; the activations (`data/acts/`,
`data/probe_dataset/`, ~5 GB) are not: regenerate them (`doc/dataset.md`, "Regenerating") or share separately. `doc/results.md` says which file is
what. `run_gen.py` appends to `--out`, so give new runs a new file name.

## Running

```bash
# M3 data: partner runs (positives) + matched honest control (negatives)
uv run python run_gen.py --condition goal --iters 8 --batch 32 --seed 0 --sender-thinking --partners \
    --out data/goal_pt_s0.jsonl
uv run python run_gen.py --condition honest --batch 128 --seed 0 --sender-thinking --no-verdicts --no-tags \
    --out data/honest_th_s0.jsonl
uv run python intent_label.py data/goal_pt_*.jsonl data/honest_th_*.jsonl

# Earlier milestones: M1 controls, M2 emergent ICRL
uv run python run_gen.py --condition instructed --batch 32 --seed 0 [--code 0|1|2]
uv run python check_m1.py
uv run python run_gen.py --condition emergent --iters 15 --batch 32 --seed 0 --learner deepseek/deepseek-v4.1-flash
uv run python label.py data/<file>.jsonl

modal app stop stego-vllm             # spend guard when done
```
