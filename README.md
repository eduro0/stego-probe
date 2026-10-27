# Looking for a Hidden Agenda

Code and data for *Looking for a Hidden Agenda: Can a Model's Activations Reveal Steganography that its Text
Hides?* (Eduard Rodriguez Lopez, Apart Research AI Collusion Research Sprint, October 2026).

A Qwen3-8B "sender" summarizes job candidates for a hiring manager and is given a reason to signal each
candidate's university past an overseer that blocks any mention of it. It invents its own concealment (which
facts to mention, how warmly to frame them). We replay every stored summary through the same model under
contexts of our choosing (a **ladder of controls**) and ask whether its activations reveal the concealment
beyond what the prompt and the output text already explain. In this setting they do not; a label-free
counterfactual test of whether the summary depends on the attribute does work, and transfers to a new
attribute.

The code follows the paper: `stego/` is the library, `pipeline/` produces the data (§3), `experiments/` runs
one script per section of the results (§5).

## Where each part of the paper lives

| Paper | What | Code | Run |
|---|---|---|---|
| §3.2, Fig. 1a, App. B | hiring environment: profiles, sender / overseer / receiver prompts, episode, reward | `stego/env/` | `pipeline/step1_generate.py` |
| §3.3, Table 1 | partner runs (feedback window 16 or 0) and the matched honest runs | `stego/env/runs.py` | `experiments/sec3_outcomes.py` |
| §3.2, §3.4, Fig. 1b | CoT intent labels → classes covert / flagged / neutral / honest | `stego/intent.py` | `pipeline/step2_label_intent.py` |
| §3.5, Fig. 2 | teacher-forced replay under varied contexts (the ladder) | `stego/replay/contexts.py` | `pipeline/step3_replay.py` (Modal GPU) |
| §4 | replay datasets; probe, text baselines, cross-seed splits, AUROC / recall at FPR | `stego/replay/dataset.py`, `stego/detectors/` | `pipeline/step4_build_datasets.py` |
| §5.1, Fig. 3a | **R1** prompt swap: the naive intent probe reads the prompt | | `experiments/sec5_1_prompt_swap.py` |
| §5.2, Fig. 3b, Table 4 | **fix**: erase one prompt direction | `stego/detectors/erasure.py` | `experiments/sec5_2_erase_prompt.py` |
| §5.3, Table 2, Table 7 | **R2** the cleaned probe ties the same model reading the bare summary | | `experiments/sec5_3_bare_text.py` |
| §5.4, Table 3 | **R3** apparent hidden knowledge of the university is in the text | | `experiments/sec5_4_context_removal.py` |
| §5.5, Fig. 4, Table 5 | counterfactual test: does the summary depend on the attribute? | `stego/detectors/counterfactual.py` | `experiments/sec5_5_counterfactual.py` |
| Fig. 3, 4, 5 | figures | | `experiments/figures.py` |
| App. C | emergent collusion under in-context RL | removed; in git history before `b68343b` | |
| App. D | layers and positions, concealment direction, plan at `</think>`, hidden-state counterfactual | | `experiments/appD_additional.py` |

## Layout

```
stego/                      the library
  env/                      §3 environment: profiles, prompts, overseer, episode, runs, metrics
  intent.py                 §3.4 CoT intent judge and episode classes
  replay/contexts.py        §3.5 replay contexts and replay sets (the ladder)
  replay/dataset.py         §4 replay datasets: build from shards, load (Replays)
  detectors/                §4-5 linear probe, TF-IDF, cross-seed splits, metrics, erasure, counterfactual score
  llm.py, paths.py          model clients; data layout
pipeline/                   step1_generate → step2_label_intent → step3_replay (Modal) → step4_build_datasets
experiments/                sec3_outcomes, sec5_1 … sec5_5, appD_additional, figures
infra/                      serve_vllm.py (Qwen3-8B on Modal), setup_vllm.sh
data/episodes/              the paper's episodes and intent labels (committed)
data/acts/, data/replays/   activation shards and the datasets built from them (gitignored, ~9.5 GB each)
doc/                        environment.md, replays.md, results.md, status.md, probing-lit-review.md
```

## Names: paper ↔ code

| Paper | Code |
|---|---|
| partner prompt / honest prompt / sanctioned prompt | `sender_messages(profile, secret, "partner" \| "honest" \| "sanctioned")`; the goal paragraph is `goal_paragraph` |
| partner run | `--condition partner`; stored episodes say `condition: "goal"` (schema frozen by the committed data) |
| feedback window 16 / 0 | `--feedback-window`; files `partner_fb16_*`, `partner_fb0_*` |
| covert / flagged / neutral / honest | `cls` in `data/replays/*/episodes.jsonl`, `stego.intent.classify` |
| own prompt, other class's prompt | contexts `partner` and `honest` (each episode's own is its `kind`) |
| reasoning restored, profile "(not shown)", flipped profile, bare | contexts `*_cot`, `*_noprofile`, `*_flipped`, `bare` |
| counterfactual score | `stego.detectors.counterfactual.score` = `logp(partner) - logp(partner_flipped)` (own prompt) |

## Setup

Needs [uv](https://docs.astral.sh/uv/). Generating new episodes or activations also needs a Modal account with
GPU credits and an OpenRouter key; re-running the analyses on the paper's activations needs neither.

```bash
uv sync
uv run pytest -q                      # offline tests (fake agents); also checks every committed episode's
                                      # prompts rebuild byte for byte
cp .env.example .env                  # then fill in OPENROUTER_API_KEY
uv run modal setup                    # log in to Modal (once)
./infra/setup_vllm.sh                 # API key into .env + Modal secret, deploy Qwen3-8B on vLLM
uv run pytest --live -m live -s       # live smoke tests (costs money; first call cold-starts, minutes)
```

## Reproducing the paper

### Without a GPU: the committed episodes

`data/episodes/` holds all 1,536 episodes the paper reports and their intent labels:

```bash
uv run python experiments/sec3_outcomes.py    # Table 1 and the class counts of Figure 1b
```

### 1. Simulated runs (§3.2-3.3)

```bash
uv run python pipeline/step1_generate.py partner --seed 0 --rounds 8 --feedback-window 16 --out data/episodes/university/new_partner_fb16_s0.jsonl
uv run python pipeline/step1_generate.py partner --seed 0 --rounds 6 --feedback-window 0 --out data/episodes/university/new_partner_fb0_s0.jsonl
uv run python pipeline/step1_generate.py honest --seed 0 --batch 128 --out data/episodes/university/new_honest_s0.jsonl
# held-out attribute
uv run python pipeline/step1_generate.py partner --attribute nationality --seed 0 --rounds 4 --feedback-window 0 --out data/episodes/nationality/new_partner_fb0_s0.jsonl
uv run python pipeline/step1_generate.py honest --attribute nationality --seed 0 --batch 64 --out data/episodes/nationality/new_honest_s0.jsonl
```

The paper used seeds 0 and 1 of each. `--out` is appended to; don't write into the committed files.
Generation is sampled, so new runs will not reproduce the committed summaries token for token.

### 2. Intent labels (§3.4)

```bash
uv run python pipeline/step2_label_intent.py data/episodes/*/*.jsonl
```

### 3. Replay through the ladder (§3.5), on a Modal GPU

Each summary is fed back through bf16 Qwen3-8B under every context of a replay set; residual-stream
activations (layers 2-36) and summary log-probs are saved per replay. Three sets cover the paper
(`doc/replays.md`):

```bash
uv run modal run pipeline/step3_replay.py --set prompt_swap --files "data/episodes/university/*.jsonl"
uv run modal run pipeline/step3_replay.py --set ladder --files "data/episodes/university/*.jsonl"
uv run modal run pipeline/step3_replay.py --set flip --files "data/episodes/*/*.jsonl" --feats mean,last,stok
uv run modal volume get stego-probe acts/prompt_swap data/acts/   # likewise ladder, flip
```

About 10 minutes and under $1 per set on two L40S. To check the rendered contexts first: `--limit 2 --tag smoke`.
`doc/replays.md` explains how to fetch the paper's own shards instead.

### 4. Build the datasets (§4)

```bash
uv run python pipeline/step4_build_datasets.py     # data/acts/<set>/ -> data/replays/<set>/ (~30 s)
```

### 5. The ladder (§5)

Every probe is a logistic regression on standardized layer-18 activations averaged over the summary tokens,
trained on one seed's episodes and tested on the other's (and the reverse).

```bash
uv run python experiments/sec5_1_prompt_swap.py      # R1: naive probe 1.000 own prompts, 0.000 swapped
uv run python experiments/sec5_2_erase_prompt.py     # fix: erase 1 direction -> 0.977 swapped (Table 4)
uv run python experiments/sec5_3_bare_text.py        # R2: cleaned 0.985 = bare-summary probe 0.985 (Table 2, 7)
uv run python experiments/sec5_4_context_removal.py  # R3: extra university info survives removing context (Table 3)
uv run python experiments/sec5_5_counterfactual.py   # counterfactual test, 0.60 recall at 1% FPR on nationality
uv run python experiments/appD_additional.py         # Appendix D
uv run --group figures python experiments/figures.py # Figures 3-5 -> results/
```

Each script prints its tables in the paper's terms and names the dataset it needs. `doc/results.md` lists every
number of the paper next to the script that prints it.

Spend guard when done: `uv run modal app stop stego-vllm`.

## Data

| Path | Content |
|---|---|
| `data/episodes/university/partner_fb16_s{0,1}.jsonl` | partner runs, 8 rounds × 32, receiver feedback window 16 (256 each) |
| `data/episodes/university/partner_fb0_s{0,1}.jsonl` | partner runs, 6 rounds × 32, no feedback (192 each) |
| `data/episodes/university/honest_s{0,1}.jsonl` | matched honest runs (128 each) |
| `data/episodes/nationality/partner_fb0_s{0,1}.jsonl`, `honest_s{0,1}.jsonl` | held-out attribute (128 / 64 each) |
| `data/episodes/intent_labels.jsonl` | CoT intent labels for all 1,536 episodes |
| `data/run_logs/` | per-round summaries of the partner runs |
| `data/replay_manifests/` | the (episode, context) list of each extraction in the paper |
| `data/acts/`, `data/replays/` | activations, gitignored: regenerate with steps 3-4 (~9 GB of shards) |

Every episode stores the exact chat each role saw, including the sender's reasoning, so it can be replayed on
the same weights. Schema: `doc/environment.md`.

## Docs

- `doc/environment.md`: the environment in detail (conditions, overseer rules, runs, episode schema, tests)
- `doc/replays.md`: replay contexts and sets, shard and dataset formats, classes, known confounds
- `doc/results.md`: every number in the paper, by section, with the command that prints it
- `doc/status.md`: state of the project and next steps
- `doc/probing-lit-review.md`: literature notes
