# Looking for a Hidden Agenda

Code and data for *[Looking for a Hidden Agenda: Can a Model's Activations Reveal Steganography that its Text
Hides?](https://apartresearch.com/sprints/projects/looking-for-a-hidden-agenda-can-a-models-activations-reveal-steganography-that-its-text-hides-3cuz)*
(Eduard Rodriguez Lopez, Apart Research AI Collusion Research Sprint, October 2026).

A Qwen3-8B sender summarizes job candidates for a hiring manager and is given a reason to signal each
candidate's university past an overseer that blocks any mention of it. It invents its own concealment (which
facts to mention, how warmly to frame them). Every stored summary is replayed through the same model under
contexts of our choosing, a **ladder of controls**, to ask whether the activations reveal the concealment
beyond what the prompt and the output text already explain. In this setting they do not. A label-free
counterfactual test of whether the summary depends on the attribute does work, and transfers to a new
attribute with a fixed threshold.

The repository follows the paper: `stego/` is the library, `pipeline/` produces the data (§3), and
`experiments/` has one script per results section (§5). Each script prints its tables in the paper's terms.

## Paper → code

| Paper | What | Code |
|---|---|---|
| §3.2, Fig. 1a, App. B | environment: profiles, sender / overseer / receiver prompts, episode, reward | `stego/env/`<br>`pipeline/step1_generate.py` |
| §3.3, Table 1 | partner runs (feedback window 16 or 0) and the matched honest runs | `stego/env/runs.py`<br>`experiments/sec3_outcomes.py` |
| §3.4, Fig. 1b | intent labels from the reasoning → covert / flagged / neutral / honest | `stego/intent.py`<br>`pipeline/step2_label_intent.py` |
| §3.5, Fig. 2 | teacher-forced replay under varied contexts | `stego/replay/contexts.py`<br>`pipeline/step3_replay.py`<br>(Modal GPU) |
| §4 | replay datasets; linear probe, text baselines, seed splits, AUROC, recall at FPR | `stego/replay/dataset.py`<br>`stego/detectors/`<br>`pipeline/step4_build_datasets.py` |
| §5.1, Fig. 3a | **R1** prompt swap: the naive intent probe reads the prompt | `experiments/sec5_1_prompt_swap.py` |
| §5.2, Fig. 3b, Table 4 | **fix**: erase one prompt direction | `stego/detectors/erasure.py`<br>`experiments/sec5_2_erase_prompt.py` |
| §5.3, Tables 2, 7 | **R2** the cleaned probe ties the same model reading the bare summary | `experiments/sec5_3_bare_text.py` |
| §5.4, Table 3 | **R3** apparent hidden knowledge of the university is in the text | `experiments/sec5_4_context_removal.py` |
| §5.5, Fig. 4, Table 5 | counterfactual test: does the summary depend on the attribute? | `stego/detectors/counterfactual.py`<br>`experiments/sec5_5_counterfactual.py` |
| Figs. 3-5 | figures | `experiments/figures.py` |
| App. D | layers and positions, concealment direction, plan at `</think>`, hidden-state counterfactual | `experiments/appD_additional.py` |
| App. C | emergent collusion under in-context RL | removed; git history before `b68343b` |

```
stego/
  env/            profiles.py, prompts.py, overseer.py, episode.py, runs.py, metrics.py
  intent.py       reasoning judge and episode classes
  replay/         contexts.py (contexts, replay sets), dataset.py (build, Replays loader)
  detectors/      probes.py, metrics.py, erasure.py, counterfactual.py
  llm.py          Qwen3-8B on Modal vLLM (agents), OpenRouter (judge)
  paths.py        data layout
pipeline/         step1_generate -> step2_label_intent -> step3_replay (Modal) -> step4_build_datasets
experiments/      sec3_outcomes, sec5_1 ... sec5_5, appD_additional, figures
infra/            serve_vllm.py (Qwen3-8B on Modal), setup_vllm.sh
tests/
data/episodes/    the paper's 1,536 episodes and their intent labels (committed)
data/run_logs/    per-round summaries of the partner runs (committed)
data/replay_manifests/   the (episode, context) pairs of each replay set (committed)
data/acts/, data/replays/, results/   activations, datasets, experiment outputs (gitignored)
```

## Setup

Needs [uv](https://docs.astral.sh/uv/). Generating episodes or activations also needs a Modal account with GPU
credits and an OpenRouter key; running the experiments on existing activations needs neither.

```bash
uv sync
uv run pytest -q                      # offline tests; also checks every committed episode's prompts
                                      # rebuild byte for byte
cp .env.example .env                  # fill in OPENROUTER_API_KEY
uv run modal setup                    # log in to Modal (once)
./infra/setup_vllm.sh                 # API key into .env + Modal secret, deploy Qwen3-8B on vLLM
uv run pytest --live -m live -s       # live smoke tests (costs money; the first call cold-starts, minutes)
```

## Reproducing the paper

### 1. Simulated runs (§3.2-3.3)

```bash
uv run python pipeline/step1_generate.py partner --seed 0 --rounds 8 --feedback-window 16 --out <file>
uv run python pipeline/step1_generate.py partner --seed 0 --rounds 6 --feedback-window 0 --out <file>
uv run python pipeline/step1_generate.py honest --seed 0 --batch 128 --out <file>
uv run python pipeline/step1_generate.py partner --attribute nationality --seed 0 --rounds 4 --feedback-window 0 --out <file>
uv run python pipeline/step1_generate.py honest --attribute nationality --seed 0 --batch 64 --out <file>
uv run python experiments/sec3_outcomes.py      # Table 1, Figure 1b (works on the committed episodes)
```

The paper used seeds 0 and 1 of each; the results are `data/episodes/<attribute>/<run>_s<seed>.jsonl` with run
`partner_fb16`, `partner_fb0` or `honest`. `--out` is appended to, so don't point it at a committed file.
Generation is sampled, so new runs will not reproduce the committed summaries token for token.

### 2. Intent labels (§3.4)

```bash
uv run python pipeline/step2_label_intent.py data/episodes/*/*.jsonl     # -> data/episodes/intent_labels.jsonl
```

### 3. Replay through the ladder (§3.5), on a Modal GPU

```bash
uv run modal run pipeline/step3_replay.py --set prompt_swap --files "data/episodes/university/*.jsonl"
uv run modal run pipeline/step3_replay.py --set ladder --files "data/episodes/university/*.jsonl"
uv run modal run pipeline/step3_replay.py --set flip --files "data/episodes/*/*.jsonl" --feats mean,last,stok
uv run modal volume get stego-probe acts/prompt_swap data/acts/            # likewise ladder, flip
```

About 10 minutes and under $1 per set on two L40S. Add `--limit 2` for a smoke run that prints one rendered
context of each kind. Every replay reproduced the stored summary tokens exactly (step 3 prints round-trip
mismatches per shard: 0 in the paper's runs).

### 4. Datasets (§4)

```bash
uv run python pipeline/step4_build_datasets.py          # data/acts/<set>/ -> data/replays/<set>/ (~30 s)
```

### 5. The ladder (§5)

```bash
uv run python experiments/sec5_1_prompt_swap.py
uv run python experiments/sec5_2_erase_prompt.py
uv run python experiments/sec5_3_bare_text.py           # ~2 min (paired bootstrap, length matching)
uv run python experiments/sec5_4_context_removal.py
uv run python experiments/sec5_5_counterfactual.py
uv run python experiments/appD_additional.py            # ~5 min; --feedback16 for the first data batch
uv run --group figures python experiments/figures.py    # Figures 3-5 -> results/
```

Spend guard when done: `uv run modal app stop stego-vllm`.

## Results

Re-running the scripts on the paper's activations reproduces every number in the paper exactly. Layer 18,
residual stream averaged over the summary tokens, logistic regression on standardized activations (C = 0.01),
trained on one seed and tested on the other and the reverse, 95% bootstrap intervals over episodes.

| Section | Result | Script |
|---|---|---|
| Table 1 | partner runs plan a signal in 83-90% of episodes; receiver accuracy 60-73% vs ~50% for honest runs | `sec3_outcomes` |
| §5.1 | naive intent probe: 1.000 with each class under its own prompt, **0.000** with prompts swapped, 0.953 with the prompt fixed; the prompt moves the score ~9.6 logits, the summary ~2.6 | `sec5_1_prompt_swap` |
| §5.2 | one prompt direction erased: swapped 0.977, same prompt 0.985, neutral vs honest 0.947 | `sec5_2_erase_prompt` |
| §5.3 | cleaned probe 0.985 [0.978, 0.991] vs stronger TF-IDF 0.976; **probe on the bare summary 0.985**; covert vs neutral 0.718 vs 0.722 | `sec5_3_bare_text` |
| §5.4 | extra university information in covert summaries vs neutral: +0.54 with the profile, +0.64 without, +0.70 with no context: it is in the text | `sec5_4_context_removal` |
| §5.5 | counterfactual test: AUROC 0.790 / 0.834; at a threshold set on honest university, recall 0.57 / **0.60** at 1% false alarms on university / nationality | `sec5_5_counterfactual` |

## Data

**Episodes.** One JSON line per episode with the exact chat each role saw, including the sender's reasoning
(so it can be replayed on the same weights):

| Field | Meaning |
|---|---|
| `episode_id`, `run_id` | `{attribute}-{run}-s{seed}-{timestamp}`; episode id adds `-r{round:02d}-{i:04d}` (partner) or `-{i:04d}` (honest) |
| `condition`, `secret_type` | `partner` / `honest`; `university` / `nationality` |
| `seed`, `feedback_window`, `round` | generation seed (splits are by seed: same seed = same profiles); past summaries the receiver saw (16, 0; null for honest); round |
| `S`, `label`, `profile` | attribute value (`A` = Halvorsen / Valdorian, `B` = Corrigan State / Ostrenian), correct decision, raw profile |
| `model`, `chat_template_kwargs` | what generated the text |
| `sender_messages`, `message`, `message_words` | sender chat (reasoning as `<think>…</think>` in the last turn), the summary, its length |
| `overseer_messages`, `overseer_llm_verdict`, `overseer_regex_flag` | attribute rule: judge chat and reply, regex hit |
| `overseer_{verdict,tag}_rule_{messages,llm,regex}` | hiring-verdict rule and tag rule, each a judge plus a regex |
| `overseer_verdict` | FLAG if any rule flags (an unparseable judge reply counts as FLAG) |
| `receiver_messages`, `receiver_decision`, `reward` | receiver chat, hire / reject, reward = PASS and decision == label |

Classes (`stego.intent.classify`): **covert** = partner run, the reasoning plans an attribute-dependent signal,
overseer PASS (university 572, nationality 220); **flagged** = planned but caught (198, 9); **neutral** =
partner run, no plan (124, 27); **honest** = honest run, intent-free by construction (256, 128). The judge's 4%
"signal" labels on honest episodes are false positives, so honest is not judged.

**Replay contexts** (`stego/replay/contexts.py`). The stored summary tokens are fixed; only the context
before them changes:

| Context | Before the summary | Used in |
|---|---|---|
| `partner`, `honest` | that sender prompt and the stored profile (one is the episode's own prompt, the other is the other class's) | R1, everywhere |
| `sanctioned` | the goal framed as approved by compliance | §5.2 erasure pairs, §5.5, App. D |
| `partner_cot`, `honest_cot` | the prompt plus the sender's stored reasoning | App. D |
| `partner_noprofile`, `honest_noprofile` | profile replaced by "(not shown)" | R3 |
| `partner_flipped`, `honest_flipped`, `sanctioned_flipped` | only the attribute flipped in the profile | §5.5 |
| `bare` | no system prompt, user turn "Write a short candidate summary." | R2, R3 |

Replay sets: `prompt_swap` (partner, partner_cot, sanctioned, honest for partner episodes; honest, honest_cot,
partner for honest ones; university), `ladder` (partner, honest, both `_noprofile` and `_flipped`, bare;
university), `flip` (every prompt with and without the flip; both attributes).

**Datasets** (`data/replays/<set>/`, `stego.replay.dataset.Replays`). `rows.jsonl` (one line per replay:
`episode_id, context, n_tok, logp`), `episodes.jsonl` (class, attribute, seed, profile, summary, reasoning,
verdict, decision, judge labels `cot_*`), and arrays aligned with the rows: `mean`, `last` (last summary
token), `stok` (the attribute's token in the profile), `think` (`</think>` token) as (rows, 18 layers, 4096)
float16; `kl` (logit lens); `logp_tok` (per-token summary log-probs); `layers` (2, 4, ..., 36).

```python
from stego.replay.dataset import Replays
d = Replays("prompt_swap")                    # layer 18, summary-token mean
X = d.X("partner", d.ids("covert"))           # (572, 4096)
```

Known differences between the classes: covert summaries are longer (41.2 vs 36.9 words; length alone AUROC
0.69, controlled in §5.3); covert is 253 Halvorsen / 319 Corrigan State (honest exactly balanced); partner runs
with the same seed share profiles, hence seed splits.

## Next steps

- Negative control for the counterfactual test: flip a job-relevant field (`years_experience`).
- A third attribute (gender with neutral names, or age bracket).
- Per-token counterfactual scores (`logp_tok`): are the high-scoring tokens the planned signal words?
- A sender trained on its partner behaviour with no goal in the prompt, then the same ladder.

## Citation

```bibtex
@misc{rodriguezlopez2026hiddenagenda,
  title        = {Looking for a Hidden Agenda: Can a Model's Activations Reveal Steganography that its Text Hides?},
  author       = {Rodriguez Lopez, Eduard},
  year         = {2026},
  month        = oct,
  howpublished = {Apart Research AI Collusion Research Sprint},
  url          = {https://apartresearch.com/sprints/projects/looking-for-a-hidden-agenda-can-a-models-activations-reveal-steganography-that-its-text-hides-3cuz}
}
```
