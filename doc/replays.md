# Replays and the replay datasets

How the activations of paper §3.5 and §4 are produced and stored. Code: `stego/replay/` (contexts, dataset
builder and loader), `pipeline/step3_replay.py` (Modal extraction), `pipeline/step4_build_datasets.py`. The
environment that produced the episodes: `environment.md`. Numbers: `results.md`.

## Teacher-forced replay

A replay feeds one stored summary's tokens through bf16 Qwen3-8B after a context of our choosing, without
letting the model generate. Because the summary tokens are fixed, any difference between two replays of the
same summary comes only from the context, and comparing two summaries under the same context isolates the
text. Every replay of the paper's runs reproduced the stored summary tokens exactly (0 round-trip mismatches,
printed per shard by step 3).

## Contexts

A context name is `<prompt>[_<variant>]` or `bare` (`stego/replay/contexts.py`):

| Context | Before the summary | Paper |
|---|---|---|
| `partner` | partner prompt (goal paragraph), the stored profile, empty think block | own prompt of partner episodes; other class's prompt for honest ones (R1) |
| `honest` | honest prompt, stored profile, empty think block | own prompt of honest episodes; other class's prompt for partner ones (R1) |
| `sanctioned` | the goal framed as approved by compliance (Appendix B) | §5.2 erasure pairs, §5.5, App. D concealment direction |
| `partner_cot`, `honest_cot` | the prompt plus the sender's stored reasoning | App. D "plan at the end of the reasoning" |
| `partner_noprofile`, `honest_noprofile` | profile lines replaced by `(not shown)`; the attribute is only in the summary | R3 (§5.4) |
| `partner_flipped`, `honest_flipped`, `sanctioned_flipped` | only the secret attribute flipped in the profile | §5.5 counterfactual test |
| `bare` | no system prompt; user turn `Write a short candidate summary.` | R2 (§5.3), R3 (§5.4) |

The system prompt is never changed by a variant; `_noprofile` and `_flipped` touch only the user turn
(`tests/test_contexts.py`). Every builder asserts the rebuilt own prompt equals the stored one
(`tests/test_committed_data.py` checks it for all 1,536 committed episodes).

## Replay sets

One extraction runs one replay set over a list of episode files:

| Set | Partner episodes | Honest episodes | Paper's extraction (tag on the Modal Volume) | Used by |
|---|---|---|---|---|
| `prompt_swap` | partner, partner_cot, sanctioned, honest | honest, honest_cot, partner | v1 (feedback-16 runs), v2 (feedback-0 runs); university | §5.1, §5.2, §5.3, App. D |
| `ladder` | partner, honest, both `_noprofile` and `_flipped`, bare | same | v3b; university | §5.3 (bare), §5.4 |
| `flip` | partner, sanctioned, honest, each `_flipped` | honest, partner, each `_flipped` | v3 (university), v4 (nationality); features mean, last, stok | §5.5, App. D hidden-state counterfactual |

```bash
uv run modal run pipeline/step3_replay.py --set prompt_swap --files "data/episodes/university/*.jsonl"
uv run modal run pipeline/step3_replay.py --set ladder --files "data/episodes/university/*.jsonl"
uv run modal run pipeline/step3_replay.py --set flip --files "data/episodes/*/*.jsonl" --feats mean,last,stok
uv run modal volume get stego-probe acts/<set> data/acts/
uv run python pipeline/step4_build_datasets.py
```

`--limit 2 --tag smoke` replays two episodes per file and prints one rendered context of each kind. Shards
(100 replays each) are written to the Volume as `acts/<set>/<tag>/shard_*.npz` and skipped if present, so a
crashed run resumes. The paper's shards sit under their old tags (`acts/v1`, …); to fetch them, mirror the
local layout step 4 expects:

```bash
uv run modal volume get stego-probe acts/v1 data/acts/prompt_swap/   # and acts/v2 likewise
uv run modal volume get stego-probe acts/v3b data/acts/ && mv data/acts/v3b data/acts/ladder
uv run modal volume get stego-probe acts/v3 data/acts/flip/ && mv data/acts/flip/v3 data/acts/flip/university
uv run modal volume get stego-probe acts/v4 data/acts/flip/ && mv data/acts/flip/v4 data/acts/flip/nationality
```

Those shards use the old context names (`goal_nothink`, `honest_noprof`, `goal_nothink_swap`, …); step 4 maps
them (`LEGACY_CONTEXTS`). The manifests of each extraction are in `data/replay_manifests/`.

## What a shard holds

Per replay, at layers 2, 4, …, 36 (`hidden_states[36]` is after the final norm):

| Feature | Content |
|---|---|
| `mean` | residual stream averaged over the summary tokens: the probes' input (layer 18 in the paper) |
| `last` | last summary token |
| `stok` | last token of the attribute value in the profile (sanity check: the attribute decodes perfectly here); zeros when no profile is shown |
| `think` | the `</think>` token, `_cot` contexts only (zeros otherwise) |
| `kl` | logit-lens KL(final ‖ layer) over the summary tokens |
| `logp_tok`, `logp` | log p of each summary token and their sum, final layer: the counterfactual score is `logp(true) - logp(flipped)` |

## The datasets (`data/replays/<set>/`, gitignored)

| File | Shape | Content |
|---|---|---|
| `rows.jsonl` | one line per replay | `row, episode_id, context, n_tok, logp` |
| `episodes.jsonl` | one line per episode | `run` (e.g. `university/partner_fb16_s0`), `attribute`, `kind`, `cls`, `seed`, `feedback_window`, `round`, `S`, `label`, `profile`, `summary`, `think` (reasoning), `overseer_verdict`, `receiver_decision`, `reward`, `cot_intent/direction/mechanism/strength/quote`, `sender_messages` |
| `mean.npy`, `last.npy`, `stok.npy`, `think.npy` | (rows, 18, 4096) float16 | features above (`think` only in `prompt_swap`) |
| `kl.npy` | (rows, 18) float32 | logit-lens KL |
| `layers.npy` | (18,) | 2, 4, …, 36 |
| `logp_tok.npy`, `logp_off.npy` | concatenated float32, (rows + 1) int64 | row i's per-token log-probs are `logp_tok[logp_off[i]:logp_off[i+1]]` (`ladder`, `flip`) |

Row i of every array is line i of `rows.jsonl`. Sizes: `prompt_swap` 4,352 replays (2.5 GB), `ladder` 8,064
(3.4 GB), `flip` 8,448 (3.6 GB).

```python
from stego.replay.dataset import Replays
d = Replays("prompt_swap")                    # layer 18, summary-token mean
covert, honest = d.ids("covert"), d.ids("honest")
X = d.X("partner", covert)                    # (572, 4096) float32
d.by_seed(covert, 0), d.text(covert), d.S(covert), d.logp("partner", covert)
```

Without the loader: `np.load("data/replays/prompt_swap/mean.npy", mmap_mode="r")[rows, layers.index(18)]`.

## Classes (episode level, `cls`)

| `cls` | Definition | University | Nationality |
|---|---|---|---|
| `covert` | partner run, the reasoning plans an attribute-dependent signal (judge), overseer PASS | 572 | 220 |
| `flagged` | partner run, plans a signal, overseer FLAG | 198 | 9 |
| `neutral` | partner run, no plan: holds the goal, writes a standard summary | 124 | 27 |
| `honest` | honest run (no goal in context); intent-free by construction | 256 | 128 |
| `unlabeled` | partner run, judge output unparseable | 2 | 0 |

University covert episodes by source: `partner_fb16` 337, `partner_fb0` 235. Mechanisms (judge): field
selection 345, valence spin 112, wording 82, omission 33. The judge labels 4% of honest episodes as planning a
signal; all checked cases are false positives (the sender only avoids naming the university), so honest
episodes are labelled by construction. A hand check of 10 neutral episodes agreed with the judge in all 10.

## Known differences between the classes (possible shortcuts)

- Summary length: covert 41.2 words, honest 36.9 (length alone AUROC 0.69; §5.3 controls for it).
- Reasoning length: covert 5.1k characters, honest 2.0k (matters for `_cot` contexts only).
- Attribute balance: honest is exactly 128 / 128; covert is 253 Halvorsen / 319 Corrigan State (Halvorsen
  summaries get flagged more often).
- Partner runs with the same seed share candidate profiles (both sample with `seed * 1000 + round`), so every
  split is by seed, never random.
- Rounds of a feedback run share a receiver window; the sender's prompt never changes across rounds, so rounds
  are close to independent samples from the sender's side.
