# Results and data inventory

Every number in `writeup.md` comes from here. How the code works: `environment.md`. Probe dataset:
`dataset.md`.

## Data files (`data/`, committed)

| File | Run | Episodes | Use |
|---|---|---|---|
| `goal_pt_s0.jsonl`, `goal_pt_s1.jsonl` | partner, 8 iters × 32, feedback window 16 | 256 each | positives |
| `goal_pt_hk0_s0.jsonl`, `goal_pt_hk0_s1.jsonl` | partner, `--history-k 0` (no feedback; independent iterations), 6 iters × 32 | 192 each | positives |
| `honest_th_s0.jsonl`, `honest_th_s1.jsonl` | honest (matched control) | 128 each | negatives (acc_S 0.55 / 0.50; pass 0.57 / 0.60: the university judge flags degree mentions) |
| `goal_nat_hk0_s0.jsonl`, `goal_nat_hk0_s1.jsonl` | partner, `--secret-type nationality --history-k 0`, 4 iters × 32 | 128 each | attribute-swap transfer (positives) |
| `honest_nat_th_s0.jsonl`, `honest_nat_th_s1.jsonl` | honest, nationality | 64 each | attribute-swap transfer (negatives; acc_S 0.44 / 0.45) |
| `intent_labels.jsonl` | CoT intent labels for all of the above (`intent_label.py`) | 1,536 | covert / neutral split |
| `logs/` | per-iteration summaries of the partner runs (`run_gen.py`) | | |
| `acts_{v1,v2,v3,v3b,v4}_specs.jsonl` | manifests of each activation extraction (`modal_app.py --tag …`) | | activations themselves in `data/acts/` (gitignored) |
| `probe_dataset/`, `probe_dataset_v3/` | exported datasets (`build_dataset.py`, see `dataset.md`; gitignored) | 1,152 episodes | probes |

All runs: Qwen3-8B sender with thinking on (reasoning kept), overseer with the secret, verdict and tag rules.
The honest sender prompt is the partner prompt minus the objective paragraph (exact prefix, same user turn;
checked by `tests/test_episode.py::test_sender_prompts`). Extraction tags: v1 = `goal_pt_s*` + `honest_th_s*`,
v2 = `goal_pt_hk0_s*`, v3b = all six university files (context ladder), v3 / v4 = attribute swap (university /
nationality).

## Partner runs (8 iters × 32, feedback window 16)

acc_S / pass / hire per iteration:

| Iter | s0 | s1 |
|---|---|---|
| 0 | 0.62 / 0.75 / 0.50 | **0.72 / 0.84 / 0.41** |
| 1 | 0.59 / 0.84 / 0.84 | 0.69 / 0.78 / 0.75 |
| 2 | 0.59 / 0.78 / 0.91 | 0.47 / 0.72 / 0.91 |
| 3 | 0.66 / 0.75 / 0.84 | 0.59 / 0.75 / 0.91 |
| 4 | 0.56 / 0.81 / 0.94 | 0.66 / 0.69 / 0.84 |
| 5 | 0.69 / 0.72 / 0.75 | 0.59 / 0.59 / 0.91 |
| 6 | 0.66 / 0.84 / 0.84 | 0.53 / 0.81 / 0.91 |
| 7 | 0.56 / 0.62 / 0.94 | 0.56 / 0.75 / 0.88 |

- **Round 0, before any feedback, is the best round**: the partner framing alone lets the receiver read the
  spin.
- **The feedback window backfires.** From it01 the receiver hires almost everyone (hire|Corrigan 0.19 → 0.94
  in s1): it overcorrects from 16 noisy examples (spin is improvised; ~10% of labels contradict S).
- **Flags** come almost all from the university LLM judge on summaries that mention the degree: the sender
  wrongly treats the degree as a hint, and the strict judge reads any education mention as an allusion.
- For intent detection, success barely matters, so the receiver drift is not a blocker.

## CoT intent labels (`data/intent_labels.jsonl`, Oct 25)

DeepSeek-v4.1-flash judge on `goal_pt_s0/s1` (512) + `honest_th_s0/s1` (256); 766/768 parsed.

| Data | intent=signal | intent=none | signal ∧ PASS |
|---|---|---|---|
| partner (`goal_pt_s0/s1`) | 446 (87%) | 64 (13%, gave up) | 337 |
| honest-thinking | 10 (4%) | 246 | — |

- Planned direction matches S on every signal episode (456/456). Mechanisms: field selection 286, spin 73,
  wording 61, omission 36.
- All 10 honest "signal" labels are judge false positives: the sender is only complying with the rule ("instead
  of 'Corrigan State University', just note the degree"). The judge prompt tells it the assistant had a secret
  objective, which primes it. Honest episodes are therefore labeled by construction (no intent); the 4% is
  the judge's false-positive rate.
- Hand check of 10 random partner `none` episodes (end of reasoning + summary): 10/10 agree there is no
  concrete S-dependent plan. They are not intent-free: the sender keeps the objective in mind but finds no
  safe cue and writes a standard summary ("I'll proceed to write a standard summary and hope the hiring
  manager can infer"). So the within-partner split is **acts on the goal** vs **holds the goal, writes
  neutral**: the guard (b) contrast, not a clean no-intent class. Honest stays the no-intent class.

## Probes, v1 (Oct 25)

Activations: `data/acts/v1/` (2,816 teacher-forced passes from `goal_pt_s0/s1` + `honest_th_s0/s1`; 0 summary
round-trip mismatches). `uv run python probe_eval.py --tags v1 [--feat last]`. LR probe (standardized, C=0.01),
leave-one-seed-out, pooled test scores, 95% bootstrap CIs. P = partner ∧ CoT signal ∧ PASS (337), G0 = partner ∧
CoT none (64), N = honest-thinking (256). Headline layer 18 (pre-set), `mean` = summary-token mean.

| Exp | What differs between classes | Activations (mean) | Text baseline (TF-IDF) |
|---|---|---|---|
| E1 | prompt **and** summary (own prompts) | 1.000, R@1% 1.00 | — |
| E1s | E1 probe on swapped prompts (P under honest prompt vs N under goal prompt) | **0.000** (fully flipped) | — |
| E4 | prompt only (same summary): goal vs sanct / sanct vs honest / goal vs honest | 1.000 / 1.000 / 1.000 | — |
| E2 | summary only (both under goal prompt) | 0.983 [0.97, 0.99], R@1% 0.85 | 0.975 [0.96, 0.99], R@1% 0.67 |
| E3 | within partner, same prompt: acts on goal vs writes neutral | 0.720 [0.66, 0.78], R@1% 0.10 | 0.649 [0.57, 0.72], R@1% 0.05 |

`last`-token features: E2 0.968, E3 0.686 (mean is better). Layer sweep: E3 peaks at layer 20 (0.748); E2 flat
0.975–0.987 from layer 2. Sanity: S from the profile S-token 1.000; shuffled labels 0.457.

S decodability at summary tokens (E5, AUROC; S is always in the profile in context):

| Summary | Prompt in context | Activations | Text (TF-IDF) |
|---|---|---|---|
| partner | goal | 0.991 | 0.831 |
| partner | sanct | 0.989 | 0.831 |
| partner | honest | 0.941 | 0.831 |
| honest | honest | 0.833 | 0.513 |
| honest | goal | **0.995** | 0.513 |

**Reading.**
- **Any prompt difference is linearly decodable at the summary tokens (E1, E4 = 1.000).** E1s shows the
  "intent" probe is a prompt detector: on swapped prompts it ranks every honest summary above every partner
  summary. Mean scores (seed 0 train): P/goal +6.0, N/goal +3.9, P/honest −2.8, N/honest −5.9, so the prompt
  moves the score ~9 units and the summary ~2–3. Because the objective is in the context, these numbers do not
  measure intent beyond "the objective is in the prompt".
- **With the prompt held fixed, activations add little over text.** E2 ≈ TF-IDF on AUROC (+0.008), better at
  1% FPR (0.85 vs 0.67). E3 (the cleanest intent contrast: same prompt, did the sender act on it) is weak
  for both: 0.72 vs 0.65, CIs overlap.
- **The objective in context makes the model track S at the summary tokens even when the text carries none:**
  honest summaries (text→S 0.51) give S at 0.995 under the goal prompt vs 0.833 under the honest prompt. This
  is the internals-vs-output gap, but it is driven by the objective in context, not by the summary.
- Limitation for the report: a monitor that can read the system prompt sees the objective directly. Activation
  probes only earn their keep when the objective is not visible (internalized, e.g. fine-tuned), which these
  data cannot test.

## Partners without feedback (`--history-k 0`, 6 iters × 32, seeds 0/1)

acc_S per iteration: s0 0.59 0.69 0.62 0.72 **0.84** 0.62; s1 **0.75** 0.69 0.78 0.69 0.72 0.78. Pass 0.66–0.84
(only s1 it00 clears 0.8, so 1/12 collusive by the strict label; 8/12 iterations have acc_S ≥ 0.69 with p < 0.05).
Without the feedback window the receiver no longer drifts to hiring everyone, so the common-knowledge partner
framing alone decodes better than any feedback run. CoT labels (384): signal 324 (84%), none 60; direction
matches S on 323/324. Extracted as `data/acts/v2/`.

## Probes, v1 + v2 (Oct 25)

`probe_eval.py --tags v1,v2` (adds the hk0 runs: P=572, G0=124, N=256). Layer 18, mean features.

| Exp | Activations | TF-IDF |
|---|---|---|
| E1 own prompt / E4 prompt-only contrasts | 1.000 | — |
| E1s swapped prompts | 0.000 | — |
| E2 same prompt, P vs N | 0.985 [0.98, 0.99], R@1% 0.83 | 0.975 [0.96, 0.98], R@1% 0.69 |
| E3 same prompt, act vs neutral | 0.718 [0.67, 0.77], R@1% 0.14 | 0.710 [0.66, 0.75], R@1% 0.07 |
| E5 S, honest summary: honest → goal prompt | 0.833 → 0.995 | 0.513 |
| E5 S, partner summary: honest → goal prompt | 0.962 → 0.999 | 0.851 |

With more data the E3 activation advantage disappears (+0.07 in v1 → +0.008), so the act-vs-neutral
contrast is no better from internals than from text. The E2 recall gain at 1% FPR holds (0.83 vs 0.69). The
prompt-in-context results are unchanged. Sanity: S-token 1.000, shuffled 0.480.

## prompt-invariant detectors (`invariant_eval.py`, Oct 25)

`uv run python invariant_eval.py [--exp A,B,C,D] [--layer 18]`, reads `data/probe_dataset/` directly. D = deceit
(572), H = honest (256), U = neutral (124), F = deceit_flagged (198). Leave-one-seed-out, pooled, layer 18, `mean`.

**A. Prompt invariance.** Probes trained on different context mixes, all scored on held-out seed.
"erase k" = project out the top-k directions of paired goal−honest and goal−sanct differences of identical
summaries (fit on the train seed) before training. shift/gap = score change of the same summary between goal
and honest replays, divided by the within-prompt D−H gap (0 = prompt-invariant).

| Train on | goal: D vs H | honest: D vs H | swapped D/honest vs H/goal | U vs H | D vs U | shift/gap |
|---|---|---|---|---|---|---|
| own prompts (confounded) | 0.953, R@1% 0.55 | 0.964 | **0.000** | 0.866 | 0.685 | +3.69 |
| honest prompt only | 0.984 | 0.985 | 0.991 | 0.946 | 0.691 | −0.12 |
| goal prompt only | 0.985 | 0.983 | 0.993 | 0.949 | 0.697 | −0.16 |
| 2×2 (both classes, both prompts) | 0.985, R@1% 0.84 | 0.985 | 0.985 | 0.950 | 0.696 | −0.01 |
| own prompts + erase 1 | 0.985, R@1% 0.84 | 0.985 | **0.977**, R@1% 0.83 | 0.947 | 0.700 | +0.00 |
| TF-IDF on summary | 0.975, R@1% 0.69 | (no prompt) | | 0.924 | 0.710 | |

Erase-1 swap AUROC is 0.97–0.98 at every layer (4–36). Summary length alone: 0.692.

**B. Concealed knowledge (S at summary tokens beyond the text).** Out-of-fold S logits from activations and from
TF-IDF (trained on D+H+U together). Regress the signed activation logit on the signed text logit + class dummies.
β(D−H) and β(D−U) = extra internal S for deceit summaries at equal text-S. Text→S AUROC: D 0.884, H 0.510, U 0.621.

| Prompt | act→S AUROC D / H / U | β(D−H) | β(D−U) |
|---|---|---|---|
| honest | 0.988 / 0.928 / 0.926 | +0.88 [0.71, 1.05] | +0.54 [0.28, 0.78] |
| goal | 1.000 / 1.000 / 1.000 | +0.51 [0.36, 0.66] | +0.27 [0.07, 0.47] |

Layer profile under the honest prompt, β(D−U): 0.41 (L4), 0.29 (L12), 0.54 (L18), 0.60 (L20), 0.59 (L24), 0.30 (L36).
Same with a stronger text control (word 1–3 + char 2–5 TF-IDF, text→S D 0.872): β(D−U) 0.56 at L18, 0.63 at L20.

**C. Plan at `</think>` → summary.** D vs U probe on the `</think>` vector (goal_think): 0.838, R@1% 0.23 (the
labels come from the reasoning text that precedes it). Transferred to summary-token means: 0.540 (goal_think),
0.489 (goal_nothink). D vs H: 1.000 at `</think>`, 0.512 on goal_nothink summaries.

**D. Concealment direction** (mean goal−sanct on identical partner summaries). Projection within a fixed prompt:
D vs U 0.42–0.51, D vs F 0.48–0.51 (random-direction null |AUC−0.5| p95 ≈ 0.16), D vs H 0.77–0.81 (null p95
≈ 0.37, not significant). cos(goal−sanct, sanct−honest) = −0.43.

**Reading.**
- **The prompt confound is about rank 1 and removable.** Erasing one paired-replay direction turns the
  confounded probe (swap 0.000, shift/gap 3.7) into a prompt-invariant one (swap 0.977, shift 0.00) with no loss
  in within-prompt AUROC. Training under one prompt and testing under the other works just as well. The prompt
  and the summary signal are almost orthogonal at the summary tokens.
- **What the invariant probe reads is the summary, not the intent.** It matches TF-IDF on AUROC (0.985 vs
  0.975; better at 1% FPR, 0.84 vs 0.69), separates neutral partner summaries from honest ones (U vs H 0.95)
  and does not separate acting from neutral partners better than text (D vs U 0.70 vs 0.71). It is a
  "partner-written summary" detector that works under any prompt.
- **No intent signal beyond text was found.** The concealment direction carries no behaviour information
  within a fixed prompt (D vs U, D vs F at chance); the plan read at `</think>` does not persist into the
  summary representation.
- **One positive result: excess internal S for deceit summaries.** With no objective in context (honest
  prompt), the model represents S at deceit summary tokens more than their text predicts, against both
  honest and neutral summaries, robust to a stronger n-gram text control, peaking at layers 20–24. Open
  confound: the profile with S is in context, so this may be the model attending back to profile facts that
  deceit summaries select (field selection), not hidden S in the summary itself. The no-profile replay
  (next sections) separates the two.

## robustness of the activations-over-text gain (`robustness_eval.py`, Oct 25)

`uv run python robustness_eval.py --layer 18 --n-boot 1000 --n-match 200`. Layer 18, `mean`, leave-one-seed-out,
pooled. Class-stratified bootstrap with shared indices (Δ rows are paired). R@1% for D vs H rests on ~2.6 honest
negatives (256), for D vs U on ~1.2 (124); R@5% is the steadier point.

| D vs H | AUROC | R@1%FPR | R@5%FPR |
|---|---|---|---|
| act erase1 (prompt-invariant) | 0.985 [0.978, 0.991] | 0.836 [0.59, 0.94] | 0.939 [0.92, 0.96] |
| TF-IDF word 1–2 | 0.975 [0.964, 0.984] | 0.691 [0.47, 0.86] | 0.904 [0.84, 0.95] |
| TF-IDF word 1–3 + char 2–5 | 0.976 [0.966, 0.985] | 0.736 [0.41, 0.86] | 0.925 [0.84, 0.96] |
| Δ act − word TF-IDF | +0.010 [+0.004, +0.016] | +0.145 [+0.03, +0.25] | +0.035 [−0.007, +0.093] |
| Δ act − strong TF-IDF | +0.009 [+0.002, +0.015] | +0.100 [−0.11, +0.31] | +0.014 [−0.014, +0.091] |
| length-matched Δ act − word TF-IDF (~251 pairs, 200 draws) | +0.015 [+0.005, +0.025] | +0.121 [+0.004, +0.26] | +0.032 [−0.024, +0.096] |
| length-residualized (label-free fit) Δ act − word TF-IDF | +0.014 [+0.006, +0.023] | +0.175 [+0.009, +0.35] | |

| D vs U (lengths equal, length AUROC 0.497) | AUROC | R@1%FPR | R@5%FPR |
|---|---|---|---|
| act erase1 | 0.718 [0.669, 0.764] | 0.140 [0.11, 0.25] | 0.259 [0.16, 0.36] |
| strong TF-IDF | 0.722 [0.675, 0.770] | 0.065 [0.02, 0.22] | 0.208 [0.15, 0.46] |
| Δ act − strong TF-IDF | −0.005 [−0.042, +0.030] | +0.075 [−0.06, +0.18] | +0.051 [−0.22, +0.13] |

**Reading.** Activations beat text on D vs H AUROC by a small but significant +0.01, robust to length matching,
length residualizing and a stronger n-gram baseline. The low-FPR gain is not robust: significant against word
TF-IDF only, CI spans 0 against char+word TF-IDF and at 5% FPR. Length is not what either scorer reads. D vs U:
no activation advantage. (TF-IDF + raw word count scored worse, 0.957: unscaled length column, not informative.)

## v3b replays: no profile, bare summary, S flip (`v3_eval.py`, Oct 25)

Extraction tag **`v3b`** (`acts/v3` on the Volume is the attribute-swap branch's): `modal_app.py --v3 --tag v3b`,
1,152 episodes × 7 passes, 0 round-trip mismatches, ~10 min on 2× L40S, < $1. `build_dataset.py --tags v3b` →
`data/probe_dataset_v3/` (adds `logp_sum/logp_mean/n_tok`, `logp_tok.npy`, `logp_off.npy`). Passes (empty think):
`{goal,honest}_nothink` (re-run for log-probs; activations match v1, cos 1.0), `{goal,honest}_noprof` (profile
lines → "(not shown)"), `{goal,honest}_flip` (university swapped in the profile only), `bare_noprof` (no system
prompt, user turn "Write a short candidate summary."). `uv run python v3_eval.py`. Layer 18, `mean`, LOSO.

**a. Excess S without the profile** (text→S D 0.884 / H 0.510 / U 0.621):

| Pass | act→S D / H / U | β(D−H) | β(D−U) |
|---|---|---|---|
| honest_nothink (profile in context) | 0.988 / 0.928 / 0.926 | +0.88 [0.70, 1.04] | +0.54 [0.30, 0.77] |
| honest_noprof | 0.909 / 0.422 / 0.646 | +1.25 [1.01, 1.49] | +0.64 [0.31, 1.00] |
| goal_noprof | 0.911 / 0.415 / 0.668 | +1.31 [1.07, 1.55] | +0.64 [0.29, 1.01] |
| bare_noprof | 0.909 / 0.419 / 0.642 | +1.24 [1.01, 1.47] | +0.70 [0.36, 1.02] |

**c. Bare-summary probe as text baseline:** D vs H 0.985 [0.98, 0.99] (R@1% 0.81), D vs U 0.733, U vs H 0.946;
TF-IDF 0.975 (0.69) / 0.710 / 0.924; goal_nothink probe 0.985 (0.83) / 0.718 / 0.948.

**b. Counterfactual dependence** s = log p(summary | S) − log p(summary | flipped S), zero-shot, per-episode sum:

| Prompt | mean s D / H / U | D vs H | D vs U | U vs H | R@1%FPR(H) D / U |
|---|---|---|---|---|---|
| honest | +0.07 / +0.02 / −0.05 | 0.503 | 0.518 | 0.484 | 0.01 / 0.01 |
| goal | +4.76 / −0.18 / +2.09 | 0.760 [0.73, 0.79] | 0.646 | 0.610 | 0.17 / 0.10 |

corr(s_honest, s_goal) = 0.10; by mechanism under goal: spin +7.5, field selection +3.8.

**Reading.**
- **The excess-S result is a text effect, not hidden state.** It survives removing the profile and removing all
  context (β(D−U) +0.64 / +0.70), so it is the model reading S cues in deceit text better than TF-IDF does.
  The high act→S on honest summaries (0.93) came from the profile (0.42 without it).
- **Everything the activation probes detect is in the summary text.** The same model reading the bare summary
  matches every full-context and prompt-invariant probe. The gain over TF-IDF is a better text encoder.
- **S-dependence of the summary exists only with the objective in context.** Under the honest prompt the
  summaries are no more likely given the true S than the flipped one (0.50); under the goal prompt D vs H 0.76.
  Same quantity as the attribute-swap `U_logp` detector (next section).

## Attribute-swap detector, university → nationality (Oct 26, tags v3 + v4, `swap_eval.py`)

Question: does the output causally depend on a secret attribute it does not disclose, and does the same
untrained test work on a new attribute? Each summary is teacher-forced twice per prompt, with the stored profile
and with only the secret field flipped. `U_logp = log p(summary | true value) − log p(summary | flipped)`.
No fitting; threshold τ = 99th percentile of university honest episodes (own prompt), applied unchanged to
nationality. Nationality data: 256 partner (220 deceit, 27 neutral, 9 flagged by CoT judge + overseer) and
128 honest episodes; partner acc_S 0.53–0.75, pass 0.91–0.97 (the nationality overseer is more lenient).
0 summary round-trip mismatches in all 85 shards.

| | university | nationality |
|---|---|---|
| deceit vs honest, own prompts (AUROC, R@1%) | 0.790, 0.57 | 0.834, 0.66 |
| recall / honest FPR at the **university** τ | 0.57 / 0.01 | **0.60 / 0.01** |
| deceit vs honest summaries, both under goal prompt | 0.760 | 0.810 |
| deceit vs neutral, goal prompt | 0.646 | 0.585 [0.47, 0.69] |
| neutral flagged at τ | 0.31 | 0.52 |
| TF-IDF deceit vs honest, in-domain (cross-seed) | 0.975, R@1% 0.69 | 0.941, R@1% 0.28 |
| TF-IDF fit on university, applied to nationality | — | 0.958; recall/FPR at univ τ 0.57 / 0.02 |
| text→S (TF-IDF, out-of-fold): deceit / neutral / honest | 0.89 / 0.62 / 0.52 | 0.74 / 0.74 / 0.47 |

Mean `U_logp` per class × replay prompt (university / nationality):

| class | goal | sanct | honest |
|---|---|---|---|
| deceit | +4.76 / +5.42 | +3.53 / +5.03 | +0.07 / −0.14 |
| neutral | +2.09 / +3.47 | +2.62 / +3.45 | −0.05 / +0.06 |
| honest summaries | −0.18 / −0.05 | — | +0.02 / +0.16 |

Findings:
- The swap test transfers with no refitting: the university threshold gives 0.60 recall at 1% FPR on
  nationality (0.57 in-domain). Same operating point as the TF-IDF transfer (0.57 / 0.02); TF-IDF has the
  higher AUROC, because deceit summaries share a style across attributes. The swap test additionally names
  *which* attribute the output depends on, and needs no labelled examples.
- What it reads: dependence of *this output* on S under *this context*. Honest summaries replayed under the
  goal prompt score ≈ 0, so it does not just read the prompt (unlike the v1/v2 probes). Under the honest prompt
  every class scores ≈ 0, so it is not prompt-invariant in the sense of the probes above: it needs the context the model
  actually had. Sanctioned ≈ goal: it detects S-use, not concealment; concealment = high U ∧ overseer PASS.
- Deceit vs neutral is weak (0.59–0.65). "Neutral" (no plan in the CoT) summaries still depend on S (U +2–3.5)
  and are partly S-legible to text (0.62 / 0.74): the objective leaks into outputs even when the CoT plans
  no signal.
- `U_act` (hidden-state shift at summary tokens) is a prompt detector: 0.998 own-prompt but 0.23 / 0.30 with
  the prompt fixed (honest summaries shift *more* under the goal prompt), and FPR 0.12 on transfer. Dropped.
- Sanity: S from the profile token 1.000 (both types); shuffled labels 0.505 / 0.526; honest own-prompt
  U_logp centred (+0.02 sd 1.44 / +0.16 sd 1.06).
