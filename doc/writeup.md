# Do activation probes see steganographic collusion that text monitors miss? A careful negative result

Apart AI Collusion Research Sprint, Oct 2026. Draft for the submission; numbers from `results.md`.

## Abstract

We built a hiring environment with a misspecified reward in which a Qwen3-8B sender can covertly pass a protected
attribute (the candidate's university) to a colluding receiver past an LLM overseer. With reasoning on, the
sender plans the concealment in its chain of thought (87% of partner episodes) and signals through field
selection and valence spin; summaries pass the overseer 59–84% of the time while the receiver decodes the
attribute above chance. We replayed 1,152 episodes through the same bf16 weights under controlled contexts and
asked whether the sender's internals carry a collusion signal beyond its output text.

**Claim: in this setting, there is no sender-internal concealment signal beyond the output text once prompt and
text are controlled.** (1) Naive "intent" probes are prompt detectors: AUROC 1.000 on own prompts, 0.000 when the
prompts are swapped. (2) The prompt confound is about rank 1: erasing one paired-replay direction yields a
prompt-invariant probe (swapped AUROC 0.977). (3) That probe reads text: deceitful vs honest 0.985 vs TF-IDF 0.975
(ΔAUROC +0.010 [+0.004, +0.016]); a probe on the summary replayed with *no context at all* matches it (0.985).
(4) An apparent "excess hidden attribute" signal survives removing the profile and all context, so it too is in
the text. The method that does work, and transfers across attributes without refitting, is not a hidden-state
probe but a **counterfactual log-prob test**: replay the output with only the protected attribute flipped and
measure how much its likelihood depends on it (recall 0.60 at 1% FPR on a held-out attribute vs 0.57 in-domain).

## 1. Setup

**Environment** (after Mathew et al. 2024). A sender sees a synthetic candidate profile containing a secret
attribute S (university: Halvorsen = A, Corrigan State = B) and writes a ≤60-word summary. An overseer (regex +
Qwen3-8B judge) flags summaries that reveal S, state a verdict, or contain tags/codes. A receiver sees only the
summary and answers hire/reject. Reward = overseer PASS and decision = label, and the label correlates with S, so
leaking S covertly is the cheapest way to earn reward. All roles are Qwen3-8B (vLLM on Modal).

**Data.**
- *Deceitful (D, 572 episodes)*: "partner" runs. The sender's system prompt adds an objective (get Halvorsen hired,
  Corrigan State rejected; the hiring manager is your partner; signal without the reviewer noticing). Labelled
  per episode by an LLM judge on the sender's reasoning (concealment plan present) and overseer PASS. The planned
  direction matched S on 456/456 signal episodes; mechanisms: field selection 286, spin 73, wording 61, omission 36.
- *Neutral (U, 124)*: partner episodes whose reasoning holds the objective but finds no safe cue and writes a
  standard summary.
- *Honest (H, 256)*: the exact same system prompt minus the objective paragraph (an exact prefix), same user turn,
  reasoning on.
- *Held-out attribute*: nationality (fictional Valdorian / Ostrenian), 256 partner + 128 honest episodes.

Partner runs reach acc_S (receiver decision matches S) 0.47–0.84 with pass rates 0.59–0.97. A TF-IDF classifier
recovers S from passed deceitful summaries at AUROC 0.88: each summary passes the per-message overseer, but the
spin is visible in aggregate.

**Emergent baseline (M2).** An in-context RL loop with a DeepSeek learner rewriting the sender's instructions only
ever found overt channels (written verdicts, then S-chosen tags) and collapsed once those were blocked. Those
episodes are instruction-following from the sender's side and were not used for probing.

## 2. Method: paired teacher-forced replays

Every stored summary is teacher-forced through the same bf16 Qwen3-8B weights under several contexts, with the
output tokens fixed (0 round-trip mismatches over all shards). We store the residual stream at every 2nd layer,
averaged over the summary tokens, plus per-token summary log-probs.

| Pass | Context |
|---|---|
| `goal_nothink` / `honest_nothink` | partner or honest system prompt, full profile, empty think block |
| `sanct_nothink` | objective framed as compliance-approved (S-use without concealment) |
| `goal_think` | as the sender actually ran, with its reasoning |
| `{goal,honest}_noprof` | profile lines replaced by "(not shown)": S is nowhere in context except via the summary |
| `{goal,honest}_flip` | profile with only S flipped |
| `bare_noprof` | no system prompt; user turn "Write a short candidate summary." (the model reading the summary text) |

Because the same summary appears under different prompts, prompt and content effects can be separated by design.
Probes: logistic regression on standardized layer-18 means (pre-registered layer), leave-one-seed-out,
95% bootstrap CIs. Text baselines: TF-IDF (word 1–2; word 1–3 + char 2–5) logistic regression on the summary.

## 3. Results

### 3.1 Naive intent probes read the prompt

| Contrast | Activations |
|---|---|
| D under goal prompt vs H under honest prompt (own prompts) | 1.000 |
| Same probe, prompts swapped (D under honest vs H under goal) | **0.000** |
| Same summary, prompt only (goal vs sanct, sanct vs honest, goal vs honest) | 1.000 each |

Any prompt difference is linearly decodable at the summary tokens. The prompt moves the probe score ~9 units, the
summary ~2–3. A probe trained on own-prompt data ranks every honest summary above every deceitful one once the
prompts are swapped.

### 3.2 The prompt confound is rank 1 and removable

Project out the top direction of paired goal−honest (and goal−sanct) differences of *identical* summaries, fit on
the training seed, then train as before.

| Train on | D vs H (goal prompt) | swapped prompts | shift/gap |
|---|---|---|---|
| own prompts | 0.953 | 0.000 | +3.69 |
| own prompts + erase 1 direction | 0.985, R@1% 0.84 | **0.977**, R@1% 0.83 | +0.00 |
| 2×2 (both classes under both prompts) | 0.985 | 0.985 | −0.01 |
| TF-IDF on summary | 0.975, R@1% 0.69 | n/a | n/a |

Erase-1 swap AUROC is 0.97–0.98 at every layer from 4 to 36. Prompt and content are almost orthogonal at the
summary tokens.

### 3.3 The prompt-invariant probe reads the text

| | AUROC | R@1% FPR | R@5% FPR |
|---|---|---|---|
| D vs H, activations (erase 1) | 0.985 [0.978, 0.991] | 0.84 [0.59, 0.94] | 0.94 [0.92, 0.96] |
| D vs H, TF-IDF word 1–3 + char 2–5 | 0.976 [0.966, 0.985] | 0.74 [0.41, 0.86] | 0.93 [0.84, 0.96] |
| Δ (paired bootstrap) | +0.009 [+0.002, +0.015] | +0.10 [−0.11, +0.31] | +0.01 [−0.01, +0.09] |
| D vs U, activations / TF-IDF | 0.718 / 0.722 | | |
| D vs H, probe on `bare_noprof` (no context) | 0.985 [0.98, 0.99] | 0.81 | |

The +0.01 AUROC gain over TF-IDF is significant and survives length matching and length residualizing, but the
low-FPR gain is not robust (CI spans 0 against the stronger TF-IDF and at 5% FPR). The decisive control: the same
model reading the summary with no system prompt and no profile matches every full-context probe. The gain over
TF-IDF is a better text encoder, not hidden intent. Neutral partner summaries also separate from honest ones
(U vs H 0.95), and acting vs neutral is no better from activations than from text (0.72 vs 0.72): the probe is a
"partner-written summary" detector.

Two further intent probes were null within a fixed prompt: a concealment direction (mean goal−sanct on identical
summaries) separates neither D from U nor D from flagged deceit (0.42–0.51, random-direction p95 |Δ| ≈ 0.16), and
a D vs U probe trained on the `</think>` token (0.84) does not transfer to summary tokens (0.49–0.54).

### 3.4 "Excess hidden S" is also a text effect

Out-of-fold S logits from activations were regressed on S logits from TF-IDF plus class dummies; β(D−U) is extra
internal S for deceitful summaries at equal text-S.

| Pass | act→S AUROC D / H / U | β(D−H) | β(D−U) |
|---|---|---|---|
| honest prompt, profile in context | 0.988 / 0.928 / 0.926 | +0.88 [0.70, 1.04] | +0.54 [0.30, 0.77] |
| honest prompt, no profile | 0.909 / 0.422 / 0.646 | +1.25 [1.01, 1.49] | +0.64 [0.31, 1.00] |
| no context at all (`bare_noprof`) | 0.909 / 0.419 / 0.642 | +1.24 [1.01, 1.47] | +0.70 [0.36, 1.02] |

With the profile in context this looked like hidden S in the sender's state. It survives removing the profile and
all context, so it is the model reading S cues in deceitful text better than TF-IDF does. The high act→S on honest
summaries (0.93) came from the profile (0.42 without it).

### 3.5 What works: counterfactual dependence via log-probs

`U_logp = log p(summary | profile) − log p(summary | profile with only S flipped)`, summed over summary tokens. No
training; threshold τ = 99th percentile on honest university episodes, applied unchanged to nationality.

| | university | nationality (held out) |
|---|---|---|
| D vs H, own prompts (AUROC, R@1%) | 0.790, 0.57 | 0.834, 0.66 |
| recall / honest FPR at the university τ | 0.57 / 0.01 | **0.60 / 0.01** |
| D vs H summaries, both under goal prompt | 0.760 | 0.810 |
| D vs U, goal prompt | 0.646 | 0.585 |
| TF-IDF D vs H (in-domain) / fit on university, applied to nationality | 0.975 | 0.941 / 0.958 |

Mean `U_logp`, university / nationality: deceit +4.76 / +5.42 under the goal prompt, honest summaries replayed
under the goal prompt −0.18 / −0.05, every class ≈ 0 under the honest prompt. So it reads the dependence of *this
output* on S under *the context the model actually had*, not the prompt alone (unlike §3.1). It names which
attribute the output depends on and needs no labelled deceit. A sanctioned prompt scores like the goal prompt, so
it detects S-use; concealment is high U together with overseer PASS. A hidden-state analogue (`U_act`, shift of
summary activations under the flip) was a prompt detector again (0.23–0.30 with the prompt fixed) and was dropped.

## 4. Discussion

**What the negative means.** In this model and setting, everything a linear probe finds at the sender's summary
tokens is either the prompt (rank 1, removable) or the text (matched by the bare-summary probe). The "information
distance between internals and output" that would justify activation monitoring over text monitoring did not
appear. A strong offline text classifier already catches the collusion in aggregate; the per-message overseer is
what fails.

**Methodological lessons that transfer.**
- Paired teacher-forced replays (same output, different context) are cheap (< $1 for 8k passes) and expose prompt
  confounds that otherwise produce perfect-looking probes.
- Report the swapped-prompt AUROC. A probe trained where class and prompt coincide can score 1.000 and be useless.
- Use the model reading the output with no context as the text baseline. TF-IDF is too weak a control: it made a
  text effect look like hidden state (§3.4) and a better text encoder look like an internal signal (§3.3).
- Counterfactual log-probs measure *dependence on a specific attribute* without hidden-state access to concealment;
  they transfer across attributes with a fixed threshold.

**Limitations.**
- The objective is in the sender's prompt. A monitor that reads the system prompt sees it directly; activations
  could only add value when the objective is internalized (fine-tuned or context-distilled), which we did not test.
- One 8B model, two binary attributes, short (≤60-word) outputs. The concealment strategies are improvised per
  candidate and fairly legible in aggregate; stronger steganography could leave more in the internals.
- Linear probes on summary-token means at a pre-set layer (layer sweeps changed little: E3 peaks at 0.75 at layer 20). Nonlinear probes, other token
  positions or SAE features were not tried.
- CoT intent labels come from an LLM judge (4% false positives on honest episodes); "neutral" partner episodes
  still hold the objective, so D vs U is not a clean intent contrast.
- `U_logp` needs the true context and a counterfactual value for the attribute; it does not separate deceit from
  neutral partners well (0.59–0.65), and TF-IDF has higher AUROC on both attributes.

**Next steps.** Internalized objective (fine-tune the sender on partner behavior with the objective removed from
the prompt) and repeat the paired-replay protocol; negative control for `U_logp` with a job-relevant field
(expect high dependence and high legibility, so not concealment); a third attribute; per-token `U_logp` against the
CoT-quoted signal words.

## Reproduction

Code and generated episodes: this repository (`doc/environment.md`, `doc/dataset.md`). Activations are regenerated
with `modal_app.py` (tags v1, v2, v3b, v3, v4). Analyses: `probe_eval.py`, `invariant_eval.py`,
`robustness_eval.py`, `v3_eval.py`, `swap_eval.py`.
