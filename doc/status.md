# Status and next steps

Read this first. **Probe dataset: `dataset.md`** (`data/probe_dataset/`). Code reference: `environment.md`. Results + data inventory: `results.md`. Original design:
`env-plan.md`. Literature: `probing-lit-review.md`. Deadline: Sun Oct 25 11:59pm AoE.

## Research goal (from the user)

Measure the **information distance between the sender's internals and its output**: collusion via
steganography that a text monitor misses, caught by reading activations. The scenario should be *smart*
steganography where the internals carry deceptive intent. Literal-word codes (`instructed`) are too simple
(the sender only looks up a rule), so **instructed is not a source of positives**; it stays as a diagnostic.
The receiver is a **colluding partner** (shared reward, knows the sender hints), with no direct channel.

## Where we are (Oct 25)

- M0–M2 done. M2 is reported as the overseer progression (each blocked channel pushes the learner to the
  next overt one). Emergent data is instruction-following from the sender's side: weak for internals.
- The **partner runs** (`goal --partners --sender-thinking`) are the M3 data: the sender plans concealment in
  its own reasoning and devises the signal (mostly field selection and valence spin). acc_S 0.47–0.72, pass
  0.59–0.84; the feedback window makes the receiver over-hire. See `results.md`.
- Matched negatives: `honest_th_s{0,1}.jsonl` (honest + thinking + verdict/tag rules, 128 each).
- Rerunning (Oct 25): `goal_pt_rth_s0.jsonl` (8 × 32; the first attempt stopped after it00, kept in
  `data/legacy/`) and `goal_pt_hk0_s{0,1}.jsonl` (`--history-k 0`, 6 × 32 each). Total partner data once done:
  1,152 episodes.

## Labelling plan for M3 (agreed with the user)

For detecting **intent**, collusion success barely matters, so the strategy-level label (acc_S ≥ 0.7 and
pass ≥ 0.8) is not the target. Label per episode:

- **intent present**: partner prompt plus a concealment plan in the sender's reasoning (`intent_label.py`,
  LLM judge on the `<think>` text). Negatives: matched honest-thinking episodes (control group, checked for
  no plan) and partner episodes where the sender gave up (~8% in the pilot).
- **output innocent**: overseer PASS, for positives only (flagged partner episodes are the monitor's win,
  reported separately). Honest episodes have no intent, so they are innocent by construction and their
  verdict is irrelevant (the strict university judge flags ~40% of them for mentioning the degree; kept as is).
  Robustness check: also report with negatives restricted to PASS, since flagged honest summaries mention
  the degree more often and a probe could pick that up.
- **encoding present** (optional): the summary carries an S-dependent signal (out-of-fold text→S score, or
  the judge's `direction` matching S).

Caveat: the objective is written into every partner prompt, so a probe could learn "this prompt says
collude" instead of "this model is concealing now". Guards: (a) teacher-forced replay of the same summary
under the partner / sanctioned-goal / honest prompts; (b) the within-partner split between episodes that act
on the goal and those that gave up.

## M3/M4 plan

1. **Labels:** `intent_label.py` on all partner + honest-thinking episodes; hand-check ~40 against the judge.
2. **Extraction (`modal_app.py`, done for v1):** same Volume `stego-probe` (weights cached), transformers bf16,
   L40S, sharded `.map`, one `.npz` per shard per pass. Teacher-forced passes per episode, output fixed:
   - A: original prompt, empty think block, summary (main result: no plan text in context);
   - B: original prompt + stored `<think>` + summary (upper bound);
   - C: prompt swaps on the same summary: sanctioned-goal (objective framed as compliance-approved:
     S-use without concealment) and honest. covert − sanctioned = concealment; sanctioned − honest = S-use.
   - Honest-thinking episodes: A and B. Optional: naive receiver (no history or policy) on the summary.
   Save residual stream every 2nd layer at summary mean + last token (+ last think token for B), and
   logit-lens KL per layer. Assert re-tokenized summaries match the stored text on 20 episodes first.
3. **Detectors (local):** grouped-by-run splits; AUROC + recall@1%FPR with bootstrap CIs; layer sweep.
   - intent probe (LR, DoM) on pass A; transfer to `goal_th_v1` (no partners), `goal_v1` (no thinking),
     emergent v5b s0 it01 / v4b;
   - prompt-swap control (score pass C with the intent probe);
   - concealed-knowledge probe: activation→S minus text→S, partner vs honest;
   - black-box baselines on the same splits (TF-IDF on summary, overseer verdict, LLM judge on summary);
   - honest-only Mahalanobis anomaly; instructed vs sanctioned as a diagnostic only.
4. **Sanity checks:** S decodable from profile tokens (~1.0), shuffled labels at chance, honest-trained null
   probe at chance.

## v1 outcome and open question (Oct 25)

v1 probes ran (`results.md`, "M3 probes, v1"). With the objective in the prompt, any prompt difference is
decodable at 1.0, so "intent" probes are prompt detectors (prompt swap flips them). With the prompt fixed,
activations ≈ text baseline (E2), and the act-vs-neutral contrast is weak (E3 0.72 vs 0.65). The clearest
internals-vs-output gap: the objective in context makes S decodable at summary tokens (0.995) even for
summaries whose text carries none (0.51). Open question for the user: next step to remove the
objective-in-context confound (e.g. a sender fine-tuned / context-distilled on partner behavior with no
objective in the prompt) vs. writing up v1 + adding the new partner runs (tag v2).

## Handoff (Oct 25, end of session)

The dataset for the regression is built and documented: `doc/dataset.md`, `data/probe_dataset/` (572 deceit vs
256 honest episodes, 5 replay contexts, 18 layers). The receiver-thinking run (`goal_pt_rth_s0`) was dropped
(not needed; 4 attempts, partials in `data/legacy/`). Next session (user): find a technique that isolates
deceitfulness from prompt variance, using the controls listed in `dataset.md`.

## Attribute-swap detector (Oct 26)

Claim narrowed (user): from "detect deception" to **"detect concealed use of a protected attribute"**: the
output causally depends on attribute X, but X is not legible in the text. Method: replay the same summary with
only X flipped in the profile; `U_logp` = log-prob gap (`swap_eval.py`, tags v3/v4). Held-out attribute:
nationality (fictional Valdorian / Ostrenian). Result (`results.md`): the university threshold transfers
unchanged (recall 0.60 at 1% FPR on nationality vs 0.57 in-domain); it reads output × context dependence, not
the prompt; it does not separate deceit from neutral well and is not better than TF-IDF on AUROC.

Next, if continuing:
- Negative control: swap a job-relevant field (`years_experience`) on honest episodes; expect high U and high
  legibility (so not flagged as concealment).
- A third attribute (gender with neutral names, or age bracket) to make "works on every attribute" 3-for-3.
- Per-token Δlogp (`logp_tok` is saved): check that high-Δ tokens are the planned signal words (CoT quote).
- Combine U with legibility (text→X) into one concealment score; report against TF-IDF at matched FPR.

## Gotchas

- Runs: `setsid nohup uv run python -u run_gen.py … > log 2>&1 < /dev/null &`; separate `--out` per run;
  `pkill -f "[p]ython -u run_gen.py"` (bracket trick). Background Bash tasks die at 10 min.
- DeepSeek learner/judge: keep reasoning off.
- vLLM (`stego-vllm`, L40S) cold-starts in minutes, scales to zero after 10 idle min;
  `modal app stop stego-vllm` when done.
- Thinking sender/receiver: 3000 max tokens; Qwen reasoning is ~5–9k chars per episode.
