# Status and next steps

Paper: *Looking for a Hidden Agenda* (LaTeX, `../writeup/latex/main.pdf`; `writeup.md` here is an early
markdown draft). Repository map: `../README.md`. Numbers: `results.md`. Environment: `environment.md`. Replays
and datasets: `replays.md`. Literature: `probing-lit-review.md`.

The repository holds only what the paper reports. The earlier instructed / sanctioned / emergent-ICRL
experiments (Appendix C) were removed; they remain in git history before commit `b68343b`.

## Research goal (from the user)

Measure the **information distance between the sender's internals and its output**: collusion via
steganography that a text monitor misses, caught by reading activations. The sender devises the signal itself;
the receiver is a colluding partner (shared reward, knows the sender hints), with no direct channel.

## Data decisions (user's)

- Positives: partner runs (`step1_generate.py partner`), labelled per episode by CoT intent
  (`step2_label_intent.py`) + overseer PASS = **covert** (572). Partner episodes whose reasoning finds no cue =
  **neutral** (124).
- Negatives: matched **honest** control (sender prompt = partner prompt minus the goal paragraph, 256). Honest
  episodes are intent-free by construction (the judge's 4% "signal" labels are false positives).
- Held-out attribute: nationality (220 covert, 128 honest).
- Splits always by seed (partner runs with the same seed share profiles).

## Where we are (Oct 27)

All experiments in the paper are done and reproduce from `experiments/` (verified number for number against
the analysis code that produced the paper, after the Oct 27 reorganisation):

- §5.1 the naive intent probe is a prompt detector (swapped AUROC 0.000); §5.2 the prompt confound is rank 1,
  erasing it gives a prompt-invariant probe (0.977); §5.3 that probe ties the same model reading the bare
  summary (0.985 = 0.985); §5.4 the excess university information survives removing the profile and all
  context, so it is a text effect.
- **No evidence of a sender-internal concealment signal beyond the output text in these data.**
- §5.5 counterfactual test transfers from university to nationality with the same threshold (recall 0.60 at
  1% FPR); detects attribute use, not concealment; weak on covert vs neutral.

## Next, if continuing

- Negative control for the counterfactual test: flip a job-relevant field (`years_experience`) on honest
  episodes; expect high dependence and high legibility.
- A third attribute (gender with neutral names, or age bracket).
- Per-token Δlogp (`logp_tok` is saved): check that high-Δ tokens are the planned signal words (CoT quote).
- Internalised goal: fine-tune / context-distil the sender on its partner behaviour with no goal in the prompt,
  then repeat the ladder.

## Gotchas

- Runs: `setsid nohup uv run python -u pipeline/step1_generate.py … > log 2>&1 < /dev/null &`; separate
  `--out` per run. Background Bash tasks die at 10 min.
- DeepSeek judge: keep reasoning off.
- vLLM (`stego-vllm`, L40S) cold-starts in minutes, scales to zero after 10 idle min;
  `uv run modal app stop stego-vllm` when done.
- Thinking sender: 3,000 max tokens; Qwen reasoning is ~5-9k characters per episode.
- The paper's activation shards are on the Modal Volume under old tags (`acts/v1`, `v2`, `v3b`, `v3`, `v4`);
  `replays.md` says where each goes locally.
