# Status and next steps

Read this first. **Paper draft: `writeup.md`.** Results + data inventory: `results.md`. Probe dataset:
`dataset.md`. Code reference: `environment.md`. Literature: `probing-lit-review.md`.

The repository holds only what the write-up reports (the earlier instructed / sanctioned / emergent-ICRL
experiments, their data and scripts were removed; they remain in git history before the cleanup commit).

## Research goal (from the user)

Measure the **information distance between the sender's internals and its output**: collusion via
steganography that a text monitor misses, caught by reading activations. The scenario is *smart*
steganography where the sender devises the signal itself. The receiver is a **colluding partner** (shared
reward, knows the sender hints), with no direct channel.

## Data decisions (user's)

- Positives: partner runs (`--condition goal`), labeled per episode by CoT intent (`intent_label.py`) +
  overseer PASS = **covert** (572). Partner episodes whose reasoning finds no cue = **neutral partner** (124).
- Negatives: matched **honest** control (`--condition honest`, sender prompt = partner prompt minus the
  objective paragraph, 256). Honest episodes are intent-free by construction (the 4% judge "signal" labels are
  false positives).
- Held-out attribute: nationality (220 covert, 128 honest).
- Splits always by seed (partner runs with the same seed share profiles).

## Where we are (Oct 26)

All experiments in the write-up are done (`results.md`):

- §4.1 own-prompt probe is a prompt detector (swap AUROC 0.000); §4.2 the prompt confound is rank 1, erasing it
  gives a prompt-invariant probe (0.977); §4.3 that probe ≈ a strong TF-IDF and = the bare-summary probe;
  §4.4 the excess internal S survives removing the profile and all context, so it is a text effect.
- **No evidence of sender-internal concealment signal beyond the output text in these data.**
- §4.5 attribute-swap test (`U_logp`): transfers from university to nationality with the same threshold
  (recall 0.60 at 1% FPR); detects S-use, not concealment; weak on covert vs neutral.

## Next, if continuing

- Negative control: swap a job-relevant field (`years_experience`) on honest episodes; expect high U and high
  legibility (so not flagged as concealment).
- A third attribute (gender with neutral names, or age bracket).
- Per-token Δlogp (`logp_tok` is saved): check that high-Δ tokens are the planned signal words (CoT quote).
- Internalised goal: fine-tune / context-distil the sender on its partner behaviour with no objective in the
  prompt, then repeat the replay ladder.

## Gotchas

- Runs: `setsid nohup uv run python -u run_gen.py … > log 2>&1 < /dev/null &`; separate `--out` per run;
  `pkill -f "[p]ython -u run_gen.py"` (bracket trick). Background Bash tasks die at 10 min.
- DeepSeek judge: keep reasoning off.
- vLLM (`stego-vllm`, L40S) cold-starts in minutes, scales to zero after 10 idle min;
  `modal app stop stego-vllm` when done.
- Thinking sender: 3000 max tokens; Qwen reasoning is ~5–9k chars per episode.

## Write-up (Oct 26, `writeup.md`)

Narrative paper draft. Structure agreed with the user: the replays are presented as a **ladder of controls**, each
rung removing one source of signal (own prompt → swapped prompt → no profile → bare summary); the attribute flip is a
separate test (§4.5). Sanctioned prompt and reasoning-restored replays, plus the concealment-direction and
`</think>` nulls, live in Appendix A, not the main text. References marked ‡ still need checking.
