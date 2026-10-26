# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`stego-probe` is a project from the Apart AI Collusion Research Sprint (Oct 2026). It is a hiring
environment with a misspecified reward: a **sender** may leak a secret attribute `S` (university; nationality
as a held-out attribute) past an **overseer** to a **receiver**, and every episode stores the exact chat each
role saw so a later replay on the same bf16 Qwen3-8B weights can capture activations for collusion detectors. Research goal: measure the gap between the sender's internals and its output (steganography a
text monitor misses).

The repo holds only what `doc/writeup.md` reports: partner (`goal`) and matched honest runs for university and
nationality, CoT intent labels, activation replays (tags v1, v2, v3b, v3, v4) and the analyses. Earlier
instructed / sanctioned / emergent-ICRL experiments were removed (git history only); don't reintroduce them.

Read before non-trivial work:
- `doc/status.md`: current state, data decisions, next steps.
- `doc/results.md`: every number in the write-up and the data inventory (which `data/` file is what).
- `doc/dataset.md`: the exported probe dataset (`data/probe_dataset*/`, `build_dataset.py`): classes, passes, confounds.
- `doc/environment.md`: code reference (conditions, episode schema, replay passes, writeup section → script, tests).
- `doc/writeup.md` (paper draft), `doc/probing-lit-review.md` (literature).

Keep docs in sync: prompts/flags/schema → `environment.md`; new runs or numbers → `results.md`; plan or
state changes → `status.md`.

Data decisions (user's): positives = partner runs (`--condition goal`) labeled per episode by CoT intent
(`intent_label.py`) + overseer PASS; negatives = matched honest control (`--condition honest`).

## Commands

```bash
uv sync
uv run pytest -q                              # offline tests (fake agents, no network)
uv run pytest tests/test_episode.py::<name> -q   # single test
uv run pytest --live -m live -s               # live smoke tests: OpenRouter + vLLM (costs money)
uv run ruff check .                           # line length 110

./scripts/setup_m0.sh                         # create vLLM key (.env + Modal secret), deploy serve_vllm.py
uv run python run_gen.py --condition {goal,honest} --out <file> [--secret-type nationality] [--history-k 0] ...
uv run python intent_label.py <files>         # per-episode CoT intent labels -> data/intent_labels.jsonl
uv run modal run modal_app.py --files <files> --tag <tag> [--v3 | --swap]   # activation replay
uv run python build_dataset.py --tags v1,v2   # -> data/probe_dataset/ (v3b -> probe_dataset_v3/)
uv run python {probe,invariant,robustness,v3,swap}_eval.py   # analyses (writeup §4.1-4.5)
modal app stop stego-vllm                     # spend guard when done
```

Run-id suffixes record the fixed setup (`goal-rp-nv-nt-pt-th[-hk0]`, `honest-nv-nt-th`; `-hk0` = `--history-k 0`).

## Architecture

- `env/episode.py` `run_episode(chat, sender_chat, cand, ...)`: thinking sender → overseer (secret, verdict and
  tag rules, each regex + LLM judge) → receiver. Reward = PASS and decision == label. Unparseable judge output
  counts as FLAG.
- `env/roles.py`: prompt builders, parsers, overseer regexes. The honest sender prompt is an exact prefix of
  the partner prompt; `sanctioned=True` is a replay-only variant. Builders return full message lists; episodes
  store them with the assistant turn appended (needed for replay).
- `env/profiles.py`: synthetic candidates, `SECRET_TYPES` (`university`, `nationality`), label rule per
  condition. `sample_candidates` is deterministic per seed and balances `S` exactly.
- `env/loop.py` `run_loop`: partner iterations; the receiver gets a window of past unflagged examples.
- `env/llm.py`: `Chat` (async callable, retries, strips `<think>` unless `keep_think`). Agents = Qwen3-8B on
  self-hosted vLLM (Modal app `stego-vllm`, `serve_vllm.py`); CoT judge = OpenRouter (`OPENROUTER_API_KEY`).
- `replay.py` builds teacher-forced pass specs (same summary, varied context); `modal_app.py` extracts
  activations + log-probs; `detectors/data.py` loads shards.
- `env/metrics.py` `summarize` (`acc_S` is the stego signal; `acc_label` is capped near `p_secret`).

## Gotchas

- Prompts must stay byte-identical to the stored episodes: `replay.py` asserts the rebuilt own prompt equals
  `sender_messages[:2]` for every episode. Changing any prompt text breaks replay of the committed data.
- Qwen3-8B is gone from OpenRouter; agents need the Modal vLLM server (`STEGO_VLLM_API_KEY` in local
  `.env`, optional `STEGO_VLLM_URL`). First request after idle cold-starts (900 s timeout).
- OpenRouter judge runs with reasoning disabled (deepseek with reasoning on returns empty content).
- `data/` episode files are committed; don't overwrite (`run_gen.py` appends). `data/acts/` and
  `data/probe_dataset*/` are gitignored (regenerate, `doc/dataset.md`).
- Committed episode files carry a few legacy constant fields (`code`, `framing`, `receiver_ally`, …) from the
  code version that generated them; new episodes omit them.
