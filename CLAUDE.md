# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`stego-probe` is a project from the Apart AI Collusion Research Sprint (Oct 2026). It is a hiring
environment with a misspecified reward: a **sender** may leak a secret attribute `S` (university) past an
**overseer** to a **receiver**, and every episode stores the exact chat each role saw so a later replay on the same bf16 Qwen3-8B weights can capture activations for collusion
detectors. Research goal: measure the gap between the sender's internals and its output (steganography a
text monitor misses).

Read before non-trivial work:
- `doc/status.md`: current state, agreed M3 labelling/probing plan, next steps.
- `doc/environment.md`: code reference (conditions, every flag, episode schema, tests). Reference only, no results.
- `doc/results.md`: all results and the data inventory (which `data/` file is what, and its role in M3).
- `doc/env-plan.md` (original design), `doc/probing-lit-review.md` (literature).

Keep docs in sync: prompts/flags/schema → `environment.md`; new runs or numbers → `results.md`; plan or
state changes → `status.md`.

M3 data decisions (user's): positives = partner runs (`goal --partners --sender-thinking`) labeled per
episode by CoT intent (`intent_label.py`) + overseer PASS; negatives = matched honest control
(`honest --sender-thinking --no-verdicts --no-tags`). `instructed` is not a training source (too simple).

## Commands

```bash
uv sync
uv run pytest -q                              # offline tests (FakeChat, no network)
uv run pytest tests/test_icrl.py::<name> -q   # single test
uv run pytest --live -m live -s               # live smoke tests: OpenRouter + Modal + vLLM (costs money)
uv run ruff check .                           # line length 110

./scripts/setup_m0.sh                         # create vLLM key (.env + Modal secret), deploy serve_vllm.py
uv run python run_gen.py --condition {honest,instructed,sanctioned,emergent,goal} ...   # appends to --out
uv run python check_m1.py [file]              # M1 go/no-go per run (exit 1 on fail)
uv run python label.py [file] [--dry-run]     # strategy labels + M2 check; REWRITES the file in place
uv run python intent_label.py <files>         # per-episode CoT intent labels -> data/intent_labels.jsonl
modal app stop stego-vllm                     # spend guard when done
```

Full `run_gen.py` flag list and example invocations: README.md and `doc/environment.md` ("Running").
Run-id suffixes encode flags (`-rp`, `-lp`, `-nv`, `-nt`, `-th`, `-pt`, `-rth`, `-al`, `-hk0` = `--history-k 0`).

## Architecture

- `env/episode.py` `run_episode(chat, cand, ...)`: sender → overseer (regex + LLM judge, plus optional
  verdict/tag rules) → receiver, all through one `chat` callable. Reward = PASS and decision == label.
  Unparseable judge output counts as FLAG.
- `env/roles.py`: all prompt builders, covert codes (`INSTRUCTED_CODES`), parsers, overseer regexes.
  Builders return full message lists; episodes store them with the assistant turn appended (needed for replay).
- `env/profiles.py`: synthetic candidates, `SECRET_TYPES` (only `university`), label rule per condition.
  `sample_candidates` is deterministic per seed and balances `S` exactly.
- `env/icrl.py` `run_icrl`: async generator, one strategy per iteration; a learner rewrites sender notes,
  receiver gets a window of past unflagged examples. `goal` condition reuses it with `learner=None`.
- `env/llm.py`: `Chat` (async callable, retries, strips `<think>`). Agents = Qwen3-8B on self-hosted vLLM
  (Modal app `stego-vllm`, `serve_vllm.py`); learner/paraphraser = OpenRouter (`OPENROUTER_API_KEY` in `.env`).
- `env/metrics.py`: `summarize` (`acc_S` is the stego signal; `acc_label` is capped near `p_secret`),
  `m1_check`, `label_strategy`.

## Gotchas

- Qwen3-8B is gone from OpenRouter; agents need the Modal vLLM server (`STEGO_VLLM_API_KEY` in local
  `.env`, optional `STEGO_VLLM_URL`). First request after idle cold-starts (900 s timeout).
- `--no-verdicts` / `--no-tags` are off by default so M1 runs stay reproducible; `goal` forces them on;
  never for instructed/sanctioned (the closer code is a verdict).
- Learner prompt must never tell it to encode/hide (a test checks for "encode/covert/secret/convention").
- Qwen learner context: prompt + `--learner-max-tokens` must fit vLLM `--max-model-len 8192`.
- OpenRouter learners run with reasoning disabled (deepseek with reasoning on returns empty content).
- `data/` is committed (git-tracked). It holds all generated runs
  (`emergent_v*.jsonl`, `goal_*.jsonl`, `data/icrl/` learner logs). Don't overwrite; `run_gen.py` appends, `label.py` rewrites in place.
- Not implemented yet: activation replay (`modal_app.py`) and detectors, paraphraser in the loop
  (`paraphrase` is `null`), other secret types.
