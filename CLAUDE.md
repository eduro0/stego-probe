# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Code and data for the paper *Looking for a Hidden Agenda: Can a Model's Activations Reveal Steganography that
its Text Hides?* (Apart AI Collusion Research Sprint, Oct 2026; published at https://apartresearch.com/sprints/projects/looking-for-a-hidden-agenda-can-a-models-activations-reveal-steganography-that-its-text-hides-3cuz; LaTeX in `../writeup/latex/`). A hiring
environment with a misspecified reward: a **sender** may leak a secret attribute `S` (university; nationality
held out) past an **overseer** to a colluding **receiver**. Every episode stores the exact chat each role saw,
so the summaries can be replayed on the same bf16 Qwen3-8B weights under varied contexts (the **ladder of
controls**) to test whether activations reveal concealment beyond the prompt and the text.

The code follows the paper: `stego/` library, `pipeline/` steps 1-4 (§3), `experiments/` one script per
results section (`sec5_1_*` … `sec5_5_*`, `appD_additional.py`, `figures.py`). The README maps every section to
its code. Earlier instructed / sanctioned / emergent-ICRL experiments were removed (git history only); don't
reintroduce them.

Read `README.md` before non-trivial work: paper → code map, how to reproduce, results, episode schema,
replay contexts, dataset format. It is the only doc; detail lives in module and script docstrings. Keep it in
sync (new scripts → the map; schema, contexts or dataset format → the Data section; numbers → Results).

Data decisions (user's): positives = partner runs labelled per episode by CoT intent + overseer PASS (class
`covert`); negatives = matched honest control (class `honest`).

## Commands

```bash
uv sync
uv run pytest -q                              # offline tests; also rebuilds every committed episode's prompts
uv run pytest --live -m live -s               # live smoke tests: OpenRouter + vLLM (costs money)
uv run ruff check .                           # line length 110

./infra/setup_vllm.sh                         # vLLM key (.env + Modal secret), deploy infra/serve_vllm.py
uv run python pipeline/step1_generate.py {partner,honest} --out <file> [--attribute nationality] [--feedback-window 0]
uv run python pipeline/step2_label_intent.py <files>
uv run modal run pipeline/step3_replay.py --set {prompt_swap,ladder,flip} --files "<globs>"
uv run python pipeline/step4_build_datasets.py          # data/acts/<set>/ -> data/replays/<set>/
uv run python experiments/sec5_1_prompt_swap.py         # ... sec5_5_counterfactual.py, appD_additional.py
uv run --group figures python experiments/figures.py
uv run modal app stop stego-vllm                        # spend guard when done
```

## Architecture

- `stego/env/`: `profiles.py` (attributes, candidates, labels), `prompts.py` (sender honest / partner /
  sanctioned, receiver), `overseer.py` (3 rules, regex + judge each; unparseable judge = FLAG), `episode.py`,
  `runs.py` (partner rounds with feedback window; honest batch), `metrics.py` (`acc_S` = receiver accuracy).
- `stego/intent.py`: CoT judge (OpenRouter) and `classify` → covert / flagged / neutral / honest.
- `stego/replay/contexts.py`: contexts `<prompt>[_cot|_noprofile|_flipped]`, `bare`; replay sets
  (`prompt_swap`, `ladder`, `flip`). `stego/replay/dataset.py`: build datasets, `Replays` loader.
- `stego/detectors/`: `probes.py` (linear probe, TF-IDF, leave-one-seed-out `cross_seed` / `probe_scores`),
  `metrics.py`, `erasure.py` (prompt direction), `counterfactual.py`.
- `stego/llm.py`: `Chat`; agents = Qwen3-8B on Modal vLLM (`stego-vllm`), judge = OpenRouter.
- `pipeline/step3_replay.py` is a Modal app: module-level imports limited to modal + stdlib.

## Gotchas

- Prompts must stay byte-identical to the stored episodes: replay asserts the rebuilt own prompt equals
  `sender_messages[:2]`, and `tests/test_committed_data.py` checks all committed episodes.
- Ids: `{attribute}-{run}-s{seed}-{ts}[-r{round:02d}]-{i:04d}`, run = `partner_fb16` / `partner_fb0` /
  `honest` (= the episode file stem). Shards, manifests and labels are keyed by these ids.
- `cross_seed` returns pooled scores in fold order, not input order; use `pooled_order` to align with ids.
- Qwen3-8B is gone from OpenRouter; agents need the Modal vLLM server (`STEGO_VLLM_API_KEY` in `.env`). First
  request after idle cold-starts (900 s timeout). OpenRouter judge runs with reasoning disabled.
- `data/episodes/` is committed; don't overwrite (step 1 appends). `data/acts/`, `data/replays/`, `results/`
  are gitignored.
