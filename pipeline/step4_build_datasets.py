"""Step 4 (paper §4): activation shards -> one dataset per replay set, the input of every experiment.

    uv run modal volume get stego-probe acts/<set> data/acts/     # after step 3
    uv run python pipeline/step4_build_datasets.py                 # all sets found under data/acts/
    uv run python pipeline/step4_build_datasets.py ladder          # one set

Reads data/acts/<set>/**/shard_*.npz, writes data/replays/<set>/ (format: doc/replays.md,
stego.replay.dataset). Sets: prompt_swap (~2.5 GB), ladder (~3.4 GB), flip (~3.5 GB).
"""
from __future__ import annotations

import argparse

from stego.paths import ACTS
from stego.replay.contexts import REPLAY_SETS
from stego.replay.dataset import build


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="*", metavar="set", help=f"one of {', '.join(REPLAY_SETS)} (default: all)")
    args = ap.parse_args()
    if unknown := set(args.sets) - set(REPLAY_SETS):
        ap.error(f"unknown replay set(s) {sorted(unknown)}")
    sets = args.sets or [s for s in REPLAY_SETS if any((ACTS / s).rglob("shard_*.npz"))]
    if not sets:
        raise SystemExit(f"no shards under {ACTS}/<set>/ (pipeline/step3_replay.py, then modal volume get)")
    for s in sets:
        build(s)


if __name__ == "__main__":
    main()
