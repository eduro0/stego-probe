"""Where data lives, and how to read the committed episode files.

    data/episodes/<attribute>/<run>_s<seed>.jsonl                one JSON line per episode (§3); run =
                                                                  partner_fb16, partner_fb0 or honest
    data/episodes/intent_labels.jsonl                             CoT intent labels (§3.4)
    data/run_logs/                                                per-round summaries of partner runs
    data/acts/<replay set>/...                                    activation shards (gitignored, §3.5)
    data/replays/<replay set>/                                    exported datasets (gitignored, §4)
    data/replay_manifests/<replay set>.jsonl                      the (episode, context) pairs replayed
    results/                                                      per-episode scores for the figures
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EPISODES = DATA / "episodes"
INTENT_LABELS = EPISODES / "intent_labels.jsonl"
RUN_LOGS = DATA / "run_logs"
ACTS = DATA / "acts"
REPLAYS = DATA / "replays"
MANIFESTS = DATA / "replay_manifests"
RESULTS = ROOT / "results"


def episode_files(attribute: str | None = None) -> list[Path]:
    """Committed episode files, optionally for one secret attribute ("university", "nationality")."""
    return sorted(EPISODES.glob(f"{attribute or '*'}/*.jsonl"))


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_episodes(files: list[Path] | None = None) -> Iterator[dict]:
    for f in files if files is not None else episode_files():
        yield from read_jsonl(f)


def read_intent_labels(path: Path = INTENT_LABELS) -> dict[str, dict]:
    return {r["episode_id"]: r for r in read_jsonl(path)} if path.exists() else {}
