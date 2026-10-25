"""Print per-run metrics + M1 go/no-go from an episodes file.  uv run python check_m1.py [data/episodes.jsonl]"""
import json
import sys
from pathlib import Path

from env.metrics import by_run, m1_check, summarize

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "episodes.jsonl")
eps = [json.loads(line) for line in path.open()]
all_ok = True
for run_id, run in by_run(eps).items():
    s = summarize(run)
    ok, msg = m1_check(s)
    all_ok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {run_id:<48} n={s['n']:<4} acc_label={s['acc_label']:.2f} "
          f"hire={s['hire_rate']:.2f} words={s['mean_words']:.0f}  {msg}")
sys.exit(0 if all_ok else 1)
