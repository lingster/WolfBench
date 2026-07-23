#!/usr/bin/env python3
"""Build a wolfbench_results-shaped JSON restricted to genuine full 89-task
terminal-bench runs, covering ALL configs (historic full-89-task runs plus
our new local Ollama full runs).

Scans both:
  - wolfbench-runs/<config>/<timestamp>/         (historic Terminal-Bench 2.0 runs)
  - /tmp/wolfbench-runs/<config>/<timestamp>/     (local Ollama full runs)

Why this script exists (instead of just using wolfbench_collect.py):
wolfbench_collect.py's classify_run() decides validity from the run's
TOP-LEVEL result.json `stats.n_trials` field. Historic harbor-evals runs
populate that field directly (e.g. {"n_trials": 89, "n_errors": 7}), so the
normal collector classifies them fine even when a few trials were dropped
mid-run (eval-level n_trials 88/87/etc.). Our newer local Ollama/vLLM runs
use a different top-level stats schema — n_completed_trials /
n_errored_trials instead of n_trials/n_errors — so `stats.get("n_trials", 0)`
silently returns 0, and classify_run() then excludes EVERY one of our local
full runs as a "test run (0 tasks)", regardless of how many tasks it
actually scored.

This script sidesteps that entirely by reading the score straight from the
eval's reward_stats (which is reliably populated in both schemas) and
computing score = (# tasks with reward 1.0) / 89 — i.e. any task that was
dropped/errored and never reached a scored verdict simply counts as a zero
against the fixed 89-task denominator. This matches how terminal-bench's own
`metrics[0].mean` already normalizes full runs (verified against several
runs with dropped trials — e.g. a run with 87 scored + 2 dropped tasks and
26 solved reports mean=26/89=0.29213..., not 26/87), so this script's score
should agree with the harness-native score whenever collect.py's classifier
allows a run through, while additionally rescuing runs the classifier drops.

A run is only treated as "full-89" if its eval_name suffix is exactly
"terminal-bench" (not "terminal-bench-sample", the 10-task subset used by
sample10_results.py). Across this dataset only those two suffixes occur.

Output conforms to the schema wolfbench-chart.py consumes (see
wolfbench_collect.py's `extract_metrics()`/`main()` for the canonical
shape): {collected_at, n_vms, n_runs, benchmark, expected_tasks, vms, runs}
where each run record carries agent/model/score/passed_tasks/etc., with
score = (# tasks solved) / 89 and expected_tasks = 89.

Usage:
    python full89_results.py [-o wolfbench_results_full89.json]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import wolfbench_collect as wc  # noqa: E402 — reuse config/agent parsing helpers

REPO_RUNS_DIR = Path(__file__).parent / "wolfbench-runs"
LOCAL_RUNS_DIR = Path("/tmp/wolfbench-runs")

N_FULL_TASKS = 89  # Terminal-Bench 2.0 full benchmark

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$")


def full89_reward(result: dict) -> tuple[str, dict[str, float]] | None:
    """Return (eval_name, {task_base_name: reward}) for a genuine full-89
    terminal-bench run, or None if this run's eval is the 10-task sample
    (or any other non-full eval)."""
    stats = result.get("stats", {})
    evals = stats.get("evals", {})
    eval_name = next(iter(evals), "")
    if not eval_name or eval_name.split("__")[-1] != "terminal-bench":
        return None
    eval_data = evals.get(eval_name, {})
    reward = eval_data.get("reward_stats", {}).get("reward", {})

    task_reward: dict[str, float] = {}
    for reward_str, task_list in reward.items():
        try:
            rv = float(reward_str)
        except (TypeError, ValueError):
            continue
        for t in task_list:
            base = t.split("__")[0]
            task_reward[base] = rv

    return eval_name, task_reward


def find_runs(base_dir: Path) -> list[Path]:
    """Mirror wolfbench_collect.find_local_runs: <config>/<timestamp>/ dirs
    with both result.json and config.json."""
    if not base_dir.exists():
        return []
    runs = []
    for ts_dir in base_dir.glob("*/*"):
        if (
            ts_dir.is_dir()
            and _TIMESTAMP_RE.match(ts_dir.name)
            and (ts_dir / "result.json").exists()
            and (ts_dir / "config.json").exists()
        ):
            runs.append(ts_dir)
    return sorted(runs)


def build_record(run_dir: Path, source: str) -> dict | None:
    """Build a chart-schema run record scoped to the full 89-task benchmark,
    scoring solved/89 directly from reward_stats (bypassing the top-level
    stats.n_trials field that wolfbench_collect's classifier relies on).
    """
    data = wc.read_local_run_data(str(run_dir))
    if data is None:
        return None
    result, config, tokens = data["result"], data["config"], data.get("tokens")

    restricted = full89_reward(result)
    if restricted is None:
        return None
    eval_name, task_reward = restricted

    n_scored = len(task_reward)
    if n_scored == 0:
        return None  # total infra failure / empty eval — nothing to score

    passed_tasks = sorted(t for t, v in task_reward.items() if v == 1.0)
    failed_tasks = sorted(t for t, v in task_reward.items() if v != 1.0)
    score = len(passed_tasks) / N_FULL_TASKS

    record = wc.extract_metrics(source, str(run_dir), result, config, tokens=tokens)

    # Look up eval-level n_errors (reliable in both stats schemas) rather
    # than trusting the top-level field, which is missing/0 for our local
    # Ollama runs.
    stats = result.get("stats", {})
    eval_data = stats.get("evals", {}).get(eval_name, {})
    n_errors = eval_data.get("n_errors", record.get("n_errors"))

    record.update({
        "eval_name": eval_name,
        "score": score,
        "n_trials": N_FULL_TASKS,
        "n_scored": n_scored,
        "n_errors": n_errors,
        "n_passed": len(passed_tasks),
        "n_failed": len(failed_tasks),
        "passed_tasks": passed_tasks,
        "failed_tasks": failed_tasks,
    })
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path(__file__).parent / "wolfbench_results_full89.json",
    )
    args = parser.parse_args()

    all_run_dirs: list[tuple[Path, str]] = (
        [(d, "repo") for d in find_runs(REPO_RUNS_DIR)]
        + [(d, "local") for d in find_runs(LOCAL_RUNS_DIR)]
    )

    records = []
    skipped = []
    for run_dir, source in all_run_dirs:
        rec = build_record(run_dir, source)
        if rec is None:
            skipped.append(str(run_dir))
            continue
        records.append(rec)

    records.sort(key=lambda r: r.get("timestamp", ""))

    print(f"Scanned {len(all_run_dirs)} run dirs "
          f"({len(find_runs(REPO_RUNS_DIR))} in {REPO_RUNS_DIR}, "
          f"{len(find_runs(LOCAL_RUNS_DIR))} in {LOCAL_RUNS_DIR})", file=sys.stderr)
    print(f"Included {len(records)} full-89 runs", file=sys.stderr)
    print(f"Skipped {len(skipped)} runs (10-task sample evals or empty):", file=sys.stderr)
    for s in skipped:
        print(f"  skipped: {s}", file=sys.stderr)

    out_data = {
        "collected_at": datetime.now().isoformat(),
        "n_vms": 0,
        "n_runs": len(records),
        "benchmark": "terminal-bench-2.0",
        "expected_tasks": N_FULL_TASKS,
        "vms": [],
        "runs": records,
    }

    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Written → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
