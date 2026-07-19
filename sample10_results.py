#!/usr/bin/env python3
"""Build a wolfbench_results-shaped JSON restricted to the 10-task
terminal-bench-sample subset, covering ALL configs (historic full-89-task
runs plus our new local Ollama runs).

Scans both:
  - wolfbench-runs/<config>/<timestamp>/         (historic Terminal-Bench 2.0 runs)
  - /tmp/wolfbench-runs/<config>/<timestamp>/     (local Ollama sample runs)

For every run, restricts scoring to the 10 sample tasks:
    chess-best-move, polyglot-c-py, qemu-startup, regex-log,
    fix-code-vulnerability, build-cython-ext, qemu-alpine-ssh,
    log-summary-date-ranges, sqlite-with-gcov, configure-git-webserver

A run is only included if its eval's reward_stats cover all 10 of those
task base-names (i.e. each one appears — pass or fail — in the run's
reward_stats). Runs that don't cover the full 10 (partial/aborted/smoke
runs) are skipped and reported.

Output conforms to the schema wolfbench-chart.py consumes (see
wolfbench_collect.py's `extract_metrics()`/`main()` for the canonical
shape): {collected_at, n_vms, n_runs, benchmark, expected_tasks, vms, runs}
where each run record carries agent/model/score/passed_tasks/etc., with
score = (# of the 10 sample tasks solved) / 10 and expected_tasks = 10.

Usage:
    python sample10_results.py [-o wolfbench_results_sample10.json]
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

SAMPLE_TASKS = {
    "chess-best-move",
    "polyglot-c-py",
    "qemu-startup",
    "regex-log",
    "fix-code-vulnerability",
    "build-cython-ext",
    "qemu-alpine-ssh",
    "log-summary-date-ranges",
    "sqlite-with-gcov",
    "configure-git-webserver",
}
N_SAMPLE_TASKS = len(SAMPLE_TASKS)

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$")


def sample10_reward(result: dict) -> tuple[str, dict[str, float]] | None:
    """Return (eval_name, {task_base_name: reward}) restricted to SAMPLE_TASKS,
    or None if the run's eval doesn't cover all 10 sample tasks."""
    stats = result.get("stats", {})
    evals = stats.get("evals", {})
    eval_name = next(iter(evals), "")
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
            if base in SAMPLE_TASKS:
                task_reward[base] = rv

    if set(task_reward.keys()) != SAMPLE_TASKS:
        return None
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
    """Build a chart-schema run record scoped to the 10-task sample.

    Reuses wolfbench_collect's read + extract_metrics for all the
    agent/model/thinking/timing/token bookkeeping (so display labels match
    what wolfbench_collect.py would produce), then overrides the
    score/task-count fields to be restricted to SAMPLE_TASKS.
    """
    data = wc.read_local_run_data(str(run_dir))
    if data is None:
        return None
    result, config, tokens = data["result"], data["config"], data.get("tokens")

    restricted = sample10_reward(result)
    if restricted is None:
        return None
    eval_name, task_reward = restricted

    passed_tasks = sorted(t for t, v in task_reward.items() if v == 1.0)
    failed_tasks = sorted(t for t, v in task_reward.items() if v != 1.0)
    score = len(passed_tasks) / N_SAMPLE_TASKS

    record = wc.extract_metrics(source, str(run_dir), result, config, tokens=tokens)

    # Override with sample-10-restricted results (the base record's
    # score/n_trials/etc. reflect the FULL run, e.g. 89 tasks for historic
    # runs — we only want the 10-task subset here).
    record.update({
        "eval_name": eval_name,
        "score": score,
        "n_trials": N_SAMPLE_TASKS,
        "n_scored": N_SAMPLE_TASKS,
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
        default=Path(__file__).parent / "wolfbench_results_sample10.json",
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
    print(f"Included {len(records)} runs covering all 10 sample tasks", file=sys.stderr)
    print(f"Skipped {len(skipped)} runs (partial sample-task coverage):", file=sys.stderr)
    for s in skipped:
        print(f"  skipped: {s}", file=sys.stderr)

    out_data = {
        "collected_at": datetime.now().isoformat(),
        "n_vms": 0,
        "n_runs": len(records),
        "benchmark": "terminal-bench-sample",
        "expected_tasks": N_SAMPLE_TASKS,
        "vms": [],
        "runs": records,
    }

    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Written → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
