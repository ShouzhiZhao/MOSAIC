"""Evaluate a saved seed-policy CSV with paired, reset-checked PromoSim replays."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

from mosaic import paths
from mosaic.policies import read_policies

from .policy_replay import run_policy_replays


def summarize_outcomes(outcomes: Path, output: Path) -> None:
    """Aggregate completed replays by movie, method, and budget."""
    grouped = defaultdict(list)
    with outcomes.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[
                (row["movie"], row.get("movie_id", ""), row["method"], int(row["budget"]))
            ].append(float(row["acceptance"]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "movie",
                "movie_id",
                "method",
                "budget",
                "replays",
                "mean_acceptance",
                "std_acceptance",
            ],
        )
        writer.writeheader()
        for (movie, movie_id, method, budget), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "movie": movie,
                    "movie_id": movie_id,
                    "method": method,
                    "budget": budget,
                    "replays": len(values),
                    "mean_acceptance": mean(values),
                    "std_acceptance": stdev(values) if len(values) > 1 else 0.0,
                }
            )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=paths.get_promosim_config())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--movie", nargs="+")
    parser.add_argument("--budget", type=int, nargs="+")
    parser.add_argument("--method", nargs="+")
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--start-time", default="2023-01-01T08:00:00")
    parser.add_argument("--log-file", default="-")
    args = parser.parse_args(argv)
    policies = read_policies(args.policy_csv, args.movie, args.budget, args.method)
    simulation_args = SimpleNamespace(
        config_file=str(args.config.resolve()),
        output_file="policy_replay_messages.json",
        log_file=args.log_file,
        log_name="policy-replay",
        play_role=False,
        memory_backend="behavioral",
    )
    report = run_policy_replays(
        policies,
        simulation_args,
        args.output_dir,
        replications=args.replications,
        base_seed=args.seed,
        replay_start_time=datetime.fromisoformat(args.start_time),
    )
    summary = args.output_dir / "policy_replay_summary.csv"
    summarize_outcomes(Path(report["outcomes"]), summary)
    print(json.dumps({**report, "summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
