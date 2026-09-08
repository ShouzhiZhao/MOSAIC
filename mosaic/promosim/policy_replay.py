"""Paired, reset-checked PromoSim evaluation of discrete seed policies."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from mosaic import paths
from mosaic.artifacts import atomic_write, environment_metadata, sha256_file, write_json
from mosaic.catalog import load_movies, resolve_policy_movie
from mosaic.policies import SeedPolicy

from .integrity import paired_rng_seed


def movie_titles_by_id(item_path: Path | None = None) -> dict[int, str]:
    """Read the bundled ``id`` column or the alternative ``item_id`` schema."""
    return {m.id: m.title for m in load_movies(item_path).values()}


def movie_id_by_name(item_path: Path | None = None) -> dict[str, int]:
    result: dict[str, int] = {}
    for movie in load_movies(item_path).values():
        result[movie.key] = movie.id
        result[f"id:{movie.id}"] = movie.id
        if movie.key == movie.title.replace("/", "_").replace(" ", "_"):
            result[movie.title] = movie.id
    return result


def run_policy_replays(
    policies: Iterable[SeedPolicy],
    simulation_args,
    output_dir: Path,
    *,
    replications: int = 10,
    base_seed: int = 2026,
    replay_start_time: datetime = datetime(2023, 1, 1, 8, 0, 0),
) -> dict[str, object]:
    """Evaluate policies with common random numbers inside each paired cell."""
    if replications < 1:
        raise ValueError("replications must be positive")

    item_path = None
    if getattr(simulation_args, "config_file", None):
        import yaml

        config = yaml.safe_load(Path(simulation_args.config_file).read_text())
        item_path = Path(config["item_path"])
        if not item_path.is_absolute():
            item_path = paths.PROMOSIM_DATA_DIR / item_path
    catalog = load_movies(item_path)
    grouped: dict[tuple[str, int], list[SeedPolicy]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for policy in policies:
        movie = resolve_policy_movie(policy, catalog)
        policy = SeedPolicy(movie.key, policy.method, policy.budget, policy.seeds, movie.id)
        key = (policy.movie, policy.budget, policy.method)
        if key in seen:
            raise ValueError(f"duplicate policy row: {key}")
        seen.add(key)
        grouped[(policy.movie, policy.budget)].append(policy)
    if not grouped:
        raise ValueError("no policies supplied")

    movie_ids = {m.key: m.id for m in catalog.values()}
    missing_movies = sorted({movie for movie, _ in grouped if movie not in movie_ids})
    if missing_movies:
        raise ValueError(f"movies absent from item catalog: {missing_movies}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "policy_replay_outcomes.csv"
    integrity_path = output_dir / "policy_replay_integrity.jsonl"
    if result_path.exists() or integrity_path.exists():
        raise FileExistsError("Use a new output directory; existing replay results are preserved")
    result_fields = (
        "movie",
        "movie_id",
        "method",
        "budget",
        "replication",
        "rng_seed",
        "seeds",
        "acceptance",
        "state_digest",
    )

    initializations = 0
    paired_cells = 0
    with (
        result_path.open("w", newline="", encoding="utf-8") as result_handle,
        integrity_path.open("w", encoding="utf-8") as integrity_handle,
    ):
        writer = csv.DictWriter(result_handle, fieldnames=result_fields)
        writer.writeheader()

        for (movie, budget), cell_policies in sorted(grouped.items()):
            for replication in range(replications):
                rng_seed = paired_rng_seed(base_seed, movie, budget, replication)
                reference_digest: str | None = None
                initialization_state = {}
                paired_cells += 1

                for policy in sorted(cell_policies, key=lambda value: value.method):
                    report_holder: dict[str, object] = {}

                    def record_integrity(report: dict) -> None:
                        report_holder.update(report)
                        snapshot_path = (
                            output_dir / "initializations" / f"{movie}-{budget}-{replication}.pt"
                        )
                        if reference_digest is None:
                            import torch

                            atomic_write(
                                snapshot_path, lambda f: torch.save(initialization_state, f)
                            )
                        report_holder["initialization_sha256"] = sha256_file(snapshot_path)
                        integrity_handle.write(
                            json.dumps(
                                {
                                    "movie": movie,
                                    "movie_id": movie_ids[movie],
                                    "method": policy.method,
                                    "budget": budget,
                                    "replication": replication + 1,
                                    **report_holder,
                                    "status": "initialized",
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        integrity_handle.flush()

                    from .generation import run_simulation

                    try:
                        flags, _, _, _ = run_simulation(
                            seed_size=budget,
                            target_movie_id=movie_ids[movie],
                            seed_agents=list(policy.seeds),
                            args=simulation_args,
                            rng_seed=rng_seed,
                            initialization_state=initialization_state,
                            replay_start_time=replay_start_time,
                            reference_state_digest=reference_digest,
                            integrity_callback=record_integrity,
                            config_overrides={
                                "interaction_path": str(
                                    output_dir / "policy_replay_interaction.csv"
                                ),
                                "simulator_restore_file_name": "",
                            },
                        )
                    except Exception as exc:
                        integrity_handle.write(
                            json.dumps(
                                {
                                    "movie": movie,
                                    "movie_id": movie_ids[movie],
                                    "method": policy.method,
                                    "budget": budget,
                                    "replication": replication + 1,
                                    "status": "failed",
                                    "error_type": type(exc).__name__,
                                    **report_holder,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        integrity_handle.flush()
                        raise
                    state_digest = str(report_holder["state_digest"])
                    if reference_digest is None:
                        reference_digest = state_digest

                    audit = {
                        "movie": movie,
                        "movie_id": movie_ids[movie],
                        "method": policy.method,
                        "budget": budget,
                        "replication": replication + 1,
                        **report_holder,
                        "status": "passed",
                    }
                    integrity_handle.write(json.dumps(audit, sort_keys=True) + "\n")
                    integrity_handle.flush()

                    writer.writerow(
                        {
                            "movie": movie,
                            "movie_id": movie_ids[movie],
                            "method": policy.method,
                            "budget": budget,
                            "replication": replication + 1,
                            "rng_seed": rng_seed,
                            "seeds": json.dumps(policy.seeds),
                            "acceptance": int(np.any(flags, axis=0).sum()),
                            "state_digest": state_digest,
                        }
                    )
                    result_handle.flush()
                    initializations += 1

    summary = {
        "environment": environment_metadata(),
        "base_seed": base_seed,
        "replications_per_policy": replications,
        "paired_cells": paired_cells,
        "policy_initializations_checked": initializations,
        "reset_integrity_failures": 0,
        "replay_start_time": replay_start_time.isoformat(),
        "outcomes": str(result_path),
        "integrity_log": str(integrity_path),
    }
    write_json(output_dir / "policy_replay_environment.json", summary)
    return summary
