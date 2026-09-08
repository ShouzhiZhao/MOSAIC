"""Load PromoSim condition records without item-level leakage."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mosaic.catalog import base_title
from mosaic.config import DataSplit
from mosaic.records import validate_replay


@dataclass(frozen=True)
class ConditionRecord:
    movie: str
    condition_id: str
    seed: np.ndarray
    content_match: np.ndarray
    acceptance: np.ndarray
    replay_count: int

    def as_array(self) -> np.ndarray:
        return np.stack([self.seed, self.content_match, self.acceptance], axis=1)


def discover_movies(simulation_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in simulation_root.iterdir()
        if path.is_dir() and (path / "temp").is_dir()
    )


def _load_condition(
    movie: str,
    condition_dir: Path,
    minimum_replays: int,
    expected_rounds: int = 30,
) -> ConditionRecord | None:
    manifest_path = condition_dir / "condition.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        if manifest.get("format") != "mosaic-condition-v2" or manifest.get("movie") != movie:
            raise ValueError(f"Invalid condition identity: {manifest_path}")
        if not (condition_dir / "complete.json").is_file():
            return None
    replay_paths = sorted(condition_dir.glob("*.npz"))
    if manifest is not None and len(replay_paths) != manifest["replications"]:
        raise ValueError(f"Completed condition is missing replays: {condition_dir}")
    if len(replay_paths) < minimum_replays:
        return None

    seeds: list[np.ndarray] = []
    matches: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    for replay_path in replay_paths:
        with np.load(replay_path) as replay:
            trajectory = np.asarray(replay["X"])
            similarity = np.asarray(replay["sim"])
            if manifest is not None and int(replay["movie_id"]) != manifest["movie_id"]:
                raise ValueError(f"Replay movie ID differs from manifest: {replay_path}")
        trajectory, similarity = validate_replay(
            trajectory,
            similarity,
            rounds=expected_rounds,
            num_nodes=len(seeds[0]) if seeds else None,
            source=str(replay_path),
        )
        seeds.append(trajectory[0].astype(np.float32))
        matches.append(similarity[0].astype(np.float32))
        outcomes.append(np.any(trajectory, axis=0).astype(np.float32))

    seed = seeds[0]
    if manifest is not None and (
        len(seed) != manifest["num_nodes"]
        or expected_rounds != manifest["rounds"]
        or sorted(np.flatnonzero(seed).tolist()) != sorted(manifest["seeds"])
    ):
        raise ValueError(f"Replay shape or seeds differ from manifest: {condition_dir}")
    if any(not np.array_equal(seed, candidate) for candidate in seeds[1:]):
        raise ValueError(f"seed vector changes across replays: {condition_dir}")

    return ConditionRecord(
        movie=movie,
        condition_id=condition_dir.name,
        seed=seed,
        content_match=np.mean(matches, axis=0, dtype=np.float64).astype(np.float32),
        acceptance=np.mean(outcomes, axis=0, dtype=np.float64).astype(np.float32),
        replay_count=len(replay_paths),
    )


def load_records(
    simulation_root: Path,
    movies: Sequence[str] | None = None,
    minimum_replays: int = 1,
    seed_budget_range: tuple[int, int] | None = (0, 10),
    expected_rounds: int = 30,
) -> list[ConditionRecord]:
    """Load complete conditions admitted by the analysis seed-budget range.

    The zero-seed control is eligible. Raw archived conditions outside the
    range remain on disk; pass ``None`` only when auditing the raw archive.
    """
    if minimum_replays < 1 or expected_rounds < 1:
        raise ValueError("minimum_replays and expected_rounds must be positive")
    if seed_budget_range is not None:
        lower, upper = seed_budget_range
        if lower < 0 or lower > upper:
            raise ValueError("invalid seed-budget range")
    selected = discover_movies(simulation_root) if movies is None else list(movies)
    records: list[ConditionRecord] = []
    for movie in selected:
        temp_dir = simulation_root / movie / "temp"
        if not temp_dir.is_dir():
            raise FileNotFoundError(f"PromoSim movie directory not found: {temp_dir}")
        for condition_dir in sorted(path for path in temp_dir.iterdir() if path.is_dir()):
            record = _load_condition(movie, condition_dir, minimum_replays, expected_rounds)
            if record is not None:
                if seed_budget_range is not None:
                    seed_count = int(np.count_nonzero(record.seed))
                    if not lower <= seed_count <= upper:
                        continue
                records.append(record)
    return records


def pretraining_movies(simulation_root: Path, split: DataSplit) -> list[str]:
    """Return movies eligible for CASO pretraining.

    The set is computed from the filesystem and then filtered by the explicit
    paper split. This makes newly added titles trainable without accidentally
    admitting the OOD or new-item targets.
    """

    return [
        movie
        for movie in discover_movies(simulation_root)
        if base_title(movie) not in split.held_out
    ]


def movie_content_match(records: Iterable[ConditionRecord]) -> np.ndarray:
    """Construct the fixed target-item content-match vector.

    PromoSim saves the match vector at the pre-intervention state in every
    replay. Numerical differences caused by repeated recommender
    initialization are averaged; seed outcomes are never used here.
    """

    values = [record.content_match for record in records]
    if not values:
        raise ValueError("at least one condition record is required")
    return np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
