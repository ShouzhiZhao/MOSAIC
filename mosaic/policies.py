"""Shared seed-policy records used by search, prediction, and simulation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mosaic.catalog import canonical_title, resolve_movie


@dataclass(frozen=True)
class SeedPolicy:
    """One target movie, method, and exact-budget intervention."""

    movie: str
    method: str
    budget: int
    seeds: tuple[int, ...]
    movie_id: int | None = None

    def __post_init__(self) -> None:
        if self.movie_id is not None and (type(self.movie_id) is not int or self.movie_id < 0):
            raise ValueError("movie_id must be a nonnegative integer")
        if not self.movie.strip() or not self.method.strip():
            raise ValueError("movie and method must be nonempty")
        if type(self.budget) is not int or self.budget < 0:
            raise ValueError("policy budget must be a nonnegative integer")
        if any(type(node) is not int or node < 0 for node in self.seeds):
            raise ValueError("seed IDs must be nonnegative integers")
        if len(self.seeds) != self.budget or len(set(self.seeds)) != self.budget:
            raise ValueError("seed IDs must be distinct and match the stated budget")


def read_policies(
    path: Path,
    movies: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
) -> list[SeedPolicy]:
    """Read canonical policies or expand a graph-only baseline CSV over movies.

    Canonical columns are movie, method, budget, and seeds. The historical
    baseline names algorithm and k are accepted at this input boundary.
    """
    policies = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        method_field = "method" if "method" in fields else "algorithm"
        budget_field = "budget" if "budget" in fields else "k"
        if not {method_field, budget_field, "seeds"} <= fields:
            raise ValueError("Policy CSV requires method, budget, and seeds columns")
        if "movie" not in fields and not movies:
            raise ValueError("Supply --movie to expand graph-only baseline seeds")
        for row in reader:
            budget = int(row[budget_field])
            method = row[method_field]
            targets = [row["movie"]] if "movie" in fields else list(movies)
            if budgets is not None and budget not in budgets:
                continue
            if methods is not None and method not in methods:
                continue
            seeds = json.loads(row["seeds"])
            if not isinstance(seeds, list):
                raise ValueError("seeds must be a JSON list of integer user IDs")
            for movie in targets:
                movie_id = int(row["movie_id"]) if row.get("movie_id", "").strip() else None
                if movies is not None:
                    aliases = {canonical_title(movie)}
                    if movie_id is not None:
                        aliases.add(f"id:{movie_id}")
                    else:
                        try:
                            resolved = resolve_movie(movie)
                        except (ValueError, FileNotFoundError):
                            pass
                        else:
                            aliases.update((resolved.key, f"id:{resolved.id}"))
                    if not aliases.intersection(canonical_title(value) for value in movies):
                        continue
                policy = SeedPolicy(movie, method, budget, tuple(seeds), movie_id)
                key = (movie_id if movie_id is not None else movie, method, budget)
                if key in seen:
                    raise ValueError(f"duplicate policy row: {key}")
                seen.add(key)
                policies.append(policy)
    if not policies:
        raise ValueError("No policies match the requested movies, methods, and budgets")
    return policies


def write_policies(path: Path, policies: Iterable[SeedPolicy]) -> None:
    """Write the common interchange format without mixing in predicted scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["movie", "movie_id", "method", "budget", "seeds"]
        )
        writer.writeheader()
        for policy in policies:
            writer.writerow(
                {
                    "movie": policy.movie,
                    "movie_id": policy.movie_id,
                    "method": policy.method,
                    "budget": policy.budget,
                    "seeds": json.dumps(policy.seeds),
                }
            )
