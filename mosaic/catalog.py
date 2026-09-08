"""Stable movie identity shared by features, policies, and simulation."""

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mosaic import paths


def canonical_title(title: str) -> str:
    return title.strip().replace("/", "_").replace(" ", "_")


def base_title(key: str) -> str:
    return re.sub(r"__id_\d+$", "", canonical_title(key))


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    genre: str
    description: str
    key: str


def load_movies(path: Path | None = None) -> dict[int, Movie]:
    with (path or paths.get_simulation_item_file()).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        id_field = "item_id" if "item_id" in fields else "id"
        if not {id_field, "title"} <= fields:
            raise ValueError("Movie catalog requires id (or item_id) and title columns")
        rows = list(reader)
    counts = Counter(canonical_title(row["title"]) for row in rows)
    movies = {}
    for row in rows:
        ident, title = int(row[id_field]), row["title"].strip()
        if ident < 0 or ident in movies or not title:
            raise ValueError("Movie IDs must be unique and nonnegative; titles must be nonempty")
        key = canonical_title(title)
        if counts[key] > 1:
            key += f"__id_{ident}"
        movies[ident] = Movie(
            ident, title, row.get("genre", ""), row.get("description", "").strip(), key
        )
    if len({movie.key for movie in movies.values()}) != len(movies):
        raise ValueError("Movie keys collide with reserved __id_ suffixes")
    return movies


def resolve_movie(value: str | int, movies: dict[int, Movie] | None = None) -> Movie:
    movies = load_movies() if movies is None else movies
    text = str(value).strip()
    if isinstance(value, int) or text.startswith("id:"):
        ident = int(text.removeprefix("id:"))
        if ident not in movies:
            raise ValueError(f"Movie ID {ident} is absent from the catalog")
        return movies[ident]
    key = canonical_title(text)
    matches = [m for m in movies.values() if key in (m.key, canonical_title(m.title))]
    if len(matches) != 1:
        if matches:
            choices = ", ".join(f"id:{m.id} ({m.key})" for m in matches)
            raise ValueError(f"Ambiguous movie {value!r}; select one of: {choices}")
        raise ValueError(f"Movie {value!r} is absent from the catalog")
    return matches[0]


def resolve_policy_movie(policy, movies=None) -> Movie:
    movies = load_movies() if movies is None else movies
    movie = resolve_movie(policy.movie_id if policy.movie_id is not None else policy.movie, movies)
    if policy.movie_id is not None and canonical_title(policy.movie) not in (
        canonical_title(movie.title),
        movie.key,
        f"id:{movie.id}",
    ):
        raise ValueError(f"Movie name and movie_id disagree: {policy.movie!r}, {policy.movie_id}")
    return movie
