"""Label-free user--item content features for zero-shot CASO inference."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from mosaic.artifacts import atomic_write, environment_metadata, sha256_file, write_json
from mosaic.catalog import canonical_title, load_movies, resolve_movie

CONTENT_FEATURE_FORMAT = "mosaic-content-match-v1"


def canonical_movie_name(title: str) -> str:
    return canonical_title(title)


def _user_descriptions(user_path: Path, num_users: int) -> list[str]:
    descriptions: list[str] = []
    name_counts: dict[str, int] = {}
    with user_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["name"]
            duplicate_index = name_counts.get(name, 0)
            if duplicate_index:
                name = f"{name}_{duplicate_index}"
            name_counts[row["name"]] = duplicate_index + 1
            profile = (
                f"agent profile: name {name}, age {row['age']}, "
                f"gender {row['gender']}, traits {row['traits']}, "
                f"interests {row['interest']}, status {row['status']}"
            )
            descriptions.append(profile + "\nshort-term memory: \nlong-term memory: ")
            if len(descriptions) == num_users:
                break
    if len(descriptions) != num_users:
        raise ValueError(f"expected {num_users} users in {user_path}, found {len(descriptions)}")
    return descriptions


def _item_descriptions(item_path: Path) -> dict[str, str]:
    return {
        m.key: f"name: {m.title}, genre: {m.genre}, description: {m.description}"
        for m in load_movies(item_path).values()
    }


def generate_content_match_catalog(
    user_path: Path,
    item_path: Path,
    output_path: Path,
    movies: Sequence[str],
    num_users: int = 1000,
    batch_size: int = 64,
    encoder_name_or_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate target-label-free cosine similarities with frozen BGE-M3."""

    if not torch.cuda.is_available():
        raise RuntimeError("content-match generation requires CUDA")
    available = _item_descriptions(item_path)
    movie_catalog = load_movies(item_path)
    selected_movies = {m.key: m for m in (resolve_movie(value, movie_catalog) for value in movies)}
    selected = list(selected_movies)
    if not selected:
        raise ValueError("Supply at least one movie")
    missing = set(selected).difference(available)
    if missing:
        raise ValueError(f"movies are absent from the item catalog: {sorted(missing)}")

    # Import lazily so ordinary CASO inference does not initialize BGE-M3.
    from sentence_transformers import SentenceTransformer

    encoder_source = str(
        encoder_name_or_path or os.environ.get("MOSAIC_TEXT_ENCODER", "BAAI/bge-m3")
    )
    encoder = SentenceTransformer(encoder_source, device="cuda")
    users = encoder.encode(
        _user_descriptions(user_path, num_users),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device="cuda",
    )
    items = encoder.encode(
        [available[movie] for movie in selected],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device="cuda",
    )
    matches = (items @ users.T).float().cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_path,
        lambda f: np.savez_compressed(
            f,
            format=np.asarray("mosaic-content-match-v2"),
            movies=np.asarray(selected),
            movie_ids=np.asarray([selected_movies[key].id for key in selected], dtype=np.int64),
            ambiguous_titles=np.asarray(
                sorted(
                    {
                        canonical_title(m.title)
                        for m in movie_catalog.values()
                        if m.key != canonical_title(m.title)
                    }
                ),
                dtype=str,
            ),
            matches=matches,
        ),
    )
    metadata = {
        "format": "mosaic-content-match-v2",
        "environment": environment_metadata(),
        "users_sha256": sha256_file(user_path),
        "items_sha256": sha256_file(item_path),
        "encoder": encoder_source,
        "encoder_source": encoder_source,
        "construction": "cosine(frozen item description, frozen static user profile)",
        "num_users": num_users,
        "movies": selected,
        "movie_ids": {key: selected_movies[key].id for key in selected},
        "target_acceptance_labels_used": False,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(),
        "output": str(output_path),
    }
    write_json(output_path.with_suffix(".json"), metadata)
    return metadata


class ContentCatalog(dict):
    """Dictionary-compatible catalog retaining item identity metadata."""

    def __init__(self, *args, movie_ids=None, ambiguous_titles=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.movie_ids = movie_ids or {}
        self.ambiguous_titles = set(ambiguous_titles)


def load_content_match_catalog(path: Path) -> ContentCatalog:
    with np.load(path, allow_pickle=False) as payload:
        version = str(payload["format"])
        if version not in {CONTENT_FEATURE_FORMAT, "mosaic-content-match-v2"}:
            raise ValueError(f"unsupported content-match catalog: {path}")
        movies = payload["movies"].astype(str).tolist()
        matches = payload["matches"].astype(np.float32)
        ids = payload["movie_ids"].tolist() if version.endswith("v2") else []
        ambiguous = (
            payload["ambiguous_titles"].astype(str).tolist() if version.endswith("v2") else []
        )
    if (
        matches.ndim != 2
        or matches.shape[0] != len(movies)
        or matches.shape[1] < 1
        or len(set(movies)) != len(movies)
        or not np.isfinite(matches).all()
        or np.any(np.abs(matches) > 1.00001)
    ):
        raise ValueError(f"invalid content-match arrays in {path}")
    if version.endswith("v2") and (
        len(ids) != len(movies)
        or len(set(ids)) != len(ids)
        or any(type(i) is not int or i < 0 for i in ids)
    ):
        raise ValueError(f"invalid movie IDs in {path}")
    return ContentCatalog(
        zip(movies, matches), movie_ids=dict(zip(movies, ids)), ambiguous_titles=ambiguous
    )


def content_movie_key(catalog: dict, movie: str) -> str:
    key = canonical_movie_name(movie)
    if key in getattr(catalog, "ambiguous_titles", set()):
        raise ValueError(f"Ambiguous movie {movie!r}; use id:<movie_id> or a qualified movie key")
    ids = getattr(catalog, "movie_ids", {})
    if key.startswith("id:"):
        wanted = int(key[3:])
        keys = [name for name, ident in ids.items() if ident == wanted]
        if not keys:
            if ids:
                raise KeyError(f"content-match catalog has no ID {wanted}")
            legacy_movie = resolve_movie(wanted)
            if legacy_movie.key != canonical_title(legacy_movie.title):
                raise ValueError("Legacy features are ambiguous; regenerate using movie IDs")
            key = legacy_movie.key
        else:
            key = keys[0]
    if not ids:
        # A legacy title-only archive cannot identify which same-name movie was encoded.
        try:
            movies = load_movies()
        except FileNotFoundError:
            movies = {}
        if any(canonical_title(m.title) == key and m.key != key for m in movies.values()):
            raise ValueError(
                f"Legacy features for {movie!r} are ambiguous; regenerate using movie IDs"
            )
    if key not in catalog:
        raise KeyError(f"content-match catalog has no entry for {key}")
    return key


def content_match_for_movie(catalog: dict[str, np.ndarray], movie: str) -> np.ndarray:
    return catalog[content_movie_key(catalog, movie)].copy()


def apply_content_match_catalog(records: Iterable, catalog: dict[str, np.ndarray]):
    """Return ConditionRecords with label-free static content features."""

    from dataclasses import replace

    return [
        replace(record, content_match=content_match_for_movie(catalog, record.movie))
        for record in records
    ]
