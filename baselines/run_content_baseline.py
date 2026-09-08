#!/usr/bin/env python3
"""Generate the item-specific ContentMatch-Degree baseline policies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from baselines.algorithms import content_match_degree
from baselines.run_baselines import load_graph
from mosaic import paths
from mosaic.caso.content import (
    content_match_for_movie,
    content_movie_key,
    load_content_match_catalog,
)

DEFAULT_OUTPUT = paths.ARTIFACTS_DIR / "baselines" / "seeds" / "all_seeds_k1-10_content.csv"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=paths.get_relationship_file())
    parser.add_argument("--content-match", type=Path, default=paths.get_content_match_catalog())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--movie", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.k < 1 or args.scale < 0:
        parser.error("k must be positive and scale must be nonnegative")

    graph = load_graph(args.data_path, args.scale)
    if args.k > len(graph):
        parser.error("seed budget exceeds the graph size")
    catalog = load_content_match_catalog(args.content_match)
    rows: list[dict[str, object]] = []
    movies = args.movie
    for movie in movies:
        movie = content_movie_key(catalog, movie)
        started = time.perf_counter()
        seeds = content_match_degree(graph, args.k, content_match_for_movie(catalog, movie))
        elapsed = time.perf_counter() - started
        for budget in range(1, args.k + 1):
            rows.append(
                {
                    "movie": movie,
                    "movie_id": catalog.movie_ids.get(movie),
                    "method": "ContentMatch-Degree",
                    "budget": budget,
                    "seeds": json.dumps(seeds[:budget]),
                    "time_seconds": elapsed,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
