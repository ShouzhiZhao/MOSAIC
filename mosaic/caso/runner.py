"""High-level CASO seed-search runner and result serialization."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from mosaic import paths
from mosaic.artifacts import sha256_file
from mosaic.config import SearchConfig
from mosaic.policies import SeedPolicy, write_policies

from .checkpoints import load_checkpoint
from .content import content_match_for_movie, content_movie_key, load_content_match_catalog
from .graph import prepare_graph, scipy_to_torch_sparse
from .search import CASOSearcher
from .training_utils import seed_everything


def optimize_movies(
    checkpoint_path: Path,
    simulation_root: Path,
    relationship_path: Path,
    movies: Sequence[str],
    budgets: Sequence[int],
    output_dir: Path,
    search_config: SearchConfig,
    device: torch.device | None = None,
    content_match_scale: float = 1.0,
    content_match_path: Path | None = None,
) -> list[dict[str, object]]:
    device = device or torch.device("cuda")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CASO search requires CUDA; CPU fallback is disabled")
    if not movies or not budgets or min(budgets) < 1:
        raise ValueError("Supply movies and positive seed budgets")
    seed_everything(search_config.random_seed)
    model, checkpoint_metadata = load_checkpoint(checkpoint_path, device)
    content_match_path = content_match_path or paths.get_content_match_catalog()
    content_catalog = load_content_match_catalog(content_match_path)
    graph = prepare_graph(
        relationship_path,
        model.config.num_nodes,
        model.config.positional_dim,
        checkpoint_metadata.get("graph_random_seed", 2026),
    )
    adjacency = scipy_to_torch_sparse(graph.normalized_adjacency, device)
    position = graph.positional_encoding.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    rows: list[dict[str, object]] = []
    for movie in movies:
        movie = content_movie_key(content_catalog, movie)
        movie_id = content_catalog.movie_ids.get(movie)
        match = (
            torch.tensor(
                content_match_for_movie(content_catalog, movie),
                dtype=torch.float32,
                device=device,
            )
            * content_match_scale
        )
        searcher = CASOSearcher(model, adjacency, position, match, search_config)
        maximum_budget = max(budgets)
        greedy_prefix, greedy_profile = searcher.greedy_prefix_profile(maximum_budget)
        for budget in sorted(set(budgets)):
            results = searcher.search_budget(budget, greedy_prefix, greedy_profile[budget])
            for result in results:
                rows.append(
                    {
                        "movie": movie,
                        "movie_id": movie_id,
                        "method": result.method,
                        "budget": result.budget,
                        "seeds": list(result.seeds),
                        "predicted_acceptance": result.predicted_acceptance,
                        "candidate_source": result.candidate_source,
                        "search_seconds": result.elapsed_seconds,
                        **environment,
                    }
                )
        movie_rows = [row for row in rows if row["movie"] == movie]
        (output_dir / f"{movie}.json").write_text(
            json.dumps(
                {
                    "framework": "MOSAIC",
                    "optimizer": "CASO",
                    "movie": movie,
                    "search_config": asdict(search_config),
                    "environment": environment,
                    "checkpoint_metadata": checkpoint_metadata,
                    "inputs": {
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "relationship_sha256": sha256_file(relationship_path),
                        "content_match_sha256": sha256_file(content_match_path),
                    },
                    "content_match_scale": content_match_scale,
                    "content_match_catalog": str(content_match_path),
                    "results": movie_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    csv_path = output_dir / "caso_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["seeds"] = json.dumps(serializable["seeds"])
            writer.writerow(serializable)
    write_policies(
        output_dir / "caso_policies.csv",
        [
            SeedPolicy(
                row["movie"], row["method"], row["budget"], tuple(row["seeds"]), row["movie_id"]
            )
            for row in rows
            if row["method"] == "CASO"
        ],
    )
    return rows
