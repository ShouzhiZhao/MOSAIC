"""Predict total acceptance for saved seed policies without running PromoSim."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from mosaic import paths
from mosaic.policies import read_policies

from .checkpoints import load_checkpoint
from .content import (
    canonical_movie_name,
    content_match_for_movie,
    content_movie_key,
    load_content_match_catalog,
)
from .graph import prepare_graph, scipy_to_torch_sparse
from .search import SurrogateEvaluator


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=paths.get_caso_checkpoint())
    parser.add_argument("--relationship", type=Path, default=paths.get_relationship_file())
    parser.add_argument("--content-match", type=Path, default=paths.get_content_match_catalog())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--movie", nargs="+")
    parser.add_argument("--budget", nargs="+", type=int)
    parser.add_argument("--method", nargs="+")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="One policy per forward pass keeps scores independent of batch composition",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output.resolve() == args.policy_csv.resolve():
        parser.error("output must differ from the input policy CSV")
    policies = read_policies(args.policy_csv, args.movie, args.budget, args.method)
    device = torch.device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    if any(node >= model.config.num_nodes for policy in policies for node in policy.seeds):
        raise ValueError("Seed ID lies outside the checkpoint's graph")
    catalog = load_content_match_catalog(args.content_match)
    resolved = []
    for policy in policies:
        key = content_movie_key(
            catalog, f"id:{policy.movie_id}" if policy.movie_id is not None else policy.movie
        )
        if policy.movie_id is not None and canonical_movie_name(policy.movie) not in {
            key,
            key.rsplit("__id_", 1)[0],
            f"id:{policy.movie_id}",
        }:
            raise ValueError(f"Movie title and movie_id disagree: {policy}")
        resolved.append(replace(policy, movie=key))
    policies = resolved
    graph = prepare_graph(
        args.relationship,
        model.config.num_nodes,
        model.config.positional_dim,
        metadata.get("graph_random_seed", 2026),
    )
    adjacency = scipy_to_torch_sparse(graph.normalized_adjacency, device)
    position = graph.positional_encoding.to(device)
    rows = []
    for movie in dict.fromkeys(policy.movie for policy in policies):
        selected = [policy for policy in policies if policy.movie == movie]
        match = torch.tensor(
            content_match_for_movie(catalog, movie), dtype=torch.float32, device=device
        )
        evaluator = SurrogateEvaluator(
            model, adjacency, position, match.reshape(1, -1), args.batch_size
        )
        values = evaluator.evaluate(policy.seeds for policy in selected)
        rows.extend(
            {
                "movie": movie,
                "movie_id": catalog.movie_ids.get(movie),
                "method": policy.method,
                "budget": policy.budget,
                "seeds": json.dumps(policy.seeds),
                "predicted_acceptance": value,
            }
            for policy, value in zip(selected, values)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "policies": len(rows),
                "output": str(args.output),
                "outcome_source": "CASO prediction; no PromoSim replay",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
