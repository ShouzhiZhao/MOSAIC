"""Search a trained CASO model for exact-budget movie-specific seed policies."""

import argparse
import json
from pathlib import Path

import torch

from mosaic import paths
from mosaic.config import SearchConfig

from .runner import optimize_movies


def main(argv=None) -> None:
    defaults = SearchConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", nargs="+", required=True)
    parser.add_argument("--budget", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--checkpoint", type=Path, default=paths.get_caso_checkpoint())
    parser.add_argument("--relationship", type=Path, default=paths.get_relationship_file())
    parser.add_argument("--content-match", type=Path, default=paths.get_content_match_catalog())
    parser.add_argument("--output-dir", type=Path, default=paths.CASO_RESULTS_DIR)
    parser.add_argument("--latent-iterations", type=int, default=defaults.latent_iterations)
    parser.add_argument("--latent-learning-rate", type=float, default=defaults.latent_learning_rate)
    parser.add_argument("--budget-penalty", type=float, default=defaults.budget_penalty)
    parser.add_argument("--restarts", type=int, default=defaults.random_restarts)
    parser.add_argument("--batch-size", type=int, default=defaults.evaluation_batch_size)
    parser.add_argument("--swap-passes", type=int, default=defaults.max_swap_passes)
    parser.add_argument("--latent-flow-steps", type=int, default=defaults.latent_flow_steps)
    parser.add_argument(
        "--latent-stability-patience", type=int, default=defaults.latent_stability_patience
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not all(1 <= budget <= 10 for budget in args.budget):
        parser.error("CASO search budgets must be between one and ten")
    if min(args.latent_iterations, args.restarts, args.batch_size, args.latent_flow_steps) < 1:
        parser.error("iterations, restarts, batch size, and flow steps must be positive")
    if (
        args.latent_learning_rate <= 0
        or min(args.budget_penalty, args.swap_passes, args.latent_stability_patience) < 0
    ):
        parser.error(
            "learning rate must be positive; penalties and pass counts must be nonnegative"
        )
    search = SearchConfig(
        max_budget=max(args.budget),
        latent_iterations=args.latent_iterations,
        latent_learning_rate=args.latent_learning_rate,
        budget_penalty=args.budget_penalty,
        random_restarts=args.restarts,
        random_seed=args.seed,
        evaluation_batch_size=args.batch_size,
        max_swap_passes=args.swap_passes,
        latent_flow_steps=args.latent_flow_steps,
        latent_stability_patience=args.latent_stability_patience,
    )
    rows = optimize_movies(
        args.checkpoint,
        paths.PROMOSIM_SIMULATION_DATA_DIR,
        args.relationship,
        args.movie,
        args.budget,
        args.output_dir,
        search,
        torch.device(args.device),
        content_match_path=args.content_match,
    )
    print(json.dumps(rows, indent=2))
