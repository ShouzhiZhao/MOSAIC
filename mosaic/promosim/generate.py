"""Generate training records for one randomly sampled movie–seed condition."""

import argparse
from pathlib import Path

from .runner import run


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-movie-id", type=int, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--resume-condition")
    args = parser.parse_args(argv)
    run(
        args.target_movie_id,
        args.config,
        args.data_dir,
        args.output_dir,
        seed=args.seed,
        replications=args.replications,
        resume_condition=args.resume_condition,
    )
