"""Construct frozen BGE-M3 content features without behavioral outcome labels."""

import argparse
import json
from pathlib import Path

from mosaic import paths

from .content import generate_content_match_catalog


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", nargs="+", required=True)
    parser.add_argument("--users", type=Path, default=paths.get_simulation_user_file())
    parser.add_argument("--items", type=Path, default=paths.get_simulation_item_file())
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder", help="Model identifier or local BGE-M3 directory")
    parser.add_argument("--output", type=Path, default=paths.get_content_match_catalog())
    args = parser.parse_args(argv)
    result = generate_content_match_catalog(
        args.users,
        args.items,
        args.output,
        args.movie,
        batch_size=args.batch_size,
        num_users=args.num_users,
        encoder_name_or_path=args.encoder,
    )
    print(json.dumps(result, indent=2))
