"""Run a small CPU training and prediction example with entirely synthetic inputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mosaic.artifacts import environment_metadata, write_json
from mosaic.caso.flow_training import main as train_flow
from mosaic.caso.inference import main as predict
from mosaic.caso.train import main as train
from mosaic.policies import SeedPolicy, write_policies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir.resolve()
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    nodes = 16
    graph = root / "relationship.csv"
    pd.DataFrame(
        [(i, (i + 1) % nodes, 0.5) for i in range(nodes)],
        columns=["user_1", "user_2", "closeness"],
    ).to_csv(graph, index=False)
    matches = rng.uniform(0, 1, (1, nodes)).astype(np.float32)
    catalog = root / "content.npz"
    np.savez_compressed(
        catalog,
        format="mosaic-content-match-v2",
        movies=["Synthetic_demo"],
        movie_ids=np.asarray([0], dtype=np.int64),
        ambiguous_titles=np.asarray([], dtype=str),
        matches=matches,
    )
    records = root / "records"
    for condition, budget in enumerate((0, 1, 2, 3, 5, 10)):
        directory = records / "Synthetic_demo" / "temp" / str(condition)
        directory.mkdir(parents=True)
        for replication in range(2):
            watching = np.zeros((31, nodes), dtype=bool)
            watching[0, :budget] = True
            watching[-1, : min(nodes, budget + replication + 2)] = True
            np.savez_compressed(
                directory / f"{replication}.npz", X=watching, sim=np.tile(matches[0], (31, 1))
            )
    train_flow(
        [
            "--output-dir",
            str(root / "flow"),
            "--device",
            "cpu",
            "--num-nodes",
            str(nodes),
            "--updates",
            "2",
            "--batch-size",
            "16",
            "--evaluate-every",
            "1",
            "--validation-size",
            "32",
            "--test-size",
            "32",
            "--generation-size",
            "32",
            "--shared-hidden-dim",
            "8",
            "--context-topk",
            "4",
            "--ode-steps",
            "4",
            "--seed",
            str(args.seed),
        ]
    )
    train(
        [
            "--flow-checkpoint",
            str(root / "flow" / "flow_pretrained.pt"),
            "--records",
            str(records),
            "--relationship",
            str(graph),
            "--content-match",
            str(catalog),
            "--output-dir",
            str(root / "trained"),
            "--num-nodes",
            str(nodes),
            "--hidden-dim",
            "8",
            "--heads",
            "2",
            "--layers",
            "1",
            "--positional-dim",
            "2",
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--seed",
            str(args.seed),
            "--device",
            "cpu",
        ]
    )
    policies = root / "policies.csv"
    write_policies(
        policies,
        [
            SeedPolicy("Synthetic_demo", "Control", 0, (), 0),
            SeedPolicy("Synthetic_demo", "Example", 2, (0, 1), 0),
        ],
    )
    predict(
        [
            "--checkpoint",
            str(root / "trained" / "best.pt"),
            "--relationship",
            str(graph),
            "--content-match",
            str(catalog),
            "--policy-csv",
            str(policies),
            "--output",
            str(root / "predictions.csv"),
            "--device",
            "cpu",
        ]
    )
    write_json(
        root / "demo.json",
        {
            "purpose": "Synthetic CPU interface demonstration; not paper results",
            "seed": args.seed,
            "nodes": nodes,
            "environment": environment_metadata(),
        },
    )
    print(f"Synthetic CPU demo completed: {root / 'predictions.csv'}")


if __name__ == "__main__":
    main()
