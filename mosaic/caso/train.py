"""Two-stage CASO training: synthetic seed flow, then total-acceptance regression."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mosaic import paths
from mosaic.artifacts import environment_metadata, sha256_file
from mosaic.catalog import base_title
from mosaic.config import DataSplit, ModelConfig, TrainingConfig

from .checkpoints import load_checkpoint, save_checkpoint
from .content import content_match_for_movie, load_content_match_catalog
from .data import ConditionRecord, load_records
from .evaluation import acceptance_metrics
from .graph import prepare_graph, repeat_sparse_graph, scipy_to_torch_sparse
from .losses import total_acceptance_loss
from .models import CASOModel
from .training_utils import seed_everything, split_records


def select_training_records(
    records: list[ConditionRecord],
    validation_fraction: float,
    seed: int,
    metadata: dict | None = None,
) -> tuple[list[ConditionRecord], list[ConditionRecord]]:
    """Split eligible movies, or reuse explicit condition keys from a checkpoint.

    Gamma Rays and the configured OOD movies never enter predictor fitting or
    checkpoint selection. A saved split must match records present on disk.
    """
    split = DataSplit()
    eligible = [record for record in records if base_title(record.movie) not in split.held_out]
    if metadata and "condition_keys" in metadata:
        by_key = {(record.movie, record.condition_id): record for record in eligible}
        keys = metadata["condition_keys"]
        train_keys = [tuple(key) for key in keys["train"]]
        validation_keys = [tuple(key) for key in keys["validation"]]
        if len(set(train_keys)) != len(train_keys) or len(set(validation_keys)) != len(
            validation_keys
        ):
            raise ValueError("Saved split contains duplicate condition keys")
        if set(train_keys) & set(validation_keys):
            raise ValueError("Saved training and validation conditions overlap")
        missing = (set(train_keys) | set(validation_keys)) - by_key.keys()
        if missing:
            raise ValueError(
                f"Saved split includes missing or held-out conditions: {sorted(missing)}"
            )
        train = [by_key[key] for key in train_keys]
        validation = [by_key[key] for key in validation_keys]
    else:
        train, validation = split_records(eligible, validation_fraction, seed)
    if not train or not validation:
        raise ValueError(
            "Provide enough pretraining conditions for nonempty training and validation sets"
        )
    return train, validation


def arguments(argv=None) -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=paths.PROMOSIM_SIMULATION_DATA_DIR)
    parser.add_argument("--relationship", type=Path, default=paths.get_relationship_file())
    parser.add_argument("--content-match", type=Path, default=paths.get_content_match_catalog())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--flow-checkpoint", type=Path, help="Reuse stage one instead of retraining it"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional architecture and saved split; predictor weights are not reused",
    )
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--positional-dim", type=int, default=8)
    parser.add_argument("--flow-updates", type=int, default=40000)
    parser.add_argument("--flow-batch-size", type=int, default=256)
    parser.add_argument(
        "--epochs", "--frozen-epochs", dest="epochs", type=int, default=defaults.epochs
    )
    parser.add_argument(
        "--patience", "--frozen-patience", dest="patience", type=int, default=defaults.patience
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = arguments(argv)
    if (
        min(
            args.epochs,
            args.patience,
            args.batch_size,
            args.flow_updates,
            args.flow_batch_size,
            args.learning_rate,
        )
        <= 0
    ):
        raise ValueError("Training counts and learning rate must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve previous training runs")

    metadata = None
    config = ModelConfig(
        num_nodes=args.num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        positional_dim=args.positional_dim,
        acceptance_output="total",
    )
    if args.checkpoint:
        template, metadata = load_checkpoint(args.checkpoint, torch.device("cpu"))
        config = replace(template.config, acceptance_output="total")
        del template
    if config.num_nodes < 10 or not 0 < config.positional_dim < config.num_nodes - 1:
        raise ValueError("Graph size must support budgets 0–10 and the positional dimension")
    records = load_records(args.records)
    train, validation = select_training_records(
        records, args.validation_fraction, args.seed, metadata
    )
    chosen = [*train, *validation]
    if any(len(record.seed) != config.num_nodes for record in chosen):
        raise ValueError("Trajectory node count does not match the model")
    catalog = load_content_match_catalog(args.content_match)
    matches = np.stack([content_match_for_movie(catalog, record.movie) for record in chosen])
    if matches.shape != (len(chosen), config.num_nodes):
        raise ValueError("Content-match catalog does not match the training graph")

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = args.flow_checkpoint
    if flow_path is None:
        from .flow_training import main as train_flow

        train_flow(
            [
                "--output-dir",
                str(args.output_dir / "flow"),
                "--num-nodes",
                str(config.num_nodes),
                "--context-topk",
                str(min(16, config.num_nodes)),
                "--updates",
                str(args.flow_updates),
                "--batch-size",
                str(args.flow_batch_size),
                "--seed",
                str(args.seed),
                "--device",
                str(device),
            ]
        )
        flow_path = args.output_dir / "flow/flow_pretrained.pt"
    flow = torch.load(flow_path, map_location="cpu", weights_only=False)
    if flow.get("format") != "mosaic-seed-flow-v1":
        raise ValueError("Expected a standalone seed-flow checkpoint")
    flow_config = ModelConfig(**flow["model_config"])
    if flow_config.num_nodes != config.num_nodes:
        raise ValueError("Seed flow and acceptance predictor must have the same node count")
    config = replace(
        config,
        **{
            name: value
            for name, value in asdict(flow_config).items()
            if name.startswith("flow_") or name == "logit_epsilon"
        },
    )

    # Predictor initialization is independent of how long stage one ran.
    seed_everything(args.seed)
    model = CASOModel(config).to(device)
    model.seed_flow.load_state_dict(flow["flow_state"])
    model.seed_flow.requires_grad_(False)
    model.seed_flow.eval()
    graph = prepare_graph(args.relationship, config.num_nodes, config.positional_dim, args.seed)
    adjacency = scipy_to_torch_sparse(graph.normalized_adjacency, device)
    position = graph.positional_encoding.to(device)
    graph_batches = {
        size: (repeat_sparse_graph(adjacency, size), position.repeat(size, 1))
        for size in range(1, args.batch_size + 1)
    }
    seeds = torch.tensor(np.stack([record.seed for record in chosen]), device=device)
    content = torch.tensor(matches, dtype=torch.float32, device=device)
    targets = torch.tensor(np.stack([record.acceptance for record in chosen]), device=device)
    with torch.no_grad():
        relaxed = torch.cat(
            [model.seed_flow.reconstruct_discrete(batch) for batch in seeds.split(args.batch_size)]
        )
    train_indices = torch.arange(len(train), device=device)
    validation_indices = torch.arange(len(train), len(chosen), device=device)

    @torch.no_grad()
    def validate() -> tuple[dict, torch.Tensor]:
        model.acceptance_predictor.eval()
        predictions = []
        for indices in validation_indices.split(args.batch_size):
            batch_graph, batch_position = graph_batches[len(indices)]
            predictions.append(
                model.predict(relaxed[indices], content[indices], batch_graph, batch_position)
            )
        prediction = torch.cat(predictions)
        return acceptance_metrics(prediction, targets[validation_indices]), prediction

    split_rows = [
        {
            "movie": record.movie,
            "condition": record.condition_id,
            "budget": int(record.seed.sum()),
            "replays": record.replay_count,
            "split": "train" if index < len(train) else "validation",
        }
        for index, record in enumerate(chosen)
    ]
    pd.DataFrame(split_rows).to_csv(args.output_dir / "condition_split.csv", index=False)
    optimizer = torch.optim.Adam(model.acceptance_predictor.parameters(), lr=args.learning_rate)
    best_metrics, _ = validate()
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = [{"epoch": 0, **best_metrics}]
    for epoch in range(1, args.epochs + 1):
        model.acceptance_predictor.train()
        order = train_indices[torch.randperm(len(train_indices), device=device)]
        loss_sum = 0.0
        for indices in order.split(args.batch_size):
            batch_graph, batch_position = graph_batches[len(indices)]
            prediction = model.predict(
                relaxed[indices], content[indices], batch_graph, batch_position
            )
            loss = total_acceptance_loss(prediction, targets[indices], weight=100.0)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite acceptance loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.acceptance_predictor.parameters(), TrainingConfig().gradient_clip
            )
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        measured, _ = validate()
        history.append({"epoch": epoch, "training_loss": loss_sum / len(train), **measured})
        if measured["total_mae"] < best_metrics["total_mae"] - 1e-6:
            best_metrics, best_epoch = measured, epoch
            best_state = copy.deepcopy(model.state_dict())
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        print(
            f"epoch={epoch} validation_MAE={measured['total_mae']:.4f} best={best_metrics['total_mae']:.4f}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    if any(
        not torch.equal(value.cpu(), flow["flow_state"][name])
        for name, value in model.seed_flow.state_dict().items()
    ):
        raise RuntimeError("Frozen flow changed during predictor training")
    checkpoint_metadata = {
        "environment": environment_metadata(),
        "inputs": {
            "relationship_sha256": sha256_file(args.relationship),
            "content_match_sha256": sha256_file(args.content_match),
        },
        "training_method": "synthetic flow pretraining, then frozen-flow total-acceptance regression",
        "frozen_seed_flow": True,
        "predictor_initialization": "random",
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_conditions": len(train),
        "validation_conditions": len(validation),
        "selection_metric": "validation total-acceptance MAE",
        "validation_metrics": best_metrics,
        "flow_pretraining": flow["metadata"],
        "flow_checkpoint": str(flow_path),
        "pretraining_movies": sorted({record.movie for record in chosen}),
        "condition_keys": {
            name: [[r.movie, r.condition_id] for r in group]
            for name, group in [("train", train), ("validation", validation)]
        },
        "training_config": {
            "random_seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "predictor_learning_rate": args.learning_rate,
        },
        "seed_input": "deterministic midpoint reconstruction",
        "content_match_catalog": str(args.content_match),
        "graph_random_seed": args.seed,
    }
    save_checkpoint(args.output_dir / "best.pt", model, checkpoint_metadata)
    _, prediction = validate()
    pd.DataFrame(
        {
            "movie": [r.movie for r in validation],
            "condition": [r.condition_id for r in validation],
            "actual_acceptance": targets[validation_indices].sum(1).cpu().numpy(),
            "predicted_acceptance": prediction.sum(1).cpu().numpy(),
        }
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    report = {**checkpoint_metadata, "elapsed_seconds": time.monotonic() - started}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output_dir / "best.pt"), **best_metrics}, indent=2))


if __name__ == "__main__":
    main()
