"""Model-level diagnostics used for tuning without PromoSim policy replay."""

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import ConditionRecord
from .graph import repeat_sparse_graph
from .models import CASOModel


@torch.no_grad()
def predictor_metrics(
    model: CASOModel,
    records: Sequence[ConditionRecord],
    adjacency: torch.Tensor,
    positional_encoding: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    relaxed_seed: bool = True,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    absolute_total_error = 0.0
    squared_total_error = 0.0
    predicted_totals: list[float] = []
    observed_totals: list[float] = []
    value_count = 0
    for offset in range(0, len(records), batch_size):
        chunk = records[offset : offset + batch_size]
        seed = torch.tensor(
            np.stack([record.seed for record in chunk]),
            dtype=torch.float32,
            device=device,
        )
        match = torch.tensor(
            np.stack([record.content_match for record in chunk]),
            dtype=torch.float32,
            device=device,
        )
        acceptance = torch.tensor(
            np.stack([record.acceptance for record in chunk]),
            dtype=torch.float32,
            device=device,
        )
        predictor_seed = model.seed_flow.reconstruct_discrete(seed) if relaxed_seed else seed
        graph = repeat_sparse_graph(adjacency, len(chunk))
        position = positional_encoding.repeat(len(chunk), 1)
        prediction = model.predict(predictor_seed, match, graph, position)
        if model.config.acceptance_output == "node":
            squared_error += F.mse_loss(prediction, acceptance, reduction="sum").item()
        predicted = prediction.sum(dim=1)
        observed = acceptance.sum(dim=1)
        absolute_total_error += torch.abs(predicted - observed).sum().item()
        squared_total_error += (predicted - observed).square().sum().item()
        predicted_totals.extend(predicted.cpu().tolist())
        observed_totals.extend(observed.cpu().tolist())
        value_count += acceptance.numel()

    correlation = float("nan")
    if len(records) > 1:
        correlation = float(np.corrcoef(predicted_totals, observed_totals)[0, 1])
    return {
        **(
            {"node_mse": squared_error / max(value_count, 1)}
            if model.config.acceptance_output == "node"
            else {}
        ),
        "total_acceptance_mse": squared_total_error / max(len(records), 1),
        "total_acceptance_rmse": (squared_total_error / max(len(records), 1)) ** 0.5,
        "total_acceptance_mae": absolute_total_error / max(len(records), 1),
        "total_acceptance_correlation": correlation,
    }


def acceptance_metrics(prediction, target):
    prediction, target = prediction.double(), target.double()
    p, y = prediction.sum(1), target.sum(1)
    error = p - y
    return {
        "conditions": len(y),
        "actual_mean": y.mean().item(),
        "predicted_mean": p.mean().item(),
        "total_mae": error.abs().mean().item(),
        "total_rmse": error.square().mean().sqrt().item(),
        "bias": error.mean().item(),
        **(
            {"node_mse": (prediction - target).square().mean().item()}
            if prediction.shape == target.shape
            else {}
        ),
        "pearson": torch.corrcoef(torch.stack([p, y]))[0, 1].item() if len(y) > 1 else None,
    }
