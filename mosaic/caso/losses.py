"""Acceptance objectives for node prediction and total-count supervision."""

import torch
import torch.nn.functional as F


def total_acceptance_loss(
    prediction: torch.Tensor, target: torch.Tensor, weight: float = 100.0
) -> torch.Tensor:
    """Scalar count regression; target contains the retained node acceptance labels."""
    if target.ndim != 2 or prediction.shape != (target.shape[0], 1):
        raise ValueError("scalar predictions must have shape [conditions, 1]")
    if target.numel() == 0 or weight <= 0:
        raise ValueError("target must be nonempty and weight positive")
    return weight * F.mse_loss(prediction[:, 0] / target.shape[1], target.mean(1))


def acceptance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    objective: str = "joint",
    total_weight: float = 100.0,
) -> torch.Tensor:
    """Supervise each condition's count, optionally retaining node supervision.

    Total counts are divided by the graph size before computing MSE, keeping
    the scale independent of N. At fixed N this is count MSE divided by N².
    ``total_weight`` controls its scale relative to the node-level objective.
    A total-only objective does not identify individual node probabilities.
    """
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be matching [conditions, nodes] arrays")
    if prediction.shape[0] == 0 or prediction.shape[1] == 0:
        raise ValueError("acceptance arrays must be nonempty")
    if total_weight <= 0:
        raise ValueError("total_weight must be positive")
    if objective == "node":
        return F.mse_loss(prediction, target)
    total_mse = F.mse_loss(prediction.mean(dim=1), target.mean(dim=1))
    if objective == "total":
        return total_weight * total_mse
    if objective == "joint":
        return F.mse_loss(prediction, target) + total_weight * total_mse
    raise ValueError(f"unknown acceptance objective: {objective}")
