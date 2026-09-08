"""Validation shared by replay writers and behavioral training readers."""

import numpy as np


def validate_replay(trajectory, similarity, *, rounds=30, num_nodes=None, source="replay"):
    x, sim = np.asarray(trajectory), np.asarray(similarity)
    if x.ndim != 2 or x.shape != sim.shape or x.shape[0] != rounds + 1 or x.shape[1] < 1:
        raise ValueError(f"{source}: expected matching ({rounds + 1}, nodes) X and sim arrays")
    if num_nodes is not None and x.shape[1] != num_nodes:
        raise ValueError(f"{source}: node count differs from the configured graph")
    if not np.isfinite(x).all() or not np.isfinite(sim).all():
        raise ValueError(f"{source}: replay arrays must contain finite values")
    if not np.isin(x, [0, 1]).all():
        raise ValueError(f"{source}: watch indicators must be binary")
    if np.any(np.abs(sim) > 1.00001):
        raise ValueError(f"{source}: cosine similarities must lie in [-1, 1]")
    return x.astype(bool), sim
