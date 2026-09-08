"""Label-free item-aware seed selection from content fit and weighted degree."""

from __future__ import annotations

import networkx as nx
import numpy as np


def _minmax(values: np.ndarray) -> np.ndarray:
    lower = float(values.min())
    span = float(values.max()) - lower
    if span <= 0:
        return np.ones_like(values)
    return (values - lower) / span


def content_match_degree(
    graph: nx.DiGraph,
    budget: int,
    content_match: np.ndarray,
) -> list[int]:
    """Rank users by normalized content match times log weighted out-degree."""

    nodes = sorted(int(node) for node in graph.nodes())
    if content_match.ndim != 1 or content_match.shape[0] <= max(nodes):
        raise ValueError(f"content-match vector cannot index graph nodes: {content_match.shape}")
    match = np.asarray([content_match[node] for node in nodes], dtype=np.float64)
    weighted_degree = np.asarray(
        [
            sum(float(data.get("weight", 0.1)) for _, _, data in graph.out_edges(node, data=True))
            for node in nodes
        ],
        dtype=np.float64,
    )
    scores = _minmax(match) * _minmax(np.log1p(weighted_degree))
    order = np.lexsort((np.asarray(nodes), -scores))[:budget]
    return [nodes[index] for index in order]
