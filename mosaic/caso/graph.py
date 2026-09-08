"""Graph preparation shared by CASO training and search."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# Orientation used by the released CASO checkpoint. It is recovered exactly
# from the checkpoint's archived validation loss and turns the otherwise
# sign-ambiguous eigenspace into a versioned graph feature contract.
CASO_POSITIONAL_ORIENTATION = np.asarray(
    [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float64
)


@dataclass(frozen=True)
class GraphContext:
    adjacency: sp.coo_matrix
    normalized_adjacency: sp.coo_matrix
    positional_encoding: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.adjacency.shape[0]


def load_directed_graph(path: Path, num_nodes: int | None = None) -> sp.coo_matrix:
    frame = pd.read_csv(path)
    required = {"user_1", "user_2", "closeness"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"relationship file lacks columns: {sorted(missing)}")
    if num_nodes is None:
        num_nodes = int(max(frame["user_1"].max(), frame["user_2"].max())) + 1
    return sp.coo_matrix(
        (
            frame["closeness"].to_numpy(dtype=np.float32),
            (
                frame["user_1"].to_numpy(dtype=np.int64),
                frame["user_2"].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )


def symmetric_normalize(adjacency: sp.spmatrix) -> sp.coo_matrix:
    adjacency = adjacency.tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 0
    inverse_sqrt[positive] = np.power(degree[positive], -0.5)
    scale = sp.diags(inverse_sqrt)
    return (scale @ adjacency @ scale).tocoo()


def laplacian_positional_encoding(
    directed_adjacency: sp.spmatrix,
    dimensions: int,
    random_seed: int = 2026,
) -> torch.Tensor:
    """Compute a reproducible PE on the graph's undirected projection.

    SciPy's ARPACK wrapper draws a random starting vector when ``v0`` is not
    supplied. CASO training seeds NumPy before graph preparation; reproducing
    that seed locally keeps checkpoint inference invariant across processes
    without changing the caller's NumPy random stream.
    """

    undirected = directed_adjacency.maximum(directed_adjacency.T).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    normalized = symmetric_normalize(undirected)
    laplacian = sp.eye(undirected.shape[0], dtype=np.float64) - normalized
    state = np.random.get_state()
    try:
        np.random.seed(random_seed)
        _, vectors = sp.linalg.eigsh(laplacian, k=dimensions + 1, which="SM")
    finally:
        np.random.set_state(state)
    vectors = vectors[:, 1 : dimensions + 1]
    # Eigenvectors are defined only up to sign. Use the largest-magnitude
    # coordinate as an orientation anchor so repeated decompositions agree.
    anchors = np.abs(vectors).argmax(axis=0)
    signs = np.sign(vectors[anchors, np.arange(dimensions)])
    signs[signs == 0] = 1.0
    vectors = vectors * signs
    if dimensions == len(CASO_POSITIONAL_ORIENTATION):
        vectors = vectors * CASO_POSITIONAL_ORIENTATION
    return torch.tensor(vectors, dtype=torch.float32)


def prepare_graph(
    path: Path,
    num_nodes: int,
    positional_dim: int,
    random_seed: int = 2026,
) -> GraphContext:
    directed = load_directed_graph(path, num_nodes=num_nodes)
    with_self_loops = (directed + sp.eye(num_nodes, dtype=np.float32)).tocoo()
    normalized = symmetric_normalize(with_self_loops)
    positional = laplacian_positional_encoding(directed, positional_dim, random_seed=random_seed)
    return GraphContext(directed, normalized, positional)


def scipy_to_torch_sparse(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    matrix = matrix.tocoo()
    indices = torch.tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long, device=device)
    values = torch.tensor(matrix.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device).coalesce()


def repeat_sparse_graph(adjacency: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size == 1:
        return adjacency
    node_count = adjacency.shape[0]
    indices = adjacency.indices()
    offsets = torch.arange(batch_size, device=indices.device) * node_count
    repeated = indices.unsqueeze(0) + offsets[:, None, None]
    repeated = repeated.permute(1, 0, 2).reshape(2, -1)
    values = adjacency.values().repeat(batch_size)
    shape = (batch_size * node_count, batch_size * node_count)
    # Ordered disjoint copies of a coalesced graph remain sorted and unique.
    return torch.sparse_coo_tensor(repeated, values, shape, is_coalesced=True)
