"""Official-architecture ToupleGDD influence-maximization baseline."""

from __future__ import annotations

import networkx as nx
import torch

from ._official_neural import (
    ToupleGDDNetwork,
    directed_source_target_embeddings,
    graph_tensors,
    load_reference_weights,
    require_cuda,
)


def touplegdd(
    graph: nx.DiGraph,
    budget: int,
    *,
    embedding_epochs: int = 30,
    random_seed: int = 123,
    **_: object,
) -> list[int]:
    if budget < 1 or budget > graph.number_of_nodes():
        raise ValueError(f"invalid seed budget {budget}")
    device = require_cuda()
    nodes, edge_index, edge_weight = graph_tensors(graph, device)
    embeddings = directed_source_target_embeddings(
        graph,
        nodes,
        device,
        epochs=embedding_epochs,
        random_seed=random_seed,
    )
    selected = torch.zeros(len(nodes), dtype=torch.bool, device=device)
    # Q scores differ by less than float32 accumulation error on dense graphs.
    # Float64 preserves the ordering of the released equations and checkpoint.
    model = ToupleGDDNetwork().to(device=device, dtype=torch.float64)
    load_reference_weights(model, "touplegdd.ckpt")
    model.eval()
    with torch.inference_mode():
        q_values = model(embeddings.double(), selected, edge_index, edge_weight.double())
        order = torch.topk(q_values, budget).indices.cpu().tolist()
    return [nodes[index] for index in order]
