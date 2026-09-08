"""Official-architecture S2V-DQN influence-maximization baseline.

This native-PyTorch port retains the structure2vec message passing,
graph/action Q head, three propagation iterations, and pretrained weights
released in the ToupleGDD reference repository.
"""

from __future__ import annotations

import networkx as nx
import torch

from ._official_neural import (
    S2VDQNNetwork,
    graph_tensors,
    load_reference_weights,
    require_cuda,
)


@torch.inference_mode()
def s2v_dqn(graph: nx.DiGraph, budget: int, **_: object) -> list[int]:
    if budget < 1 or budget > graph.number_of_nodes():
        raise ValueError(f"invalid seed budget {budget}")
    device = require_cuda()
    nodes, edge_index, edge_weight = graph_tensors(graph, device)
    selected = torch.zeros(len(nodes), dtype=torch.float32, device=device)
    node_features = torch.ones((len(nodes), 2), device=device)
    node_features[:, 1] = 1.0 - selected
    source, target = edge_index
    edge_features = torch.ones((edge_index.shape[1], 4), device=device)
    edge_features[:, 0] = selected[source]
    edge_features[:, 1] = edge_weight
    edge_features[:, 2] = (selected[source] - selected[target]).abs()

    model = S2VDQNNetwork().to(device)
    load_reference_weights(model, "s2vdqn.ckpt")
    model.eval()
    q_values = model(node_features, edge_index, edge_features)
    order = torch.topk(q_values, budget).indices.cpu().tolist()
    return [nodes[index] for index in order]
