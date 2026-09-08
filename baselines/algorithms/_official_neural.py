"""CUDA ports of the neural architectures released with ToupleGDD.

The original implementation depends on PyTorch Geometric and torch-scatter.
This module preserves the released equations and parameter names while using
native PyTorch scatter operations, which keeps the baseline runnable with the
project's current CUDA stack.
"""

from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_DIR = Path(
    os.environ.get(
        "MOSAIC_BASELINE_CHECKPOINT_DIR", Path(__file__).resolve().parents[1] / "checkpoints"
    )
).expanduser()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("neural influence baselines require CUDA")
    return torch.device("cuda")


def graph_tensors(
    graph: nx.DiGraph, device: torch.device
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    nodes = sorted(int(node) for node in graph.nodes())
    index = {node: position for position, node in enumerate(nodes)}
    edges = sorted(
        (index[int(source)], index[int(target)], float(data.get("weight", 0.1)))
        for source, target, data in graph.edges(data=True)
    )
    edge_index = (
        torch.tensor(
            [(source, target) for source, target, _ in edges],
            dtype=torch.long,
            device=device,
        )
        .t()
        .contiguous()
    )
    edge_weight = torch.tensor(
        [weight for _, _, weight in edges], dtype=torch.float32, device=device
    )
    return nodes, edge_index, edge_weight


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output_shape = (size, *values.shape[1:])
    output = torch.zeros(output_shape, dtype=values.dtype, device=values.device)
    output.index_add_(0, index, values)
    return output


def scatter_softmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    maxima = torch.full((size,), -torch.inf, dtype=values.dtype, device=values.device)
    maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    exponent = torch.exp(values - maxima[index])
    denominator = torch.zeros(size, dtype=values.dtype, device=values.device)
    denominator.index_add_(0, index, exponent)
    return exponent / denominator[index].clamp_min(1e-12)


class S2VDQNNetwork(nn.Module):
    """Structure2vec Q-network from the ToupleGDD reference implementation."""

    def __init__(self, iterations: int = 3) -> None:
        super().__init__()
        self.iterations = iterations
        self.w_n2l = nn.Parameter(torch.empty(2, 64))
        self.w_e2l = nn.Parameter(torch.empty(4, 64))
        self.p_node_conv = nn.Parameter(torch.empty(64, 64))
        self.trans_node_1 = nn.Parameter(torch.empty(64, 64))
        self.trans_node_2 = nn.Parameter(torch.empty(64, 64))
        self.h1_weight = nn.Parameter(torch.empty(128, 32))
        self.h2_weight = nn.Parameter(torch.empty(32, 1))
        self.last_w = self.h2_weight

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        node_state = F.relu(node_features @ self.w_n2l)
        edge_state = edge_features @ self.w_e2l
        for _ in range(self.iterations):
            messages = F.relu((node_state @ self.p_node_conv)[source] + edge_state)
            aggregate = scatter_sum(messages, target, node_state.shape[0])
            node_state = F.relu(aggregate @ self.trans_node_1 + node_state @ self.trans_node_2)
        graph_state = node_state.sum(dim=0, keepdim=True).expand_as(node_state)
        state_action = torch.cat((node_state, graph_state), dim=-1)
        return (F.relu(state_action @ self.h1_weight) @ self.last_w).squeeze(-1)


class ToupleGDDNetwork(nn.Module):
    """The released Tripling state/source/target GNN and Q head."""

    def __init__(self) -> None:
        super().__init__()
        dimensions = [50, 50, 50, 50]
        self.trans_weights = nn.ParameterList()
        self.influgate_etas = nn.ParameterList()
        self.state_weights_self = nn.ParameterList()
        self.state_weights_neibor = nn.ParameterList()
        self.state_weights_attention = nn.ParameterList()
        self.state_weights_edge = nn.ParameterList()
        self.source_betas = nn.ParameterList()
        self.sourcegate_layer1s = nn.ModuleList()
        self.sourcegate_layer2s = nn.ModuleList()
        self.source_weights_self = nn.ParameterList()
        self.source_weights_neibor = nn.ParameterList()
        self.source_weights_state = nn.ParameterList()
        self.source_weights_attention = nn.ParameterList()
        self.source_weights_edge = nn.ParameterList()
        self.target_taus = nn.ParameterList()
        self.targetgate_layer1s = nn.ModuleList()
        self.targetgate_layer2s = nn.ModuleList()
        self.target_weights_self = nn.ParameterList()
        self.target_weights_neibor = nn.ParameterList()
        self.target_weights_state = nn.ParameterList()
        self.target_weights_attention = nn.ParameterList()
        self.target_weights_edge = nn.ParameterList()

        for layer in range(3):
            input_dim, output_dim = dimensions[layer : layer + 2]
            self.trans_weights.append(nn.Parameter(torch.empty(input_dim, output_dim)))
            self.influgate_etas.append(nn.Parameter(torch.empty(2 * output_dim, 1)))
            self.state_weights_self.append(nn.Parameter(torch.empty(1)))
            self.state_weights_neibor.append(nn.Parameter(torch.empty(1)))
            self.state_weights_attention.append(nn.Parameter(torch.empty(1)))
            self.state_weights_edge.append(nn.Parameter(torch.empty(1)))
            self.source_betas.append(nn.Parameter(torch.empty(2 * output_dim, 1)))
            self.sourcegate_layer1s.append(nn.Linear(input_dim, 128))
            self.sourcegate_layer2s.append(nn.Linear(128, output_dim))
            self.source_weights_self.append(nn.Parameter(torch.empty(1)))
            self.source_weights_neibor.append(nn.Parameter(torch.empty(1)))
            self.source_weights_state.append(nn.Parameter(torch.empty(1)))
            self.source_weights_attention.append(nn.Parameter(torch.empty(1)))
            self.source_weights_edge.append(nn.Parameter(torch.empty(1)))
            self.target_taus.append(nn.Parameter(torch.empty(2 * output_dim, 1)))
            self.targetgate_layer1s.append(nn.Linear(input_dim, 128))
            self.targetgate_layer2s.append(nn.Linear(128, output_dim))
            self.target_weights_self.append(nn.Parameter(torch.empty(1)))
            self.target_weights_neibor.append(nn.Parameter(torch.empty(1)))
            self.target_weights_state.append(nn.Parameter(torch.empty(1)))
            self.target_weights_attention.append(nn.Parameter(torch.empty(1)))
            self.target_weights_edge.append(nn.Parameter(torch.empty(1)))

        self.theta1 = nn.Parameter(torch.empty(150, 1))
        self.theta2 = nn.Parameter(torch.empty(50, 50))
        self.theta3 = nn.Parameter(torch.empty(50, 50))
        self.theta4 = nn.Parameter(torch.empty(50, 50))

    def forward(
        self,
        embeddings: torch.Tensor,
        selected: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        source_index, target_index = edge_index
        source_influence = embeddings[:, :50]
        target_influence = embeddings[:, 50:100]
        initial_selected = selected.float()
        state = initial_selected

        for layer in range(3):
            source_transformed = source_influence @ self.trans_weights[layer]
            target_transformed = target_influence @ self.trans_weights[layer]
            paired = torch.cat(
                (source_transformed[source_index], target_transformed[target_index]),
                dim=-1,
            )

            state_attention = scatter_softmax(
                F.leaky_relu((paired @ self.influgate_etas[layer]).squeeze(-1), 0.2),
                target_index,
                embeddings.shape[0],
            )
            state_attention = (
                state_attention * self.state_weights_attention[layer]
                + edge_weight * self.state_weights_edge[layer]
            )
            incoming_state = scatter_sum(
                state_attention * state[source_index],
                target_index,
                embeddings.shape[0],
            )
            next_state = torch.sigmoid(
                state * self.state_weights_self[layer]
                + incoming_state * self.state_weights_neibor[layer]
            )
            next_state = next_state * (1.0 - initial_selected) + initial_selected

            source_attention = scatter_softmax(
                F.leaky_relu((paired @ self.source_betas[layer]).squeeze(-1), 0.2),
                source_index,
                embeddings.shape[0],
            )
            source_attention = (
                source_attention * self.source_weights_attention[layer]
                + edge_weight * self.source_weights_edge[layer]
            )
            source_gate = F.leaky_relu(
                self.sourcegate_layer2s[layer](
                    F.leaky_relu(
                        self.sourcegate_layer1s[layer](target_influence[target_index]),
                        0.2,
                    )
                ),
                0.2,
            )
            outgoing = scatter_sum(
                source_attention.unsqueeze(-1) * source_gate,
                source_index,
                embeddings.shape[0],
            )
            next_source = F.leaky_relu(
                source_transformed * self.source_weights_self[layer]
                + outgoing * self.source_weights_neibor[layer]
                + state.unsqueeze(-1) * self.source_weights_state[layer],
                0.01,
            )

            target_attention = scatter_softmax(
                F.leaky_relu((paired @ self.target_taus[layer]).squeeze(-1), 0.2),
                target_index,
                embeddings.shape[0],
            )
            target_attention = (
                target_attention * self.target_weights_attention[layer]
                + edge_weight * self.target_weights_edge[layer]
            )
            target_gate = F.leaky_relu(
                self.targetgate_layer2s[layer](
                    F.leaky_relu(
                        self.targetgate_layer1s[layer](source_influence[source_index]),
                        0.2,
                    )
                ),
                0.2,
            )
            incoming = scatter_sum(
                target_attention.unsqueeze(-1) * target_gate,
                target_index,
                embeddings.shape[0],
            )
            next_target = F.leaky_relu(
                target_transformed * self.target_weights_self[layer]
                + incoming * self.target_weights_neibor[layer]
                + state.unsqueeze(-1) * self.target_weights_state[layer],
                0.01,
            )
            state, source_influence, target_influence = (
                next_state,
                next_source,
                next_target,
            )

        candidate_source = source_influence.clone()
        candidate_source[selected] = 0.0
        selected_source = source_influence.clone()
        selected_source[~selected] = 0.0
        selected_sum = selected_source.sum(dim=0, keepdim=True).expand_as(source_influence).clone()
        selected_sum[selected] = 0.0
        remaining_target = target_influence.clone()
        remaining_target[selected] = 0.0
        target_without_candidate = remaining_target.sum(dim=0, keepdim=True) - remaining_target
        q_features = torch.cat(
            (
                candidate_source @ self.theta2,
                selected_sum @ self.theta4,
                target_without_candidate @ self.theta3,
            ),
            dim=-1,
        )
        return (F.leaky_relu(q_features, 0.01) @ self.theta1).squeeze(-1)


def load_reference_weights(model: nn.Module, filename: str) -> None:
    checkpoint = CHECKPOINT_DIR / filename
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"missing official baseline checkpoint {checkpoint}; see baselines/README.md"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def directed_source_target_embeddings(
    graph: nx.DiGraph,
    nodes: list[int],
    device: torch.device,
    *,
    epochs: int = 30,
    walks_per_node: int = 50,
    random_seed: int = 123,
) -> torch.Tensor:
    """Train the directed DeepWalk-negative-sampling initializer in ToupleGDD."""

    node_index = {node: position for position, node in enumerate(nodes)}
    children = [[node_index[int(target)] for target in graph.successors(node)] for node in nodes]
    rng = random.Random(random_seed)
    contexts: list[list[int]] = []
    for start in range(len(nodes)):
        reachable = {start}
        frontier = deque([start])
        for _ in range(5):
            next_frontier: deque[int] = deque()
            while frontier:
                current = frontier.popleft()
                for target in children[current]:
                    if target not in reachable:
                        reachable.add(target)
                        next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break
        reachable_list = sorted(reachable)
        for _ in range(walks_per_node):
            current = start
            positive: list[int] = []
            for _ in range(2):
                if children[current] and rng.random() >= 0.15:
                    current = rng.choice(children[current])
                else:
                    current = start
                positive.append(current)
            positive.extend(rng.choices(reachable_list, k=5))
            contexts.extend((start, target) for target in positive)

    positive_pairs = torch.tensor(contexts, dtype=torch.long, device=device)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    embedding = nn.Embedding(len(nodes), 101, sparse=True, device=device)
    generator = torch.Generator(device=device).manual_seed(random_seed)
    optimizer = torch.optim.SparseAdam(embedding.parameters(), lr=0.01)
    batch_size = 65_536
    for _ in range(epochs):
        permutation = torch.randperm(positive_pairs.shape[0], device=device, generator=generator)
        for offset in range(0, positive_pairs.shape[0], batch_size):
            batch = positive_pairs[permutation[offset : offset + batch_size]]
            source = embedding(batch[:, 0])
            positive = embedding(batch[:, 1])
            positive_score = (
                source[:, 100] * (source[:, :50] * positive[:, 50:100]).sum(dim=-1)
                + positive[:, 100]
            )
            negative_index = torch.randint(
                len(nodes),
                (batch.shape[0], 5),
                device=device,
                generator=generator,
            )
            negative = embedding(negative_index)
            negative_score = (
                source[:, None, 100] * (source[:, None, :50] * negative[:, :, 50:100]).sum(dim=-1)
                + negative[:, :, 100]
            )
            loss = -F.logsigmoid(positive_score).mean() - F.logsigmoid(-negative_score).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return embedding.weight.detach()
