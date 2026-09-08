"""GCOMB: GraphSAGE candidate pruning followed by Q-learning seed selection."""

from __future__ import annotations

import random
from collections import deque, namedtuple

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F

from ._official_neural import graph_tensors, require_cuda


class _RRSetOracle:
    def __init__(
        self,
        graph: nx.DiGraph,
        nodes: list[int],
        device: torch.device,
        samples: int,
        random_seed: int,
    ) -> None:
        index = {node: position for position, node in enumerate(nodes)}
        parents: list[list[tuple[int, float]]] = [[] for _ in nodes]
        for source, target, data in graph.edges(data=True):
            parents[index[int(target)]].append((index[int(source)], float(data.get("weight", 0.1))))
        rng = random.Random(random_seed)
        incidence = torch.zeros((samples, len(nodes)), dtype=torch.bool)
        for row in range(samples):
            root = rng.randrange(len(nodes))
            reached = {root}
            queue = deque([root])
            while queue:
                target = queue.popleft()
                for source, probability in parents[target]:
                    if source not in reached and rng.random() <= probability:
                        reached.add(source)
                        queue.append(source)
            incidence[row, list(reached)] = True
        self.incidence = incidence.to(device)
        self.num_nodes = len(nodes)

    def singleton_scores(self) -> torch.Tensor:
        return self.num_nodes * self.incidence.float().mean(dim=0)

    def spread(self, selected: torch.Tensor) -> torch.Tensor:
        if not selected.any():
            return torch.zeros((), device=self.incidence.device)
        covered = self.incidence[:, selected].any(dim=1)
        return self.num_nodes * covered.float().mean()

    def marginal_gains(self, selected: torch.Tensor) -> torch.Tensor:
        covered = (
            self.incidence[:, selected].any(dim=1)
            if selected.any()
            else torch.zeros(
                self.incidence.shape[0],
                dtype=torch.bool,
                device=self.incidence.device,
            )
        )
        gains = self.incidence[~covered].float().sum(dim=0)
        gains *= self.num_nodes / self.incidence.shape[0]
        gains[selected] = -torch.inf
        return gains


class _GraphSAGEPruner(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.self_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)]
        )
        self.neighbor_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)]
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = features
        for self_layer, neighbor_layer in zip(self.self_layers, self.neighbor_layers):
            neighbor_sum = torch.sparse.mm(adjacency, hidden)
            degrees = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1.0)
            neighbor_mean = neighbor_sum / degrees.unsqueeze(-1)
            hidden = F.relu(self_layer(hidden) + neighbor_layer(neighbor_mean))
        return hidden

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(features, adjacency)
        return self.score(embeddings).squeeze(-1), embeddings


class _GCOMBQNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.selected = nn.Linear(embedding_dim, embedding_dim)
        self.remaining = nn.Linear(embedding_dim, embedding_dim)
        self.candidate = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(3 * embedding_dim, 1)

    def forward(
        self,
        selected_mean: torch.Tensor,
        remaining_mean: torch.Tensor,
        candidate_embedding: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                self.selected(selected_mean),
                self.remaining(remaining_mean),
                self.candidate(candidate_embedding),
            ),
            dim=-1,
        )
        return self.output(F.relu(combined)).squeeze(-1)


Transition = namedtuple("Transition", ("selected", "action", "reward", "next_selected", "done"))


def _structural_features(graph: nx.DiGraph, nodes: list[int], device: torch.device) -> torch.Tensor:
    values = []
    for node in nodes:
        out_edges = list(graph.out_edges(node, data=True))
        in_edges = list(graph.in_edges(node, data=True))
        values.append(
            (
                len(out_edges),
                len(in_edges),
                sum(float(data.get("weight", 0.1)) for _, _, data in out_edges),
                sum(float(data.get("weight", 0.1)) for _, _, data in in_edges),
            )
        )
    features = torch.tensor(values, dtype=torch.float32, device=device)
    maximum = features.max(dim=0).values.clamp_min(1.0)
    return features / maximum


def _sparse_adjacency(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    reverse = edge_index.flip(0)
    indices = torch.cat((edge_index, reverse), dim=1)
    values = torch.cat((edge_weight, edge_weight))
    adjacency = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), device=edge_weight.device
    ).coalesce()
    return adjacency


def _state_means(
    embeddings: torch.Tensor, states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = states.float()
    remaining = 1.0 - selected
    selected_mean = (selected @ embeddings) / selected.sum(dim=1, keepdim=True).clamp_min(1.0)
    remaining_mean = (remaining @ embeddings) / remaining.sum(dim=1, keepdim=True).clamp_min(1.0)
    return selected_mean, remaining_mean


def gcomb(
    graph: nx.DiGraph,
    budget: int,
    *,
    rr_samples: int = 10_000,
    pruning_epochs: int = 250,
    episodes: int = 400,
    candidate_fraction: float = 0.1,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.8,
    random_seed: int = 2026,
    **_: object,
) -> list[int]:
    """Run the two stages of GCOMB on the supplied IC graph."""

    if budget < 1 or budget > graph.number_of_nodes():
        raise ValueError(f"invalid seed budget {budget}")
    device = require_cuda()
    nodes, edge_index, edge_weight = graph_tensors(graph, device)
    num_nodes = len(nodes)
    oracle = _RRSetOracle(graph, nodes, device, samples=rr_samples, random_seed=random_seed)
    adjacency = _sparse_adjacency(edge_index, edge_weight, num_nodes)
    features = _structural_features(graph, nodes, device)
    labels = oracle.singleton_scores()
    labels = labels / labels.max().clamp_min(1.0)

    pruner = _GraphSAGEPruner(features.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.Adam(pruner.parameters(), lr=learning_rate)
    for _ in range(pruning_epochs):
        prediction, _ = pruner(features, adjacency)
        loss = F.mse_loss(prediction, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    pruner.eval()
    with torch.no_grad():
        predicted_gain, embeddings = pruner(features, adjacency)
    candidate_count = min(num_nodes, max(budget, int(round(candidate_fraction * num_nodes))))
    candidates = torch.topk(predicted_gain, candidate_count).indices
    candidate_embeddings = embeddings[candidates].detach()

    q_network = _GCOMBQNetwork(hidden_dim).to(device)
    target_network = _GCOMBQNetwork(hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    replay: deque[Transition] = deque(maxlen=20_000)
    rng = random.Random(random_seed)

    def train_batch(batch_size: int = 64) -> None:
        if len(replay) < batch_size:
            return
        batch = rng.sample(replay, batch_size)
        states = torch.stack([transition.selected for transition in batch])
        actions = torch.tensor([transition.action for transition in batch], device=device)
        rewards = torch.tensor([transition.reward for transition in batch], device=device)
        next_states = torch.stack([transition.next_selected for transition in batch])
        dones = torch.tensor([transition.done for transition in batch], device=device)
        selected_mean, remaining_mean = _state_means(candidate_embeddings, states)
        q_values = q_network(selected_mean, remaining_mean, candidate_embeddings[actions])
        with torch.no_grad():
            next_selected_mean, next_remaining_mean = _state_means(
                candidate_embeddings, next_states
            )
            batch_count = next_states.shape[0]
            all_q = target_network(
                next_selected_mean[:, None, :].expand(-1, candidate_count, -1),
                next_remaining_mean[:, None, :].expand(-1, candidate_count, -1),
                candidate_embeddings[None, :, :].expand(batch_count, -1, -1),
            )
            all_q[next_states] = -torch.inf
            next_q = all_q.max(dim=1).values
            next_q = torch.where(dones, torch.zeros_like(next_q), next_q)
            targets = rewards + gamma * next_q
        loss = F.mse_loss(q_values, targets)
        q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), 5.0)
        q_optimizer.step()

    for episode in range(episodes):
        state = torch.zeros(candidate_count, dtype=torch.bool, device=device)
        full_selected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        previous_spread = 0.0
        epsilon = max(0.05, 1.0 - episode / max(episodes * 0.8, 1.0))
        for _ in range(budget):
            available = (~state).nonzero(as_tuple=False).flatten()
            if rng.random() < epsilon:
                action = int(available[rng.randrange(len(available))])
            else:
                with torch.no_grad():
                    selected_mean, remaining_mean = _state_means(
                        candidate_embeddings, state.unsqueeze(0)
                    )
                    q_values = q_network(
                        selected_mean.expand(len(available), -1),
                        remaining_mean.expand(len(available), -1),
                        candidate_embeddings[available],
                    )
                    action = int(available[q_values.argmax()])
            next_state = state.clone()
            next_state[action] = True
            full_selected[candidates[action]] = True
            spread = float(oracle.spread(full_selected))
            reward = (spread - previous_spread) / num_nodes
            done = int(next_state.sum()) == budget
            replay.append(Transition(state.clone(), action, reward, next_state.clone(), done))
            state = next_state
            previous_spread = spread
            train_batch()
        if (episode + 1) % 25 == 0:
            target_network.load_state_dict(q_network.state_dict())

    q_network.eval()
    state = torch.zeros(candidate_count, dtype=torch.bool, device=device)
    order: list[int] = []
    with torch.inference_mode():
        for _ in range(budget):
            available = (~state).nonzero(as_tuple=False).flatten()
            selected_mean, remaining_mean = _state_means(candidate_embeddings, state.unsqueeze(0))
            q_values = q_network(
                selected_mean.expand(len(available), -1),
                remaining_mean.expand(len(available), -1),
                candidate_embeddings[available],
            )
            action = int(available[q_values.argmax()])
            state[action] = True
            order.append(nodes[int(candidates[action])])
    return order
