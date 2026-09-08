"""Exact-budget CASO latent, greedy, and swap search."""

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.optim import Adam

from mosaic.config import SearchConfig

from .graph import repeat_sparse_graph
from .models import CASOModel


@dataclass(frozen=True)
class SearchResult:
    method: str
    budget: int
    seeds: tuple[int, ...]
    predicted_acceptance: float
    candidate_source: str
    elapsed_seconds: float


class SurrogateEvaluator:
    def __init__(
        self,
        model: CASOModel,
        adjacency: torch.Tensor,
        positional_encoding: torch.Tensor,
        content_match: torch.Tensor,
        batch_size: int,
    ):
        self.model = model
        self.adjacency = adjacency
        self.position = positional_encoding
        self.content_match = content_match
        self.batch_size = batch_size
        self.num_nodes = model.config.num_nodes

    @torch.no_grad()
    def evaluate(self, seed_sets: Iterable[Iterable[int]]) -> list[float]:
        sets = [tuple(seed_set) for seed_set in seed_sets]
        values: list[float] = []
        for offset in range(0, len(sets), self.batch_size):
            chunk = sets[offset : offset + self.batch_size]
            count = len(chunk)
            seed_array = np.zeros((count, self.num_nodes), dtype=np.float32)
            for row, nodes in enumerate(chunk):
                seed_array[row, list(nodes)] = 1.0
            seed = torch.from_numpy(seed_array).to(self.content_match.device)
            relaxed_seed = self.model.seed_flow.reconstruct_discrete(seed)
            match = self.content_match.expand(count, -1)
            adjacency = repeat_sparse_graph(self.adjacency, count)
            position = self.position.repeat(count, 1)
            prediction = self.model.predict_total(relaxed_seed, match, adjacency, position)
            values.extend(prediction.tolist())
        return values


class CASOSearcher:
    def __init__(
        self,
        model: CASOModel,
        adjacency: torch.Tensor,
        positional_encoding: torch.Tensor,
        content_match: torch.Tensor,
        config: SearchConfig,
    ):
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.adjacency = adjacency
        self.position = positional_encoding
        self.content_match = content_match.reshape(1, -1)
        self.config = config
        self.num_nodes = model.config.num_nodes
        self.evaluator = SurrogateEvaluator(
            model,
            adjacency,
            positional_encoding,
            self.content_match,
            config.evaluation_batch_size,
        )

    def greedy_prefix_profile(self, maximum_budget: int) -> tuple[list[int], dict[int, float]]:
        selected: list[int] = []
        elapsed_by_budget: dict[int, float] = {}
        start = time.perf_counter()
        for _ in range(maximum_budget):
            candidates = [node for node in range(self.num_nodes) if node not in selected]
            scores = self.evaluator.evaluate(selected + [node] for node in candidates)
            selected.append(candidates[int(np.argmax(scores))])
            elapsed_by_budget[len(selected)] = time.perf_counter() - start
        return selected, elapsed_by_budget

    def greedy_prefix(self, maximum_budget: int) -> list[int]:
        return self.greedy_prefix_profile(maximum_budget)[0]

    def swap_refine(self, seeds: Iterable[int]) -> tuple[list[int], float]:
        selected = list(seeds)
        current = self.evaluator.evaluate([selected])[0]
        for _ in range(self.config.max_swap_passes):
            improved = False
            for position in range(len(selected)):
                candidates = [node for node in range(self.num_nodes) if node not in selected]
                trials = []
                for node in candidates:
                    trial = list(selected)
                    trial[position] = node
                    trials.append(trial)
                scores = self.evaluator.evaluate(trials)
                best = int(np.argmax(scores))
                if scores[best] > current + 1e-8:
                    selected = trials[best]
                    current = scores[best]
                    improved = True
            if not improved:
                break
        return selected, current

    def _degree_warm_start(self, budget: int) -> torch.Tensor:
        dense_degree = torch.sparse.sum(self.adjacency, dim=1).to_dense()
        nodes = dense_degree.topk(budget).indices
        seed = torch.zeros(1, self.num_nodes, device=self.content_match.device)
        seed[0, nodes] = 1.0
        with torch.no_grad():
            return self.model.seed_flow.encode(seed)

    def latent_candidate(self, budget: int) -> list[int]:
        generator = torch.Generator(device=self.content_match.device)
        generator.manual_seed(self.config.random_seed + budget)
        restart_count = self.config.random_restarts
        initial = self._degree_warm_start(budget)
        latent = initial.expand(restart_count, -1).clone()
        if restart_count > 1:
            latent[1:] += 0.1 * torch.randn(
                (restart_count - 1, self.num_nodes),
                generator=generator,
                device=latent.device,
                dtype=latent.dtype,
            )
        latent.requires_grad_(True)
        optimizer = Adam([latent], lr=self.config.latent_learning_rate)
        adjacency = repeat_sparse_graph(self.adjacency, restart_count)
        position = self.position.repeat(restart_count, 1)
        content_match = self.content_match.expand(restart_count, -1)

        # Restarts are independent rows in one tensor. Adam keeps separate
        # element-wise moments while the batched graph pass uses the GPU fully.
        previous_sets = None
        stable_steps = 0
        self.last_latent_iterations = 0
        for iteration in range(self.config.latent_iterations):
            original_steps = self.model.seed_flow.steps
            try:
                if self.config.latent_flow_steps:
                    self.model.seed_flow.steps = self.config.latent_flow_steps
                relaxed = self.model.seed_flow.decode(latent, budget=budget)
            finally:
                self.model.seed_flow.steps = original_steps
            prediction = self.model.predict_total(
                relaxed,
                content_match,
                adjacency,
                position,
            )
            relaxed_count = relaxed.sum(dim=1)
            if self.model.config.acceptance_output == "total":
                width = self.model.seed_flow.dequantization_width
                relaxed_count = (relaxed_count - self.num_nodes * width / 2) / (1 - width)
            excess = torch.relu(relaxed_count - budget)
            objective_by_restart = -prediction + self.config.budget_penalty * excess.square()
            optimizer.zero_grad(set_to_none=True)
            objective_by_restart.sum().backward()
            optimizer.step()
            self.last_latent_iterations = iteration + 1
            if self.config.latent_stability_patience:
                current_sets = relaxed.detach().topk(budget, dim=1).indices.sort(1).values
                stable_steps = (
                    stable_steps + 1
                    if previous_sets is not None and torch.equal(current_sets, previous_sets)
                    else 0
                )
                previous_sets = current_sets
                if stable_steps >= self.config.latent_stability_patience:
                    break

        with torch.no_grad():
            relaxed = self.model.seed_flow.decode(latent, budget=budget)
            candidates = relaxed.topk(budget, dim=1).indices.tolist()
        scores = self.evaluator.evaluate(candidates)
        return candidates[int(np.argmax(scores))]

    def search_budget(
        self,
        budget: int,
        greedy_prefix: list[int] | None = None,
        greedy_seconds: float | None = None,
    ) -> list[SearchResult]:
        if budget < 1 or budget > self.num_nodes:
            raise ValueError(f"budget must be in [1, {self.num_nodes}]")
        if greedy_prefix is None or len(greedy_prefix) < budget:
            greedy_prefix, profile = self.greedy_prefix_profile(budget)
            greedy_seconds = profile[budget]
        if greedy_seconds is None:
            raise ValueError("greedy_seconds is required with a precomputed prefix")
        greedy = greedy_prefix[:budget]
        caso_start = time.perf_counter()
        greedy_swap, _ = self.swap_refine(greedy)

        latent = self.latent_candidate(budget)
        latent_swap, _ = self.swap_refine(latent)

        # CUDA sparse reductions can differ slightly between separate calls.
        # Score the matched greedy policy and the complete CASO portfolio in
        # one batch so the final comparison uses an identical numeric pass.
        portfolio = [greedy, greedy_swap, latent_swap]
        final_scores = self.evaluator.evaluate(portfolio)
        greedy_score = final_scores[0]
        caso_index = int(np.argmax(final_scores))
        caso_seeds = portfolio[caso_index]
        caso_score = final_scores[caso_index]
        caso_source = "latent+swap" if caso_index == 2 else "greedy+swap"
        caso_seconds = greedy_seconds + (time.perf_counter() - caso_start)

        return [
            SearchResult(
                "DualBranch-Greedy",
                budget,
                tuple(sorted(greedy)),
                greedy_score,
                "pure-greedy",
                greedy_seconds,
            ),
            SearchResult(
                "CASO",
                budget,
                tuple(sorted(caso_seeds)),
                caso_score,
                caso_source,
                caso_seconds,
            ),
        ]
