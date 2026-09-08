"""Paper-aligned CASO seed relaxation and acceptance predictor."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from mosaic.config import ModelConfig

from .seed_sampling import rank_couple_endpoint


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimensions // 2
        frequencies = torch.exp(torch.linspace(0, -math.log(10_000.0), half, device=time.device))
        arguments = time[:, None] * frequencies[None, :]
        embedding = torch.cat([arguments.sin(), arguments.cos()], dim=-1)
        return F.pad(embedding, (0, self.dimensions - embedding.shape[-1]))


class VelocityField(nn.Module):
    def __init__(self, dimensions: int, hidden_dim: int, time_dim: int):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(dimensions + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimensions),
        )

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, self.time_embedding(time)], dim=-1))


class EquivariantVelocityField(nn.Module):
    """Shared coordinate dynamics with permutation-invariant set context.

    Uniform random seed subsets have no privileged node ordering. A shared
    coordinate path also avoids forcing a full-dimensional velocity through
    the original dense model's narrower global bottleneck.
    """

    def __init__(self, hidden_dim: int, context_topk: int = 0, budget_conditioned: bool = False):
        super().__init__()
        self.context_topk = context_topk
        self.budget_conditioned = budget_conditioned
        if context_topk < 0:
            raise ValueError("context top-k must be nonnegative")
        self.local = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + 4 + context_topk + int(budget_conditioned), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, time: torch.Tensor, budget: torch.Tensor | None = None
    ) -> torch.Tensor:
        t = time[:, None].expand_as(state)
        features = torch.stack(
            [
                state,
                torch.tanh(state),
                state.square() / (1 + state.square()),
                t,
                t.square(),
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
                state * t,
            ],
            dim=-1,
        )
        hidden = self.local(features)
        context = torch.stack(
            [
                state.mean(1),
                state.square().mean(1).clamp_min(1e-8).sqrt(),
                torch.sigmoid(4 * state).mean(1),
                torch.logsumexp(4 * state, dim=1) / 4 - math.log(state.shape[1]) / 4,
            ],
            dim=-1,
        )
        if self.context_topk:
            top = state.topk(min(self.context_topk, state.shape[1]), dim=1).values
            top = F.pad(top, (0, self.context_topk - top.shape[1]))
            context = torch.cat([context, top], dim=-1)
        if self.budget_conditioned:
            if budget is None:
                raise ValueError("budget-conditioned velocity requires the requested budget")
            context = torch.cat([context, budget[:, None].to(state.dtype) / 10], dim=-1)
        context = context[:, None, :].expand(-1, state.shape[1], -1)
        return self.output(torch.cat([hidden, context], dim=-1)).squeeze(-1)


class RectifiedFlowSeedModel(nn.Module):
    """Dimension-preserving conditional-flow-matching seed relaxation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.steps = config.flow_steps
        self.logit_epsilon = config.logit_epsilon
        self.solver = config.flow_solver
        self.budget_conditioned = config.flow_budget_conditioned
        self.dequantization_width = config.flow_dequantization_width
        if not 0 < self.dequantization_width <= 0.5:
            raise ValueError("dequantization width must be in (0, 0.5]")
        if self.steps < 1 or self.solver not in {"euler", "rk4"}:
            raise ValueError("flow requires positive integration steps and an euler or rk4 solver")
        if config.flow_velocity_variant == "mlp":
            if self.budget_conditioned:
                raise ValueError("budget conditioning is available for the equivariant velocity")
            self.velocity = VelocityField(
                config.num_nodes, config.flow_hidden_dim, config.flow_time_dim
            )
        elif config.flow_velocity_variant == "equivariant":
            self.velocity = EquivariantVelocityField(
                config.flow_shared_hidden_dim,
                config.flow_context_topk,
                config.flow_budget_conditioned,
            )
        else:
            raise ValueError(f"unknown flow velocity variant: {config.flow_velocity_variant}")

    def logit_dequantize(
        self,
        seed: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        use_jitter = self.training if stochastic is None else stochastic
        if use_jitter:
            jitter = torch.rand(
                seed.shape, dtype=seed.dtype, device=seed.device, generator=generator
            )
        else:
            jitter = torch.full_like(seed, 0.5)
        unit = ((1 - self.dequantization_width) * seed + self.dequantization_width * jitter).clamp(
            self.logit_epsilon, 1.0 - self.logit_epsilon
        )
        return torch.log(unit) - torch.log1p(-unit)

    def velocity_at(
        self, state: torch.Tensor, time: torch.Tensor, budget: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        if not self.budget_conditioned:
            return self.velocity(state, time)
        if budget is None:
            raise ValueError("budget-conditioned flow requires a budget for decoding")
        budget = torch.as_tensor(budget, device=state.device, dtype=state.dtype)
        if budget.ndim == 0:
            budget = budget.expand(state.shape[0])
        if budget.shape != (state.shape[0],):
            raise ValueError("budget must be a scalar or one value per batch row")
        return self.velocity(state, time, budget)

    def integrate(
        self, initial: torch.Tensor, reverse: bool = False, budget: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        state = initial
        if self.budget_conditioned and budget is not None:
            # Convert once: constructing a CUDA scalar from a Python budget at
            # every RK substep otherwise synchronizes host and device repeatedly.
            budget = torch.as_tensor(budget, device=initial.device, dtype=initial.dtype)
            if budget.ndim == 0:
                budget = budget.expand(initial.shape[0])
        step = 1.0 / self.steps
        for index in range(self.steps):
            value = 1.0 - index * step if reverse else index * step
            time = torch.full((state.shape[0],), value, device=state.device, dtype=state.dtype)
            direction = -step if reverse else step
            k1 = self.velocity_at(state, time, budget)
            if self.solver == "euler":
                state = state + direction * k1
            else:
                k2 = self.velocity_at(state + direction * k1 / 2, time + direction / 2, budget)
                k3 = self.velocity_at(state + direction * k2 / 2, time + direction / 2, budget)
                k4 = self.velocity_at(state + direction * k3, time + direction, budget)
                state = state + direction * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state

    def encode(
        self,
        seed: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.integrate(
            self.logit_dequantize(seed, stochastic=stochastic, generator=generator),
            reverse=True,
            budget=seed.sum(1),
        )

    def decode(
        self, latent: torch.Tensor, budget: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        return torch.sigmoid(self.integrate(latent, reverse=False, budget=budget))

    def reconstruct(
        self,
        seed: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.decode(
            self.encode(seed, stochastic=stochastic, generator=generator), budget=seed.sum(1)
        )

    @torch.no_grad()
    def reconstruct_discrete(self, seed: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction, caching permutation-equivalent binary inputs.

        A shared equivariant velocity gives identical coordinates within each
        seed/non-seed class at the midpoint. Only one representative per budget
        needs ODE integration. Cache invalidation tracks parameter versions.
        Stochastic reconstruction and latent optimization always use the ODE.
        """
        if not isinstance(self.velocity, EquivariantVelocityField):
            return self.reconstruct(seed, stochastic=False)
        if not torch.all((seed == 0) | (seed == 1)):
            raise ValueError("discrete reconstruction requires binary seed masks")
        key = (
            seed.device,
            seed.dtype,
            self.steps,
            self.solver,
            tuple(p._version for p in self.parameters()),
        )
        if getattr(self, "_midpoint_cache_key", None) != key:
            self._midpoint_cache = {}
            self._midpoint_cache_key = key
        budgets = seed.sum(1).long()
        missing = [int(k) for k in budgets.unique().tolist() if int(k) not in self._midpoint_cache]
        if missing:
            reference = torch.zeros(
                len(missing), self.num_nodes, device=seed.device, dtype=seed.dtype
            )
            for row, k in enumerate(missing):
                reference[row, :k] = 1
            rebuilt = self.reconstruct(reference, stochastic=False)
            for row, k in enumerate(missing):
                self._midpoint_cache[k] = torch.stack(
                    [
                        rebuilt[row, k] if k < self.num_nodes else rebuilt[row, 0],
                        rebuilt[row, 0] if k else rebuilt[row, -1],
                    ]
                )
        endpoints = torch.stack([self._midpoint_cache[int(k)] for k in budgets.tolist()])
        return endpoints[:, :1] + seed * (endpoints[:, 1:] - endpoints[:, :1])

    def flow_matching_loss(
        self,
        seed: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
        coupling: str = "independent",
        tail_weight: float = 0.0,
    ) -> torch.Tensor:
        endpoint = self.logit_dequantize(seed, stochastic=stochastic, generator=generator)
        noise = torch.randn(
            endpoint.shape,
            dtype=endpoint.dtype,
            device=endpoint.device,
            generator=generator,
        )
        if coupling == "rank":
            endpoint = rank_couple_endpoint(noise, endpoint)
        elif coupling != "independent":
            raise ValueError(f"unknown flow coupling: {coupling}")
        time = torch.rand(endpoint.shape[0], device=endpoint.device, generator=generator)
        path = (1.0 - time[:, None]) * noise + time[:, None] * endpoint
        target_velocity = endpoint - noise
        predicted_velocity = self.velocity_at(path, time, seed.sum(1))
        if tail_weight < 0:
            raise ValueError("tail weight must be nonnegative")
        if tail_weight:
            weight = 1 + tail_weight * torch.sigmoid((path - 2.5 * (1 - time[:, None])) / 0.15)
            return ((predicted_velocity - target_velocity).square() * weight).sum() / weight.sum()
        return F.mse_loss(predicted_velocity, target_velocity)


class GINBranch(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_weight = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_state: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.sparse.mm(adjacency, node_state)
        return self.mlp((1.0 + self.self_weight) * node_state + neighbors)


class DualBranchLayer(nn.Module):
    """Parallel local GIN and global attention branches.

    Both branches consume the same layer input. Their outputs are fused with
    the residual and normalized, matching the paper's stated update rule.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        variant: str = "dual_branch",
    ):
        super().__init__()
        if variant not in {"dual_branch", "local_only", "global_only"}:
            raise ValueError(f"unknown predictor variant: {variant}")
        self.hidden_dim = hidden_dim
        self.variant = variant
        self.local = GINBranch(hidden_dim)
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        adjacency: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        layer_input = node_state
        if self.variant != "global_only":
            local_output = self.local(layer_input, adjacency)
        else:
            local_output = torch.zeros_like(layer_input)
        if self.variant != "local_only":
            global_input = layer_input.view(batch_size, num_nodes, self.hidden_dim)
            attention = self.global_attention
            qkv = F.linear(global_input, attention.in_proj_weight, attention.in_proj_bias)
            qkv = qkv.view(
                batch_size,
                num_nodes,
                3,
                attention.num_heads,
                self.hidden_dim // attention.num_heads,
            ).permute(2, 0, 3, 1, 4)
            attended = F.scaled_dot_product_attention(
                qkv[0], qkv[1], qkv[2], dropout_p=attention.dropout if self.training else 0.0
            )
            attended = attended.transpose(1, 2).reshape(batch_size, num_nodes, self.hidden_dim)
            global_output = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
            global_output = global_output.reshape(-1, self.hidden_dim)
        else:
            global_output = torch.zeros_like(layer_input)
        return self.normalization(
            layer_input + self.dropout(local_output) + self.dropout(global_output)
        )


class DualBranchAcceptancePredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.positional_dim = config.positional_dim
        self.acceptance_output = config.acceptance_output
        if self.acceptance_output not in {"node", "total"}:
            raise ValueError("acceptance_output must be node or total")
        self.node_embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.position_embedding = nn.Linear(config.positional_dim, config.hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            DualBranchLayer(
                config.hidden_dim,
                config.num_heads,
                config.dropout,
                config.predictor_variant,
            )
            for _ in range(config.num_layers)
        )
        self.output_head = nn.Sequential(
            nn.Linear(
                config.hidden_dim * (2 if self.acceptance_output == "total" else 1),
                config.hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        positional_encoding: torch.Tensor,
    ) -> torch.Tensor:
        total_nodes = node_features.shape[0]
        if total_nodes % self.num_nodes:
            raise ValueError("node feature count is not divisible by graph size")
        batch_size = total_nodes // self.num_nodes
        state = self.node_embedding(node_features)
        state = state + self.position_embedding(positional_encoding)
        for layer in self.layers:
            state = layer(state, adjacency, batch_size, self.num_nodes)
        if self.acceptance_output == "total":
            nodes = state.view(batch_size, self.num_nodes, -1)
            pooled = torch.cat([nodes.mean(1), nodes.amax(1)], dim=-1)
            return self.num_nodes * self.output_head(pooled)
        return self.output_head(state)


class CASOModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.seed_flow = RectifiedFlowSeedModel(config)
        self.acceptance_predictor = DualBranchAcceptancePredictor(config)

    def predict(
        self,
        relaxed_seed: torch.Tensor,
        content_match: torch.Tensor,
        adjacency: torch.Tensor,
        positional_encoding: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([relaxed_seed, content_match], dim=-1).reshape(-1, 2)
        prediction = self.acceptance_predictor(features, adjacency, positional_encoding)
        return prediction.view(relaxed_seed.shape[0], -1)

    def predict_total(self, relaxed_seed, content_match, adjacency, positional_encoding):
        """One expected acceptance count per graph, including legacy checkpoints."""
        return self.predict(relaxed_seed, content_match, adjacency, positional_encoding).sum(1)
