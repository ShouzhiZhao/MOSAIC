"""Generate exact-budget seed sets without simulator calls or outcome labels."""

import torch


def sample_random_seed_masks(
    num_nodes: int,
    batch_size: int,
    *,
    maximum_budget: int = 10,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw k uniformly from 0..maximum_budget, then select k distinct nodes."""
    if batch_size < 1 or not 0 <= maximum_budget <= num_nodes:
        raise ValueError("positive batch size and a budget between zero and N are required")
    budgets = torch.randint(
        0, maximum_budget + 1, (batch_size,), device=device, generator=generator
    )
    return sample_seed_masks(num_nodes, budgets, generator=generator)


def rank_couple_endpoint(noise: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
    """Pair coordinate ranks, valid for an exchangeable synthetic target prior.

    Each endpoint's multiset is preserved exactly. The rank permutation of
    iid continuous Gaussian noise is uniform, so assigning the sorted target
    through that permutation preserves an exchangeable target distribution.
    This must not be used to preserve a nonuniform, node-specific seed prior.
    """
    if noise.ndim != 2 or endpoint.shape != noise.shape:
        raise ValueError("matching [batch, nodes] arrays are required")
    return torch.empty_like(endpoint).scatter_(1, noise.argsort(dim=1), endpoint.sort(dim=1).values)


def sample_seed_masks(
    num_nodes: int,
    budgets: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform sampling without replacement conditional on each row's budget.

    Rows represent unordered sets, so different orderings of the same selected
    users do not create different examples. Repeated draws can repeat sets,
    especially for k=0 (only one possible set) and k=1.
    """
    if num_nodes < 1 or budgets.ndim != 1 or budgets.numel() == 0:
        raise ValueError("positive node count and a nonempty budget vector are required")
    if budgets.dtype not in (torch.int32, torch.int64):
        raise ValueError("budgets must be integers")
    if torch.any(budgets < 0) or torch.any(budgets > num_nodes):
        raise ValueError("budgets must lie between zero and the node count")
    maximum = int(budgets.max().item())
    mask = torch.zeros(len(budgets), num_nodes, device=budgets.device)
    if maximum:
        scores = torch.rand(len(budgets), num_nodes, device=budgets.device, generator=generator)
        indices = scores.topk(maximum, dim=1, sorted=True).indices
        chosen = torch.arange(maximum, device=budgets.device)[None, :] < budgets[:, None]
        mask.scatter_(1, indices, chosen.to(mask.dtype))
    return mask
