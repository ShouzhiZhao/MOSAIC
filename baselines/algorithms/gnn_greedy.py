"""
GNN-Greedy: GNN-based Influence Surrogate + Greedy Seed Selection.

References:
    Chen et al. "Maximizing Influence with Graph Neural Networks."
    arXiv:2108.04623 (extended 2023). — GLIE approach.

    Ling et al. "Deep Graph Representation Learning and Optimization
    for Influence Maximization." ICML 2023. — DeepIM proxy concept.

Core idea:
    1. Generate labelled data by sampling random seed sets and computing
       per-node activation frequencies via Monte-Carlo IC simulation.
    2. Train a 3-layer GCN to predict per-node activation probability
       given (graph, seed indicator).
    3. Use the trained GCN as a fast differentiable proxy inside a
       standard greedy loop (replacing expensive MC evaluation).
"""

import random
import time

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class _GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        return adj @ self.W(x) + self.bias


class _InfluencePredictor(nn.Module):
    """3-layer GCN → per-node activation probability."""

    def __init__(self, feat_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = _GCNConv(feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = _GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv3 = _GCNConv(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, adj):
        h = torch.relu(self.norm1(self.conv1(x, adj)))
        h = torch.relu(self.norm2(self.conv2(h, adj)))
        h = torch.relu(self.norm3(self.conv3(h, adj)))
        return self.head(h).squeeze(-1)  # (N,)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _build_norm_adj(G, nodes, device):
    """Return D^{-1/2}(A+I)D^{-1/2} as dense tensor + node→idx map."""
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}

    A = torch.zeros(n, n, device=device)
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            A[idx[u], idx[v]] = d.get("weight", 0.1)

    A = A + torch.eye(n, device=device)
    deg = A.sum(1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[d_inv_sqrt == float("inf")] = 0
    A = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
    return A, idx


def _mc_node_activation(G, seeds, p, mc, node_to_idx, n):
    """Run *mc* IC simulations and return per-node activation frequency."""
    counts = np.zeros(n)
    for _ in range(mc):
        active = set(seeds)
        frontier = list(seeds)
        while frontier:
            nxt = []
            for u in frontier:
                for v in G.successors(u):
                    if v not in active:
                        prob = G[u][v].get("weight", p or 0.1)
                        if random.random() <= prob:
                            active.add(v)
                            nxt.append(v)
            frontier = nxt
        for nd in active:
            if nd in node_to_idx:
                counts[node_to_idx[nd]] += 1
    return counts / mc


def _static_features(G, nodes, device):
    """Normalised degree + weighted out-degree as static node features."""
    degs = np.array([G.degree(nd) for nd in nodes], dtype=np.float32)
    w_degs = np.array(
        [sum(G[nd][v].get("weight", 0.1) for v in G.successors(nd)) for nd in nodes],
        dtype=np.float32,
    )
    degs /= max(degs.max(), 1)
    w_degs /= max(w_degs.max(), 1)
    return torch.tensor(np.stack([degs, w_degs], axis=1), dtype=torch.float32, device=device)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def gnn_greedy(G, k, p=None, monte_carlo=50, n_train=300, n_epochs=80, hidden_dim=64):
    """
    GNN proxy + greedy Influence Maximisation.

    Parameters
    ----------
    G           : nx.DiGraph – input social network
    k           : int        – seed budget
    p           : float      – default edge propagation probability
    monte_carlo : int        – MC runs per training sample
    n_train     : int        – number of training samples to generate
    n_epochs    : int        – GCN training epochs
    hidden_dim  : int        – GCN hidden dimension

    Returns
    -------
    list[int] – selected seed nodes
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for GNN-Greedy.")

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = list(G.nodes())
    n = len(nodes)

    print(f"GNN-Greedy: n={n}, k={k}, generating {n_train} samples ({monte_carlo} MC each)")

    adj, idx_map = _build_norm_adj(G, nodes, device)
    static = _static_features(G, nodes, device)  # (N, 2)

    # ---- Generate training data ----
    t0 = time.time()
    samples = []
    max_ss = min(k * 3, max(n // 5, k))
    for i in range(n_train):
        ss = random.randint(1, max_ss)
        seeds = random.sample(nodes, ss)
        act = _mc_node_activation(G, seeds, p, monte_carlo, idx_map, n)
        ind = np.zeros(n, dtype=np.float32)
        for s in seeds:
            ind[idx_map[s]] = 1.0
        samples.append((ind, act))
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_train} samples")
    print(f"  Data generation: {time.time() - t0:.1f}s")

    # ---- Train GCN ----
    # feat_dim = 3: [is_seed, deg_norm, w_deg_norm]
    model = _InfluencePredictor(3, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(n_epochs):
        random.shuffle(samples)
        total_loss = 0.0
        for ind, act in samples:
            ind_t = torch.tensor(ind, device=device).unsqueeze(-1)  # (N, 1)
            x = torch.cat([ind_t, static], dim=-1)  # (N, 3)
            target = torch.tensor(act, device=device, dtype=torch.float32)

            pred = model(x, adj)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep + 1}/{n_epochs}: loss={total_loss / len(samples):.6f}")

    # ---- Greedy selection via GCN proxy ----
    print("  Greedy selection via GCN proxy ...")
    model.eval()
    selected = []
    seed_ind = torch.zeros(n, 1, device=device)

    for step in range(k):
        best_node, best_spread = None, -1.0
        for nd in nodes:
            if nd in selected:
                continue
            i = idx_map[nd]
            seed_ind[i, 0] = 1.0
            x = torch.cat([seed_ind, static], dim=-1)
            with torch.no_grad():
                spread = model(x, adj).sum().item()
            if spread > best_spread:
                best_spread = spread
                best_node = nd
            seed_ind[i, 0] = 0.0

        selected.append(best_node)
        seed_ind[idx_map[best_node], 0] = 1.0
        print(f"    Step {step + 1}: node {best_node} (predicted spread: {best_spread:.2f})")

    elapsed = time.time() - start
    print(f"  GNN-Greedy done in {elapsed:.1f}s")
    return selected
