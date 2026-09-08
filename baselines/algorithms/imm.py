"""
IMM — Influence Maximization via Martingales.

Reference:
    Youze Tang, Yanchen Shi, Xiaokui Xiao.
    "Influence Maximization in Near-Linear Time: A Martingale Approach."
    ACM SIGMOD 2015.

(1 − 1/e − ε)-approximate with adaptive RR-set sampling.
RR-set generation is parallelised across CPU cores.
"""

import math
import multiprocessing
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ── RR-set generation (IC model, parallel) ─────────────────────────────


def _generate_rr_set(G, nodes_list, p=None, rng=None):
    rng = rng or random
    v = rng.choice(nodes_list)
    visited = {v}
    queue = [v]
    while queue:
        u = queue.pop(0)
        for w in G.predecessors(u):
            if w not in visited:
                prob = G[w][u].get("weight", p)
                if prob is None:
                    prob = 0.1
                if rng.random() <= prob:
                    visited.add(w)
                    queue.append(w)
    return visited


def _rr_batch_worker(args):
    G, nodes_list, p, count, seed, offset = args
    return [
        _generate_rr_set(G, nodes_list, p, random.Random(seed + i))
        for i in range(offset, offset + count)
    ]


def _generate_rr_sets_parallel(G, nodes_list, p, total, n_jobs=None):
    if total <= 0:
        return []
    if n_jobs is not None and n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    n_jobs = min(n_jobs or multiprocessing.cpu_count(), total)
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    seed = random.getrandbits(64)
    if n_jobs == 1 or total < 4:
        return _rr_batch_worker((G, nodes_list, p, total, seed, 0))
    chunk, rem = divmod(total, n_jobs)
    tasks = []
    offset = 0
    for worker in range(n_jobs):
        count = chunk + int(worker < rem)
        tasks.append((G, nodes_list, p, count, seed, offset))
        offset += count
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        batches = list(pool.map(_rr_batch_worker, tasks))
    return [rr for batch in batches for rr in batch]


# ── Greedy max-coverage on RR sets ─────────────────────────────────────


def _greedy_coverage(rr_sets, k, n_nodes, node_rr):
    covered = set()
    seeds = []
    gains = {nd: len(indices) for nd, indices in node_rr.items()}

    for _ in range(k):
        if not gains:
            break
        best_node = max(gains, key=gains.get)
        gains.pop(best_node)
        real_gain = len(node_rr.get(best_node, set()) - covered)

        while gains:
            runner_up = max(gains, key=gains.get)
            if real_gain >= gains[runner_up]:
                break
            rg = len(node_rr.get(runner_up, set()) - covered)
            gains[runner_up] = rg
            if rg > real_gain:
                gains[best_node] = real_gain
                best_node = runner_up
                real_gain = rg
                gains.pop(best_node)

        seeds.append(best_node)
        covered |= node_rr.get(best_node, set())

    return seeds, len(covered)


def _build_index(rr_sets):
    node_rr = defaultdict(set)
    for idx, rr in enumerate(rr_sets):
        for nd in rr:
            node_rr[nd].add(idx)
    return node_rr


# ── Helpers ─────────────────────────────────────────────────────────────


def _log_comb(n, k):
    val = 0.0
    for i in range(k):
        val += math.log(n - i) - math.log(i + 1)
    return val


# ── IMM main ───────────────────────────────────────────────────────────


def imm(G, k, p=None, epsilon=0.5, l=1.0, n_jobs=None):  # noqa: E741 - published IMM parameter
    """
    IMM algorithm for Influence Maximization.

    Parameters
    ----------
    G       : nx.DiGraph
    k       : int        – number of seeds
    p       : float|None – default edge propagation probability
    epsilon : float      – approximation parameter ε ∈ (0, 1)
    l       : float      – confidence parameter
    n_jobs  : int|None   – parallel workers (default: all cores)
    """
    n = G.number_of_nodes()
    nodes_list = list(G.nodes())

    if n == 0 or k == 0:
        return []

    l_adj = l * (1.0 + math.log(2) / math.log(max(n, 2)))
    eps_prime = math.sqrt(2) * epsilon
    lcn = _log_comb(n, k)

    lambda_prime = (
        (2.0 + 2.0 * eps_prime / 3.0)
        * (lcn + l_adj * math.log(n) + math.log(math.log2(max(n, 2))))
        * n
        / (eps_prime**2)
    )

    alpha = math.sqrt(l_adj * math.log(n) + math.log(2))
    beta = math.sqrt((1.0 - 1.0 / math.e) * (lcn + l_adj * math.log(n) + math.log(2)))
    lambda_star = 2.0 * n * ((1.0 - 1.0 / math.e) * alpha + beta) ** 2 / (epsilon**2)

    # ── Phase 1: lower-bound estimation ──
    rr_sets = []
    LB = 1.0
    max_iter = max(1, int(math.log2(n)))

    print(f"IMM: n={n}, k={k}, ε={epsilon}")

    for i in range(1, max_iter):
        x = n / (2.0**i)
        theta_i = int(math.ceil(lambda_prime / x))

        need = theta_i - len(rr_sets)
        if need > 0:
            rr_sets.extend(_generate_rr_sets_parallel(G, nodes_list, p, need, n_jobs))

        node_rr = _build_index(rr_sets)
        seeds_i, n_covered = _greedy_coverage(rr_sets, k, n, node_rr)
        frac = n_covered / len(rr_sets)

        if n * frac >= (1.0 + eps_prime) * x:
            LB = n * frac / (1.0 + eps_prime)
            print(f"  Phase-1 converged at i={i}: LB={LB:.1f}, |R|={len(rr_sets)}")
            break

    # ── Phase 2: final sampling ──
    theta = int(math.ceil(lambda_star / max(LB, 1.0)))
    theta = max(theta, len(rr_sets))

    need = theta - len(rr_sets)
    if need > 0:
        print(f"  Phase-2: generating {need} more RR sets (total θ={theta})")
        rr_sets.extend(_generate_rr_sets_parallel(G, nodes_list, p, need, n_jobs))

    node_rr = _build_index(rr_sets)
    seeds, n_covered = _greedy_coverage(rr_sets, k, n, node_rr)

    est_spread = n * n_covered / len(rr_sets)
    print(f"  IMM done: {len(rr_sets)} RR sets, est. spread={est_spread:.1f}")

    return seeds
