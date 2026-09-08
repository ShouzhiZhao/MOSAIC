"""
D-SSA — Dynamic Stop-and-Stare Algorithm for Influence Maximization.

Reference:
    Hung T. Nguyen, My T. Thai, Thang N. Dinh.
    "Stop-and-Stare: Optimal Sampling Algorithms for Viral Marketing
     in Billion-scale Networks."
    ACM SIGMOD 2016.

RR-set generation is parallelised across CPU cores.
"""

import math
import multiprocessing
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ── RR-set generation (IC, parallel) ───────────────────────────────────


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


# ── Greedy coverage ────────────────────────────────────────────────────


def _build_index(rr_sets):
    node_rr = defaultdict(set)
    for idx, rr in enumerate(rr_sets):
        for nd in rr:
            node_rr[nd].add(idx)
    return node_rr


def _greedy_coverage(rr_sets, k, node_rr):
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

    return seeds, covered


def _coverage_fraction(rr_sets, seeds):
    seed_set = set(seeds)
    covered = sum(1 for rr in rr_sets if seed_set & rr)
    return covered / max(len(rr_sets), 1)


# ── Helpers ─────────────────────────────────────────────────────────────


def _log_comb(n, k):
    val = 0.0
    for i in range(k):
        val += math.log(n - i) - math.log(i + 1)
    return val


def _compute_lambda(n, k, epsilon, delta):
    lcn = _log_comb(n, k)
    return (2.0 + 2.0 * epsilon / 3.0) * (lcn + math.log(1.0 / delta)) * n / (epsilon**2)


# ── D-SSA main ─────────────────────────────────────────────────────────


def d_ssa(G, k, p=None, epsilon=0.5, delta=None, n_jobs=None):
    """
    D-SSA algorithm for Influence Maximization.

    Parameters
    ----------
    G       : nx.DiGraph
    k       : int
    p       : float|None – default edge probability
    epsilon : float      – approximation ε ∈ (0, 1)
    delta   : float|None – failure probability (default 1/n)
    n_jobs  : int|None   – parallel workers (default: all cores)
    """
    n = G.number_of_nodes()
    nodes_list = list(G.nodes())

    if n == 0 or k == 0:
        return []

    if delta is None:
        delta = 1.0 / n

    eps_2 = epsilon * (1.0 - 1.0 / math.e) / (3.0 - epsilon)

    lambda_max = _compute_lambda(n, k, epsilon / 3.0, delta / 3.0)
    theta_max = min(int(math.ceil(lambda_max)), 10 * n)

    print(f"D-SSA: n={n}, k={k}, ε={epsilon}, θ_max={theta_max}")

    R1, R2 = [], []
    batch = max(64, int(math.sqrt(n)))
    best_seeds = None
    iteration = 0

    while True:
        iteration += 1

        # Generate batch for both collections in parallel
        new1 = _generate_rr_sets_parallel(G, nodes_list, p, batch, n_jobs)
        new2 = _generate_rr_sets_parallel(G, nodes_list, p, batch, n_jobs)
        R1.extend(new1)
        R2.extend(new2)

        # Greedy on R1
        node_rr = _build_index(R1)
        seeds, covered_set = _greedy_coverage(R1, k, node_rr)

        # Independent estimation on R2
        cov_R1 = len(covered_set) / len(R1)
        cov_R2 = _coverage_fraction(R2, seeds)

        est_R1 = n * cov_R1
        est_R2 = n * cov_R2

        delta_i = delta / (3.0 * max(iteration, 1))
        conf_width = math.sqrt(math.log(4.0 / delta_i) / (2.0 * len(R2)))
        rel_err = conf_width / max(cov_R2, 1e-10)

        if iteration <= 5 or iteration % 5 == 0:
            print(
                f"  iter={iteration}: |R|={len(R1)}, "
                f"est(R1)={est_R1:.1f}, est(R2)={est_R2:.1f}, "
                f"rel_err={rel_err:.4f}"
            )

        best_seeds = seeds

        if rel_err <= eps_2 or len(R1) >= theta_max:
            break

        batch = len(R1)  # double

    est_final = n * _coverage_fraction(R2, best_seeds)
    print(
        f"  D-SSA done: {len(R1)}+{len(R2)} RR sets ({iteration} iters), "
        f"est. spread={est_final:.1f}"
    )

    return best_seeds
