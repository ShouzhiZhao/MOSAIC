"""
TIM — Two-phase Influence Maximization using Reverse Reachable sets.

Reference:
    Tang et al. "Influence Maximization: Near-Optimal Time Complexity
    Meets Practical Efficiency." ACM SIGMOD 2014.
"""

import heapq
import multiprocessing
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def _generate_rr_set(G, nodes_list, p, rng=None):
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
    return list(visited)


def _rr_batch_worker(args):
    """Each worker generates *count* RR sets (graph pickled once per worker)."""
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


def tim(G, k, p=None, num_rr_sets=10000, n_jobs=None):
    """
    TIM with parallel RR-set generation.
    """
    nodes_list = list(G.nodes())

    print(
        f"TIM: generating {num_rr_sets} RR sets "
        f"({n_jobs or multiprocessing.cpu_count()} workers) ..."
    )
    rr_sets = _generate_rr_sets_parallel(G, nodes_list, p, num_rr_sets, n_jobs)

    # Inverted index
    node_rr_indices = defaultdict(list)
    for i, rr in enumerate(rr_sets):
        for node in rr:
            node_rr_indices[node].append(i)

    # Greedy seed selection (lazy forward)
    print("TIM: selecting seeds ...")
    S = []
    covered = set()

    gains = []
    for node in G.nodes():
        heapq.heappush(gains, (-len(node_rr_indices[node]), node))

    while len(S) < k:
        if not gains:
            break

        gain, node = heapq.heappop(gains)
        gain = -gain

        cur = set(node_rr_indices[node])
        mg = len(cur - covered)

        if not gains:
            if mg > 0:
                S.append(node)
                covered.update(cur)
            break

        next_gain = -gains[0][0]
        if mg >= next_gain:
            S.append(node)
            covered.update(cur)
        else:
            heapq.heappush(gains, (-mg, node))

    return S
