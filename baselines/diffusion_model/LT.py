import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor


def _run_LT_worker(args):
    """
    Worker function for parallel LT simulation.
    args: (G, S, monte_carlo_chunk)
    """
    G, S, monte_carlo, seed, offset = args
    spread = 0

    for sample in range(offset, offset + monte_carlo):
        rng = random.Random(seed + sample)
        thresholds = {node: rng.random() for node in G.nodes()}

        active = set(S)
        new_active = list(S)

        current_weights = {}

        while new_active:
            current_new_active = []

            for u in new_active:
                for v in G.successors(u):
                    if v not in active:
                        weight = G[u][v].get("weight", 0.1)  # Default weight

                        # Update weight
                        if v not in current_weights:
                            current_weights[v] = 0.0
                        current_weights[v] += weight

                        # Check threshold
                        if current_weights[v] >= thresholds[v]:
                            active.add(v)
                            current_new_active.append(v)

            new_active = current_new_active

        spread += len(active)

    return spread


def run_LT(G, S, monte_carlo=100, n_jobs=None):
    """
    Linear Threshold (LT) Model simulation.
    Supports parallel execution.
    """
    if monte_carlo < 1 or (n_jobs is not None and n_jobs < 1):
        raise ValueError("monte_carlo and n_jobs must be positive")
    n_jobs = min(n_jobs or multiprocessing.cpu_count(), monte_carlo)
    seed = random.getrandbits(64)
    if monte_carlo < 100 or n_jobs == 1:
        return _run_LT_worker((G, S, monte_carlo, seed, 0)) / monte_carlo
    chunk, remainder = divmod(monte_carlo, n_jobs)
    tasks = []
    offset = 0
    for worker in range(n_jobs):
        count = chunk + int(worker < remainder)
        tasks.append((G, S, count, seed, offset))
        offset += count
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_run_LT_worker, tasks))
    return sum(results) / monte_carlo
