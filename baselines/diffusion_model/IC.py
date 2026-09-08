import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor


def _run_IC_worker(args):
    """
    Worker function for parallel IC simulation.
    args: (G, S, p, monte_carlo_chunk)
    """
    G, S, p, monte_carlo, seed, offset = args
    spread = 0

    for sample in range(offset, offset + monte_carlo):
        rng = random.Random(seed + sample)
        new_active = list(S)
        active = set(S)
        while new_active:
            current_new_active = []
            for u in new_active:
                for v in G.successors(u):
                    if v not in active:
                        prob = G[u][v].get("weight", p)
                        if prob is None:
                            prob = 0.1
                        if rng.random() <= prob:
                            active.add(v)
                            current_new_active.append(v)
            new_active = current_new_active
        spread += len(active)
    return spread


def run_IC(G, S, p=None, monte_carlo=100, n_jobs=None):
    """
    Independent Cascade Model simulation.
    Supports parallel execution if monte_carlo is large.
    """
    if monte_carlo < 1 or (n_jobs is not None and n_jobs < 1):
        raise ValueError("monte_carlo and n_jobs must be positive")
    n_jobs = min(n_jobs or multiprocessing.cpu_count(), monte_carlo)
    seed = random.getrandbits(64)
    if monte_carlo < 100 or n_jobs == 1:
        return _run_IC_worker((G, S, p, monte_carlo, seed, 0)) / monte_carlo
    chunk, remainder = divmod(monte_carlo, n_jobs)
    tasks = []
    offset = 0
    for worker in range(n_jobs):
        count = chunk + int(worker < remainder)
        tasks.append((G, S, p, count, seed, offset))
        offset += count
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_run_IC_worker, tasks))
    return sum(results) / monte_carlo
