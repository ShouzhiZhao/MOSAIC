import heapq
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor

from baselines.diffusion_model import run_IC, run_LT


def _celf_worker(args):
    node, G, p, monte_carlo, model, seed = args
    random.seed(seed)
    gain = _evaluate(G, [node], p, monte_carlo, model, n_jobs=1)
    return (-gain, node)


def _evaluate(G, seeds, p, monte_carlo, model, n_jobs=None):
    if model == "LT":
        return run_LT(G, seeds, monte_carlo, n_jobs=n_jobs)
    return run_IC(G, seeds, p, monte_carlo, n_jobs=n_jobs)


def celf(G, k, p=None, monte_carlo=100, n_jobs=None, model="IC"):
    """
    Cost-Effective Lazy Forward (CELF) optimization for Greedy algorithm.
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    marginal_gains = []

    # Calculate initial marginal gains for all nodes in parallel
    print(f"Initializing CELF with {n_jobs} workers (model={model})...")
    nodes = list(G.nodes())

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Prepare arguments
        args_list = [(node, G, p, monte_carlo, model, random.getrandbits(64)) for node in nodes]
        results = list(executor.map(_celf_worker, args_list))

    for res in results:
        heapq.heappush(marginal_gains, res)

    S = []
    spread = 0

    print("Selecting seeds...")
    while len(S) < k:
        gain, node = heapq.heappop(marginal_gains)
        gain = -gain

        if len(S) == 0:
            S.append(node)
            spread = gain
            print(f"Selected {node} with spread {spread:.2f}")
            continue

        # Re-evaluate marginal gain
        current_spread = _evaluate(G, S + [node], p, monte_carlo, model, n_jobs=n_jobs)
        marginal_gain = current_spread - spread

        if not marginal_gains:
            S.append(node)
            spread = current_spread
            break

        next_gain, _ = marginal_gains[0]
        next_gain = -next_gain

        if marginal_gain >= next_gain:
            S.append(node)
            spread = current_spread
            print(f"Selected {node} with spread {spread:.2f}")
        else:
            heapq.heappush(marginal_gains, (-marginal_gain, node))

    return S
