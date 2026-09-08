import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor

from baselines.diffusion_model import run_IC, run_LT


def _naive_worker(args):
    node, G, S, p, monte_carlo, model, seed = args
    random.seed(seed)
    # Use n_jobs=1 to avoid nested process pools
    if model == "LT":
        current_spread = run_LT(G, S + [node], monte_carlo, n_jobs=1)
    else:
        current_spread = run_IC(G, S + [node], p, monte_carlo, n_jobs=1)
    return (current_spread, node)


def naive_greedy(G, k, p=None, monte_carlo=100, n_jobs=None, model="IC"):
    """
    Naive Greedy algorithm for Influence Maximization.
    Evaluates the marginal gain of every node in each step.
    Parallelized version.
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    S = []
    print(f"Naive Greedy initialized with {n_jobs} workers (model={model}).")

    for i in range(k):
        best_node = None
        max_gain = -1

        # In each step, evaluate all nodes not in S
        nodes = [node for node in G.nodes() if node not in S]

        print(f"Step {i + 1}/{k}: Evaluating {len(nodes)} candidates...")

        # Prepare arguments
        args_list = [(node, G, S, p, monte_carlo, model, random.getrandbits(64)) for node in nodes]

        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_naive_worker, args_list))

        for spread, node in results:
            if spread > max_gain:
                max_gain = spread
                best_node = node

        if best_node is not None:
            S.append(best_node)
            print(f"Selected {best_node} with spread {max_gain:.2f}")

    return S
