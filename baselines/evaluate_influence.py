#!/usr/bin/env python3
"""
evaluate_influence.py — Evaluate all baseline seeds under IC and LT models.

For each algorithm:
  - IC evaluation: uses IC-selected seeds, evaluates with IC simulation
  - LT evaluation: uses LT-selected seeds if available, else IC seeds,
                    evaluates with LT simulation

Evaluates k=1..10 for all algorithms. Parallelized at the task level
for maximum throughput on multi-core machines.

Usage:
    python -m baselines.evaluate_influence --help
    python evaluate_influence.py
    python evaluate_influence.py --mc 2000    # more MC sims for precision
    python evaluate_influence.py --workers 32  # limit parallelism
"""

import argparse
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import pandas as pd

from mosaic import paths

# ── IC / LT simulation (self-contained for pickling) ─────────────────────


def _simulate_IC(G, seeds, mc):
    total = 0
    for _ in range(mc):
        active = set(seeds)
        frontier = list(seeds)
        while frontier:
            nxt = []
            for u in frontier:
                for v in G.successors(u):
                    if v not in active:
                        p = G[u][v].get("weight", 0.1)
                        if p is None:
                            p = 0.1
                        if random.random() <= p:
                            active.add(v)
                            nxt.append(v)
            frontier = nxt
        total += len(active)
    return total


def _simulate_LT(G, seeds, mc):
    total = 0
    for _ in range(mc):
        thresholds = {n: random.random() for n in G.nodes()}
        active = set(seeds)
        frontier = list(seeds)
        acc_weights = {}
        while frontier:
            nxt = []
            for u in frontier:
                for v in G.successors(u):
                    if v not in active:
                        w = G[u][v].get("weight", 0.1)
                        if w is None:
                            w = 0.1
                        acc_weights[v] = acc_weights.get(v, 0.0) + w
                        if acc_weights[v] >= thresholds[v]:
                            active.add(v)
                            nxt.append(v)
            frontier = nxt
        total += len(active)
    return total


def _worker(task):
    """Worker: runs MC simulations for a single (algo, k, model) task."""
    G, seed_list, k, model, mc, seed = task
    random.seed(seed)
    seeds = seed_list[:k]
    if model == "IC":
        spread = _simulate_IC(G, seeds, mc)
    else:
        spread = _simulate_LT(G, seeds, mc)
    return spread / mc


# ── Data loading ──────────────────────────────────────────────────────────


def load_graph(path, scale=1.0):
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        u, v = int(row["user_1"]), int(row["user_2"])
        prob = min(float(row["closeness"]) * scale, 1.0)
        G.add_edge(u, v, weight=prob)
    return G


def load_seeds(results_dir, k_max=10):
    """Load all algorithm seeds. Returns dict[algo] = {'IC': [...], 'LT': [...] or None}"""
    algos = {}
    for algo_dir in sorted(Path(results_dir).iterdir()):
        if not algo_dir.is_dir():
            continue
        name = algo_dir.name
        ic_file = algo_dir / f"k_{k_max}_IC.json"
        lt_file = algo_dir / f"k_{k_max}_LT.json"
        if not ic_file.exists():
            continue
        with open(ic_file) as f:
            ic_data = json.load(f)
        ic_seeds = ic_data["seeds"]

        lt_seeds = None
        if lt_file.exists():
            with open(lt_file) as f:
                lt_data = json.load(f)
            lt_seeds = lt_data["seeds"]

        algos[name] = {"IC": ic_seeds, "LT": lt_seeds}
    return algos


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline seeds under IC and LT")
    parser.add_argument("--mc", type=int, default=1000, help="MC simulations per evaluation")
    parser.add_argument("--k_max", type=int, default=10, help="Max seed budget")
    parser.add_argument("--scale", type=float, default=0.1, help="Edge weight scaling")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.mc < 1 or args.k_max < 1 or (args.workers is not None and args.workers < 1):
        parser.error("mc, k_max, and workers must be positive")
    random.seed(args.seed)

    n_cpu = multiprocessing.cpu_count()
    n_workers = args.workers or n_cpu
    print(f"CPUs: {n_cpu}, Workers: {n_workers}, MC: {args.mc}\n")

    graph_path = str(paths.get_relationship_file())
    print(f"Loading graph: {graph_path}")
    G = load_graph(graph_path, args.scale)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print(f"Loading seeds from {args.results_dir}/")
    algo_seeds = load_seeds(args.results_dir, args.k_max)
    print(f"  Found {len(algo_seeds)} algorithms:")
    for name, info in algo_seeds.items():
        lt_tag = f"LT seeds: {len(info['LT'])}" if info["LT"] else "LT: use IC seeds"
        print(f"    {name:14s}  IC seeds: {len(info['IC']):2d}  {lt_tag}")
    print()

    # Build task list
    tasks = []  # (G, seeds, k, model, mc)
    task_meta = []  # (algo_name, k, eval_model, seed_source)

    for algo, info in algo_seeds.items():
        ic_seeds = info["IC"]
        lt_seeds = info["LT"]

        for k in range(1, args.k_max + 1):
            # IC evaluation: always use IC seeds
            tasks.append((G, ic_seeds, k, "IC", args.mc, random.getrandbits(64)))
            task_meta.append((algo, k, "IC", "IC"))

            # LT evaluation: use LT seeds if available, else IC seeds
            lt_eval_seeds = lt_seeds if lt_seeds else ic_seeds
            seed_src = "LT" if lt_seeds else "IC"
            tasks.append((G, lt_eval_seeds, k, "LT", args.mc, random.getrandbits(64)))
            task_meta.append((algo, k, "LT", seed_src))

    n_tasks = len(tasks)
    print(f"Total evaluation tasks: {n_tasks}")
    print(f"  ({len(algo_seeds)} algos × {args.k_max} k-values × 2 models)")
    print(f"  Each task: {args.mc} MC simulations\n")

    # Run in parallel
    print(f"Running evaluations with {n_workers} workers ...")
    t0 = time.time()

    results = [None] * n_tasks
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {pool.submit(_worker, tasks[i]): i for i in range(n_tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % 50 == 0 or done == n_tasks:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_tasks - done)
                print(f"  {done}/{n_tasks} done  ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nAll evaluations completed in {total_time:.1f}s\n")

    # Assemble results
    eval_data = {}  # algo -> {IC: {k: influence}, LT: {k: influence, seed_source: ...}}
    for i, (algo, k, eval_model, seed_src) in enumerate(task_meta):
        if algo not in eval_data:
            eval_data[algo] = {"IC": {}, "LT": {}, "LT_seed_source": None}
        eval_data[algo][eval_model][k] = round(results[i], 4)
        if eval_model == "LT":
            eval_data[algo]["LT_seed_source"] = seed_src

    # Save per-algorithm JSON
    out_dir = Path(args.results_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for algo, data in eval_data.items():
        record = {
            "algorithm": algo,
            "k_max": args.k_max,
            "mc_simulations": args.mc,
            "random_seed": args.seed,
            "IC": {
                "seed_source": "IC",
                "influence_by_k": data["IC"],
            },
            "LT": {
                "seed_source": data["LT_seed_source"],
                "influence_by_k": data["LT"],
            },
        }
        fpath = out_dir / f"{algo}_eval.json"
        with open(fpath, "w") as f:
            json.dump(record, f, indent=2)

    # Save summary CSV
    rows = []
    for algo, data in eval_data.items():
        for model in ["IC", "LT"]:
            seed_src = "IC" if model == "IC" else data["LT_seed_source"]
            for k in range(1, args.k_max + 1):
                rows.append(
                    {
                        "algorithm": algo,
                        "k": k,
                        "eval_model": model,
                        "seed_source": seed_src,
                        "influence": data[model].get(k, 0),
                    }
                )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "influence_evaluation.csv"
    df.to_csv(csv_path, index=False)

    # Print summary table
    print(f"{'=' * 80}")
    print(f"  Results saved to {out_dir}/")
    print(f"{'=' * 80}\n")

    header = f"  {'Algorithm':14s}  {'Seeds':5s}  {'IC k=5':>8s}  {'IC k=10':>8s}  {'LT k=5':>8s}  {'LT k=10':>8s}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for algo in sorted(eval_data.keys()):
        d = eval_data[algo]
        src = d["LT_seed_source"]
        ic5 = d["IC"].get(5, 0)
        ic10 = d["IC"].get(10, 0)
        lt5 = d["LT"].get(5, 0)
        lt10 = d["LT"].get(10, 0)
        print(f"  {algo:14s}  {src:5s}  {ic5:8.1f}  {ic10:8.1f}  {lt5:8.1f}  {lt10:8.1f}")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  MC simulations per task: {args.mc}")
    print()


if __name__ == "__main__":
    main()
