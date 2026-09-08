#!/usr/bin/env python3
"""
run_baselines.py — Run IM baselines and save seed selections for k=1..k_max.

Every algorithm produces an *ordered* seed list.  Running once with k_max
gives solutions for all smaller budgets: seeds[:k] is the k-seed answer.

Usage:
    python run_baselines.py                       # all algos, k=1..10
    python run_baselines.py --k 5                 # k=1..5
    python run_baselines.py --algo degree celf voterank clusterrank  # subset
    python run_baselines.py --mc 500 --scale 0.1  # tune MC / edge scaling

Output:
    results/all_seeds_k1-{k}_{model}.csv   — tidy CSV (algorithm, k, seeds)
    results/<Algo>/k_{k}_{model}.json      — per-algorithm JSON (full k_max list)
"""

import argparse
import json
import multiprocessing
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from baselines.algorithms import (
    celf,
    clusterrank,
    d_ssa,
    degree_discount,
    gcomb,
    gnn_greedy,
    high_degree,
    imm,
    naive_greedy,
    random_selection,
    s2v_dqn,
    tim,
    touplegdd,
    voterank,
)
from mosaic import paths
from mosaic.catalog import resolve_movie
from mosaic.policies import SeedPolicy, write_policies

DEFAULT_RESULTS_DIR = paths.ARTIFACTS_DIR / "baselines" / "seeds"


# ── Graph loading ────────────────────────────────────────────────────────


def load_graph(path, scale=1.0):
    print(f"Loading graph from {path}  (scale={scale})")
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        u = int(row["user_1"])
        v = int(row["user_2"])
        prob = min(float(row["closeness"]) * scale, 1.0)
        G.add_edge(u, v, weight=prob)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G


# ── Result I/O ───────────────────────────────────────────────────────────


def save_algo_json(algo_name, k, seeds, elapsed, model, results_dir):
    """Per-algorithm JSON (backward compatible with old format)."""
    algo_dir = Path(results_dir) / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "algorithm": algo_name,
        "k": k,
        "model": model,
        "seeds": [int(s) for s in seeds],
        "time_seconds": round(elapsed, 4),
    }
    filepath = algo_dir / f"k_{k}_{model}.json"
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def save_all_csv(all_results, k_max, model, results_dir):
    """Tidy CSV: one row per (algorithm, k)."""
    rows = []
    for display, info in all_results.items():
        seeds = info["seeds"]
        t = info["time"]
        for kk in range(1, k_max + 1):
            rows.append(
                {
                    "algorithm": display,
                    "k": kk,
                    "seeds": [int(s) for s in seeds[:kk]],
                    "time_seconds": round(t, 4),
                }
            )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"all_seeds_k1-{k_max}_{model}.csv"
    df = pd.DataFrame(rows)
    if csv_path.exists():
        previous = pd.read_csv(csv_path)
        previous = previous[previous["algorithm"].isin(set(ALGO_REGISTRY.values()))]
        previous = previous[~previous["algorithm"].isin(all_results)]
        df = pd.concat([previous, df], ignore_index=True)
    order = {name: index for index, name in enumerate(ALGO_REGISTRY.values())}
    df["_order"] = df["algorithm"].map(order)
    df = df.sort_values(["_order", "k"]).drop(columns="_order")
    df.to_csv(csv_path, index=False)
    return csv_path


# ── Algorithm registry ───────────────────────────────────────────────────

ALGO_REGISTRY = {
    "degree": "Degree",
    "discount": "Degree Discount",
    "random": "Random",
    "celf": "CELF",
    "naive": "Naive Greedy",
    "tim": "TIM",
    "imm": "IMM",
    "dssa": "D-SSA",
    "s2v_dqn": "S2V-DQN",
    "gcomb": "GCOMB",
    "gnn_greedy": "GNN-Greedy",
    "touplegdd": "ToupleGDD",
    "voterank": "VoteRank",
    "clusterrank": "ClusterRank",
}

IC_ONLY_ALGOS = {
    "tim",
    "imm",
    "dssa",
    "s2v_dqn",
    "gcomb",
    "gnn_greedy",
    "touplegdd",
}


def run_algo(name, G, k, scale, mc, model, workers):
    """Run a single algorithm.  Returns ordered list of seed node IDs."""
    if name == "degree":
        return high_degree(G, k)
    elif name == "discount":
        return degree_discount(G, k)
    elif name == "random":
        return random_selection(G, k)
    elif name == "celf":
        return celf(
            G,
            k,
            monte_carlo=mc,
            n_jobs=workers,
            model=model,
        )
    elif name == "naive":
        return naive_greedy(G, k, monte_carlo=mc, n_jobs=workers, model=model)
    elif name == "tim":
        return tim(G, k, p=scale, num_rr_sets=10000, n_jobs=workers)
    elif name == "imm":
        return imm(G, k, p=scale, epsilon=0.5, n_jobs=workers)
    elif name == "dssa":
        return d_ssa(G, k, p=scale, epsilon=0.5, n_jobs=workers)
    elif name == "s2v_dqn":
        return s2v_dqn(G, k)
    elif name == "gcomb":
        return gcomb(G, k)
    elif name == "gnn_greedy":
        return gnn_greedy(G, k, p=scale, monte_carlo=min(mc, 50), n_train=300, n_epochs=80)
    elif name == "touplegdd":
        return touplegdd(G, k)
    elif name == "voterank":
        return voterank(G, k)
    elif name == "clusterrank":
        return clusterrank(G, k)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


# ── Main ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run IM baselines, save seeds for k=1..k")
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        type=str,
        default=str(paths.get_relationship_file()),
    )
    parser.add_argument("--k", type=int, default=10, help="Max seed budget (saves for k=1..k)")
    parser.add_argument(
        "--mc", type=int, default=1000, help="MC simulations for stochastic algorithms"
    )
    parser.add_argument("--algo", nargs="*", default=None, help="Algorithms to run (default: all)")
    parser.add_argument("--scale", type=float, default=0.1, help="Edge weight scaling")
    parser.add_argument("--model", type=str, default="IC", choices=["IC", "LT"])
    parser.add_argument(
        "--seed", type=int, default=2026, help="Random seed reset before each method"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="CPU workers for Monte Carlo and RR methods"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--movie", nargs="+", help="Export movie-specific policies for prediction or replay"
    )
    args = parser.parse_args(argv)
    if (
        args.k < 1
        or args.mc < 1
        or args.scale < 0
        or (args.workers is not None and args.workers < 1)
    ):
        parser.error("k, mc, and workers must be positive; scale must be nonnegative")

    movies = [resolve_movie(m) for m in args.movie] if args.movie else []
    G = load_graph(args.data_path, args.scale)
    if args.k > len(G):
        parser.error("seed budget exceeds the graph size")
    n_cores = multiprocessing.cpu_count()
    workers = args.workers or n_cores
    print(f"CPU cores: {n_cores}; workers: {workers}")
    print(
        f"CUDA: {torch.cuda.is_available()}"
        + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
    )

    algos = args.algo if args.algo else list(ALGO_REGISTRY.keys())

    for a in algos:
        if a not in ALGO_REGISTRY:
            parser.error(f"Unknown algorithm '{a}'. Choose from: {list(ALGO_REGISTRY.keys())}")

    all_results = {}
    total_start = time.time()

    for algo in algos:
        display = ALGO_REGISTRY[algo]

        if args.model == "LT" and algo in IC_ONLY_ALGOS:
            print(f"  [skip] {display} (LT not supported)\n")
            continue

        print(f"{'─' * 60}")
        print(f"  {display}")
        print(f"{'─' * 60}")

        start = time.time()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        seeds = run_algo(algo, G, args.k, args.scale, args.mc, args.model, workers)
        elapsed = time.time() - start
        seeds = [int(s) for s in seeds]
        if len(seeds) != args.k or len(set(seeds)) != args.k:
            raise ValueError(f"{display} did not return the requested number of distinct seeds")

        all_results[display] = {"seeds": seeds, "time": elapsed}
        save_algo_json(display, args.k, seeds, elapsed, args.model, args.results_dir)

        print(f"  {elapsed:>7.2f}s  seeds = {seeds}\n")

    total_elapsed = time.time() - total_start

    if all_results:
        csv_path = save_all_csv(all_results, args.k, args.model, args.results_dir)

        if args.movie:
            topology = {"Random", "Degree", "Degree Discount", "VoteRank", "ClusterRank"}
            policies = [
                SeedPolicy(
                    movie.key,
                    display if display in topology else f"{display} ({args.model})",
                    budget,
                    tuple(info["seeds"][:budget]),
                    movie.id,
                )
                for movie in movies
                for display, info in all_results.items()
                for budget in range(1, args.k + 1)
            ]
            write_policies(args.results_dir / f"baseline_policies_{args.model}.csv", policies)

        print(f"{'═' * 60}")
        print(f"  DONE — {len(all_results)} algorithms, total {total_elapsed:.1f}s")
        print(f"  {csv_path}")
        print(f"{'═' * 60}")

        hdr = f"  {'Algorithm':<12s}  {'Time':>8s}  {'Seeds (k=' + str(args.k) + ')'}"
        print(f"\n{hdr}")
        print(f"  {'─' * len(hdr)}")
        for display, info in all_results.items():
            seeds_str = str(info["seeds"])
            if len(seeds_str) > 50:
                seeds_str = seeds_str[:47] + "...]"
            print(f"  {display:<12s}  {info['time']:>7.2f}s  {seeds_str}")
        print()


if __name__ == "__main__":
    main()
