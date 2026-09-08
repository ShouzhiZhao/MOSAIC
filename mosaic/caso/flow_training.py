#!/usr/bin/env python3
"""Train and diagnose CASO seed flow using only synthetic k=0..10 subsets."""

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch

from mosaic.caso.models import RectifiedFlowSeedModel
from mosaic.caso.seed_sampling import (
    rank_couple_endpoint,
    sample_random_seed_masks,
    sample_seed_masks,
)
from mosaic.caso.training_utils import seed_everything
from mosaic.config import ModelConfig


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--num-nodes", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variant", choices=["mlp", "equivariant"], default="equivariant")
    p.add_argument("--coupling", choices=["independent", "rank"], default="rank")
    p.add_argument("--updates", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    p.add_argument("--evaluate-every", type=int, default=500)
    p.add_argument("--validation-size", type=int, default=4096)
    p.add_argument("--test-size", type=int, default=8192)
    p.add_argument("--generation-size", type=int, default=4096)
    p.add_argument("--shared-hidden-dim", type=int, default=64)
    p.add_argument("--context-topk", type=int, default=16)
    p.add_argument("--budget-conditioned", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dequantization-width", type=float, default=0.1)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--solver", choices=["euler", "rk4"], default="rk4")
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument(
        "--tail-weight",
        type=float,
        default=100.0,
        help="Extra weight for rare high coordinates; depends only on the current state/time",
    )
    p.add_argument("--generation-budget-weight", type=float, default=0.0)
    p.add_argument("--generation-every", type=int, default=10)
    p.add_argument("--generation-batch-size", type=int, default=16)
    p.add_argument("--generation-training-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--diagnose-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if (
        min(
            args.updates,
            args.batch_size,
            args.evaluate_every,
            args.validation_size,
            args.test_size,
            args.generation_size,
            args.shared_hidden_dim,
            args.ode_steps,
        )
        < 1
    ):
        raise ValueError("sizes and counts must be positive")
    if not 0 <= args.ema < 1 or not 0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("invalid EMA or learning rate")
    if min(args.tail_weight, args.generation_budget_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if min(args.generation_every, args.generation_batch_size, args.generation_training_steps) < 1:
        raise ValueError("generation regularization sizes must be positive")
    if (args.output_dir / "report.json").exists() or (
        args.output_dir / "flow_pretrained.pt"
    ).exists():
        raise FileExistsError("Use a new output directory")
    if args.diagnose_only and not args.resume:
        raise ValueError("diagnose-only requires a checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.num_nodes < 10 or not 0 <= args.context_topk <= args.num_nodes:
        raise ValueError("Seed budgets require at least 10 nodes; context-topk must fit the graph")
    config = ModelConfig(
        num_nodes=args.num_nodes,
        flow_velocity_variant=args.variant,
        flow_shared_hidden_dim=args.shared_hidden_dim,
        flow_context_topk=args.context_topk,
        flow_budget_conditioned=args.budget_conditioned,
        flow_dequantization_width=args.dequantization_width,
        flow_steps=args.ode_steps,
        flow_solver=args.solver,
    )
    source = None
    if args.resume:
        source = torch.load(args.resume, map_location=device, weights_only=False)
        if source.get("format") != "mosaic-seed-flow-v1":
            raise ValueError("Expected a standalone seed-flow checkpoint")
        source_config = ModelConfig(**source["model_config"])
        if source_config.flow_velocity_variant != args.variant:
            raise ValueError("Resume architecture must match --variant")
        config = replace(source_config, flow_steps=args.ode_steps, flow_solver=args.solver)
    model = RectifiedFlowSeedModel(config).to(device)
    if source:
        model.load_state_dict(source["flow_state"])
    initial = copy.deepcopy(model)
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    n = config.num_nodes
    source_hash = hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None

    def generator(offset):
        return torch.Generator(device=device).manual_seed(args.seed + offset)

    def sample(size, rng):
        seed = sample_random_seed_masks(n, size, device=device, generator=rng)
        endpoint = model.logit_dequantize(seed, stochastic=True, generator=rng)
        noise = torch.randn(endpoint.shape, device=device, generator=rng)
        if args.coupling == "rank":
            endpoint = rank_couple_endpoint(noise, endpoint)
            seed = (endpoint >= 0).float()
        t = torch.rand(size, device=device, generator=rng)
        state = (1 - t[:, None]) * noise + t[:, None] * endpoint
        return seed, state, t, endpoint - noise

    validation = sample(args.validation_size, generator(301))
    gen_validation_rng = generator(306)
    gen_validation_budgets = torch.randint(
        0, 11, (512,), device=device, generator=gen_validation_rng
    )
    gen_validation_noise = torch.randn(512, n, device=device, generator=gen_validation_rng)

    @torch.no_grad()
    def measure_generation(net):
        counts = torch.cat(
            [
                (
                    net.decode(gen_validation_noise[idx], budget=gen_validation_budgets[idx]) >= 0.5
                ).sum(1)
                for idx in torch.arange(512, device=device).split(128)
            ]
        )
        return {
            "generation_budget_mae": (counts - gen_validation_budgets).float().abs().mean().item(),
            "generation_exact_budget_fraction": (counts == gen_validation_budgets)
            .float()
            .mean()
            .item(),
        }

    def weights(state, t):
        # State/time weighting preserves the population conditional-mean
        # velocity optimum. Weighting by the unknown endpoint's seed bit
        # instead would change the target vector field and is not used here.
        return 1 + args.tail_weight * torch.sigmoid((state - 2.5 * (1 - t[:, None])) / 0.15)

    @torch.no_grad()
    def measure(net, data):
        seed, state, t, target = data
        totals = torch.zeros(5, dtype=torch.float64, device=device)
        for idx in torch.arange(len(seed), device=device).split(128):
            error = (
                (net.velocity_at(state[idx], t[idx], seed[idx].sum(1)) - target[idx])
                .square()
                .double()
            )
            weight = weights(state[idx], t[idx]).double()
            totals += torch.stack(
                [
                    error.sum(),
                    (error * seed[idx]).sum(),
                    (error * (1 - seed[idx])).sum(),
                    (error * weight).sum(),
                    weight.sum(),
                ]
            )
        seed_count = seed.sum().item()
        return {
            "flow_mse": totals[0].item() / seed.numel(),
            "weighted_flow_mse": (totals[3] / totals[4]).item(),
            "seed_coordinate_mse": totals[1].item() / seed_count,
            "nonseed_coordinate_mse": totals[2].item() / (seed.numel() - seed_count),
        }

    initial_metrics = measure(model, validation)
    if args.generation_budget_weight:
        if not model.budget_conditioned:
            raise ValueError("generation budget regularization requires a conditioned model")
        initial_metrics.update(measure_generation(model))
    best_metrics = initial_metrics
    selection_key = (
        "generation_budget_mae"
        if args.generation_budget_weight
        else "weighted_flow_mse"
        if args.tail_weight
        else "flow_mse"
    )
    best = best_metrics[selection_key]
    best_step, best_kind = 0, "initial"
    best_state = copy.deepcopy(model.state_dict())
    history = [
        {
            "step": 0,
            "samples_drawn": 0,
            "raw_validation_mse": best,
            "ema_validation_mse": initial_metrics["flow_mse"],
            "best_selection_value": best,
        }
    ]
    start = time.monotonic()
    budget_counts = torch.zeros(11, dtype=torch.long, device=device)
    generation_draws = 0
    if not args.diagnose_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.updates, eta_min=args.minimum_learning_rate
        )
        rng = generator(302)
        for step in range(1, args.updates + 1):
            sampled_seed, state, t, target = sample(args.batch_size, rng)
            budget_counts += torch.bincount(sampled_seed.sum(1).long(), minlength=11)
            error = (model.velocity_at(state, t, sampled_seed.sum(1)) - target).square()
            weight = weights(state, t)
            loss = (error * weight).sum() / weight.sum()
            if args.generation_budget_weight and step % args.generation_every == 0:
                count_budget = torch.randint(
                    0, 11, (args.generation_batch_size,), device=device, generator=rng
                )
                count_noise = torch.randn(
                    args.generation_batch_size, n, device=device, generator=rng
                )
                saved_steps = model.steps
                model.steps = args.generation_training_steps
                generated_logits = model.integrate(count_noise, budget=count_budget)
                model.steps = saved_steps
                soft_count = torch.sigmoid(generated_logits / 0.25).sum(1)
                loss = (
                    loss
                    + args.generation_budget_weight * (soft_count - count_budget).square().mean()
                )
                generation_draws += args.generation_batch_size
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, 1 - args.ema)
            if step == 1 or step % args.evaluate_every == 0 or step == args.updates:
                raw_metrics = measure(model, validation)
                ema_metrics = measure(ema, validation)
                if args.generation_budget_weight:
                    raw_metrics.update(measure_generation(model))
                    ema_metrics.update(measure_generation(ema))
                for kind, candidate, measured in [
                    ("raw", model, raw_metrics),
                    ("ema", ema, ema_metrics),
                ]:
                    if measured[selection_key] < best:
                        best, best_metrics, best_step, best_kind = (
                            measured[selection_key],
                            measured,
                            step,
                            kind,
                        )
                        best_state = copy.deepcopy(candidate.state_dict())
                history.append(
                    {
                        "step": step,
                        "samples_drawn": step * args.batch_size,
                        "training_loss": loss.item(),
                        "raw_validation_mse": raw_metrics["flow_mse"],
                        "ema_validation_mse": ema_metrics["flow_mse"],
                        "best_selection_value": best,
                        "raw_weighted_validation_mse": raw_metrics["weighted_flow_mse"],
                        "ema_weighted_validation_mse": ema_metrics["weighted_flow_mse"],
                        "raw_generation_budget_mae": raw_metrics.get("generation_budget_mae"),
                        "ema_generation_budget_mae": ema_metrics.get("generation_budget_mae"),
                        "elapsed_seconds": time.monotonic() - start,
                    }
                )
                print(
                    f"step={step} draws={step * args.batch_size} raw={raw_metrics['flow_mse']:.6f} ema={ema_metrics['flow_mse']:.6f} best={best:.6f}",
                    flush=True,
                )
                pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
                # Allow recovery after interruption without selecting on test data.
                torch.save(
                    {
                        "format": "mosaic-seed-flow-v1",
                        "model_config": asdict(config),
                        "flow_state": best_state,
                        "metadata": {
                            "best_step": best_step,
                            "samples_drawn": step * args.batch_size,
                            "budgets": [0, 10],
                            "best_validation_flow_mse": best_metrics["flow_mse"],
                            "simulation_labels_used": 0,
                            "selection_metric": selection_key,
                            "best_selection_value": best,
                            "coupling": args.coupling,
                        },
                    },
                    args.output_dir / "best_so_far.pt",
                )
    model.load_state_dict(best_state)
    model.eval()
    # A separately sampled test stream is first evaluated after selection.
    test = sample(args.test_size, generator(303))

    @torch.no_grad()
    def diagnose(net, noise, reference, solver, steps):
        original_solver, original_steps = net.solver, net.steps
        net.solver, net.steps = solver, steps
        requested = torch.arange(len(noise), device=device) % 11
        generated = torch.cat(
            [
                net.decode(noise[idx], budget=requested[idx])
                for idx in torch.arange(len(noise), device=device).split(128)
            ]
        )
        counts = (generated >= 0.5).sum(1)
        freq = torch.bincount(counts, minlength=n + 1).double() / len(counts)
        uniform = torch.zeros(n + 1, dtype=torch.float64, device=device)
        uniform[:11] = 1 / 11
        rebuilt = torch.cat([net.reconstruct(s, stochastic=False) for s in reference.split(128)])
        budgets = reference.sum(1).long()
        ranks = rebuilt.topk(10, dim=1).indices
        mask = torch.arange(10, device=device)[None, :] < budgets[:, None]
        hits = (reference.gather(1, ranks) * mask).sum(1)
        width = net.dequantization_width
        expected = (1 - width) * reference + width / 2
        # Uniform dequantization within each binary half should also be reproduced.
        nonseed = generated[generated < 0.5]
        positive = generated[generated >= 0.5]
        q = torch.linspace(0.01, 0.99, 99, device=device)
        p_quantile_error = (
            (torch.quantile(positive, q) - (1 - width + q * width)).abs().max().item()
            if len(positive)
            else None
        )
        original_noise = noise[:128]
        cycled = net.integrate(
            net.integrate(original_noise, budget=requested[:128]),
            reverse=True,
            budget=requested[:128],
        )
        measured = {
            "solver": solver,
            "steps": steps,
            "velocity_calls": steps * (4 if solver == "rk4" else 1),
            "samples": len(counts),
            "generated_budget_mean": counts.float().mean().item(),
            "generated_budget_std": counts.float().std(unbiased=False).item(),
            "generated_budget_min": counts.min().item(),
            "generated_budget_max": counts.max().item(),
            "fraction_in_0_to_10": (counts <= 10).float().mean().item(),
            "empty_fraction": (counts == 0).float().mean().item(),
            "budget_total_variation_from_uniform": ((freq - uniform).abs().sum() / 2).item(),
            "budget_wasserstein_from_uniform": (freq.cumsum(0) - uniform.cumsum(0))
            .abs()
            .sum()
            .item(),
            "budget_histogram": {
                str(k): int(v * len(counts)) for k, v in enumerate(freq.cpu().tolist()) if v
            },
            "nonseed_quantile_max_error": (torch.quantile(nonseed, q) - q * width)
            .abs()
            .max()
            .item()
            if len(nonseed)
            else None,
            "seed_quantile_max_error": p_quantile_error,
            "roundtrip_midpoint_mse": (rebuilt - expected).square().mean().item(),
            "roundtrip_max_error": (rebuilt - expected).abs().max().item(),
            "roundtrip_bit_accuracy": ((rebuilt >= 0.5) == reference.bool()).float().mean().item(),
            "roundtrip_exact_set_fraction": (hits == budgets).float().mean().item(),
            "topk_seed_recall": (hits[budgets > 0] / budgets[budgets > 0]).float().mean().item(),
            "noise_roundtrip_mse": (cycled - original_noise).square().mean().item(),
        }
        if net.budget_conditioned:
            measured["conditional_budget_mae"] = (counts - requested).float().abs().mean().item()
            measured["conditional_exact_budget_fraction"] = (
                (counts == requested).float().mean().item()
            )
            measured["by_requested_budget"] = {
                str(k): {
                    "samples": int((requested == k).sum()),
                    "mean_generated_budget": counts[requested == k].float().mean().item(),
                    "exact_budget_fraction": (counts[requested == k] == k).float().mean().item(),
                }
                for k in range(11)
            }
        net.solver, net.steps = original_solver, original_steps
        return measured

    noise = torch.randn(args.generation_size, n, device=device, generator=generator(304))
    reference = sample_seed_masks(
        n, torch.arange(512, device=device) % 11, generator=generator(305)
    )
    diagnostics = []
    for solver, steps in [("euler", 6), ("euler", 64), ("rk4", 32), ("rk4", 64)]:
        measured = diagnose(model, noise, reference, solver, steps)
        diagnostics.append(measured)
        print(
            "GENERATION",
            json.dumps({k: v for k, v in measured.items() if k != "budget_histogram"}),
            flush=True,
        )
    draws = 0 if args.diagnose_only else args.updates * args.batch_size
    metadata = {
        "budgets": [0, 10],
        "samples_drawn": draws,
        "coupling": args.coupling,
        "samples_drawn_per_budget": {
            str(k): count for k, count in enumerate(budget_counts.cpu().tolist())
        },
        "samples_are_unique": False,
        "simulation_labels_used": 0,
        "best_step": best_step,
        "best_weights": best_kind,
        "samples_at_best_step": best_step * args.batch_size,
        "best_validation_flow_mse": best_metrics["flow_mse"],
        "selection": "minimum independent validation loss across raw and EMA weights",
        "selection_metric": selection_key,
        "best_selection_value": best,
        "tail_weight": args.tail_weight,
        "generation_budget_weight": args.generation_budget_weight,
        "generation_regularization_draws": generation_draws,
        "source_checkpoint": str(args.resume) if args.resume else None,
        "source_checkpoint_sha256": source_hash,
        "source_metadata": source.get("metadata") if source else None,
        "sampling": "k drawn iid uniformly from integers 0..10, then a uniform subset of k distinct nodes",
        "random_seed": args.seed,
    }
    torch.save(
        {
            "format": "mosaic-seed-flow-v1",
            "model_config": asdict(config),
            "flow_state": model.state_dict(),
            "metadata": metadata,
        },
        args.output_dir / "flow_pretrained.pt",
    )
    report = {
        "metadata": metadata,
        "config": asdict(config),
        "initial_validation": initial_metrics,
        "selected_validation": best_metrics,
        "initial_test": measure(initial, test),
        "selected_test": measure(model, test),
        "generation_diagnostics": diagnostics,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": time.monotonic() - start,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    if args.resume:
        assert hashlib.sha256(args.resume.read_bytes()).hexdigest() == source_hash
    print(
        "FINISHED",
        json.dumps(
            {
                "validation": best_metrics,
                "test": report["selected_test"],
                "best_step": best_step,
                "seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
