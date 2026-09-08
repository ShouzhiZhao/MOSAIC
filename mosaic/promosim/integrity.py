"""Random-stream and reset-integrity controls for paired PromoSim replays."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


def capture_recommender(model: Any) -> dict:
    """Copy initialization only; do not share mutable tensors between policies."""
    return {
        "weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "runtime": {
            name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else None
            for name in ("user_content_emb", "item_content_emb", "graph")
            if hasattr(model, name)
            for value in [getattr(model, name)]
        },
    }


def restore_recommender(model: Any, state: dict) -> None:
    model.load_state_dict(state["weights"])
    device = next(model.parameters()).device
    for name, value in state["runtime"].items():
        setattr(model, name, value.to(device).clone() if value is not None else None)


def paired_rng_seed(
    base_seed: int,
    movie: str,
    budget: int,
    replication: int,
) -> int:
    """Derive a method-independent RNG seed for one paired replay cell."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if replication < 0:
        raise ValueError("replication must be non-negative")
    payload = f"{base_seed}\0{movie}\0{budget}\0{replication}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_replay(seed: int) -> None:
    """Reset every stochastic backend used while constructing a replay."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    tensors = dict(model.state_dict())
    for name in ("user_content_emb", "item_content_emb", "graph"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            tensors[f"runtime:{name}"] = value
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        if tensor.is_sparse:
            tensor = tensor.coalesce()
            digest.update(tensor.indices().numpy().tobytes())
            digest.update(tensor.values().numpy().tobytes())
        else:
            tensor = tensor.contiguous()
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initial_state_snapshot(sim: Any) -> dict[str, Any]:
    """Capture non-policy state immediately before the seed intervention."""
    agents = []
    for agent_id, agent in sorted(sim.agents.items()):
        memory = agent.memory
        sensory = getattr(getattr(memory, "sensoryMemory", None), "buffer", [])
        short = getattr(getattr(memory, "shortTermMemory", None), "short_memories", [])
        long_term = getattr(
            getattr(
                getattr(memory, "longTermMemory", None),
                "memory_retriever",
                None,
            ),
            "memory_stream",
            [],
        )
        fallback_stream = getattr(getattr(memory, "memory_retriever", None), "memory_stream", [])
        agents.append(
            {
                "id": int(agent_id),
                "profile": (
                    agent.name,
                    agent.age,
                    agent.gender,
                    agent.traits,
                    agent.status,
                    agent.interest,
                    agent.feature,
                    tuple(sorted(agent.relationships.items())),
                    float(agent.active_prob),
                ),
                "watched": tuple(agent.watched_history),
                "heard": tuple(agent.heared_history),
                "memory_lengths": (
                    len(sensory),
                    len(short),
                    len(long_term),
                    len(fallback_stream),
                ),
                "event": (
                    agent.event.action_type,
                    agent.event.start_time.isoformat(),
                    agent.event.end_time.isoformat() if agent.event.end_time else None,
                ),
                "no_action_round": int(agent.no_action_round),
            }
        )

    recsys = sim.recsys
    return {
        "round_cnt": int(sim.round_cnt),
        "now": sim.now.isoformat(),
        "users": sim.data.users,
        "items": sim.data.items,
        "relationship_count": int(sim.data.get_relationship_num()),
        "agents": agents,
        "recommender": {
            "model": _model_digest(recsys.model),
            "record": tuple(
                (int(key), tuple(value)) for key, value in sorted(recsys.record.items())
            ),
            "positive": tuple(
                (int(key), tuple(value)) for key, value in sorted(recsys.positive.items())
            ),
            "round_record": tuple(
                (int(key), tuple(tuple(entry) for entry in value))
                for key, value in sorted(recsys.round_record.items())
            ),
            "train_data": tuple(recsys.train_data),
            "inter_num": int(recsys.inter_num),
        },
    }


def assert_initial_state(
    sim: Any,
    reference_digest: str | None = None,
) -> dict[str, Any]:
    """Assert empty mutable state and equality with the paired replay start."""
    snapshot = initial_state_snapshot(sim)
    for agent in snapshot["agents"]:
        if agent["watched"] or agent["heard"] or any(agent["memory_lengths"]):
            raise AssertionError("replay started with non-empty agent history")

    recsys = snapshot["recommender"]
    collection_fields = ("record", "positive", "round_record")
    if (
        any(values for field in collection_fields for _, values in recsys[field])
        or recsys["train_data"]
    ):
        raise AssertionError("replay started with non-empty recommender state")
    if recsys["inter_num"] != 0:
        raise AssertionError("replay started with non-zero recommender interactions")

    digest = _state_digest(snapshot)
    if reference_digest is not None and digest != reference_digest:
        raise AssertionError(
            f"paired replay initial states differ: expected {reference_digest}, observed {digest}"
        )
    return {
        "state_digest": digest,
        "agents_checked": len(snapshot["agents"]),
        "history_entries": 0,
        "recommender_interactions": 0,
    }
