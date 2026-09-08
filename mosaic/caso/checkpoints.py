"""Versioned CASO checkpoint I/O."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mosaic.artifacts import atomic_write
from mosaic.config import ModelConfig

from .models import CASOModel

CHECKPOINT_FORMAT = "mosaic-caso-v1"


def save_checkpoint(
    path: Path,
    model: CASOModel,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        lambda f: torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "model_config": asdict(model.config),
                "model_state": model.state_dict(),
                "metadata": metadata,
            },
            f,
        ),
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[CASOModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}; expected {CHECKPOINT_FORMAT}")
    model = CASOModel(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
