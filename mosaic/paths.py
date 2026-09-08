"""Configurable filesystem locations for the standalone MOSAIC package."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Editable checkouts use their repository root; wheels use the caller's workspace.
_SOURCE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()

DATA_DIR = Path(os.environ.get("MOSAIC_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
PROMOSIM_DATA_DIR = (
    Path(os.environ.get("MOSAIC_PROMOSIM_DATA_DIR", DATA_DIR / "promosim")).expanduser().resolve()
)
PROMOSIM_SIMULATION_DATA_DIR = (
    Path(os.environ.get("MOSAIC_SIMULATION_DATA_DIR", DATA_DIR / "promosim_records"))
    .expanduser()
    .resolve()
)
ARTIFACTS_DIR = (
    Path(os.environ.get("MOSAIC_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).expanduser().resolve()
)
CASO_ARTIFACTS_DIR = ARTIFACTS_DIR / "caso"
CASO_CHECKPOINTS_DIR = CASO_ARTIFACTS_DIR / "checkpoints"
CASO_RESULTS_DIR = CASO_ARTIFACTS_DIR / "results"
PROMOSIM_CONFIG_DIR = PACKAGE_ROOT / "promosim" / "config"


def get_relationship_file(filename: str = "relationship_1000.csv") -> Path:
    return PROMOSIM_DATA_DIR / filename


def get_caso_checkpoint(filename: str = "caso.pt") -> Path:
    return CASO_CHECKPOINTS_DIR / filename


def get_content_match_catalog(
    filename: str = "content_match_zero_shot.npz",
) -> Path:
    return CASO_ARTIFACTS_DIR / filename


def get_promosim_config(filename: str = "default.yaml") -> Path:
    return PROMOSIM_CONFIG_DIR / filename


def get_simulation_output_dir(target_movie: str | None = None) -> Path:
    if target_movie is None:
        return PROMOSIM_SIMULATION_DATA_DIR
    safe_name = target_movie.replace("/", "_").replace(" ", "_")
    return PROMOSIM_SIMULATION_DATA_DIR / safe_name


def get_simulation_item_file() -> Path:
    return PROMOSIM_DATA_DIR / "item.csv"


def get_simulation_user_file() -> Path:
    return PROMOSIM_DATA_DIR / "user_1000.csv"


def get_simulation_interaction_file() -> Path:
    return PROMOSIM_DATA_DIR / "interaction.csv"


def get_simulation_faiss_index_dir() -> Path:
    return PROMOSIM_DATA_DIR / "faiss_index"
