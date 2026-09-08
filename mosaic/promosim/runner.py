"""Launch the packaged behavioral simulator through the PromoSim API."""

import os
import subprocess
import sys
from pathlib import Path

from mosaic import paths


def run(
    target_movie_id: int,
    config_file: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    seed: int = 2026,
    replications: int = 10,
    resume_condition: str | None = None,
) -> None:
    """Run one PromoSim data-generation condition.

    The simulator runs in a fresh Python process because its GPU-backed text
    encoder and LLM singletons are intentionally process scoped.
    """

    environment = os.environ.copy()
    environment["TARGET_MOVIE_ID"] = str(target_movie_id)
    if data_dir is not None:
        environment["MOSAIC_PROMOSIM_DATA_DIR"] = str(Path(data_dir).resolve())
    if output_dir is not None:
        environment["MOSAIC_SIMULATION_DATA_DIR"] = str(Path(output_dir).resolve())
    config = Path(config_file) if config_file else paths.get_promosim_config()
    command = [
        sys.executable,
        "-m",
        "mosaic.promosim.generation",
        "--config_file",
        str(config.resolve()),
        "--output_file",
        "promosim_messages.json",
    ]
    command.extend(["--seed", str(seed), "--replications", str(replications)])
    if resume_condition:
        command.extend(["--resume-condition", resume_condition])
    subprocess.run(
        command,
        cwd=paths.PROJECT_ROOT,
        env=environment,
        check=True,
    )
