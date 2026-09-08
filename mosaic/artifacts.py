"""Atomic artifact publication and reproducible run provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(
        path,
        lambda f: f.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        ),
    )


def environment_metadata() -> dict:
    packages = {}
    for name in (
        "mosaic-caso",
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "networkx",
        "openai",
        "langchain",
        "transformers",
        "sentence-transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
