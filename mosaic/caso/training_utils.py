"""Reproducible random state and movie-stratified condition splitting."""

import random
from typing import Sequence

import numpy as np
import torch

from .data import ConditionRecord


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(
    records: Sequence[ConditionRecord],
    validation_fraction: float,
    random_seed: int,
) -> tuple[list[ConditionRecord], list[ConditionRecord]]:
    """Movie-stratified condition split used only inside the pretraining pool."""

    generator = random.Random(random_seed)
    by_movie: dict[str, list[ConditionRecord]] = {}
    for record in records:
        by_movie.setdefault(record.movie, []).append(record)

    train: list[ConditionRecord] = []
    validation: list[ConditionRecord] = []
    for movie_records in by_movie.values():
        shuffled = list(movie_records)
        generator.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])
    generator.shuffle(train)
    generator.shuffle(validation)
    return train, validation
