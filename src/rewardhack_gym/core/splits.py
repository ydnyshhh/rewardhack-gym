from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar


SUPPORTED_DATASET_SPLITS = ("train", "dev", "eval", "stress", "heldout")

T = TypeVar("T")


def validate_dataset_split(split: str) -> str:
    if split not in SUPPORTED_DATASET_SPLITS:
        raise ValueError(f"Unknown dataset split {split!r}. Expected one of {SUPPORTED_DATASET_SPLITS}.")
    return split


def split_seed(base_seed: int, *, split: str, index: int = 0, salt: str = "") -> int:
    validate_dataset_split(split)
    payload = f"{base_seed}:{split}:{index}:{salt}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def split_order(items: Sequence[T], *, split: str, seed: int, salt: str = "") -> list[T]:
    validate_dataset_split(split)
    ordered = list(items)
    if len(ordered) <= 1:
        return ordered
    rng = random.Random(split_seed(seed, split=split, salt=salt))
    rng.shuffle(ordered)
    return ordered

