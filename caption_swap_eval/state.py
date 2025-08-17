from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dataset_loader import sample_unique_cartoons
from models import (
    AssignmentEntry,
    Caption,
    FullState,
    PrivateGroundTruth,
    PublicState,
    ensure_derangement,
)


@dataclass
class InitConfig:
    n: int
    seed: int = 42
    run_id: str | None = None


def _random_derangement(ids: List[int], rng: random.Random) -> List[int]:
    """Return a derangement of the given ids (no fixed points)."""
    if len(ids) < 2:
        raise ValueError("Need at least 2 items to create a derangement")
    while True:
        perm = ids[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            return perm


def initialize_state(n: int, seed: int, run_id: str | None = None) -> Tuple[FullState, Dict[int, Caption]]:
    rng = random.Random(seed)
    cartoons, captions = sample_unique_cartoons(n=n, seed=seed)

    image_ids = sorted([c.image_id for c in cartoons])
    perm = _random_derangement(image_ids, rng)
    assert ensure_derangement([caption_id for caption_id in perm])

    assignments: List[AssignmentEntry] = []
    for image_id, assigned_caption in zip(image_ids, perm):
        assignments.append(
            AssignmentEntry(image_id=image_id, caption_id_assigned=assigned_caption, is_correct=False)
        )

    if run_id is None:
        run_id = f"run_{int(time.time())}_{seed}_{n}"

    public = PublicState(
        n=n,
        run_id=run_id,
        seed=seed,
        turn_index=0,
        done=False,
        assignments=assignments,
        captions=captions,
    )

    # Private truth: correct caption for image_id is the same id
    private = PrivateGroundTruth(mapping={i: i for i in image_ids})

    full = FullState(public=public, private=private)
    full.set_correctness()  # will be all False by construction

    return full, captions
