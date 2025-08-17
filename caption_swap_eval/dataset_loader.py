from __future__ import annotations

import random
from typing import Dict, List, Tuple

from datasets import load_dataset

from models import Caption, Cartoon


DATASET_NAME = "jmhessel/newyorker_caption_contest"


def sample_unique_cartoons(
    n: int, seed: int
) -> Tuple[List[Cartoon], Dict[int, Caption]]:
    """
    Load the dataset and sample n unique cartoons by contest_number (image id),
    with one caption string per image from `caption_choices`.

    Returns:
      - list of Cartoon objects
      - captions dict keyed by caption_id (equal to image_id)
    """
    rng = random.Random(seed)

    ds = load_dataset(DATASET_NAME, name="explanation", split="train")

    # Map contest_number -> first seen row index
    seen: Dict[int, int] = {}
    for i, row in enumerate(ds):
        cid = int(row["contest_number"])  # image id
        if cid not in seen:
            seen[cid] = i

    unique_ids = list(seen.keys())
    if len(unique_ids) < n:
        raise ValueError(
            f"Requested n={n} but only {len(unique_ids)} unique cartoons available"
        )

    rng.shuffle(unique_ids)
    chosen_ids = sorted(unique_ids[:n])

    cartoons: List[Cartoon] = []
    captions: Dict[int, Caption] = {}

    for image_id in chosen_ids:
        idx = seen[image_id]
        row = ds[int(idx)]
        caption_text = str(row["caption_choices"])  # per task statement
        cartoons.append(Cartoon(image_id=image_id))
        captions[image_id] = Caption(caption_id=image_id, text=caption_text)

    return cartoons, captions
