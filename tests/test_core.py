from __future__ import annotations

from caption_swap_eval.models import (
    AssignmentEntry,
    FullState,
    PrivateGroundTruth,
    PublicState,
    Swap,
    ensure_derangement,
)


def test_ensure_derangement():
    assert ensure_derangement([1, 2, 3, 4]) is False
    assert ensure_derangement([2, 1]) is True
    assert ensure_derangement([2, 3, 1]) is True


def make_small_state():
    # image ids: 10, 20, 30
    public = PublicState(
        n=3,
        run_id="test",
        seed=0,
        assignments=[
            AssignmentEntry(image_id=10, caption_id_assigned=20),
            AssignmentEntry(image_id=20, caption_id_assigned=30),
            AssignmentEntry(image_id=30, caption_id_assigned=10),
        ],
        captions={},
    )
    private = PrivateGroundTruth(mapping={10: 10, 20: 20, 30: 30})
    full = FullState(public=public, private=private)
    full.set_correctness()
    return full


def test_apply_swap_and_correctness():
    full = make_small_state()

    # Initially nothing is correct in this 3-cycle derangement
    assert all(not a.is_correct for a in full.public.assignments)

    # Swap 10 and 30 => assignments become: 10<-10 (correct), 20<-30 (wrong), 30<-20 (wrong)
    full.apply_swap(Swap(image_id_a=10, image_id_b=30))

    cm = full.correctness_map()
    assert cm[10] is True
    assert cm[20] is False
    assert cm[30] is False
    assert full.public.turn_index == 1

