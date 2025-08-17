from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2


class Cartoon(BaseModel):
    image_id: int  # contest_number


class Caption(BaseModel):
    caption_id: int  # equals the image_id of its cartoon
    text: str


class AssignmentEntry(BaseModel):
    image_id: int
    caption_id_assigned: int
    is_correct: bool = False


class Swap(BaseModel):
    image_id_a: int
    image_id_b: int

    @classmethod
    def distinct(cls, v, values):
        if "image_id_a" in values and v == values["image_id_a"]:
            raise ValueError("image_id_a and image_id_b must be distinct")
        return v


class SwapProposal(BaseModel):
    swap: Swap


class TurnRecord(BaseModel):
    turn_index: int
    prompt: str
    model_raw: str
    parsed: Optional[SwapProposal] = None
    applied_swap: Optional[Swap] = None
    correctness_summary: Dict[int, bool] = Field(default_factory=dict)


class PrivateGroundTruth(BaseModel):
    # image_id -> caption_id (equal to image_id for 1-1 mapping)
    mapping: Dict[int, int]


class PublicState(BaseModel):
    n: int
    run_id: str
    seed: int
    model: str = MODEL_NAME
    temperature: float = TEMPERATURE
    turn_index: int = 0
    done: bool = False
    # entries ordered by image_id increasing for stability
    assignments: List[AssignmentEntry] = Field(default_factory=list)
    # captions store for current assignment display
    captions: Dict[int, Caption] = Field(default_factory=dict)


class FullState(BaseModel):
    public: PublicState
    private: PrivateGroundTruth

    def correctness_map(self) -> Dict[int, bool]:
        correct: Dict[int, bool] = {}
        for a in self.public.assignments:
            correct[a.image_id] = (
                self.private.mapping.get(a.image_id) == a.caption_id_assigned
            )
        return correct

    def set_correctness(self) -> None:
        m = self.correctness_map()
        for a in self.public.assignments:
            a.is_correct = m[a.image_id]
        self.public.done = all(a.is_correct for a in self.public.assignments)

    def apply_swap(self, swap: Swap) -> None:
        # find entries by image_id
        idx_a = next(
            i
            for i, a in enumerate(self.public.assignments)
            if a.image_id == swap.image_id_a
        )
        idx_b = next(
            i
            for i, a in enumerate(self.public.assignments)
            if a.image_id == swap.image_id_b
        )
        a_entry = self.public.assignments[idx_a]
        b_entry = self.public.assignments[idx_b]
        # swap assigned captions
        a_entry.caption_id_assigned, b_entry.caption_id_assigned = (
            b_entry.caption_id_assigned,
            a_entry.caption_id_assigned,
        )
        # update list
        self.public.assignments[idx_a] = a_entry
        self.public.assignments[idx_b] = b_entry
        # recompute correctness and advance turn
        self.set_correctness()
        self.public.turn_index += 1


class TranscriptEntry(BaseModel):
    turn_index: int
    # Added prompt and model_raw for richer traceability
    prompt: Optional[str] = None
    model_raw: Optional[str] = None
    proposal: Optional[SwapProposal] = None
    applied_swap: Optional[Swap] = None
    correctness_after: Dict[int, bool] = Field(default_factory=dict)
    error: Optional[str] = None


def ensure_derangement(permutation: List[int]) -> bool:
    # permutation is over 1-based ids; treat index positions as 1-based
    return all((i + 1) != p for i, p in enumerate(permutation))
