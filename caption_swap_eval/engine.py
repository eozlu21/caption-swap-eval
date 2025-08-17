from __future__ import annotations

import json
from typing import Callable, Tuple

from llm import build_prompt, propose_swap
from models import FullState, SwapProposal, TranscriptEntry
from persistence import append_transcript, ensure_run_dir, save_state


ProposerFn = Callable[[FullState], Tuple[SwapProposal, str]]


def default_proposer(full: FullState) -> Tuple[SwapProposal, str]:
    # Build messages and call the model
    messages = build_prompt(full.public)
    proposal, raw = propose_swap(full.public, messages=messages)
    return proposal, raw


def run_until_solved(full: FullState, out_root: str = "runs", max_turns: int | None = 10000) -> str:
    """
    Run the loop until all assignments are correct or max_turns reached.
    Persists state.json and transcript.jsonl after each step.

    Returns the output directory used for persistence.
    """
    out_dir = ensure_run_dir(out_root, full.public.run_id)

    # Save initial
    save_state(full, out_dir)

    turns = 0
    while not full.public.done:
        if max_turns is not None and turns >= max_turns:
            break

        messages = build_prompt(full.public)
        prompt_serialized = json.dumps(messages)
        entry = TranscriptEntry(turn_index=full.public.turn_index, prompt=prompt_serialized)

        try:
            proposal, raw = propose_swap(full.public, messages=messages)
            entry.model_raw = raw
            entry.proposal = proposal

            # Apply
            full.apply_swap(proposal.swap)
            entry.applied_swap = proposal.swap
            entry.correctness_after = full.correctness_map()
        except Exception as e:  # noqa: BLE001
            entry.error = str(e)
            # Persist the error and break
            append_transcript(entry, out_dir)
            save_state(full, out_dir)
            break

        # Persist after successful step
        append_transcript(entry, out_dir)
        save_state(full, out_dir)
        turns += 1

    # Final save
    save_state(full, out_dir)
    return out_dir
