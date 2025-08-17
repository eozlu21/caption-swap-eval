from __future__ import annotations

import json
import os

from models import FullState, TranscriptEntry


def ensure_run_dir(base_dir: str, run_id: str) -> str:
    out_dir = os.path.join(base_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_state(state: FullState, out_dir: str) -> str:
    path = os.path.join(out_dir, "state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, sort_keys=True)
    return path


def save_state_step(
    state: FullState, out_dir: str, step_index: int | None = None
) -> str:
    """Persist a snapshot of the state with a step index (0 = initial).

    This does not replace the rolling state.json written by save_state; it
    creates a separate file alongside it, named by the step index for easy
    time-series analysis and to avoid overwrites.
    """
    idx = state.public.turn_index if step_index is None else step_index
    # Zero-pad to 5 digits to keep lexicographic order for long runs
    filename = f"state_step_{idx:05d}.json"
    path = os.path.join(out_dir, filename)
    # Write atomically-ish by writing directly; acceptable for local runs
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, sort_keys=True)
    return path


def append_transcript(entry: TranscriptEntry, out_dir: str) -> str:
    path = os.path.join(out_dir, "transcript.jsonl")
    line = json.dumps(entry.model_dump(), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
