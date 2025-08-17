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


def append_transcript(entry: TranscriptEntry, out_dir: str) -> str:
    path = os.path.join(out_dir, "transcript.jsonl")
    line = json.dumps(entry.model_dump(), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
