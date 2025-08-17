# caption-swap-eval

A minimal, deterministic Python project that runs an iterative caption–cartoon matching loop using the gpt-4o-mini API model. The loop maintains JSON state, asks the model to propose swapping exactly two assignments per turn (strict JSON only), updates correctness for each image, persists state and a transcript each step, and stops when all matches are correct.

Note: You asked for file creation only. No code is executed here. When you choose to run it later, follow the steps below.

## Highlights
- Model: `gpt-4o-mini`
- Temperature: `0.2`
- Response format: strict JSON matching `{ "swap": { "image_id_a": int, "image_id_b": int } }` (no prose)
- Prompt shows only correctness feedback and current assignment snapshot; no ground truth rule or caption texts are exposed.
- Dataset: `jmhessel/newyorker_caption_contest` (unique `contest_number` used as `image_id`; `caption_choices` used as caption text). No duplicate cartoons are sampled.
- Full state and a turn-by-turn transcript persisted to disk after each step.

## Structure
- `caption_swap_eval/models.py` – Pydantic models for state, swaps, and transcript entries; correctness logic and swap application.
- `caption_swap_eval/dataset_loader.py` – Samples `n` unique cartoons from the HF dataset and builds caption objects.
- `caption_swap_eval/state.py` – Deterministic initialization with a derangement (no initial correct matches).
- `caption_swap_eval/llm.py` – Builds a safe prompt and calls OpenAI Chat Completions with JSON-only responses.
- `caption_swap_eval/engine.py` – Orchestrates the loop, applies swaps, updates correctness, persists outputs.
- `caption_swap_eval/persistence.py` – Saves `state.json` and appends `transcript.jsonl`.
- `caption_swap_eval/cli.py` – Simple CLI wrapper to run the loop.
- `.env.example` – Environment variables template.
- `requirements.txt` – Minimal dependencies.

## Install (when you’re ready to run)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configure
* Create `.env` from the template and fill in your key:
```
cp .env.example .env
```
Edit `.env` and set `OPENAI_API_KEY`.

* Optionally set a seed for deterministic sampling and derangement:
```
SEED=42
```

## Run
The CLI will sample `n` unique cartoons, initialize a wrong assignment derangement, then iterate until all are correct.

```bash
python -m caption_swap_eval.cli --n 6 --seed 42 --out runs
```
- Output directory: `runs/<run_id>/`
  - `state.json` – Full state (public + private) after the latest step
  - `transcript.jsonl` – One JSON line per turn with prompt, raw model JSON, parsed swap, applied swap, and correctness map

You can cap iterations (safety) with `--max-turns`.

## JSON Contract
- Model output must strictly match:
```json
{"swap": {"image_id_a": 123, "image_id_b": 456}}
```
- Exactly two distinct `image_id`s that exist in the current state.
- No additional keys or prose. The SDK enforces `response_format: {"type": "json_object"}`.

## Prompt Safety
- The prompt exposes only: `(image_id, assigned_caption_id, is_correct)` for each image, plus loop metadata.
- It does not include any caption text nor the private rule tying images to captions.

## Determinism
- Dataset sampling order and derangement are seeded.
- The model call uses temperature `0.2` and JSON-only responses for consistency, but remote model behavior can still vary slightly.

## Tests (optional to run)
Basic unit tests are provided:
- `tests/test_core.py` – Derangement property and swap/correctness mechanics.

Run them with:
```bash
pytest -q
```

## Troubleshooting
- Ensure `OPENAI_API_KEY` is set.
- The first dataset load will download from Hugging Face. If you need offline runs, pre-download or vendor a small subset.
- If the model returns invalid JSON or unknown IDs, the loop records the error in the transcript and stops.

## License
MIT
