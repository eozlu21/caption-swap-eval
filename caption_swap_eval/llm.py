from __future__ import annotations

import json
from typing import List, Tuple, Union

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONObject

from models import MODEL_NAME, TEMPERATURE, PublicState, SwapProposal


MessageParam = Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]


def build_prompt(state: PublicState) -> List[MessageParam]:
    """
    Build a prompt that reveals only correctness feedback and the current permutation structure,
    not the caption texts nor the ground truth mapping.

    The assistant must respond with strict JSON: {"swap": {"image_id_a": int, "image_id_b": int}}
    """
    # Order assignments by image_id for stability
    assignments = sorted(state.assignments, key=lambda a: a.image_id)

    # Build a safe snapshot exposing only image ids, assigned caption ids, and correctness flags.
    # This does not expose any caption text or the explicit ground truth rule.
    snapshot = [
        {
            "image_id": a.image_id,
            "assigned_caption_id": a.caption_id_assigned,
            "is_correct": a.is_correct,
        }
        for a in assignments
    ]

    system = (
        "You are a JSON-only tool that proposes exactly one swap between two image assignments per turn. "
        "Reply with strict JSON only, matching the schema {\"swap\": {\"image_id_a\": int, \"image_id_b\": int}}. "
        "Do not include any extra keys or text."
    )

    user = {
        "objective": "Propose swapping exactly two image assignments to increase the number of correct matches.",
        "temperature": TEMPERATURE,
        "model": MODEL_NAME,
        "instructions": [
            "Return strict JSON only.",
            "Choose two distinct image_id values present below.",
            "Propose a swap that you believe will improve correctness.",
        ],
        "state": {
            "turn_index": state.turn_index,
            "n": state.n,
            "snapshot": snapshot,
        },
        "response_schema": {"swap": {"image_id_a": "int", "image_id_b": "int"}},
    }

    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=json.dumps(user)),
    ]


def propose_swap(
    state: PublicState,
    client: OpenAI | None = None,
    messages: List[MessageParam] | None = None,
) -> Tuple[SwapProposal, str]:
    if client is None:
        client = OpenAI()

    if messages is None:
        messages = build_prompt(state)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,  # type: ignore[arg-type]
        temperature=TEMPERATURE,
        response_format=ResponseFormatJSONObject(type="json_object"),
    )
    content = resp.choices[0].message.content or "{}"

    # Strict parse
    data = json.loads(content)
    proposal = SwapProposal.model_validate(data)

    # Validate image ids exist in state
    ids = {a.image_id for a in state.assignments}
    a = proposal.swap.image_id_a
    b = proposal.swap.image_id_b
    if a not in ids or b not in ids:
        raise ValueError("Proposed image ids not in state")

    return proposal, content
