"""The structured answer every agent must produce: a single move, C or D."""

import json
import re
from typing import Literal

import axelrod as axl
from pydantic import BaseModel, ConfigDict, ValidationError

C, D = axl.Action.C, axl.Action.D

# JSON schema sent to the model (Anthropic `output_config.format`, OpenAI-compatible
# `response_format`). Kept hand-written so it is byte-identical across requests.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {"move": {"type": "string", "enum": ["C", "D"]}},
    "required": ["move"],
    "additionalProperties": False,
}

_ALIASES = {"c": "C", "cooperate": "C", "d": "D", "defect": "D"}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[^{}]*\}")


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move: Literal["C", "D"]

    def action(self) -> axl.Action:
        return C if self.move == "C" else D


class InvalidDecision(ValueError):
    """The model answered, but not with an actionable move."""


def parse_decision(text: str) -> Decision:
    """Parse a model's text output into a Decision.

    Structured outputs should make the text exactly `{"move": "C"}`; this also
    tolerates what OpenAI-compatible servers sometimes return around it
    (`<think>` blocks, code fences, surrounding prose).
    """
    if not text or not text.strip():
        raise InvalidDecision("empty response")
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    candidates = [cleaned] + _OBJECT_RE.findall(cleaned)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("move"), str):
            data = {**data, "move": _ALIASES.get(data["move"].strip().lower(), data["move"])}
        try:
            return Decision.model_validate(data)
        except ValidationError:
            continue
    raise InvalidDecision(f"no valid move in response: {text[:200]!r}")
