"""The structured answer every agent must produce: a single move, C or D."""

import json
import re
from typing import Literal

import axelrod as axl
from pydantic import BaseModel, ConfigDict, ValidationError

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
        return axl.Action.from_char(self.move)


class InvalidDecision(ValueError):
    """The model answered, but not with an actionable move."""


def parse_decision(text: str) -> Decision:
    """Strict parse, for servers that enforce the schema."""
    try:
        return Decision.model_validate_json(text)
    except ValidationError as exc:
        raise InvalidDecision(f"no valid move in response: {text[:200]!r}") from exc


def parse_decision_lenient(text: str) -> Decision:
    """For OpenAI-compatible servers that don't fully enforce `response_format`:
    tolerates `<think>` blocks, code fences, surrounding prose and spelled-out moves."""
    cleaned = _THINK_RE.sub("", text or "").strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for candidate in [cleaned, *_OBJECT_RE.findall(cleaned)]:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("move"), str):
            data["move"] = _ALIASES.get(data["move"].strip().lower(), data["move"])
        try:
            return Decision.model_validate(data)
        except ValidationError:
            continue
    raise InvalidDecision(f"no valid move in response: {text[:200]!r}")
