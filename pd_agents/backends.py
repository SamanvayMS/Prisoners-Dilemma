"""Model backends. Each one turns (system, user) prompts into a Decision.

`Backend.decide` owns the only retry loop (the SDK clients' own retries are
off): an answer that is not an actionable move, a refusal, a truncated answer
or a transient API error is retried with exponential backoff, honouring the
server's retry-after. If every attempt fails it raises AgentDecisionError,
which stops the tournament. Errors that retrying cannot fix (bad credentials,
unknown model, malformed request) stop it immediately.
"""

import dataclasses
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import axelrod as axl

from .decision import DECISION_SCHEMA, Decision, InvalidDecision, parse_decision, parse_decision_lenient

PROVIDERS = ("anthropic", "ollama", "openai", "mock")
MODES = ("fast", "reasoning")
MOCK_BEHAVIOURS = ("tft", "allc", "alld", "random", "flaky", "broken")

# provider -> (base-URL env var, default base URL, API-key env var, default API key)
_OPENAI_COMPAT = {
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1", None, "ollama"),
    "openai": ("OPENAI_BASE_URL", None, "OPENAI_API_KEY", "not-needed"),
}
# Error classes, by name, that both SDKs raise for requests retrying can't fix.
_FATAL_ERRORS = ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                 "BadRequestError", "UnprocessableEntityError")
_MAX_BACKOFF_S = 60.0


class AgentDecisionError(RuntimeError):
    """An agent could not produce an actionable move; the tournament must stop."""


def validate_agent(provider: str, mode: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


@dataclass
class DecisionRequest:
    system: str
    user: str
    opp: Sequence[axl.Action] = ()


@dataclass
class DecisionResult:
    decision: Decision
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    attempts: int = 1
    latency_s: float = 0.0
    errors: list = field(default_factory=list)


class Backend(ABC):
    sdk = None  # SDK module whose error classes `_is_fatal` checks

    def __init__(self, model: str, mode: str, retries: int = 6, backoff_s: float = 1.0):
        self.model = model
        self.mode = mode
        self.retries = retries
        self.backoff_s = backoff_s

    @abstractmethod
    def _attempt(self, request: DecisionRequest) -> DecisionResult:
        """One model call. Raise InvalidDecision or an SDK error on failure."""

    def _is_fatal(self, exc: Exception) -> bool:
        return self.sdk is not None and isinstance(exc, tuple(getattr(self.sdk, n) for n in _FATAL_ERRORS))

    def _delay(self, attempt: int, exc: Exception) -> float:
        delay = min(self.backoff_s * 2 ** (attempt - 1), _MAX_BACKOFF_S) * (1 + random.random() * 0.25)
        response = getattr(exc, "response", None)
        retry_after = getattr(response, "headers", {}).get("retry-after") if response is not None else None
        try:
            return max(delay, float(retry_after)) if retry_after else delay
        except ValueError:
            return delay

    def decide(self, request: DecisionRequest) -> DecisionResult:
        errors = []
        start = time.monotonic()
        for attempt in range(1, self.retries + 1):
            try:
                result = self._attempt(request)
            except Exception as exc:  # noqa: BLE001 - classified below
                if self._is_fatal(exc):
                    raise AgentDecisionError(f"{self.model}: {type(exc).__name__}: {exc}") from exc
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.retries:
                    time.sleep(self._delay(attempt, exc))
                continue
            return dataclasses.replace(result, attempts=attempt, latency_s=time.monotonic() - start,
                                       errors=errors)
        raise AgentDecisionError(
            f"{self.model} ({self.mode}) gave no actionable move after {self.retries} attempts: "
            + " | ".join(errors[-3:])
        )


def anthropic_request_params(model: str, mode: str, effort: str) -> dict:
    """Thinking/effort settings that make `mode` mean the same thing on every Claude model."""
    if mode == "reasoning":
        if model.startswith("claude-haiku"):
            return {"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}}
        return {"max_tokens": 16000, "thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    # fast / System 1: no deliberation before the answer.
    if model.startswith("claude-haiku"):
        return {"max_tokens": 256}
    if model.startswith(("claude-opus-5-5", "claude-fable", "claude-mythos")):
        # Thinking can't be disabled on these models; low effort is the closest setting.
        return {"max_tokens": 2048, "output_config": {"effort": "low"}}
    if model.startswith("claude-sonnet-5-5"):
        return {"max_tokens": 256, "thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}
    return {"max_tokens": 256, "thinking": {"type": "disabled"}}


class AnthropicBackend(Backend):
    def __init__(self, model: str, mode: str, effort: str = "high", **kw):
        import anthropic

        super().__init__(model, mode, **kw)
        self.sdk = anthropic
        self.params = anthropic_request_params(model, mode, effort)
        self.params["output_config"] = {
            **self.params.get("output_config", {}),
            "format": {"type": "json_schema", "schema": DECISION_SCHEMA},
        }
        self._client = None

    def _attempt(self, request):
        if self._client is None:
            self._client = self.sdk.Anthropic(max_retries=0)
        response = self._client.messages.create(
            model=self.model,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            **self.params,
        )
        if response.stop_reason == "refusal":
            raise InvalidDecision(f"refusal: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise InvalidDecision("answer truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return DecisionResult(
            decision=parse_decision(text),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible chat completions endpoint: Ollama, vLLM, LM Studio, OpenAI."""

    def __init__(self, model: str, mode: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 effort: str = "high", extra_body: Optional[dict] = None, **kw):
        import openai

        super().__init__(model, mode, **kw)
        self.sdk = openai
        self.base_url = base_url
        self.api_key = api_key
        self.params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": DECISION_SCHEMA, "strict": True},
            },
            "max_tokens": 8192 if mode == "reasoning" else 256,
            "extra_body": extra_body,
        }
        if mode == "reasoning":
            self.params["reasoning_effort"] = effort
        self._client = None

    def _attempt(self, request):
        if self._client is None:
            self._client = self.sdk.OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=0)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            **self.params,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            hint = "" if self.mode == "reasoning" else " (if this is a thinking model, run it in reasoning mode)"
            raise InvalidDecision("answer truncated at max_tokens" + hint)
        usage = response.usage
        return DecisionResult(
            decision=parse_decision_lenient(choice.message.content or ""),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=choice.finish_reason,
        )


class MockBackend(Backend):
    """Offline backend for tests and dry runs. `model` is BEHAVIOUR or BEHAVIOUR:LABEL, where
    BEHAVIOUR is tft, allc, alld, random, flaky (garbage on the first attempt of every
    decision, then tit-for-tat) or broken (never answers)."""

    def __init__(self, model: str = "tft", mode: str = "fast", **kw):
        kw.setdefault("backoff_s", 0.0)
        super().__init__(model, mode, **kw)
        self.behaviour = model.split(":", 1)[0]
        if self.behaviour not in MOCK_BEHAVIOURS:
            raise ValueError(f"unknown mock behaviour {self.behaviour!r}; choose from {MOCK_BEHAVIOURS}")
        self._rng = random.Random()
        self._calls = 0

    def _attempt(self, request):
        self._calls += 1
        if self.behaviour == "broken" or (self.behaviour == "flaky" and self._calls % 2 == 1):
            raise InvalidDecision("mock: not a move")
        if self.behaviour in ("allc", "alld"):
            move = self.behaviour[-1].upper()
        elif self.behaviour == "random":
            move = self._rng.choice("CD")
        else:  # tft, flaky
            move = str(request.opp[-1]) if request.opp else "C"
        return DecisionResult(decision=Decision(move=move), input_tokens=len(request.user) // 4, output_tokens=5)


def make_backend(provider: str, model: str, mode: str, effort: str = "high", base_url: Optional[str] = None,
                 extra_body: Optional[dict] = None, retries: int = 6) -> Backend:
    validate_agent(provider, mode)
    if provider == "anthropic":
        return AnthropicBackend(model, mode, effort=effort, retries=retries)
    if provider == "mock":
        return MockBackend(model, mode, retries=retries)
    url_env, default_url, key_env, default_key = _OPENAI_COMPAT[provider]
    return OpenAICompatBackend(
        model, mode,
        base_url=base_url or os.environ.get(url_env, default_url),
        api_key=os.environ.get(key_env, default_key) if key_env else default_key,
        effort=effort, extra_body=extra_body, retries=retries,
    )
