"""Model backends. Each one turns (system, user) prompts into a Decision.

`Backend.decide` owns the retry loop: an answer that is not an actionable move,
a refusal, a truncated answer or a transient API error is retried with
exponential backoff. If every attempt fails it raises AgentDecisionError, which
stops the tournament. Errors that retrying cannot fix (bad credentials, unknown
model, malformed request) stop it immediately.
"""

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import axelrod as axl

from .decision import DECISION_SCHEMA, Decision, InvalidDecision, parse_decision

PROVIDERS = ("anthropic", "ollama", "openai", "mock")
MODES = ("fast", "reasoning")
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class AgentDecisionError(RuntimeError):
    """An agent could not produce an actionable move; the tournament must stop."""


@dataclass
class DecisionRequest:
    system: str
    user: str
    own: Sequence[axl.Action] = ()
    opp: Sequence[axl.Action] = ()


@dataclass
class DecisionResult:
    decision: Decision
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: Optional[str] = None
    errors: list = field(default_factory=list)


@dataclass
class _Attempt:
    decision: Decision
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None


class Backend(ABC):
    def __init__(self, model: str, mode: str, retries: int = 6, backoff_s: float = 1.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.model = model
        self.mode = mode
        self.retries = retries
        self.backoff_s = backoff_s

    @abstractmethod
    def _attempt(self, request: DecisionRequest) -> _Attempt:
        """One model call. Raise InvalidDecision or a retryable error on failure."""

    def _is_fatal(self, exc: Exception) -> bool:
        return False

    def decide(self, request: DecisionRequest) -> DecisionResult:
        errors = []
        start = time.monotonic()
        for attempt in range(1, self.retries + 1):
            try:
                out = self._attempt(request)
            except Exception as exc:  # noqa: BLE001 - classified below
                if self._is_fatal(exc):
                    raise AgentDecisionError(f"{self.model}: {type(exc).__name__}: {exc}") from exc
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.retries:
                    time.sleep(min(self.backoff_s * 2 ** (attempt - 1), 30.0) * (1 + random.random() * 0.25))
                continue
            return DecisionResult(
                decision=out.decision,
                attempts=attempt,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                latency_s=time.monotonic() - start,
                stop_reason=out.stop_reason,
                errors=errors,
            )
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
        super().__init__(model, mode, **kw)
        self.params = anthropic_request_params(model, mode, effort)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=2)
        return self._client

    def _is_fatal(self, exc):
        import anthropic

        return isinstance(
            exc,
            (
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
                anthropic.NotFoundError,
                anthropic.BadRequestError,
                anthropic.UnprocessableEntityError,
            ),
        )

    def _attempt(self, request):
        params = dict(self.params)
        output_config = {**params.pop("output_config", {}), "format": {"type": "json_schema", "schema": DECISION_SCHEMA}}
        response = self._get_client().messages.create(
            model=self.model,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            output_config=output_config,
            **params,
        )
        if response.stop_reason == "refusal":
            raise InvalidDecision(f"refusal: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise InvalidDecision("answer truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return _Attempt(
            decision=parse_decision(text),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible chat completions endpoint: Ollama, vLLM, LM Studio, OpenAI."""

    def __init__(
        self,
        model: str,
        mode: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        effort: str = "high",
        extra_body: Optional[dict] = None,
        **kw,
    ):
        super().__init__(model, mode, **kw)
        self.base_url = base_url
        self.api_key = api_key
        self.effort = effort
        self.extra_body = extra_body or {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=2)
        return self._client

    def _is_fatal(self, exc):
        import openai

        return isinstance(
            exc,
            (
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
                openai.BadRequestError,
                openai.UnprocessableEntityError,
            ),
        )

    def _attempt(self, request):
        kwargs = {}
        if self.mode == "reasoning":
            kwargs["reasoning_effort"] = self.effort
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": DECISION_SCHEMA, "strict": True},
            },
            max_tokens=8192 if self.mode == "reasoning" else 256,
            extra_body=self.extra_body or None,
            **kwargs,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            hint = "" if self.mode == "reasoning" else " (if this is a thinking model, run it in reasoning mode)"
            raise InvalidDecision("answer truncated at max_tokens" + hint)
        usage = response.usage
        return _Attempt(
            decision=parse_decision(choice.message.content or ""),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=choice.finish_reason,
        )


class MockBackend(Backend):
    """Offline backend for tests and dry runs. `model` picks the behaviour:
    tft, allc, alld, random, or flaky (garbage on the first attempt of every decision,
    then tit-for-tat), or broken (never answers)."""

    def __init__(self, model: str = "tft", mode: str = "fast", seed: Optional[int] = None, **kw):
        kw.setdefault("backoff_s", 0.0)
        super().__init__(model, mode, **kw)
        self._rng = random.Random(seed)
        self._calls = 0

    def _attempt(self, request):
        self._calls += 1
        if self.model == "broken" or (self.model == "flaky" and self._calls % 2 == 1):
            raise InvalidDecision("mock: not a move")
        if self.model == "allc":
            move = "C"
        elif self.model == "alld":
            move = "D"
        elif self.model == "random":
            move = self._rng.choice("CD")
        else:  # tft / flaky
            move = "C" if not request.opp else str(request.opp[-1])
        return _Attempt(decision=Decision(move=move), input_tokens=len(request.user) // 4, output_tokens=5)


def make_backend(
    provider: str,
    model: str,
    mode: str,
    effort: str = "high",
    base_url: Optional[str] = None,
    extra_body: Optional[dict] = None,
    retries: int = 6,
) -> Backend:
    if provider == "anthropic":
        return AnthropicBackend(model, mode, effort=effort, retries=retries)
    if provider == "ollama":
        url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        return OpenAICompatBackend(model, mode, base_url=url, api_key="ollama", effort=effort,
                                   extra_body=extra_body, retries=retries)
    if provider == "openai":
        return OpenAICompatBackend(model, mode, base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                                   api_key=os.environ.get("OPENAI_API_KEY", "not-needed"), effort=effort,
                                   extra_body=extra_body, retries=retries)
    if provider == "mock":
        return MockBackend(model, mode, retries=retries)
    raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")
