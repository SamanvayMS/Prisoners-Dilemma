from types import SimpleNamespace as NS

import anthropic
import httpx2
import openai
import pytest

from pd_agents.backends import (
    AgentDecisionError, AnthropicBackend, DecisionRequest, MockBackend, OpenAICompatBackend,
    anthropic_request_params,
)

REQ = DecisionRequest(system="rules", user="Round 1.")


class FakeCreate:
    """Stands in for client.messages.create / client.chat.completions.create."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def claude_response(text, stop_reason="end_turn"):
    return NS(
        stop_reason=stop_reason,
        stop_details=None,
        content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
        usage=NS(input_tokens=120, output_tokens=7),
    )


def oai_response(text, finish_reason="stop"):
    return NS(
        choices=[NS(finish_reason=finish_reason, message=NS(content=text))],
        usage=NS(prompt_tokens=90, completion_tokens=4),
    )


def anthropic_backend(create, **kw):
    b = AnthropicBackend(kw.pop("model", "claude-opus-5-5"), kw.pop("mode", "reasoning"),
                         backoff_s=0, **kw)
    b._client = NS(messages=NS(create=create))
    return b


def oai_backend(create, mode="fast", **kw):
    b = OpenAICompatBackend("qwen3:8b", mode, base_url="http://localhost:11434/v1", api_key="ollama",
                            backoff_s=0, **kw)
    b._client = NS(chat=NS(completions=NS(create=create)))
    return b


def status_error(cls, code):
    req = httpx2.Request("POST", "http://test")
    return cls("nope", response=httpx2.Response(code, request=req), body=None)


@pytest.mark.parametrize("model, mode, expected", [
    ("claude-haiku-4-5", "fast", {"max_tokens": 256}),
    ("claude-haiku-4-5", "reasoning", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    ("claude-opus-5-5", "reasoning", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}),
    ("claude-opus-5-5", "fast", {"output_config": {"effort": "low"}}),
    ("claude-sonnet-5-5", "fast", {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}),
    ("claude-opus-4-8", "fast", {"thinking": {"type": "disabled"}}),
])
def test_anthropic_mode_params(model, mode, expected):
    params = anthropic_request_params(model, mode, "max")
    assert expected.items() <= params.items()
    if mode == "fast" and model == "claude-haiku-4-5":
        assert "thinking" not in params


def test_anthropic_sends_schema_and_effort_together():
    create = FakeCreate(claude_response('{"move": "D"}'))
    result = anthropic_backend(create, effort="high").decide(REQ)
    assert result.decision.move == "D" and result.attempts == 1
    assert (result.input_tokens, result.output_tokens) == (120, 7)
    sent = create.calls[0]
    assert sent["output_config"]["effort"] == "high"
    assert sent["output_config"]["format"]["schema"]["properties"]["move"]["enum"] == ["C", "D"]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["system"] == "rules"


def test_anthropic_retries_refusal_truncation_and_transient_errors():
    create = FakeCreate(
        claude_response("", stop_reason="refusal"),
        claude_response('{"mo', stop_reason="max_tokens"),
        status_error(anthropic.InternalServerError, 500),
        claude_response('{"move": "C"}'),
    )
    result = anthropic_backend(create).decide(REQ)
    assert result.decision.move == "C"
    assert result.attempts == 4
    assert len(result.errors) == 3


def test_anthropic_gives_up_after_retries():
    create = FakeCreate(*[claude_response("no idea")] * 3)
    with pytest.raises(AgentDecisionError, match="after 3 attempts"):
        anthropic_backend(create, retries=3).decide(REQ)


@pytest.mark.parametrize("exc", [
    status_error(anthropic.AuthenticationError, 401),
    status_error(anthropic.NotFoundError, 404),
    status_error(anthropic.BadRequestError, 400),
])
def test_anthropic_fatal_errors_stop_without_retrying(exc):
    create = FakeCreate(exc, claude_response('{"move": "C"}'))
    with pytest.raises(AgentDecisionError):
        anthropic_backend(create).decide(REQ)
    assert len(create.calls) == 1


def test_openai_compat_fast_mode():
    create = FakeCreate(oai_response('{"move": "C"}'))
    result = oai_backend(create, mode="fast").decide(REQ)
    assert result.decision.move == "C"
    sent = create.calls[0]
    assert "reasoning_effort" not in sent
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["messages"][0] == {"role": "system", "content": "rules"}


def test_openai_compat_reasoning_mode_strips_think_block():
    create = FakeCreate(oai_response('<think>long thoughts</think>{"move": "D"}'))
    result = oai_backend(create, mode="reasoning", effort="medium", extra_body={"keep_alive": "5m"}).decide(REQ)
    assert result.decision.move == "D"
    assert create.calls[0]["reasoning_effort"] == "medium"
    assert create.calls[0]["extra_body"] == {"keep_alive": "5m"}
    assert create.calls[0]["max_tokens"] > 1000


def test_openai_compat_retries_truncated_and_aborts_on_unknown_model():
    create = FakeCreate(oai_response("", finish_reason="length"), oai_response('{"move": "C"}'))
    result = oai_backend(create).decide(REQ)
    assert result.attempts == 2 and "thinking model" in result.errors[0]

    create = FakeCreate(status_error(openai.NotFoundError, 404))
    with pytest.raises(AgentDecisionError):
        oai_backend(create).decide(REQ)


def test_retry_after_header_sets_minimum_backoff():
    exc = status_error(anthropic.RateLimitError, 429)
    exc.response.headers["retry-after"] = "7"
    assert anthropic_backend(FakeCreate())._delay(1, exc) == 7.0


def test_mock_backends():
    assert MockBackend("flaky").decide(REQ).attempts == 2
    assert MockBackend("alld:label").decide(REQ).decision.move == "D"
    with pytest.raises(AgentDecisionError):
        MockBackend("broken", retries=2).decide(REQ)
    with pytest.raises(ValueError, match="unknown mock behaviour"):
        MockBackend("titfortat")
