"""LLMPlayer: an axelrod Player whose move each round comes from a language model."""

import math
from typing import Optional

import axelrod as axl

from .backends import DecisionRequest, make_backend
from .prompt import system_prompt, user_prompt
from .telemetry import log_decision


class LLMPlayer(axl.Player):
    """Sees the full game history and answers C or D through a model backend.

    axelrod re-creates players from their constructor arguments for every match
    (`clone()`), and may run matches in worker processes, so the constructor only
    takes plain values and the API client is created on the first move.
    """

    name = "LLM"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,  # also stops axelrod caching the match as deterministic
        "long_run_time": True,
        "makes_use_of": {"game", "length"},
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5",
        mode: str = "fast",
        effort: str = "high",
        base_url: Optional[str] = None,
        extra_body: Optional[dict] = None,
        retries: int = 6,
        history_window: int = 0,
        reveal_length: bool = False,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self.mode = mode
        self.effort = effort
        self.base_url = base_url
        self.extra_body = extra_body
        self.retries = retries
        self.history_window = history_window
        self.reveal_length = reveal_length
        self.log_path = log_path
        self.name = f"LLM {mode} ({provider}:{model})"
        self._backend = None

    def __repr__(self) -> str:
        return self.name

    def backend(self):
        if self._backend is None:
            self._backend = make_backend(
                self.provider, self.model, self.mode, effort=self.effort,
                base_url=self.base_url, extra_body=self.extra_body, retries=self.retries,
            )
        return self._backend

    def _known_length(self) -> Optional[int]:
        length = self.match_attributes["length"]
        if not self.reveal_length or length is None or length < 0 or math.isinf(length):
            return None
        return int(length)

    def strategy(self, opponent: axl.Player) -> axl.Action:
        game = self.match_attributes["game"]
        own, opp = list(self.history), list(opponent.history)
        request = DecisionRequest(
            system=system_prompt(game, self._known_length(), self.match_attributes["noise"]),
            user=user_prompt(game, own, opp, self.history_window),
            own=own,
            opp=opp,
        )
        result = self.backend().decide(request)
        log_decision(self.log_path, {
            "agent": self.name,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "opponent": str(opponent),
            "round": len(own) + 1,
            "prev": f"{own[-1]}{opp[-1]}" if own else None,
            "move": result.decision.move,
            "attempts": result.attempts,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_s": round(result.latency_s, 4),
            "stop_reason": result.stop_reason,
            "errors": result.errors,
        })
        return result.decision.action()
