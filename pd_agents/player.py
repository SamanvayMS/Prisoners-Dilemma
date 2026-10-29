"""LLMPlayer: an axelrod Player whose move each round comes from a language model."""

import math
from typing import Optional

import axelrod as axl

from .backends import DecisionRequest, make_backend
from .prompt import system_prompt, user_prompt
from .telemetry import log_decision

# Constructor arguments that configure the prompt/logging rather than the backend.
_PLAYER_ONLY_ARGS = ("history_window", "reveal_length", "log_path")


class LLMPlayer(axl.Player):
    """Sees the full game history and answers C or D through a model backend.

    axelrod re-creates players from their constructor arguments for every match
    (`clone()`, via `self.init_kwargs`), and may run matches in worker processes,
    so the constructor only takes plain values and the backend is created on the
    first move.
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
        self.name = f"LLM {mode} ({provider}:{model})"
        self._backend = None

    def __repr__(self) -> str:
        return self.name

    def backend(self):
        if self._backend is None:
            kwargs = {k: v for k, v in self.init_kwargs.items() if k not in _PLAYER_ONLY_ARGS}
            self._backend = make_backend(**kwargs)
        return self._backend

    def _known_length(self) -> Optional[int]:
        length = self.match_attributes["length"]
        if not self.init_kwargs["reveal_length"] or length is None or length < 0 or math.isinf(length):
            return None
        return int(length)

    def strategy(self, opponent: axl.Player) -> axl.Action:
        cfg = self.init_kwargs
        game = self.match_attributes["game"]
        request = DecisionRequest(
            system=system_prompt(game, self._known_length(), self.match_attributes["noise"]),
            user=user_prompt(game, self.history, opponent.history, cfg["history_window"]),
            opp=opponent.history,
        )
        result = self.backend().decide(request)
        log_decision(cfg["log_path"], {
            "agent": self.name,
            "provider": cfg["provider"],
            "model": cfg["model"],
            "mode": cfg["mode"],
            "opponent": str(opponent),
            "round": len(self.history) + 1,
            "move": result.decision.move,
            "attempts": result.attempts,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_s": round(result.latency_s, 4),
            "stop_reason": result.stop_reason,
            "errors": result.errors,
        })
        return result.decision.action()
