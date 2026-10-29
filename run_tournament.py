"""Run an Axelrod tournament, optionally with LLM agents (--with-agents).

Examples:
    python run_tournament.py
    python run_tournament.py --with-agents --mock-agents --turns 20 --repetitions 1
    python run_tournament.py --with-agents \\
        --agent anthropic:claude-haiku-4-5@fast --agent anthropic:claude-opus-5-5@reasoning \\
        --agent ollama:llama3.1:8b@fast --agent ollama:qwen3:8b@reasoning
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import axelrod as axl

from pd_agents.backends import MODES, PROVIDERS, AgentDecisionError
from pd_agents.player import LLMPlayer
from pd_agents.telemetry import load_decisions, price_for, summarise_agents, write_report
from pd_agents.tournament import ROSTERS, AgentTournament, is_cheater

DEFAULT_AGENTS = ["anthropic:claude-haiku-4-5@fast", "anthropic:claude-opus-5-5@reasoning"]

# Rough per-call token sizes for the pre-run cost estimate.
_PROMPT_BASE_TOKENS = 250
_TOKENS_PER_HISTORY_ROW = 8
_OUTPUT_TOKENS = {"fast": 10, "reasoning": 600}


@dataclass
class AgentSpec:
    provider: str
    model: str
    mode: str

    @classmethod
    def parse(cls, text: str) -> "AgentSpec":
        """PROVIDER:MODEL[@MODE]; MODEL may itself contain ':' (Ollama tags)."""
        provider, sep, rest = text.partition(":")
        if not sep or not rest:
            raise argparse.ArgumentTypeError(f"agent must look like PROVIDER:MODEL[@MODE], got {text!r}")
        model, _, mode = rest.rpartition("@") if "@" in rest else (rest, "", "fast")
        if provider not in PROVIDERS:
            raise argparse.ArgumentTypeError(f"unknown provider {provider!r}; choose from {PROVIDERS}")
        if mode not in MODES:
            raise argparse.ArgumentTypeError(f"unknown mode {mode!r}; choose from {MODES}")
        return cls(provider, model, mode)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("tournament")
    g.add_argument("--roster", choices=sorted(ROSTERS), default="first",
                   help="preset strategies (default: Axelrod's first tournament)")
    g.add_argument("--turns", type=int, default=200,
                   help="expected match length; exact length with --fixed-length (default: 200)")
    g.add_argument("--fixed-length", action="store_true",
                   help="every match lasts exactly --turns rounds (default: random length, "
                        "ending after each round with probability 1/turns)")
    g.add_argument("--repetitions", type=int, default=5)
    g.add_argument("--noise", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=None, help="seeds axelrod; LLM answers are not seeded")
    g.add_argument("--processes", type=int, default=1, help="parallel worker processes (default: 1)")
    g.add_argument("--output-dir", default="results")
    g.add_argument("--plots", action="store_true", help="save axelrod boxplot/winplot/payoff PNGs")

    a = p.add_argument_group("agents")
    a.add_argument("--with-agents", action="store_true", help="add LLM agents to the tournament")
    a.add_argument("--agent", action="append", type=AgentSpec.parse, metavar="PROVIDER:MODEL[@MODE]",
                   help="repeatable. PROVIDER: anthropic, ollama, openai (any OpenAI-compatible "
                        "endpoint) or mock. MODE: fast (System 1, default) or reasoning. "
                        f"Default: {' and '.join(DEFAULT_AGENTS)}")
    a.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="high",
                   help="effort for reasoning-mode agents (OpenAI-compatible servers get it as "
                        "reasoning_effort; xhigh/max are Claude-only)")
    a.add_argument("--reveal-length", action="store_true",
                   help="tell agents the match length (only with --fixed-length)")
    a.add_argument("--history-window", type=int, default=0,
                   help="show agents only the last N rounds plus a summary (0 = all rounds)")
    a.add_argument("--agent-retries", type=int, default=6,
                   help="attempts per move before the tournament is stopped (default: 6)")
    a.add_argument("--ollama-url", default=None,
                   help="Ollama OpenAI-compatible URL (default: $OLLAMA_BASE_URL or http://localhost:11434/v1)")
    a.add_argument("--openai-base-url", default=None, help="base URL for provider 'openai' (default: $OPENAI_BASE_URL)")
    a.add_argument("--openai-extra-body", type=json.loads, default=None, metavar="JSON",
                   help="extra request fields for ollama/openai agents, e.g. '{\"reasoning_effort\": \"none\"}'")
    a.add_argument("--mock-agents", action="store_true",
                   help="replace every agent's backend with an offline tit-for-tat mock")
    a.add_argument("--max-agent-calls", type=int, default=50_000,
                   help="refuse to start if the estimated number of model calls is higher")
    a.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    return p


def build_agents(args, log_path):
    specs = args.agent or [AgentSpec.parse(s) for s in DEFAULT_AGENTS]
    agents = []
    for spec in specs:
        provider, model = (spec.provider, spec.model)
        if args.mock_agents:
            provider, model = "mock", f"tft-as-{spec.model}"
        base_url = {"ollama": args.ollama_url, "openai": args.openai_base_url}.get(provider)
        agents.append(LLMPlayer(
            provider=provider, model=model, mode=spec.mode, effort=args.reasoning_effort,
            base_url=base_url, extra_body=args.openai_extra_body, retries=args.agent_retries,
            history_window=args.history_window, reveal_length=args.reveal_length,
            log_path=str(log_path),
        ))
    return agents


def estimate(agents, n_players, args):
    """Expected model calls and rough cost per agent. Each agent plays the other
    n-1 players and itself; in self-play both sides call the model."""
    turns = args.turns
    calls = args.repetitions * turns * (n_players + 1)
    window = min(turns / 2, args.history_window or turns / 2)
    in_tokens = calls * (_PROMPT_BASE_TOKENS + _TOKENS_PER_HISTORY_ROW * window)
    rows = []
    for agent in agents:
        out_tokens = calls * _OUTPUT_TOKENS[agent.mode]
        price = price_for(agent.provider, agent.model)
        cost = None if price is None else (in_tokens * price[0] + out_tokens * price[1]) / 1e6
        rows.append((agent.name, calls, cost))
    return rows


def confirm(rows, args) -> bool:
    total = sum(r[1] for r in rows)
    print(f"Estimated model calls (expected match length {args.turns}, {args.repetitions} repetitions):")
    for name, calls, cost in rows:
        cost_s = "unknown price" if cost is None else f"~${cost:,.2f}"
        print(f"  {name}: {calls:,} calls, {cost_s}")
    print(f"  total: {total:,} calls (reasoning-mode output is a rough guess)")
    if total > args.max_agent_calls:
        print(f"Refusing to start: estimate exceeds --max-agent-calls={args.max_agent_calls:,}. "
              "Lower --turns/--repetitions or raise the limit.", file=sys.stderr)
        return False
    if args.yes or all(cost == 0 for _, _, cost in rows):  # mock / local models cost nothing
        return True
    if not sys.stdin.isatty():
        print("Not a terminal; pass --yes to confirm.", file=sys.stderr)
        return False
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def save_plots(results, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    plot = axl.Plot(results)
    for kind in ("boxplot", "winplot", "payoff"):
        getattr(plot, kind)().savefig(out_dir / f"{kind}.png", bbox_inches="tight")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.reveal_length and not args.fixed_length:
        print("--reveal-length needs --fixed-length (a random-length match has no length to reveal).",
              file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "agent_decisions.jsonl"

    players = ROSTERS[args.roster]()
    agents = []
    if args.with_agents:
        dropped = [str(p) for p in players if is_cheater(p)]
        if dropped:
            print(f"Dropping strategies that inspect their opponent's code: {', '.join(dropped)}")
        players = [p for p in players if not is_cheater(p)]
        agents = build_agents(args, log_path)
        if not confirm(estimate(agents, len(players) + len(agents), args), args):
            return 1
        log_path.unlink(missing_ok=True)
        players += agents

    length = {"turns": args.turns} if args.fixed_length else {"prob_end": 1 / args.turns}
    tournament = AgentTournament(players, repetitions=args.repetitions, noise=args.noise,
                                 seed=args.seed, **length)
    try:
        results = tournament.play(processes=args.processes if args.processes > 1 else None)
    except AgentDecisionError as exc:
        print(f"\nTournament stopped: {exc}", file=sys.stderr)
        print(f"Decisions made before the stop are in {log_path}", file=sys.stderr)
        return 3

    ranked = results.ranked_names
    (out_dir / "ranked.txt").write_text("\n".join(ranked) + "\n", encoding="utf-8")
    results.write_summary(str(out_dir / "summary.csv"))
    with open(out_dir / "payoff_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"players": [str(p) for p in players], "payoff_matrix": results.payoff_matrix}, f, indent=1)

    print("\nRanking:")
    for i, name in enumerate(ranked, 1):
        print(f"{i:>3}. {name}")

    if agents:
        summary = summarise_agents(load_decisions(log_path))
        write_report(summary, ranked, out_dir / "agent_report.md")
        print(f"\nAgent report: {out_dir / 'agent_report.md'}")
    if args.plots:
        save_plots(results, out_dir)
    print(f"Results written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
