"""Per-decision log (JSONL) of what axelrod doesn't record (tokens, retries,
latency), and the per-agent report."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import axelrod as axl

# USD per million tokens (input, output) for cost estimates. Local models are free.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
}
_STATES = ((axl.Action.C, axl.Action.C), (axl.Action.C, axl.Action.D),
           (axl.Action.D, axl.Action.C), (axl.Action.D, axl.Action.D))


def estimate_cost(provider: str, model: str, input_tokens: float, output_tokens: float) -> Optional[float]:
    """USD, or None when the model's price is unknown."""
    if provider in ("ollama", "mock"):
        return 0.0
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1e6


def log_decision(path: Optional[str], record: dict) -> None:
    if not path:
        return
    # One short line per write in append mode, so lines from parallel worker
    # processes don't interleave.
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_decisions(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise_usage(records: list) -> dict:
    """Per-agent decisions, retries, tokens, latency and cost from the decision log."""
    usage = defaultdict(Counter)
    meta = {}
    for r in records:
        meta[r["agent"]] = (r["provider"], r["model"])
        u = usage[r["agent"]]
        u["decisions"] += 1
        u["retried"] += r["attempts"] > 1
        for key in ("attempts", "input_tokens", "output_tokens", "latency_s"):
            u[key] += r[key]
    out = {}
    for name, u in usage.items():
        n = u["decisions"]
        out[name] = {
            "decisions": n,
            "retried": u["retried"],
            "mean_attempts": u["attempts"] / n,
            "mean_latency_s": u["latency_s"] / n,
            "input_tokens": u["input_tokens"],
            "output_tokens": u["output_tokens"],
            "cost_usd": estimate_cost(*meta[name], u["input_tokens"], u["output_tokens"]),
        }
    return out


def memory_one_fingerprint(results, index: int) -> dict:
    """P(C | previous round as (own, opponent)) for one player, pooled over its matches."""
    counts = sum(results.state_to_action_distribution[index], Counter())
    fingerprint = {}
    for state in _STATES:
        coop, total = counts[(state, axl.Action.C)], counts[(state, axl.Action.C)] + counts[(state, axl.Action.D)]
        fingerprint[f"{state[0]}{state[1]}"] = coop / total if total else None
    return fingerprint


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def write_report(results, usage: dict, path) -> None:
    ranked = results.ranked_names
    lines = ["# Agent report", ""]
    for name, u in usage.items():
        index = results.players.index(name)
        fp = memory_one_fingerprint(results, index)
        cost = "n/a" if u["cost_usd"] is None else f"${u['cost_usd']:.4f}"
        lines += [
            f"## {name}",
            "",
            f"- Rank: {ranked.index(name) + 1} of {len(ranked)}",
            f"- Decisions: {u['decisions']}, cooperation rate {results.cooperating_rating[index]:.2%}",
            f"- Decisions that needed a retry: {u['retried']} (mean attempts {u['mean_attempts']:.2f})",
            f"- Tokens: {u['input_tokens']} in / {u['output_tokens']} out, cost {cost}",
            f"- Mean latency per decision: {u['mean_latency_s']:.2f}s",
            "",
            "Memory-one fingerprint, P(C | previous round as (own, opponent)):",
            "",
            "| CC | CD | DC | DD |",
            "|---|---|---|---|",
            f"| {_fmt(fp['CC'])} | {_fmt(fp['CD'])} | {_fmt(fp['DC'])} | {_fmt(fp['DD'])} |",
            "",
            "For reference: TFT = 1, 0, 1, 0; WSLS = 1, 0, 0, 1; Grim = 1, 0, 0, 0.",
            "",
        ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
