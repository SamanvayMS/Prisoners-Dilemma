"""Per-decision log (JSONL) and the per-agent report built from it."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

# USD per million tokens (input, output) for cost estimates. Local models are free.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
}


def price_for(provider: str, model: str) -> Optional[tuple]:
    if provider in ("ollama", "mock"):
        return (0.0, 0.0)
    return PRICES.get(model)


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


def summarise_agents(records: list) -> dict:
    """Per-agent stats, including the empirical memory-one fingerprint:
    P(cooperate | previous round was CC, CD, DC, DD) from the agent's point of view."""
    agents = defaultdict(lambda: {
        "decisions": 0, "cooperations": 0, "attempts": 0, "retried": 0,
        "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0,
        "provider": None, "model": None, "mode": None,
        "fingerprint_counts": {k: [0, 0] for k in ("CC", "CD", "DC", "DD")},
    })
    for r in records:
        a = agents[r["agent"]]
        a["provider"], a["model"], a["mode"] = r["provider"], r["model"], r["mode"]
        a["decisions"] += 1
        a["cooperations"] += r["move"] == "C"
        a["attempts"] += r["attempts"]
        a["retried"] += r["attempts"] > 1
        a["input_tokens"] += r["input_tokens"]
        a["output_tokens"] += r["output_tokens"]
        a["latency_s"] += r["latency_s"]
        if r.get("prev"):
            counts = a["fingerprint_counts"][r["prev"]]
            counts[0] += r["move"] == "C"
            counts[1] += 1
    out = {}
    for name, a in agents.items():
        n = a["decisions"] or 1
        price = price_for(a["provider"], a["model"])
        cost = None
        if price is not None:
            cost = (a["input_tokens"] * price[0] + a["output_tokens"] * price[1]) / 1e6
        out[name] = {
            **{k: a[k] for k in ("provider", "model", "mode", "decisions", "retried",
                                 "input_tokens", "output_tokens")},
            "cooperation_rate": a["cooperations"] / n,
            "mean_attempts": a["attempts"] / n,
            "mean_latency_s": a["latency_s"] / n,
            "cost_usd": cost,
            "fingerprint": {
                k: (c / t if t else None) for k, (c, t) in a["fingerprint_counts"].items()
            },
        }
    return out


def write_report(summary: dict, ranked_names: list, path) -> None:
    lines = ["# Agent report", ""]
    for name, s in summary.items():
        rank = ranked_names.index(name) + 1 if name in ranked_names else "?"
        cost = "n/a" if s["cost_usd"] is None else f"${s['cost_usd']:.4f}"
        fp = s["fingerprint"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        lines += [
            f"## {name}",
            "",
            f"- Rank: {rank} of {len(ranked_names)}",
            f"- Decisions: {s['decisions']}, cooperation rate {s['cooperation_rate']:.2%}",
            f"- Decisions that needed a retry: {s['retried']} (mean attempts {s['mean_attempts']:.2f})",
            f"- Tokens: {s['input_tokens']} in / {s['output_tokens']} out, cost {cost}",
            f"- Mean latency per decision: {s['mean_latency_s']:.2f}s",
            "",
            "Memory-one fingerprint, P(C | previous round as (own, opponent)):",
            "",
            "| CC | CD | DC | DD |",
            "|---|---|---|---|",
            f"| {fmt(fp['CC'])} | {fmt(fp['CD'])} | {fmt(fp['DC'])} | {fmt(fp['DD'])} |",
            "",
            "For reference: TFT = 1, 0, 1, 0; WSLS = 1, 0, 0, 1; Grim = 1, 0, 0, 0.",
            "",
        ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
