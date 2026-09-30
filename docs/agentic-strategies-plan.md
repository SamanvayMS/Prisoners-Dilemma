# Plan: LLM agents in Axelrod's Tournament (`--with-agents`)

Branch: `feature/agentic-strategies` (off `main`)

## Goal

Extend the Axelrod tournament so that, alongside the preset `axelrod` strategies,
LLM-backed agents can enter. Each agent sees the full game history and must
return **one structured move: Cooperate or Defect**. Agents come in two tiers:

| Tier | Intent | Default model | Thinking config |
|---|---|---|---|
| `fast` (System 1) | Fast, intuitive answer with no deliberation | `claude-haiku-4-5` | no thinking |
| `reasoning` | Deliberates before answering | `claude-opus-5-5` | adaptive thinking, `effort` configurable (default `high`) |

Both tiers go through the **same harness** and the **same output contract**:
`{"move": "C" | "D"}`, enforced by structured outputs (`output_config.format`
JSON schema, via `client.messages.parse` with a Pydantic model). A reasoning
model can think as long as it wants in its thinking blocks, but its final answer
is still just the enum. Users can change the model for each tier on the CLI.

## Current state (what exists on `main`)

- Two notebooks (`AxelrodsTournament.ipynb`, `Axelrodsoriginal.ipynb`) that build
  `axl.Tournament` by hand, plus `human_play.py`.
- The README says to run `python run_tournament.py`, **but that file doesn't exist**,
  and neither does `requirements.txt`. Both get created as part of this work.

## Proposed layout

```
requirements.txt              # axelrod, anthropic, pydantic, matplotlib, pytest
run_tournament.py             # CLI entry point (preset-only by default)
pd_agents/
  __init__.py
  roster.py                   # preset rosters: "first" (axl.axelrod_first_strategies), "classic" (README list), "notebook"
  decision.py                 # Decision pydantic model {move: Literal["C","D"]} -> axl.Action
  prompt.py                   # renders rules + payoffs + history into the prompt
  backends.py                 # Backend protocol; AnthropicBackend; MockBackend
  profiles.py                 # AgentProfile presets: fast / reasoning (model, thinking, effort)
  player.py                   # LLMPlayer(axl.Player) -- the axelrod integration
  telemetry.py                # per-decision JSONL log + token/cost accounting
  analysis.py                 # memory-one fingerprint, cooperation stats for agents
tests/
  test_decision.py
  test_player.py              # clone/pickle/strategy with MockBackend
  test_tournament_smoke.py    # end-to-end run with --with-agents --mock-agents
```

## Core design

### 1. `LLMPlayer(axl.Player)`

- `strategy(self, opponent) -> axl.Action`. It reads `self.history` and
  `opponent.history`, builds the prompt, calls the backend and returns `C`/`D`.
- **Clone-safe:** axelrod calls `player.clone()` for every match, which
  rebuilds the player from `init_kwargs`. So the constructor takes only plain
  values (`profile: str`, `model: str | None`, `effort`, ...) and the API client
  is created **lazily** on the first call. That also keeps players picklable for
  axelrod's multiprocessing pool.
- **Classifier:** `stochastic=True`, which stops axelrod caching the match as
  deterministic. Also `long_run_time=True`, `makes_use_of={"game"}` (plus
  `"length"` when the horizon is revealed), and `inspects_source=False`,
  `manipulates_source=False`, `manipulates_state=False`.
- **Name:** e.g. `LLM-fast[claude-haiku-4-5]` / `LLM-reasoning[claude-opus-5-5]`,
  so ranked results tell the tiers apart.
- **Noise:** axelrod flips actions after `strategy()` returns, so the agent sees
  the actions that were actually realised. That is the correct semantics; no special handling.

### 2. What the agent sees (the prompt)

The system prompt (stable, so it caches well) holds:
- the game rules and payoff matrix taken from `self.match_attributes["game"]` (R, P, S, T),
- the instruction to maximise the agent's own total score across the match,
- the output contract (answer with C or D only).

The user message (appended as the match goes on) holds:
- the current round number,
- the full history as a compact table (`round | you | opponent | your pts | their pts`),
- cumulative scores.

Controls:
- `--reveal-length`: off by default. When it's on and the horizon is fixed, the
  agent is told the total number of rounds. Off matches Axelrod's setup; on lets
  you test whether the model backward-inducts into end-game defection.
- `--history-window N`: 0 means full history. For long matches, show only the
  last N rounds plus summary statistics.
- The opponent is **anonymous**: no strategy names are leaked.

### 3. Backends and the structured answer

```python
class Backend(Protocol):
    def decide(self, system: str, user: str) -> DecisionResult: ...
    # DecisionResult: move, raw_stop_reason, input/output tokens, latency, fallback_used
```

- `AnthropicBackend(profile)`:
  - `client.messages.parse(..., output_format=Decision)`, which gives a validated
    `Decision` back.
  - The **fast** tier on Haiku 4.5 runs with no `thinking` param and a small `max_tokens`.
  - The **reasoning** tier on Opus 5.5 runs with `thinking={"type": "adaptive"}` and
    `output_config.effort` set from `--reasoning-effort`. `max_tokens` is high enough
    to leave room for thinking, and it streams if needed.
  - Stop reasons get checked: if the answer is a `refusal`, is cut off at
    `max_tokens`, fails to parse, or the API call fails after the SDK's retries,
    the **fallback policy** (below) applies and the event is logged. It never crashes the tournament.
- `MockBackend`: deterministic and offline (plays tit-for-tat or seeded random). It is used by
  tests and by `--mock-agents`, so the whole pipeline runs without an API key.
- The `Backend` protocol leaves room to add other providers later (OpenAI,
  local vLLM/Ollama) without touching the player. v1 ships Anthropic + Mock only.

**Fallback policy** (`--agent-fallback {tft,cooperate,defect}`, default `tft`):
this is what happens when a structured answer can't be obtained. The fallback rate
per agent is reported next to results, so a high rate stands out instead of
quietly skewing scores.

### 4. `run_tournament.py` CLI

Without `--with-agents` it runs a preset-only tournament, like the notebooks do today.

```
python run_tournament.py
  --roster {first,classic,notebook}   (default: first)
  --turns 200 | --prob-end P
  --repetitions 5
  --noise 0.0
  --seed 0
  --processes 1
  --output-dir results/
  --plots

  --with-agents                        # add LLM agents to the roster
  --agents fast,reasoning              # which tiers to enter (default: both)
  --fast-model claude-haiku-4-5        # override per tier
  --reasoning-model claude-opus-5-5
  --reasoning-effort {low,medium,high,xhigh,max}  (default: high)
  --reveal-length
  --history-window 0
  --agent-fallback {tft,cooperate,defect}
  --mock-agents                        # offline backend (CI / dry runs)
  --max-agent-calls N                  # hard budget; abort before start if the estimate exceeds it
  --yes                                # skip the cost-estimate confirmation
```

### 5. Cost and rate-limit control

This matters. Each agent makes about `repetitions x turns x (N_opponents + 1)` calls.
With the first-tournament roster (15 strategies + 2 agents), 200 turns and
5 reps, that's roughly **17k calls per agent**, and every call is sequential
within a match.

- Before the run starts, print the estimated call count and a rough token/cost
  estimate for each tier. Ask for confirmation unless `--yes`, and abort if the
  estimate exceeds `--max-agent-calls`.
- When `--with-agents` is set, use smaller defaults for agent runs (for example,
  suggest `--turns 50 --repetitions 2`) and print that recommendation.
- Parallelism comes from axelrod's `processes` (whole matches run in parallel).
  Rate-limit errors (429) rely on the SDK's built-in retries plus
  `max_retries` tuning.
- Prompt caching: the system prompt stays byte-stable and history is only ever
  appended. Caching only takes effect once the prefix passes the model's minimum
  cacheable size, so it helps with long matches, not short ones.
- The Batches API doesn't fit here, because each move depends on the previous one.

### 6. Outputs and analysis

In `--output-dir`:
- `ranked.csv` / `summary.csv` from `results.summarise()`.
- `payoff_matrix.csv`.
- `agent_decisions.jsonl`: one row per LLM decision (match, opponent, rep,
  round, tier, model, move, fallback flag, stop reason, tokens, latency).
- `agent_report.md`: for each agent, the cooperation rate, fallback rate, tokens and cost,
  and its **empirical memory-one fingerprint** `P(C | CC), P(C | CD), P(C | DC), P(C | DD)`.
  That places each LLM in strategy space next to TFT / WSLS / Grim /
  ZD strategies, and shows whether the two tiers play measurably differently.
- Optional axelrod plots (boxplot, winplot, payoff) saved as PNG.

### 7. Testing

- Unit tests: parsing the decision and handling each fallback path (refusal, max_tokens,
  bad JSON, API error), prompt rendering (payoffs, history window, reveal-length), and
  `LLMPlayer.clone()` / `pickle` round-trips.
- Smoke test: `run_tournament.py --with-agents --mock-agents --turns 10 --repetitions 1`
  runs end to end and writes every output file.
- One manual live run with a real key (small roster, 20 turns) before merging.

## Implementation phases

1. **Scaffolding:** add `requirements.txt`, `pd_agents/roster.py`, and a preset-only
   `run_tournament.py`, and check it matches what the notebook produces.
2. **Agent core:** `Decision`, the prompt, `MockBackend`, `LLMPlayer`, and the unit tests.
3. **Anthropic backend:** the fast and reasoning profiles, structured outputs,
   error, refusal and fallback handling, and token accounting.
4. **CLI and guardrails:** `--with-agents` and its options, the cost estimate and budget,
   and the JSONL telemetry.
5. **Analysis and docs:** the memory-one fingerprint report and plots, plus
   README updates (fixing the missing-script and requirements references).

## Open questions (defaults assumed above)

1. **Providers:** Anthropic only for v1, or should OpenAI / local models
   (vLLM, Ollama) be first-class tiers now?
2. **Default models:** Haiku 4.5 for fast and Opus 5.5 for reasoning? The cheaper
   alternative for reasoning is Sonnet 5.5.
3. **Horizon:** keep the match length hidden by default (Axelrod's setup)?
4. **Fallback:** is TFT the right default when no valid answer comes back, or should
   the tournament abort instead?
