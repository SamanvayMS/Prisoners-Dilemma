# LLM agents in Axelrod's Tournament (`--with-agents`)

Branch: `feature/agentic-strategies`. Usage is documented in the README; this file
records the design and the decisions behind it.

## Decisions

| Question | Decision |
|---|---|
| Providers in v1 | Anthropic, plus Ollama (and any other server) through the OpenAI-compatible chat completions API. A `mock` provider runs offline. |
| Agent kinds | `fast` (System 1: answers with no deliberation) and `reasoning` (thinks first), on every provider. |
| Model roles | The reasoning model is a tournament player. Defaults: Claude Haiku 4.5 (`fast`) and Claude Opus 5.5 (`reasoning`), both overridable. |
| Match length | Hidden from agents and random by default (`prob_end = 1/turns`). `--fixed-length` gives exact lengths; `--reveal-length` tells agents. |
| Invalid answers | No fallback move. Retry with backoff (`--agent-retries`, default 6), then stop the tournament (exit code 3). Unfixable errors (auth, unknown model, bad request) stop it at once. |

## Layout

```
run_tournament.py          CLI; preset-only unless --with-agents
pd_agents/decision.py      Decision {move: "C"|"D"}, its JSON schema, strict and lenient parsers
pd_agents/prompt.py        system prompt (rules, payoffs) and per-round history prompt
pd_agents/backends.py      retry loop; Anthropic, OpenAI-compatible and mock backends
pd_agents/player.py        LLMPlayer(axl.Player)
pd_agents/tournament.py    preset rosters; AgentTournament (parallel mode that can abort)
pd_agents/telemetry.py     per-decision JSONL (tokens, retries, latency), agent report
tests/                     57 tests, no network needed
```

## How it works

**The answer contract.** Every backend sends the same JSON schema: one property,
`move`, with enum `["C", "D"]`. Anthropic gets it as `output_config.format`, and
OpenAI-compatible servers get it as `response_format`. Anthropic answers are parsed
strictly. Answers from OpenAI-compatible servers, which don't all enforce the schema,
are parsed leniently: `<think>` blocks, code fences and `cooperate`/`defect`
spellings are accepted. Anything else counts as an invalid answer and is retried.

**Retries.** `Backend.decide` is the only retry loop; the SDK clients' own retries
are turned off, so `--agent-retries` is the real number of attempts per move.
Backoff is exponential (capped at 60 s) and never shorter than the server's
`retry-after`.

**What `fast` and `reasoning` mean per model.**

| Model | fast | reasoning |
|---|---|---|
| Claude Haiku 4.5 | no thinking | thinking with a 4096-token budget |
| Claude Sonnet 5.5 | `thinking: between_tools`, low effort | adaptive thinking, `--reasoning-effort` |
| Claude Opus 5.5 / Fable | low effort (thinking can't be disabled on these models) | adaptive thinking, `--reasoning-effort` |
| Other Claude models | `thinking: disabled` | adaptive thinking, `--reasoning-effort` |
| OpenAI-compatible (Ollama) | no reasoning parameters, 256 max tokens | `reasoning_effort`, 8192 max tokens |

On Ollama, choose a non-thinking model for `fast`. If a thinking model runs out of
tokens, the error says so. `--openai-extra-body` passes server-specific fields
through, for example `{"reasoning_effort": "none"}` on servers that support it.

**What the agent sees.** The payoff matrix (R, P, S, T from the match's game),
the noise level if there is any, the round number, the history table of both
players' moves, and both running scores. It never sees the opponent's name or, by
default, the match length. `--history-window N` replaces older rounds with a
cooperation summary.

**Integration with axelrod.**
- axelrod rebuilds each player from its constructor arguments for every match, and
  can run matches in worker processes. So `LLMPlayer` only takes plain values and
  creates its API client on its first move, inside the worker.
- `LLMPlayer` is marked `stochastic`, so axelrod never caches its matches as deterministic.
- Stock axelrod hand-rolls its worker processes and waits forever if one raises.
  `AgentTournament` overrides only `_run_parallel`, running matches on a
  `multiprocessing.Pool`: the worker's exception is re-raised in the parent and the
  other workers are terminated. Serial mode needed no change.
- Strategies that read their opponent's code (`axl.Classifiers.obey_axelrod` is
  false) are dropped when agents play, because they would call the agent's model to
  peek at its next move.
- Cooperation rate and the memory-one fingerprint come from axelrod's `ResultSet`
  (`cooperating_rating`, `state_to_action_distribution`), i.e. from the moves
  actually played. The JSONL log only adds what axelrod doesn't record.
- axelrod is pinned to 4.13.1. Version 4.14 adds a hard PyTorch dependency for one
  strategy that this project doesn't use.

**Cost control.** Moves depend on earlier moves, so calls can't be batched. Each
agent makes about `repetitions x turns x (players + 1)` calls. With the defaults
(first-tournament roster, 200 turns, 5 repetitions) that is 18,000 calls per agent;
the estimate puts Opus 5.5 in reasoning mode at about $290. The script prints this
estimate, asks before starting and enforces `--max-agent-calls`. Use smaller
`--turns`/`--repetitions` when iterating, and `--mock-agents` to test the pipeline
for free.

**Refusal fallbacks are off on purpose.** The API's server-side fallback would
silently move a refused request to a different model, so a different player
would be making moves mid-match. A refusal is retried on the same model instead.

## Verified

- 57 unit and integration tests: answer parsing, prompts, Anthropic and
  OpenAI-compatible request shapes against fake clients, the retry and abort
  paths, clone/pickle, serial and parallel tournaments, and the CLI.
- End-to-end CLI runs with mock agents, including `--plots` and an abort in
  4-process parallel mode (exits with code 3 and leaves no stray processes).

## Not yet verified

- No live calls. This container has no Anthropic API key and no Ollama server.
  Before a full run, do a short live run, e.g.
  `--roster classic --turns 10 --repetitions 1`, against each provider.
- Ollama's handling of `response_format` json_schema and of `reasoning_effort`
  varies by version and model. The retry loop and tolerant parser cover the
  common variations, but check them against the models you plan to use.
