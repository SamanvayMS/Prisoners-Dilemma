import json
import pickle

import axelrod as axl
import pytest

import run_tournament
from pd_agents.backends import AgentDecisionError
from pd_agents.player import LLMPlayer
from pd_agents.tournament import AgentTournament


def mock_player(model="tft", **kw):
    return LLMPlayer(provider="mock", model=model, **kw)


def test_clone_and_pickle_keep_configuration():
    p = mock_player(mode="reasoning", history_window=5, reveal_length=True)
    for copy in (p.clone(), pickle.loads(pickle.dumps(p))):
        assert copy.name == "LLM reasoning (mock:tft)"
        assert (copy.mode, copy.history_window, copy.reveal_length) == ("reasoning", 5, True)
        assert str(copy) == copy.name


def test_mock_agent_plays_tit_for_tat_and_logs(tmp_path):
    log = tmp_path / "log.jsonl"
    agent = mock_player(log_path=str(log))
    match = axl.Match((agent, axl.Alternator()), turns=6)
    match.play()
    assert [str(a) for a in agent.history] == list("CCDCDC")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["round"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert rows[0]["prev"] is None and rows[1]["prev"] == "CC"
    assert rows[0]["opponent"] == "Alternator"


def test_length_is_hidden_unless_revealed_and_fixed():
    revealed = mock_player(reveal_length=True)
    revealed.set_match_attributes(length=50)
    assert revealed._known_length() == 50
    revealed.set_match_attributes(length=float("inf"))  # random-length match
    assert revealed._known_length() is None
    hidden = mock_player()
    hidden.set_match_attributes(length=50)
    assert hidden._known_length() is None


@pytest.mark.parametrize("processes", [None, 2])
def test_tournament_with_agents(processes):
    players = [axl.TitForTat(), axl.Defector(), mock_player(), mock_player("alld", mode="reasoning")]
    results = AgentTournament(players, prob_end=0.2, repetitions=2, seed=3).play(
        processes=processes, progress_bar=False)
    assert set(results.ranked_names) == {str(p) for p in players}


@pytest.mark.parametrize("processes", [None, 2])
def test_tournament_stops_when_agent_cannot_answer(processes):
    players = [axl.TitForTat(), axl.Cooperator(), mock_player("broken", retries=2)]
    with pytest.raises(AgentDecisionError, match="no actionable move"):
        AgentTournament(players, turns=5, repetitions=1).play(processes=processes, progress_bar=False)


def test_agent_spec_parsing():
    spec = run_tournament.AgentSpec.parse("ollama:qwen3:8b@reasoning")
    assert (spec.provider, spec.model, spec.mode) == ("ollama", "qwen3:8b", "reasoning")
    spec = run_tournament.AgentSpec.parse("anthropic:claude-haiku-4-5")
    assert spec.mode == "fast"
    for bad in ("claude-haiku-4-5", "nope:model", "ollama:llama3@slow"):
        with pytest.raises(Exception):
            run_tournament.AgentSpec.parse(bad)


def test_cli_end_to_end_with_mock_agents(tmp_path, capsys):
    code = run_tournament.main([
        "--roster", "classic", "--with-agents", "--mock-agents", "--turns", "10",
        "--repetitions", "1", "--seed", "0", "--output-dir", str(tmp_path),
    ])
    assert code == 0
    for name in ("ranked.txt", "summary.csv", "payoff_matrix.json", "agent_decisions.jsonl", "agent_report.md"):
        assert (tmp_path / name).exists(), name
    report = (tmp_path / "agent_report.md").read_text()
    assert "LLM fast (mock:tft-as-claude-haiku-4-5)" in report
    assert "LLM reasoning (mock:tft-as-claude-opus-5-5)" in report


def test_cli_stops_with_exit_code_on_agent_failure(tmp_path):
    code = run_tournament.main([
        "--roster", "classic", "--with-agents", "--agent", "mock:broken", "--agent-retries", "2",
        "--turns", "5", "--repetitions", "1", "--output-dir", str(tmp_path),
    ])
    assert code == 3


def test_cli_refuses_over_budget_and_reveal_without_fixed_length(tmp_path):
    common = ["--with-agents", "--output-dir", str(tmp_path)]
    assert run_tournament.main(common + ["--max-agent-calls", "10", "--yes"]) == 1
    assert run_tournament.main(common + ["--reveal-length"]) == 2


def test_cli_preset_only_by_default(tmp_path):
    code = run_tournament.main(["--roster", "classic", "--turns", "10", "--repetitions", "1",
                                "--output-dir", str(tmp_path)])
    assert code == 0
    assert not (tmp_path / "agent_report.md").exists()
    assert "LLM" not in (tmp_path / "ranked.txt").read_text()
