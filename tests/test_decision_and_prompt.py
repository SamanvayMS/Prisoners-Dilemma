import axelrod as axl
import pytest

from pd_agents.decision import C, D, InvalidDecision, parse_decision
from pd_agents.prompt import system_prompt, user_prompt


@pytest.mark.parametrize("text, move", [
    ('{"move": "C"}', "C"),
    ('{"move":"D"}', "D"),
    ('<think>they defected twice, so...</think>\n{"move": "D"}', "D"),
    ('```json\n{"move": "C"}\n```', "C"),
    ('Sure. {"move": "cooperate"}', "C"),
    ('{"move": "d"}', "D"),
])
def test_parse_decision_accepts_actionable_answers(text, move):
    assert parse_decision(text).move == move


@pytest.mark.parametrize("text", ["", "   ", "C", '{"move": "X"}', '{"action": "C"}', '{"move": "C", "why": 1}'])
def test_parse_decision_rejects_non_moves(text):
    with pytest.raises(InvalidDecision):
        parse_decision(text)


def test_decision_maps_to_axelrod_actions():
    assert parse_decision('{"move": "C"}').action() == C
    assert parse_decision('{"move": "D"}').action() == D


def test_system_prompt_hides_length_unless_given():
    game = axl.Game()
    hidden = system_prompt(game, None, 0)
    assert "not told how many rounds" in hidden
    assert "flipped" not in hidden
    assert "(5, 0)" not in hidden and "3, 3" in hidden  # R=3 for mutual cooperation
    shown = system_prompt(game, 200, 0.05)
    assert "exactly 200 rounds" in shown
    assert "probability 0.05" in shown


def test_user_prompt_history_and_scores():
    game = axl.Game()
    assert user_prompt(game, [], []).startswith("Round 1.")
    text = user_prompt(game, [C, D, C], [C, C, D])
    assert "Round 4." in text
    # CC -> 3/3, DC -> 5/0, CD -> 0/5
    assert "you 8, opponent 8" in text
    assert text.count(" | ") == 3 * 2 + 2  # header row + 3 history rows


def test_user_prompt_history_window_summarises_old_rounds():
    game = axl.Game()
    own, opp = [C] * 10, [D] * 10
    text = user_prompt(game, own, opp, history_window=3)
    assert "Rounds 1-7 (not shown)" in text
    assert "you cooperated 7/7 rounds, opponent cooperated 0/7 rounds" in text
    assert "   8 |" in text and "   7 |" not in text
