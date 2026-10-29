"""Renders the game rules and history the agent sees each round."""

from typing import Optional, Sequence

import axelrod as axl

SYSTEM_TEMPLATE = """\
You are playing an iterated Prisoner's Dilemma against one opponent.

Each round, you and your opponent simultaneously choose C (cooperate) or D (defect).
Points per round (yours, opponent's):
- you C, opponent C: {R}, {R}
- you C, opponent D: {S}, {T}
- you D, opponent C: {T}, {S}
- you D, opponent D: {P}, {P}

{length_line}{noise_line}Your goal is to maximise your own total score over the whole match.
You are told nothing about who your opponent is beyond the moves they make.

Answer with your move for the current round as JSON: {{"move": "C"}} or {{"move": "D"}}."""


def system_prompt(game: axl.Game, length: Optional[int], noise: float) -> str:
    R, P, S, T = game.RPST()
    if length is not None:
        length_line = f"The match lasts exactly {length} rounds.\n"
    else:
        length_line = "You are not told how many rounds the match lasts.\n"
    noise_line = ""
    if noise:
        noise_line = (
            f"Each move is flipped with probability {noise}; the history shows the moves "
            "that were actually played.\n"
        )
    return SYSTEM_TEMPLATE.format(R=R, P=P, S=S, T=T, length_line=length_line, noise_line=noise_line)


def user_prompt(
    game: axl.Game,
    own: Sequence[axl.Action],
    opp: Sequence[axl.Action],
    history_window: int = 0,
) -> str:
    """History table plus running scores. `history_window=0` shows every round."""
    round_no = len(own) + 1
    if not own:
        return "Round 1. No moves have been played yet. Choose your move."

    my_score = their_score = 0
    rows = []
    for i, (a, b) in enumerate(zip(own, opp), start=1):
        mine, theirs = game.score((a, b))
        my_score += mine
        their_score += theirs
        rows.append(f"{i:>4} | {a}   | {b}")

    lines = [f"Round {round_no}."]
    shown = rows
    if history_window and len(rows) > history_window:
        hidden = len(rows) - history_window
        lines.append(
            f"Rounds 1-{hidden} (not shown): you cooperated {_coop(own[:hidden])}, "
            f"opponent cooperated {_coop(opp[:hidden])}."
        )
        shown = rows[hidden:]
    lines += ["Round | You | Opponent", *shown]
    lines.append(f"Total score so far: you {my_score}, opponent {their_score}.")
    lines.append(f"Choose your move for round {round_no}.")
    return "\n".join(lines)


def _coop(actions: Sequence[axl.Action]) -> str:
    n = sum(1 for a in actions if a == axl.Action.C)
    return f"{n}/{len(actions)} rounds"
