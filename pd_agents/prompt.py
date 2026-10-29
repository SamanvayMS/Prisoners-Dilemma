"""Renders the game rules and history the agent sees each round."""

from typing import Optional, Sequence

import axelrod as axl
from axelrod.interaction_utils import compute_final_score

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
    """History table, then running scores and the current round.

    The table only grows at the end, so consecutive rounds share a prefix.
    `history_window=0` shows every round.
    """
    if not own:
        return "No moves have been played yet. Choose your move for round 1."

    lines = []
    start = 0
    if history_window and len(own) > history_window:
        start = len(own) - history_window
        lines.append(
            f"Rounds 1-{start} (not shown): you cooperated {list(own[:start]).count(axl.Action.C)}/{start} "
            f"rounds, opponent cooperated {list(opp[:start]).count(axl.Action.C)}/{start} rounds."
        )
    lines.append("Round | You | Opponent")
    lines += [f"{i:>4} | {own[i - 1]}   | {opp[i - 1]}" for i in range(start + 1, len(own) + 1)]
    mine, theirs = compute_final_score(list(zip(own, opp)), game)
    round_no = len(own) + 1
    lines.append(f"Total score so far: you {mine}, opponent {theirs}.")
    lines.append(f"Choose your move for round {round_no}.")
    return "\n".join(lines)
