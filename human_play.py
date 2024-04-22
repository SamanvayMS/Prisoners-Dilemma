import axelrod as axl
import sys

print(sys.path)
turns = int(sys.argv[1])

me = axl.Human(name='me')
players = [axl.TitForTat(), me]
match = axl.Match(players, turns=turns)
match.play()
print('match final score:- ',match.final_score())
print('match final score per turn:-',match.final_score_per_turn())
print('match winner:-',match.winner())