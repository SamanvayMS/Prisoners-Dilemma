"""Preset rosters, and a Tournament that stops cleanly when an agent fails."""

from functools import partial
from multiprocessing import Pool

import axelrod as axl
from axelrod.tournament import _close_objects


def classic_roster():
    """The strategies described in the README."""
    return [
        axl.TitForTat(), axl.GTFT(), axl.TitFor2Tats(), axl.WinStayLoseShift(),
        axl.Cooperator(), axl.Defector(), axl.Random(), axl.Grudger(),
        axl.HardTitForTat(), axl.ZDExtort2(), axl.ZDGTFT2(),
    ]


def notebook_roster():
    """The mixed roster used in AxelrodsTournament.ipynb, with duplicates removed and
    names updated to the current axelrod release (which no longer ships Geller)."""
    return [
        axl.Cooperator(), axl.Defector(), axl.TitForTat(), axl.Grudger(), axl.Random(),
        axl.FirstByTidemanAndChieruzzi(), axl.FirstByNydegger(), axl.FirstByGrofman(),
        axl.FirstByShubik(), axl.FirstBySteinAndRapoport(), axl.FirstByJoss(),
        axl.FirstByDavis(), axl.FirstByGraaskamp(), axl.FirstByDowning(), axl.FirstByFeld(),
        axl.FirstByTullock(), axl.SoftJoss(), axl.AntiTitForTat(),
        axl.PSOGambler2_2_2(), axl.EvolvedANN(), axl.EvolvedFSM16(), axl.EvolvedLookerUp1_1_1(),
        axl.EvolvedHMM5(), axl.EvolvedLookerUp2_2_2(), axl.EvolvedANN5(),
    ]


def axelrod_first_roster():
    return [s() for s in axl.axelrod_first_strategies]


ROSTERS = {"first": axelrod_first_roster, "classic": classic_roster, "notebook": notebook_roster}


class AgentTournament(axl.Tournament):
    """axl.Tournament whose parallel mode stops when a match raises.

    Stock axelrod hand-rolls its worker processes and waits forever for one that
    died mid-chunk, so an agent that gives up after its retries would hang the
    run. A Pool re-raises the worker's exception here, and leaving the `with`
    block terminates the other workers. Serial mode needs no change.
    """

    def _run_parallel(self, processes: int = 2, build_results: bool = True) -> bool:
        out_file, writer = self._get_file_objects(build_results)
        progress_bar = self._get_progress_bar()
        play = partial(self._play_matches, build_results=build_results)
        try:
            with Pool(self._n_workers(processes=processes)) as pool:
                for results in pool.imap_unordered(play, self.match_generator.build_match_chunks()):
                    self._write_interactions_to_file(results, writer)
                    if self.use_progress_bar:
                        progress_bar.update(1)
        finally:
            _close_objects(out_file, progress_bar)
        return True
