"""Preset rosters, and a Tournament that stops cleanly when an agent fails."""

import axelrod as axl
from axelrod.tournament import _close_objects

from .backends import AgentDecisionError

_ERROR = "AGENT_ERROR"


def classic_roster():
    """The strategies described in the README."""
    return [
        axl.TitForTat(), axl.GTFT(), axl.TitFor2Tats(), axl.WinStayLoseShift(),
        axl.Cooperator(), axl.Defector(), axl.Random(), axl.Grudger(),
        axl.HardTitForTat(), axl.ZDExtort2(), axl.ZDGTFT2(),
    ]


def notebook_roster():
    """The mixed roster used in AxelrodsTournament.ipynb, with duplicates removed,
    names updated to the current axelrod release, and Geller (a cheater) dropped."""
    return [
        axl.Cooperator(), axl.Defector(), axl.TitForTat(), axl.Grudger(), axl.Random(),
        axl.FirstByTidemanAndChieruzzi(), axl.FirstByNydegger(), axl.FirstByGrofman(),
        axl.FirstByShubik(), axl.FirstBySteinAndRapoport(), axl.FirstByJoss(),
        axl.FirstByDavis(), axl.FirstByGraaskamp(), axl.FirstByDowning(), axl.FirstByFeld(),
        axl.FirstByTullock(), axl.SoftJoss(), axl.AntiTitForTat(),
        axl.PSOGambler2_2_2(), axl.EvolvedANN(), axl.EvolvedFSM16(), axl.EvolvedLookerUp1_1_1(),
        axl.EvolvedHMM5(), axl.EvolvedLookerUp2_2_2(), axl.EvolvedANN5(),
    ]


def is_cheater(player) -> bool:
    """Strategies that read or alter the opponent's code. Against an LLM agent they
    would call its strategy (an API call) to peek at its next move."""
    c = player.classifier
    return c.get("inspects_source") or c.get("manipulates_source") or c.get("manipulates_state")


ROSTERS = {
    "first": lambda: [s() for s in axl.axelrod_first_strategies],
    "classic": classic_roster,
    "notebook": notebook_roster,
}


class AgentTournament(axl.Tournament):
    """axl.Tournament whose parallel mode stops when a worker raises.

    Stock axelrod waits forever for a worker that died mid-chunk, so an agent
    that gives up after its retries would hang the run instead of ending it.
    Serial mode needs no change: the exception propagates out of `play()`.
    """

    def _start_workers(self, workers, work_queue, done_queue, build_results=True):
        from multiprocessing import Process

        self._procs = []
        self._queues = (work_queue, done_queue)
        for _ in range(workers):
            proc = Process(target=self._worker, args=(work_queue, done_queue, build_results))
            work_queue.put("STOP")
            proc.start()
            self._procs.append(proc)
        return True

    def _worker(self, work_queue, done_queue, build_results=True):
        try:
            for chunk in iter(work_queue.get, "STOP"):
                done_queue.put(self._play_matches(chunk, build_results))
        except AgentDecisionError as exc:
            done_queue.put((_ERROR, str(exc)))
        except Exception as exc:  # noqa: BLE001 - forwarded to the parent
            done_queue.put((_ERROR, f"{type(exc).__name__}: {exc}"))
        done_queue.put("STOP")
        return True

    def _abort_workers(self):
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            proc.join()
        # Unplayed chunks are still buffered for the dead workers; without this the
        # queue's feeder thread blocks interpreter exit trying to flush them.
        for queue in self._queues:
            queue.cancel_join_thread()

    def _process_done_queue(self, workers, done_queue, build_results=True):
        out_file, writer = self._get_file_objects(build_results)
        progress_bar = self._get_progress_bar()
        stops = 0
        try:
            while stops < workers:
                results = done_queue.get()
                if isinstance(results, tuple) and results and results[0] == _ERROR:
                    self._abort_workers()
                    raise AgentDecisionError(results[1])
                if results == "STOP":
                    stops += 1
                    continue
                self._write_interactions_to_file(results, writer)
                if self.use_progress_bar:
                    progress_bar.update(1)
        finally:
            _close_objects(out_file, progress_bar)
        return True
