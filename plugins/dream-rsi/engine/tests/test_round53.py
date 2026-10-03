"""Round 53: a restart that leaves no attempts behind.

- Each restart of the live loop (seven on 2026-09-30 to 10-03, one more at 17:55 UTC on 10-03) stopped the run
  just after a round's dream, a few seconds into the next round's first batch. Its six worker calls were killed and
  recorded as six failed attempts (agent_error: exit -9, no result in the stream), 40 records that did no work and
  counted against the models that were given them. `drsi stop --after-round` now asks the run to end at the round
  boundary instead: the round under way finishes, its world is frozen and its dream runs, and the run exits before it
  starts another (`--wait` returns once it has). A request left from an earlier run never stops a new one.
- An attempt still running when a run is stopped is cut short by the stop, not by its model: it is recorded as the
  loop's failure (orchestrator_error, worker.stopped_by_run), and a half build is not committed as its work.
- Review (Opus): an attempt queued for the scorer when the run stopped lost its finished build (the stopping run
  refuses to start the scorer, and the error took the orchestration path), and such a build could never be rescored;
  a call that failed on its own before the stop could be relabelled if the stop landed while its half build was being
  committed; a novelty check the stop interrupted could read as a duplicate (a failed confirmation stands as one);
  a run started by an older drsi never sees the request, so `--after-round` refuses it instead of waiting forever; the
  request names its run, so a run that starts as it is written cannot drop it; `--wait` exits 1 when the run ended
  some other way, and is refused without `--after-round`.
"""
import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from drsi import cli, guardian, live
from drsi.agent import AgentResult
from drsi.guardian import Registry
from drsi.live import LiveRunner, run_cycles
from drsi.question import ROOT
from drsi.rescore import scorable
from tests import test_live as tl
from tests import test_round10 as r10
from tests import test_round13 as r13
from tests.test_live import fixed_checker, stub_worker


def developer(sb, prompt):
    return AgentResult(ok=True)  # leaves the policy unchanged


class RoundBoundaryTest(unittest.TestCase):
    setUp = tl.LiveTest.setUp
    tearDown = tl.LiveTest.tearDown

    def run_asking(self, rounds: int):
        """run_cycles, with `drsi stop --after-round` asked during the first call of the first round."""
        w = stub_worker()

        def asking(workspace, prompt, system):
            if not w.calls:
                live.request_stop_after_round(self.camp, os.getpid())
            return w(workspace, prompt, system)
        lines = []
        rep = run_cycles(self.camp, rounds, worker_fn=asking, developer=developer, indexer=lambda ids: None,
                         progress=lines.append)
        return rep, lines

    def test_a_stop_asked_for_at_the_round_boundary_ends_the_run_after_the_rounds_dream(self):
        rep, lines = self.run_asking(3)
        self.assertEqual([r["round_id"] for r in rep["rounds"]], ["iter0001"])
        self.assertEqual(len(list((self.camp.root / "logs").glob("dream-*.json"))), 1)  # its dream ran
        self.assertFalse([n for n in self.camp.tree.nodes() if n["id"].startswith("iter0002")])
        self.assertFalse(live.stop_requested(self.camp))
        self.assertEqual(live.stopped_at_boundary(self.camp, os.getpid()), "iter0001")  # for `--wait` to read
        self.assertTrue(any("round boundary" in m for m in lines), lines)

    def test_a_request_made_in_the_runs_last_round_is_met_when_it_ends(self):
        rep, lines = self.run_asking(1)
        self.assertEqual(len(rep["rounds"]), 1)
        self.assertFalse(live.stop_requested(self.camp))  # taken, so `--wait` reports the boundary

    def test_a_request_made_before_the_first_round_is_met_after_it(self):
        live.request_stop_after_round(self.camp, os.getpid())  # as the run starts: it is this run's
        rep = run_cycles(self.camp, 3, worker_fn=stub_worker(), developer=developer, indexer=lambda ids: None)
        self.assertEqual(len(rep["rounds"]), 1)

    def test_a_request_for_another_run_never_stops_this_one(self):
        w = stub_worker()

        def asking(workspace, prompt, system):
            if not w.calls:  # a request that names some other process appears mid-run
                live.request_stop_after_round(self.camp, 1)
            return w(workspace, prompt, system)
        rep = run_cycles(self.camp, 2, worker_fn=asking, developer=developer, indexer=lambda ids: None)
        self.assertEqual(len(rep["rounds"]), 2)

    def test_a_request_left_for_an_earlier_run_does_not_stop_a_new_one(self):
        live.request_stop_after_round(self.camp, 1)  # another process, long gone
        rep = run_cycles(self.camp, 2, worker_fn=stub_worker(), developer=developer, indexer=lambda ids: None)
        self.assertEqual(len(rep["rounds"]), 2)
        self.assertFalse(live.stop_requested(self.camp))


class StopAfterRoundCliTest(unittest.TestCase):
    setUp = r13.RunAndStopReviewTest.setUp
    tearDown = r13.RunAndStopReviewTest.tearDown

    def fake_run(self, secs: float, supports: bool = True, recorded=None):
        """A process that holds the run lock for `secs`, recorded as the campaign's run (or `recorded` is)."""
        logs = self.camp.root / "logs"
        run = subprocess.Popen([sys.executable, "-c",
                                "import fcntl, time\n"
                                f"fh = open({str(logs / 'run.lock')!r}, 'w'); fcntl.flock(fh, fcntl.LOCK_EX)\n"
                                f"time.sleep({secs})\n"], start_new_session=True)
        self.procs.append(run)
        time.sleep(0.5)
        reg = Registry(logs / guardian.REGISTRY, self.camp.root / "work")
        reg.open(orchestrator=(recorded or run).pid)
        if supports:
            reg.note(stops_after_round=True)
        return run

    def stop(self, *extra):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["stop", "-c", str(self.camp.root), *extra])
        return code, out.getvalue().strip().splitlines()

    def test_with_no_run_nothing_is_asked(self):
        code, _ = self.stop("--after-round")
        self.assertEqual(code, 0)
        self.assertFalse(live.stop_requested(self.camp))

    def test_the_run_is_asked_by_name_and_left_running(self):
        run = self.fake_run(60)
        sent = []
        real_kill = os.kill

        def kill(pid, sig):  # liveness probes (signal 0) pass through; nothing may signal the run
            if sig:
                sent.append((pid, sig))
            return real_kill(pid, sig)
        with mock.patch.object(cli.os, "kill", side_effect=kill):
            code, lines = self.stop("--after-round")
        self.assertEqual(code, 0)
        self.assertEqual(sent, [])
        self.assertIsNone(run.poll())
        self.assertEqual(live._requested_run(self.camp), run.pid)
        self.assertIn("round boundary", lines[-1])

    def test_wait_returns_once_the_run_has_stopped_at_the_boundary(self):
        run = self.fake_run(2)

        def boundary():  # the run takes the request at its round boundary, then exits
            time.sleep(1)
            live.stop_at_boundary(self.camp, run.pid, "iter0007")
        threading.Thread(target=boundary).start()
        t0 = time.time()
        code, lines = self.stop("--after-round", "--wait")
        self.assertEqual(code, 0)
        self.assertIsNotNone(run.poll())
        self.assertGreater(time.time() - t0, 1)
        self.assertEqual(lines[-1], f"run {run.pid} stopped at the round boundary, after iter0007")

    def test_wait_says_so_when_the_run_ended_another_way(self):
        run = self.fake_run(1)  # ends before any round boundary: the request is never taken
        (self.camp.root / "logs" / live.STOPPED_AT).write_text(json.dumps({"run": 1, "after": "iter0003"}))  # another's
        code, lines = self.stop("--after-round", "--wait")
        self.assertEqual(code, 1)
        self.assertIsNotNone(run.poll())
        self.assertIn("ended before", lines[-1])
        self.assertFalse(live.stop_requested(self.camp))  # a request no run took is not left for the next

    def test_wait_follows_the_run_not_the_lock(self):
        run = self.fake_run(1)
        logs = self.camp.root / "logs"

        def next_run():  # the run dies; another process takes the lock and clears the request, as a new run starts
            run.wait()
            nxt = subprocess.Popen([sys.executable, "-c",
                                    "import fcntl, time\n"
                                    f"fh = open({str(logs / 'run.lock')!r}, 'w'); fcntl.flock(fh, fcntl.LOCK_EX)\n"
                                    "time.sleep(60)\n"], start_new_session=True)
            self.procs.append(nxt)
            time.sleep(0.3)
            live.take_stop_request(self.camp)
        threading.Thread(target=next_run).start()
        t0 = time.time()
        code, lines = self.stop("--after-round", "--wait")
        self.assertEqual(code, 1)  # it never reported a boundary stop it did not see, nor waited out the next run
        self.assertLess(time.time() - t0, 30)
        self.assertIn("ended before", lines[-1])

    def test_a_run_from_an_older_drsi_is_not_asked(self):
        run = self.fake_run(60, supports=False)  # it would never read the request: --wait would wait forever
        code, lines = self.stop("--after-round", "--wait")
        self.assertEqual(code, 2)
        self.assertFalse(live.stop_requested(self.camp))
        self.assertIsNone(run.poll())

    def test_a_lock_holder_that_is_not_the_recorded_run_is_not_asked(self):
        gone = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self.procs.append(gone)
        time.sleep(0.3)
        self.fake_run(60, recorded=gone)  # the registry names a run that has since died
        gone.kill()
        gone.wait()
        code, _ = self.stop("--after-round")
        self.assertEqual(code, 1)
        self.assertFalse(live.stop_requested(self.camp))

    interrupted_run = r13.RunAndStopReviewTest.interrupted_run

    def test_a_run_records_that_it_stops_at_a_round_boundary_when_asked(self):
        seen = {}

        def body():  # read the run's registry while the run is live
            seen.update(json.loads((self.camp.root / "logs" / guardian.REGISTRY).read_text()))
        self.interrupted_run(body)
        self.assertIs(seen.get("stops_after_round"), True)

    def test_wait_alone_is_refused(self):
        run = self.fake_run(60)
        code, _ = self.stop("--wait")
        self.assertEqual(code, 2)
        self.assertIsNone(run.poll())


DUPLICATE = {"verdict": "duplicate", "proposal": "", "ticket": "t", "rule": "", "family": "F00", "rationale": "r",
             "what_differs": "", "warnings": [], "doubts": "", "nearest": [], "targets_gate": "",
             "addresses_stopper": False, "exit_code": 4}


class StoppedAttemptTest(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def stopped_batch(self, second_ok=True, score=None, outside=False, check_after_stop=False):
        """Two attempts: the first is interrupted the way a stop interrupts the run (its exception reaches the batch);
        the second is mid-build at that moment (or mid-check, with check_after_stop), and its call ends once the run
        kills the workers: a failed call with half a build left in its checkout, or one that had finished. `score`
        replaces the scorer. Returns the second's record and whether the run was stopping when it killed the
        workers."""
        camp = self.campaign({"worker_models": ["opus"]})
        released, stopping_at_kill = threading.Event(), []
        second_under_way = threading.Event()  # the stop comes once the second is mid-build (or mid-check)

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            if workspace.name.endswith("-001"):
                second_under_way.wait(30)
                raise KeyboardInterrupt("signal 15")
            workspace.joinpath("value.txt").write_text("2\n")
            if not second_ok:
                workspace.joinpath("notes.txt").write_text("half a build\n")
            if outside:
                workspace.joinpath("elsewhere.txt").write_text("out of scope\n")
            second_under_way.set()
            released.wait(30)
            if second_ok:
                return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                        "notes": ""})
            return AgentResult(ok=False, error="exit -9; no result in the stream", secs=4.2)

        novel = fixed_checker("novel")

        def checker(proposal, node):
            if check_after_stop and node.endswith("-002"):  # its judge calls end with the run
                second_under_way.set()
                released.wait(30)
                return dict(DUPLICATE, proposal=proposal)
            return novel(proposal, node)
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=checker)
        if score is not None:
            r._score = score

        def kill():
            stopping_at_kill.append(r._aborting)
            released.set()
        with mock.patch.object(live, "kill_all_children", side_effect=kill), self.assertRaises(KeyboardInterrupt):
            r.run_batch([f"{ROOT}0", f"{ROOT}1"])
        end = time.time() + 30
        while time.time() < end:  # the stopped batch does not wait for its workers: the record comes when it comes
            node = {n["id"]: n for n in camp.tree.nodes()}.get("iter0001-002")
            if node is not None:
                return node, stopping_at_kill
            time.sleep(0.1)
        return None, stopping_at_kill

    def assert_stopped(self, node):
        self.assertIsNotNone(node, "the stopped attempt was never recorded")
        self.assertEqual(node["fail_class"], "orchestrator_error")
        self.assertTrue(node["worker"].get("stopped_by_run"))
        self.assertEqual(node["worker"].get("model"), "opus")

    def test_an_attempt_the_runs_stop_cut_short_is_the_loops_failure_and_keeps_no_half_build(self):
        node, stopping_at_kill = self.stopped_batch(second_ok=False)
        self.assert_stopped(node)
        self.assertEqual(stopping_at_kill, [True])  # flagged before its call was killed
        self.assertEqual(node["artifacts"]["changed"], [])
        self.assertIn("stopped", node["text"].get("orchestrator_error", ""))
        self.assertIn("exit -9", node["text"].get("worker_error", ""))

    def test_an_attempt_whose_call_and_scoring_succeed_keeps_its_record(self):
        node, _ = self.stopped_batch(second_ok=True)
        self.assertIsNotNone(node)
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertFalse(node["worker"].get("stopped_by_run"))
        self.assertIn("value.txt", node["artifacts"]["changed"])

    def test_an_attempt_whose_scoring_the_stop_killed_is_the_loops_failure_and_keeps_its_build(self):
        node, _ = self.stopped_batch(score=lambda commit, nid: {
            "score": None, "valid": False, "gates": {}, "fail_class": "scorer_error",
            "error": "the scorer was killed (signal 9)"})
        self.assert_stopped(node)
        self.assertEqual(node["artifacts"]["changed"], ["value.txt"])  # its call finished: the build is its own
        self.assertTrue(scorable(node))  # and `drsi rescore` can score it

    def test_an_attempt_queued_for_the_scorer_keeps_its_build_when_the_stopping_run_refuses_the_scorer(self):
        def refused(commit, nid):
            raise RuntimeError("the run is stopping; no new process is started")
        node, _ = self.stopped_batch(score=refused)
        self.assert_stopped(node)
        self.assertEqual(node["artifacts"]["changed"], ["value.txt"])
        self.assertIn("stopped before", node["artifacts"]["scorer_summary"])
        self.assertTrue(scorable(node))

    def test_an_out_of_scope_build_keeps_its_verdict(self):
        node, _ = self.stopped_batch(outside=True)
        self.assertEqual(node["fail_class"], "out_of_scope")
        self.assertFalse(node["worker"].get("stopped_by_run"))

    def test_a_novelty_check_the_stop_interrupted_gives_no_verdict(self):
        node, _ = self.stopped_batch(check_after_stop=True)
        self.assert_stopped(node)  # never a not_novel record from a confirmation the stop killed
        self.assertIn("novelty check", node["text"].get("orchestrator_error", ""))

    def test_a_call_that_failed_before_the_stop_stays_its_models_failure(self):
        camp = self.campaign({"worker_models": ["opus"]})

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=False, error="the model gave up")
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        real = r.ws.snapshot

        def snapshot(*a, **k):  # the stop lands while the failed call's half build is committed
            r._aborting = True
            return real(*a, **k)
        with mock.patch.object(r.ws, "snapshot", side_effect=snapshot):
            out = r.run_batch([f"{ROOT}0"])
        node = {n["id"]: n for n in camp.tree.nodes()}[out[0]["id"]]
        self.assertEqual(node["fail_class"], "agent_error")
        self.assertFalse(node["worker"].get("stopped_by_run"))


class RescoreTest(unittest.TestCase):
    def node(self, changed, stopped=True, fail_class="orchestrator_error"):
        return {"source": "live", "fail_class": fail_class, "worker": {"stopped_by_run": stopped},
                "artifacts": {"commit": "abc", "changed": changed}}

    def test_a_stopped_attempt_is_rescored_only_when_it_has_a_build(self):
        self.assertTrue(scorable(self.node(["value.txt"])))
        self.assertFalse(scorable(self.node([])))
        self.assertFalse(scorable(self.node(["value.txt"], stopped=False)))  # any other orchestration failure


if __name__ == "__main__":
    unittest.main()
