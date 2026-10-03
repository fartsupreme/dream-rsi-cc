"""Round 53: a restart that leaves no attempts behind.

- Each restart of the live loop (seven on 2026-09-30 to 10-03, one more at 17:55 UTC on 10-03) stopped the run
  just after a round's dream, a few seconds into the next round's first batch. Its six worker calls were killed and
  recorded as six failed attempts (agent_error: exit -9, no result in the stream), 40 records that did no work and
  counted against the models that were given them. `drsi stop --after-round` now asks the run to end at the round
  boundary instead: the round under way finishes, its world is frozen and its dream runs, and the run exits before it
  starts another (`--wait` returns once it has). A request left from an earlier run never stops a new one.
- An attempt still running when a run is stopped is cut short by the stop, not by its model: it is recorded as the
  loop's failure (orchestrator_error, worker.stopped_by_run), and a half build is not committed as its work.
"""
import io
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
from tests import test_live as tl
from tests import test_round10 as r10
from tests import test_round13 as r13
from tests.test_live import fixed_checker, stub_worker


def developer(sb, prompt):
    return AgentResult(ok=True)  # leaves the policy unchanged


class RoundBoundaryTest(unittest.TestCase):
    setUp = tl.LiveTest.setUp
    tearDown = tl.LiveTest.tearDown

    def test_a_stop_asked_for_at_the_round_boundary_ends_the_run_after_the_rounds_dream(self):
        w = stub_worker()

        def asking(workspace, prompt, system):
            if not w.calls:  # the first call of the first round: someone runs `drsi stop --after-round`
                live.request_stop_after_round(self.camp)
            return w(workspace, prompt, system)
        lines = []
        rep = run_cycles(self.camp, 3, worker_fn=asking, developer=developer, indexer=lambda ids: None,
                         progress=lines.append)
        self.assertEqual([r["round_id"] for r in rep["rounds"]], ["iter0001"])
        self.assertEqual(len(list((self.camp.root / "logs").glob("dream-*.json"))), 1)  # its dream ran
        self.assertFalse([n for n in self.camp.tree.nodes() if n["id"].startswith("iter0002")])
        self.assertFalse(live.stop_requested(self.camp))
        self.assertTrue(any("round boundary" in m for m in lines), lines)

    def test_a_request_left_by_an_earlier_run_does_not_stop_a_new_one(self):
        live.request_stop_after_round(self.camp)
        rep = run_cycles(self.camp, 2, worker_fn=stub_worker(), developer=developer, indexer=lambda ids: None)
        self.assertEqual(len(rep["rounds"]), 2)
        self.assertFalse(live.stop_requested(self.camp))


class StopAfterRoundCliTest(unittest.TestCase):
    setUp = r13.RunAndStopReviewTest.setUp
    tearDown = r13.RunAndStopReviewTest.tearDown

    def fake_run(self, secs: float):
        """A process that holds the run lock for `secs` and is recorded as the campaign's run."""
        logs = self.camp.root / "logs"
        run = subprocess.Popen([sys.executable, "-c",
                                "import fcntl, time\n"
                                f"fh = open({str(logs / 'run.lock')!r}, 'w'); fcntl.flock(fh, fcntl.LOCK_EX)\n"
                                f"time.sleep({secs})\n"], start_new_session=True)
        self.procs.append(run)
        time.sleep(0.5)
        Registry(logs / guardian.REGISTRY, self.camp.root / "work").open(orchestrator=run.pid)
        return run

    def test_with_no_run_nothing_is_asked(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--after-round"]), 0)
        self.assertFalse(live.stop_requested(self.camp))

    def test_the_run_is_asked_and_left_running(self):
        run = self.fake_run(60)
        out = io.StringIO()
        sent = []
        real_kill = os.kill

        def kill(pid, sig):  # liveness probes (signal 0) pass through; nothing may signal the run
            if sig:
                sent.append((pid, sig))
            return real_kill(pid, sig)
        with redirect_stdout(out), mock.patch.object(cli.os, "kill", side_effect=kill):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--after-round"]), 0)
        self.assertEqual(sent, [])
        self.assertIsNone(run.poll())
        self.assertTrue(live.stop_requested(self.camp))
        self.assertIn("round", out.getvalue())

    def test_wait_returns_once_the_run_has_ended(self):
        run = self.fake_run(2)

        def boundary():  # the run takes the request at its round boundary, then exits
            time.sleep(1)
            live.take_stop_request(self.camp)
        threading.Thread(target=boundary).start()
        out = io.StringIO()
        t0 = time.time()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--after-round", "--wait"]), 0)
        self.assertIsNotNone(run.poll())
        self.assertGreater(time.time() - t0, 1)
        self.assertIn("round boundary", out.getvalue())

    def test_wait_says_so_when_the_run_ended_another_way(self):
        run = self.fake_run(1)  # ends before any round boundary: the request is never taken
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--after-round", "--wait"]), 0)
        self.assertIsNotNone(run.poll())
        self.assertIn("before", out.getvalue())
        self.assertFalse(live.stop_requested(self.camp))  # a request no run took is not left for the next


class StoppedAttemptTest(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def stopped_batch(self, second_ok: bool, scoring_killed: bool = False):
        """Two attempts: the first is interrupted the way a stop interrupts the run (its exception reaches the batch);
        the second is mid-build at that moment, and its call ends once the run kills the workers (a failed call with
        half a build left in its checkout, or one that had already finished). Returns the second's record."""
        camp = self.campaign({"worker_models": ["opus"]})
        released = threading.Event()

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            if workspace.name.endswith("-001"):
                raise KeyboardInterrupt("signal 15")
            workspace.joinpath("value.txt").write_text("2\n")
            if not second_ok:
                workspace.joinpath("notes.txt").write_text("half a build\n")
            released.wait(30)
            if second_ok:
                return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                        "notes": ""})
            return AgentResult(ok=False, error="exit -9; no result in the stream", secs=4.2)
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        if scoring_killed:  # the build finished, then the stop killed its scorer
            r._score = lambda commit, nid: {"score": None, "valid": False, "gates": {}, "fail_class": "scorer_error",
                                            "error": "the scorer was killed (signal 9)"}
        with mock.patch.object(live, "kill_all_children", side_effect=released.set), \
                self.assertRaises(KeyboardInterrupt):
            r.run_batch([f"{ROOT}0", f"{ROOT}1"])
        end = time.time() + 30
        while time.time() < end:  # the stopped batch does not wait for its workers: the record comes when it comes
            node = {n["id"]: n for n in camp.tree.nodes()}.get("iter0001-002")
            if node is not None:
                return node, camp
            time.sleep(0.1)
        return None, camp

    def test_an_attempt_the_runs_stop_cut_short_is_the_loops_failure_and_keeps_no_half_build(self):
        node, camp = self.stopped_batch(second_ok=False)
        self.assertIsNotNone(node, "the stopped attempt was never recorded")
        self.assertEqual(node["fail_class"], "orchestrator_error")
        self.assertTrue(node["worker"].get("stopped_by_run"))
        self.assertEqual(node["worker"].get("model"), "opus")
        self.assertEqual(node["artifacts"]["changed"], [])
        self.assertIn("stopped", node["text"].get("orchestrator_error", ""))
        self.assertIn("exit -9", node["text"].get("worker_error", ""))

    def test_an_attempt_that_finished_its_call_before_the_stop_keeps_its_record(self):
        node, camp = self.stopped_batch(second_ok=True)
        self.assertIsNotNone(node)
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertFalse(node["worker"].get("stopped_by_run"))
        self.assertIn("value.txt", node["artifacts"]["changed"])


    def test_an_attempt_whose_scoring_the_stop_killed_is_the_loops_failure_and_keeps_its_build(self):
        node, camp = self.stopped_batch(second_ok=True, scoring_killed=True)
        self.assertIsNotNone(node)
        self.assertEqual(node["fail_class"], "orchestrator_error")
        self.assertTrue(node["worker"].get("stopped_by_run"))
        self.assertEqual(node["artifacts"]["changed"], ["value.txt"])  # its call finished: the build is its own


if __name__ == "__main__":
    unittest.main()
