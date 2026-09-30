"""Round 38: the last check before 0.4.1 (Opus).

- When a worker's background task ends, Claude Code starts a turn of its own (its result carries origin
  {"kind": "task-notification"}). If that automatic turn failed (an overloaded API, say), the worker's finished
  attempt was failed with it and its commit never scored; and if the automatic turn's result carried no report, the
  worker's report was lost. The worker's own last successful result now stands unless an automatic turn succeeds with
  a report of its own. An error in the worker's own turn still fails the call.
- A runner that failed to open the stderr file left the transcript's handle open.
- An attempt whose worker wrote no proposal recorded no session and no time.
- as_recorded accepted an attempt that is its own parent, or a cycle of parents: replay reaches neither. Every
  attempt must now reach a root through its parents.
"""
import json
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult, _stream_result
from drsi.worlds import comparable
from tests.test_round33 import RESULT, stream

REPORT = {"proposal": "p", "summary": "s", "self_reported_score": None, "notes": ""}
OWN = dict(RESULT, structured_output=REPORT, result_index=0)
AUTO = {"kind": "task-notification", "producer": "session-task"}


def auto(**kw):
    return dict(RESULT, result_index=1, origin=AUTO, **kw)


class AutomaticTurnTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def result_of(self, *events):
        self.n += 1
        path = Path(self.tmp.name) / f"{self.n}.jsonl"
        path.write_text(stream(*events))
        return _stream_result(path)

    def test_a_failed_automatic_turn_leaves_the_workers_finished_turn(self):
        env, why = self.result_of(OWN, {"type": "system", "subtype": "task_notification"},
                                  auto(subtype="error_during_execution", is_error=True, structured_output=None))
        self.assertEqual(env, OWN, why)

    def test_an_automatic_turn_without_a_report_leaves_the_workers_report(self):
        env, why = self.result_of(OWN, auto(structured_output=None))
        self.assertEqual(env["structured_output"], REPORT, why)

    def test_an_automatic_turn_with_a_report_is_the_result(self):
        later = auto(structured_output=dict(REPORT, summary="after the background task"))
        env, why = self.result_of(OWN, later)
        self.assertEqual(env, later, why)

    def test_an_error_in_the_workers_own_turn_still_fails(self):
        env, why = self.result_of(dict(OWN, subtype="error_during_execution", is_error=True), OWN)
        self.assertIsNone(env)
        own_error = dict(OWN, subtype="error_max_turns", is_error=True)
        env, why = self.result_of(own_error, auto(structured_output=REPORT))
        self.assertEqual(env, own_error)  # the worker's own error is the call's result: ClaudeAgent.run fails it

    def test_the_workers_own_error_as_the_only_result_is_returned_for_the_caller_to_report(self):
        bad = dict(OWN, subtype="error_max_turns", is_error=True)
        env, why = self.result_of(bad, auto(subtype="error_during_execution", is_error=True))
        self.assertEqual(env, bad)


class HandleTest(unittest.TestCase):
    def test_a_failed_stderr_open_leaves_no_open_transcript(self):
        import gc
        import sys
        import warnings
        from drsi.agent import run_group
        with tempfile.TemporaryDirectory() as d:
            out, err = Path(d) / "t.jsonl", Path(d) / "t.stderr.txt"
            err.mkdir()  # a directory where the stderr file should go: opening it fails
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                with self.assertRaises(OSError):
                    run_group([sys.executable, "-c", "pass"], timeout=10, stdout_path=out, stderr_path=err)
                gc.collect()
            self.assertFalse([w for w in seen if issubclass(w.category, ResourceWarning)])


class NoProposalTest(unittest.TestCase):
    def test_a_worker_that_wrote_no_proposal_records_the_calls_session_and_time(self):
        from drsi.live import live_round
        from tests.test_live import LiveTest, fixed_checker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        self.addCleanup(base.tearDown)
        base.camp.update_config(lambda c: c["live"].update(require_check=True, max_proposals=1) or c)

        def worker(workspace, prompt, system):
            return AgentResult(ok=True, structured={}, session_id="s-none", secs=3.5)
        r = base.runner(worker, checker=fixed_checker("novel"))
        live_round(base.camp, base.policy, r)
        nodes = [base.camp.tree.get(i) for i in r.ids]
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual((n["worker"]["session"], n["worker"]["secs"]), ("s-none", 3.5))


class RootedTest(unittest.TestCase):
    def world(self, extra):
        return {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "a", "parent": None, "cell": "root:0", "score": 0.1, "valid": True}] + extra}

    def test_an_attempt_that_is_its_own_parent_is_not_what_live_records(self):
        self.assertEqual(comparable([self.world([{"id": "s", "parent": "s", "cell": "s", "score": 1.0,
                                                  "valid": True}])]), [])

    def test_a_cycle_of_parents_is_not_what_live_records(self):
        self.assertEqual(comparable([self.world([
            {"id": "x", "parent": "y", "cell": "y", "score": 1.0, "valid": True},
            {"id": "y", "parent": "x", "cell": "x", "score": 0.5, "valid": True}])]), [])

    def test_a_chain_from_a_root_is(self):
        self.assertEqual(len(comparable([self.world([
            {"id": "b", "parent": "a", "cell": "a", "score": 0.2, "valid": True},
            {"id": "c", "parent": "b", "cell": "b", "score": 0.3, "valid": True}])])), 1)


if __name__ == "__main__":
    unittest.main()
