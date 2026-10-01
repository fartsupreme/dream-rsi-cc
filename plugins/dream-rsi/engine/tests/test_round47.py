"""Round 47: the last check before 0.4.4 (Opus), each finding reproduced first.

- When the fallback was refused as well, the attempt kept its own model but recorded the fallback's call: its session,
  transcript and error, beside `model` naming the attempt's own. The record now keeps the attempt's own refused call,
  and names the refused fallback in worker.fallback_refused; the memory-cap kills of both calls are kept.
"""
import unittest
from unittest import mock
from pathlib import Path

from drsi import live
from drsi.agent import AgentResult
from drsi.live import LiveRunner
from drsi.question import ROOT
from tests import test_round10 as r10
from tests.test_live import fixed_checker


class RefusedTwiceTest(unittest.TestCase):
    def setUp(self):
        r10.WorkerModelsTest.setUp(self)
        # round 49: a refusal no model can cover waits for the reset; these tests stop the run during that wait
        stop = mock.patch.object(live, "wait_unless_stopping", return_value=False)
        stop.start()
        self.addCleanup(stop.stop)
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def test_the_record_keeps_the_attempts_own_refused_call(self):
        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            return AgentResult(ok=False, error=f"{model} refused", limited=True, session_id=f"session-{model}",
                               transcript=f"/t/{model}.jsonl",
                               mem_kills=[{"pid": len(model), "gb": 1.0, "total_gb": 1.0, "command": model}])
        camp = self.campaign({"worker_models": ["fable"], "worker_fallback": "opus"})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        out = r.run_batch([f"{ROOT}0"])
        node = camp.tree.get(out[0]["id"])
        w = node["worker"]
        self.assertEqual(w["model"], "fable")
        self.assertEqual(w["fallback_refused"], "opus")
        self.assertEqual(w["session"], "session-fable")
        self.assertEqual(w["transcript"], "/t/fable.jsonl")
        self.assertIn("fable refused", node["text"]["worker_error"])
        self.assertEqual(sorted(k["command"] for k in w["mem_kills"]), ["fable", "opus"])


if __name__ == "__main__":
    unittest.main()
