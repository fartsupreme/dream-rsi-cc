"""Round 46: the review of round 45 (Opus), each finding reproduced first.

- A call that ran and then crossed its usage limit (a 429 result after turns of work) fell back like one refused
  before it began, and the fallback started from what the refused call left: a half-written proposal was judged and
  recorded, a half-edited checkout was scored as the fallback's. Only a refusal before the model ran falls back now:
  the call's result says no turn of work was done (num_turns at most 1, duration_api_ms 0, no model usage) and the
  stream's last usage event says rejected (a real refusal, captured: status "rejected", a synthetic assistant
  message, a 429 result after 0 ms of API time). Any 429 had counted, a short-term rate limit included. Before the
  fallback's proposal call, the proposal file is cleared, as before any proposal call.
- When the fallback was refused too, the attempt was recorded as the fallback's work, which no model did: it keeps
  its own model and records the refused fallback. A fallback names the phase it took over.
- Memory-cap kills of the refused call were lost from the record.
"""
import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from drsi import live
from drsi.agent import AgentResult, ClaudeAgent
from drsi.live import LiveRunner
from drsi.question import ROOT
from tests import test_round10 as r10
from tests.test_live import fixed_checker

INIT = {"type": "system", "subtype": "init"}
REJECTED = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "seven_day_overage_included"}}
ALLOWED = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "rateLimitType": "seven_day"}}
SYNTHETIC = {"type": "assistant", "error": "rate_limit", "message": {"model": "<synthetic>", "content": []}}
REFUSED = {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
           "api_error": "model_requires_usage_credits", "num_turns": 1, "duration_api_ms": 0, "modelUsage": {},
           "result": "You've reached your Fable limit."}
WORKED = dict(REFUSED, num_turns=14, duration_api_ms=812_000, modelUsage={"claude-fable-5-1": {"inputTokens": 9}})


class Runner:
    def __init__(self, *events):
        self.text = "".join(json.dumps(e) + "\n" for e in events)

    def __call__(self, args, stdout_path=None, stderr_path=None, **kw):
        with open(stdout_path, "a") as fh:
            fh.write(self.text)
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")


class SignalTest(unittest.TestCase):
    def limited(self, *events):
        with tempfile.TemporaryDirectory() as d:
            return ClaudeAgent(model="fable", tools="Read", runner=Runner(*events)).run(
                d, "p", transcript=Path(d) / "t.jsonl").limited

    def test_a_refusal_before_the_model_ran_is_limited(self):
        self.assertTrue(self.limited(INIT, REJECTED, SYNTHETIC, REFUSED))

    def test_a_limit_crossed_after_work_is_not(self):
        self.assertFalse(self.limited(INIT, ALLOWED, {"type": "assistant", "message": {}}, REJECTED, WORKED))

    def test_a_429_without_a_rejected_usage_status_is_not(self):
        self.assertFalse(self.limited(INIT, ALLOWED, dict(REFUSED, api_error="rate_limit_error")))
        self.assertFalse(self.limited(INIT, dict(REFUSED, api_error="rate_limit_error")))

    def test_a_rejected_status_without_429_still_counts(self):
        self.assertTrue(self.limited(INIT, REJECTED, {k: v for k, v in REFUSED.items() if k != "api_error_status"}))


class RecordTest(unittest.TestCase):
    def setUp(self):
        r10.WorkerModelsTest.setUp(self)
        # round 49: a refusal no model can cover waits for the reset; these tests stop the run during that wait
        stop = mock.patch.object(live, "wait_unless_stopping", return_value=False)
        stop.start()
        self.addCleanup(stop.stop)
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def worker(self, calls, refuse=("fable",), leave_proposal=False, kills=None):
        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            calls.append((workspace.name, phase, model))
            if model in refuse:
                if leave_proposal and phase == "propose":  # what a stale file from an earlier call would look like
                    pdir = self.camp_root / "work" / "_proposals" / workspace.name
                    pdir.mkdir(parents=True, exist_ok=True)
                    (pdir / "proposal.txt").write_text("STALE PROPOSAL")
                return AgentResult(ok=False, error="limit", limited=True, mem_kills=list(kills or []))
            if phase == "propose":
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        return run

    def run_one(self, llm, **kw):
        camp = self.campaign(llm)
        self.camp_root = camp.root
        calls = []
        r = LiveRunner(camp, self.worker(calls, **kw), indexer=lambda ids: None, round_id="iter0001",
                       checker=fixed_checker("novel"))
        out = r.run_batch([f"{ROOT}0"])
        return camp.tree.get(out[0]["id"]), calls

    def test_a_refused_fallback_leaves_the_attempt_its_own_model(self):
        node, calls = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"}, refuse=("fable", "opus"))
        self.assertFalse(node["valid"])
        self.assertEqual(node["worker"]["model"], "fable")
        self.assertEqual(node["worker"].get("fallback_refused"), "opus")
        self.assertNotIn("fell_back_from", node["worker"])

    def test_a_fallback_names_the_phase_it_took_over(self):
        node, _ = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"})
        self.assertEqual(node["worker"]["fell_back_from"], "fable")
        self.assertEqual(node["worker"]["fell_back_in"], "propose")

    def test_the_fallbacks_proposal_is_never_one_left_before_it(self):
        node, _ = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"}, leave_proposal=True)
        self.assertNotIn("STALE PROPOSAL", json.dumps(node))

    def test_kills_in_the_refused_call_are_kept(self):
        kill = {"pid": 5, "gb": 4.0, "total_gb": 4.0, "command": "python3"}
        node, _ = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"}, kills=[kill])
        self.assertEqual(node["worker"].get("mem_kills"), [kill])


if __name__ == "__main__":
    unittest.main()
