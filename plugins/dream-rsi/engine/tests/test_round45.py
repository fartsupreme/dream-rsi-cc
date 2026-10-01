"""Round 45: a model whose usage limit is reached falls back to another (llm.worker_fallback).

In the first campaign one model ran into its own weekly limit a day into the week: from then on every attempt on it
was refused within seconds (an error result with api_error_status 429, "You've reached your ... limit"), half the
seats of every batch recorded nothing, and its working attempts had been the campaign's best. A call refused for its
usage limit is now made again on llm.worker_fallback, the rest of that attempt runs on the fallback, and the attempt
records the model that did the work and the one it fell back from. Every attempt tries its own model first, so the
model comes back on its own when its limit resets or the account changes.
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult, ClaudeAgent
from drsi.live import LiveRunner
from drsi.question import ROOT
from tests.test_live import fixed_checker
from tests import test_round10 as r10

LIMIT = {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
         "result": "You've reached your Fable limit. Switch to another model, or manage usage credits."}
OTHER = {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 500, "result": "API Error: 500"}
OK = {"type": "result", "subtype": "success", "is_error": False, "result": "{}", "structured_output": {}}


class Runner:
    def __init__(self, event):
        self.event = event

    def __call__(self, args, stdout_path=None, stderr_path=None, **kw):
        out = json.dumps(self.event) + "\n"
        if stdout_path is not None and self.event is LIMIT:  # a real refusal's stream (round 46) says rejected first
            out = json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}}) + "\n" + out
        if stdout_path is not None:
            with open(stdout_path, "a") as fh:
                fh.write(out)
            out = ""
        return subprocess.CompletedProcess(args, 0 if not self.event.get("is_error") else 1, stdout=out, stderr="")


class LimitedTest(unittest.TestCase):
    def test_a_usage_limit_refusal_is_marked_limited(self):
        with tempfile.TemporaryDirectory() as d:
            for event, limited in ((LIMIT, True), (OTHER, False), (OK, False)):
                for transcript in (None, Path(d) / f"{event['api_error_status'] if 'api_error_status' in event else 0}.jsonl"):
                    if transcript is not None and transcript.exists():
                        transcript.unlink()
                    res = ClaudeAgent(model="fable", tools="Read", runner=Runner(event)).run(d, "p", transcript=transcript)
                    self.assertEqual(res.limited, limited, (event, transcript))


class FallbackTest(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp  # its campaign set-up, not its tests
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def worker(self, calls, refuse=("fable",), how=LIMIT):
        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            calls.append((workspace.name, phase, model))
            if model in refuse:
                return AgentResult(ok=False, error=str(how["result"]), limited=how is LIMIT)
            if phase == "propose":
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        return run

    def runner(self, camp, calls, **kw):
        return LiveRunner(camp, self.worker(calls, **kw), indexer=lambda ids: None, round_id="iter0001",
                          checker=fixed_checker("novel"))

    def test_a_limited_model_falls_back_and_the_attempt_records_both(self):
        camp = self.campaign({"worker_models": ["opus", "fable"], "worker_fallback": "opus"})
        calls = []
        out = self.runner(camp, calls).run_batch([f"{ROOT}0", f"{ROOT}1"])
        nodes = [camp.tree.get(o["id"]) for o in out]
        self.assertTrue(all(n["valid"] for n in nodes), [n.get("fail_class") for n in nodes])
        fable_node = next(n for n in nodes if n["worker"].get("fell_back_from"))
        self.assertEqual(fable_node["worker"]["model"], "opus")
        self.assertEqual(fable_node["worker"]["fell_back_from"], "fable")
        mine = [(phase, model) for nid, phase, model in calls if nid == fable_node["id"]]
        # refused once, then the proposal and the build on the fallback: the refused model is not asked again
        self.assertEqual(mine, [("propose", "fable"), ("propose", "opus"), ("implement", "opus")])
        other = next(n for n in nodes if n is not fable_node)
        self.assertNotIn("fell_back_from", other["worker"])

    def test_without_a_fallback_a_limited_attempt_fails_as_before(self):
        camp = self.campaign({"worker_models": ["fable"]})
        calls = []
        out = self.runner(camp, calls).run_batch([f"{ROOT}0"])
        node = camp.tree.get(out[0]["id"])
        self.assertFalse(node["valid"])
        self.assertEqual(len(calls), 1)

    def test_only_a_usage_limit_falls_back(self):
        camp = self.campaign({"worker_models": ["fable"], "worker_fallback": "opus"})
        calls = []
        out = self.runner(camp, calls, how=OTHER).run_batch([f"{ROOT}0"])
        self.assertFalse(camp.tree.get(out[0]["id"])["valid"])
        self.assertEqual([m for _, _, m in calls], ["fable"])

    def test_a_limited_fallback_is_tried_once(self):
        camp = self.campaign({"worker_models": ["fable"], "worker_fallback": "opus"})
        calls = []
        out = self.runner(camp, calls, refuse=("fable", "opus")).run_batch([f"{ROOT}0"])
        self.assertFalse(camp.tree.get(out[0]["id"])["valid"])
        self.assertEqual([m for _, _, m in calls], ["fable", "opus"])

    def test_a_model_never_falls_back_onto_itself(self):
        camp = self.campaign({"worker_models": ["opus"], "worker_fallback": "opus"})
        calls = []
        out = self.runner(camp, calls, refuse=("opus",)).run_batch([f"{ROOT}0"])
        self.assertFalse(camp.tree.get(out[0]["id"])["valid"])
        self.assertEqual([m for _, _, m in calls], ["opus"])

    def test_the_next_attempt_tries_its_own_model_again(self):
        camp = self.campaign({"worker_models": ["fable"], "worker_fallback": "opus"})
        calls = []
        r = self.runner(camp, calls)
        r.run_batch([f"{ROOT}0"])
        r.run_batch([f"{ROOT}1"])
        firsts = [m for nid, phase, m in calls if phase == "propose"][::2]
        self.assertEqual(firsts, ["fable", "fable"])

    def test_a_bad_fallback_setting_is_refused(self):
        for i, bad in enumerate((["opus"], "", 3)):
            camp = self.campaign({"worker_models": ["fable"], "worker_fallback": bad}, name=f"fb{i}")
            with self.assertRaises(ValueError):
                self.runner(camp, [])


if __name__ == "__main__":
    unittest.main()
