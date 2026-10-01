"""Round 50: the last check before 0.4.5 (Opus), each finding reproduced first.

- After a proposal on the fallback, a build refused on the fallback counted as "the fallback is the same model" and
  waited for the fallback alone: the attempt's own model, maybe free again, was never tried, and the batch waited on
  that one attempt until the fallback's reset (days, for a weekly cap). The other model of the attempt is tried:
  its own model once it runs on the fallback, the fallback otherwise.
- ClaudeCLI.json's second try wrapped a refusal as "two attempts failed", so a limit that began between the two
  tries (the first a timeout or a 500) was no limit to the check.
- The indexer's skip of attempts that did no work also skipped a checked proposal whose build failed; and family
  assignment still asked about attempts never fingerprinted (their fingerprint holds only outcome and killed_by).
  Fingerprinting now skips an attempt with no proposal (a failed call's text is no longer one, round 48), and
  assignment takes only fingerprinted attempts.
- Memory-cap kills of calls before a wait are kept when the attempt then goes on (the stop path alone was tested).
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import live
from drsi.agent import AgentResult
from drsi.live import LiveRunner
from drsi.llm import ClaudeCLI, LLMLimited
from drsi.question import ROOT
from drsi.store import Campaign, make_node
from tests import test_round10 as r10
from tests.test_live import fixed_checker


class Base(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def run_one(self, llm, refuse, kills=None):
        """refuse: {(model, phase): count} refusals, each before the model ran; kills: {model: [kill, ...]}."""
        camp = self.campaign(llm)
        calls, waits, left = [], [], dict(refuse)

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            calls.append((phase, model))
            if left.get((model, phase)):
                left[(model, phase)] -= 1
                return AgentResult(ok=False, error=f"{model} refused", limited=True, limit_type="five_hour",
                                   mem_kills=list((kills or {}).get(model, [])))
            if phase == "propose":
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        with mock.patch.object(live, "wait_unless_stopping", side_effect=lambda s: waits.append(s) or True):
            out = r.run_batch([f"{ROOT}0"])
        return camp.tree.get(out[0]["id"]), calls, waits


class OtherModelTest(Base):
    def test_a_build_refused_on_the_fallback_goes_back_to_its_own_model(self):
        node, calls, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                          {("fable", "propose"): 1, ("opus", "implement"): 1})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(calls, [("propose", "fable"), ("propose", "opus"), ("implement", "opus"),
                                 ("implement", "fable")])
        self.assertEqual(waits, [])
        self.assertEqual(node["worker"]["model"], "fable")
        self.assertEqual(node["worker"]["fell_back_in"], "propose")

    def test_when_neither_can_run_both_are_tried_after_the_wait(self):
        node, calls, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                          {("fable", "propose"): 1, ("opus", "implement"): 2,
                                           ("fable", "implement"): 1})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(len(waits), 1)
        self.assertEqual(calls[2:], [("implement", "opus"), ("implement", "fable"), ("implement", "opus"),
                                     ("implement", "fable")])

    def test_a_stop_while_its_own_model_is_refused_again_names_no_refused_fallback(self):
        camp = self.campaign({"worker_models": ["fable"], "worker_fallback": "opus"})
        left = {("fable", "propose"): 1, ("opus", "implement"): 9, ("fable", "implement"): 9}

        def run(workspace, prompt, system, model=None):
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            if left.get((model, phase)):
                left[(model, phase)] -= 1
                return AgentResult(ok=False, error=f"{model} refused", limited=True, limit_type="five_hour")
            return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "", "self_reported_score": None,
                                                    "notes": ""})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        with mock.patch.object(live, "wait_unless_stopping", return_value=False):
            out = r.run_batch([f"{ROOT}0"])
        node = camp.tree.get(out[0]["id"])
        self.assertFalse(node["valid"])
        self.assertEqual(node["worker"]["model"], "opus")
        self.assertEqual(node["worker"]["fell_back_from"], "fable")
        self.assertNotIn("fallback_refused", node["worker"])  # the refused one was its own model, not a fallback

    def test_kills_before_a_wait_are_kept_when_the_attempt_goes_on(self):
        kill = {"pid": 7, "gb": 2.5, "total_gb": 2.5, "command": "python3"}
        node, calls, waits = self.run_one({"worker_models": ["opus"]}, {("opus", "propose"): 2},
                                          kills={"opus": [kill]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(node["worker"].get("mem_kills"), [kill, kill])


class SecondTryTest(unittest.TestCase):
    def test_a_refusal_on_the_second_try_is_a_limit(self):
        outs = [json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 500,
                            "num_turns": 1, "duration_api_ms": 0, "result": "API Error: 500"}),
                json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                            "num_turns": 1, "duration_api_ms": 0, "modelUsage": {},
                            "result": "You've hit your session limit"})]

        def runner(args, **kw):
            return subprocess.CompletedProcess(args, 1, stdout=outs.pop(0), stderr="")
        with self.assertRaises(LLMLimited):
            ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"})


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camp = Campaign.create("f", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(self.tmp.name))
        t = self.camp.tree
        t.add(make_node(id="iter0001-001", parent=None, source="live", proposal="", valid=False, score=None,
                        fail_class="agent_error", artifacts={"changed": []},
                        fingerprint={"outcome": "inconclusive", "killed_by": "agent_error"},
                        text={"worker_error": "You've hit your session limit"}))
        t.add(make_node(id="iter0001-002", parent=None, source="live", proposal="a checked idea whose build failed",
                        valid=False, score=None, fail_class="agent_error", artifacts={"changed": []},
                        fingerprint={"outcome": "inconclusive", "killed_by": "agent_error"},
                        text={"worker_error": "agent timed out after 5400s"}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_attempt_with_no_proposal_is_not_fingerprinted_and_a_failed_build_is(self):
        from drsi.fingerprint import fingerprint_nodes
        asked = []

        class LLM:
            def json(self, prompt, schema):
                asked.append(prompt)
                return {"items": []}
        for ids in (None, ["iter0001-001", "iter0001-002"]):
            asked.clear()
            fingerprint_nodes(self.camp.tree, LLM(), ids=ids, batch=20, workers=1)
            text = "\n".join(asked)
            self.assertNotIn("iter0001-001", text, ids)
            self.assertIn("iter0001-002", text, ids)

    def test_families_are_asked_only_about_fingerprinted_attempts(self):
        from drsi.families import assign_families
        asked = []

        class LLM:
            def json(self, prompt, schema):
                asked.append(prompt)
                return {"assignments": []}
        families = {"families": [{"id": "F01", "name": "one", "description": "d"}]}
        assign_families(self.camp.tree, families, LLM(), only_unassigned=True)
        self.assertNotIn("iter0001-001", "\n".join(asked))
        self.assertNotIn("iter0001-002", "\n".join(asked))


if __name__ == "__main__":
    unittest.main()
