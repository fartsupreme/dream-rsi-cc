"""Round 49: the review of round 48 (Opus), each finding reproduced first.

- Claude Code 2.1.286 names six kinds of limit (five_hour, seven_day, seven_day_opus, seven_day_sonnet,
  seven_day_overage_included, overage; the kind is optional). `overage`, the account's usage-credit cap, binds every
  model as five_hour and seven_day do, and a refusal may name no kind: both still failed attempts within seconds.
  The kind cannot say which models a refusal binds anyway (Grok, from the binary: Claude Code reports the exceeded
  window that resets last, so a five-hour block under a longer Opus weekly cap arrives as seven_day_opus), and
  round 48 judged it on the fallback's refusal alone: its own model's five-hour refusal with the fallback refused for
  a limit of its own was recorded as a failure. A refusal before any work is now never a recorded failure: with
  llm.worker_fallback set it runs there, and when no model the attempt may use can run (no fallback, the same model,
  or the fallback refused as well) the attempt waits for the earliest reset among the refusals, whatever their kind.
- The dream's policy developer ignored a refusal, so a dream in a limit lost its revisions and logged "kept the
  incumbent"; a refused developer call now waits and is made again.
- The novelty check's calls had no limit handling: a refused check recorded the attempt as an orchestration
  failure, and its confirmation step let a duplicate verdict stand on a refused call. A check refused for a usage
  limit (ClaudeCLI recognises the refusal, exit status 1 with an error result of status 429 and no work, and does
  not retry it at once) now waits and checks again, and the checker's own fallbacks let that refusal through.
- The indexer fingerprinted attempts that did no work, so a refusal's text could become a mechanism on the map
  (round 50 moved the rule into fingerprint_nodes).
"""
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import live, novelty
from drsi.agent import AgentResult
from drsi.live import LiveRunner
from drsi.llm import ClaudeCLI, LLMError, LLMLimited
from drsi.question import ROOT
from tests import test_round10 as r10
from tests.test_live import fixed_checker

REFUSED_JSON = json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                           "num_turns": 1, "duration_api_ms": 0, "modelUsage": {},
                           "result": "You've hit your session limit · resets 9:20am (UTC)"})


class Base(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def worker(self, calls, refusals, resets=None):
        """refusals: {model: [kind, ...]}; each call of that model takes the next kind (None: no kind named)."""
        left = {m: list(k) for m, k in refusals.items()}

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            calls.append((phase, model))
            if left.get(model):
                kind = left[model].pop(0)
                return AgentResult(ok=False, error=f"{model} refused", limited=True, limit_type=kind,
                                   limit_reset=(resets or {}).get(model))
            if phase == "propose":
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        return run

    def run_one(self, llm, refusals, wait=lambda s: True, resets=None, checker=None):
        camp = self.campaign(llm)
        calls, waits = [], []

        def waiter(secs):
            waits.append(secs)
            return wait(secs)
        r = LiveRunner(camp, self.worker(calls, refusals, resets), indexer=lambda ids: None, round_id="iter0001",
                       checker=checker or fixed_checker("novel"))
        with mock.patch.object(live, "wait_unless_stopping", side_effect=waiter):
            out = r.run_batch([f"{ROOT}0"])
        return camp.tree.get(out[0]["id"]), calls, waits


class KindsTest(Base):
    def test_the_usage_credit_cap_is_waited_out(self):
        node, calls, waits = self.run_one({"worker_models": ["opus"]}, {"opus": ["overage"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(len(waits), 1)

    def test_a_refusal_naming_no_kind_is_waited_out(self):
        node, calls, waits = self.run_one({"worker_models": ["opus"]}, {"opus": [None]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(len(waits), 1)

    def test_an_account_refusal_with_the_fallback_refused_for_its_own_limit_is_waited_out(self):
        node, calls, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                          {"fable": ["five_hour"], "opus": ["seven_day_opus"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(calls[:3], [("propose", "fable"), ("propose", "opus"), ("propose", "fable")])
        self.assertEqual(len(waits), 1)
        self.assertNotIn("fallback_refused", node["worker"])

    def test_a_models_own_limit_without_a_fallback_is_waited_out(self):
        node, calls, waits = self.run_one({"worker_models": ["fable"]}, {"fable": ["seven_day_overage_included"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(len(waits), 1)

    def test_a_models_own_limit_with_the_fallback_the_same_model_is_waited_out(self):
        node, calls, waits = self.run_one({"worker_models": ["opus"], "worker_fallback": "opus"},
                                          {"opus": ["seven_day_opus"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(len(waits), 1)

    def test_a_models_own_limit_the_fallback_cannot_cover_is_waited_out(self):
        node, calls, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                          {"fable": ["seven_day_overage_included"], "opus": ["seven_day_opus"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(calls[:3], [("propose", "fable"), ("propose", "opus"), ("propose", "fable")])
        self.assertEqual(len(waits), 1)

    def test_a_stop_while_both_are_refused_names_the_refused_fallback(self):
        node, calls, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                          {"fable": ["seven_day_overage_included"], "opus": ["seven_day_opus"]},
                                          wait=lambda s: False)
        self.assertFalse(node["valid"])
        self.assertEqual(node["worker"]["model"], "fable")
        self.assertEqual(node["worker"].get("fallback_refused"), "opus")

    def test_the_wait_is_until_the_earliest_reset(self):
        now = time.time()
        node, calls, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                          {"fable": ["seven_day_overage_included"], "opus": ["seven_day_opus"]},
                                          resets={"fable": int(now) + 5 * 86400, "opus": int(now) + 100})
        self.assertEqual(len(waits), 1)
        self.assertLess(waits[0], 200)


class CheckTest(Base):
    def test_the_cli_recognises_a_refusal_and_does_not_retry_it_at_once(self):
        calls = []

        def runner(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, stdout=REFUSED_JSON, stderr="")
        with self.assertRaises(LLMLimited):
            ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"})
        self.assertEqual(len(calls), 1)

    def test_other_failures_are_not_a_limit(self):
        def runner(args, **kw):
            return subprocess.CompletedProcess(args, 1, stdout=json.dumps(
                {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 500, "num_turns": 1,
                 "duration_api_ms": 0, "result": "API Error: 500"}), stderr="")
        with self.assertRaises(LLMError) as cm:
            ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"})
        self.assertNotIsInstance(cm.exception, LLMLimited)

    def test_a_check_refused_for_a_limit_waits_and_checks_again(self):
        state = {"n": 0}
        good = fixed_checker("novel")

        def checker(proposal, node):
            state["n"] += 1
            if state["n"] == 1:
                raise LLMLimited("claude -p refused for a usage limit")
            return good(proposal, node)
        node, calls, waits = self.run_one({"worker_models": ["opus"]}, {}, checker=checker)
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(state["n"], 2)
        self.assertEqual(len(waits), 1)

    def test_the_checkers_own_fallbacks_let_a_limit_through(self):
        class Refusing:
            def json(self, prompt, schema):
                raise LLMLimited("refused")
        with self.assertRaises(LLMLimited):
            novelty._query_fingerprint(Refusing(), "an idea", "goal")
        with self.assertRaises(LLMLimited):
            novelty._confirm(Refusing(), "goal", "an idea", [], [], [])


class DreamTest(unittest.TestCase):
    def test_a_refused_developer_call_waits_and_is_made_again(self):
        from drsi import dream
        from tests.test_dream import DREAM_CFG, DreamTest as Base_, Dev
        case = Base_("test_unchanged_file_is_not_deployed")
        case.setUp()
        self.addCleanup(case.tearDown)
        inner, n = Dev(), {"calls": 0}

        def dev(sandbox, prompt):
            n["calls"] += 1
            if n["calls"] == 1:
                return AgentResult(ok=False, error="refused", limited=True, limit_type="five_hour")
            return inner(sandbox, prompt)
        waits = []
        with mock.patch.object(dream, "wait_unless_stopping", side_effect=lambda s: waits.append(s) or True):
            rep = dream.run_dream(case.pdir, case.worlds, dev, DREAM_CFG, case.logs)
        self.assertEqual([r["stage"] for r in rep["revisions"]], ["unchanged", "unchanged"])
        self.assertEqual(len(waits), 1)


# The indexer's skip of attempts with no work moved into fingerprint_nodes, as a skip of attempts with no proposal
# (round 50, whose FingerprintTest covers it).


if __name__ == "__main__":
    unittest.main()
