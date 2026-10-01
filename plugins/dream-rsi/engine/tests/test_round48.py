"""Round 48: what a day on 0.4.4 showed.

- The account reached its five-hour usage limit, which binds every model: every worker call was refused within
  seconds, the fallback was refused as well, and the loop ran seven rounds in sixteen minutes, recording 169 attempts
  that did no work (each an agent_error child of a real attempt, its "proposal" the refusal's text) and freezing
  six empty worlds. A worker call refused for a limit of the account's own windows (five_hour, seven_day: the
  stream's rate_limit_event names it) now waits for the limit to reset, checking again at least every
  LIMIT_POLL_S (an account switch ends the wait sooner), and the attempt goes on; a stopping run ends the wait. A
  model's own limit (Fable's, say) is still the fallback's to cover, and fails the attempt as before when there is
  no fallback or it is refused for a limit of its own.
- An attempt whose call failed recorded the failure's text as its proposal, and the map listed it as one.
- `drsi prune --dry-run` reported "frozen worlds removed 0, rewritten 0" for rounds whose worlds held nothing but
  the attempts it would remove: the dry run returned before the world step. It now reports what the run would do.
- Workers gave the helper they started in the background the shell's timeout, and a background command is stopped
  at its timeout, and its run with it (checked with a real call); others gave the command a path on this machine,
  which the compute host does not have. The brief says both.
"""
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import live, offload
from drsi.agent import AgentResult, ClaudeAgent
from drsi.live import LiveRunner
from drsi.question import ROOT
from tests import test_round10 as r10
from tests.test_live import fixed_checker

RESET = 1790864400


def refusal(kind: str, resets: int = RESET) -> list[dict]:
    """A real refusal's stream (2026-10-01, the account's five-hour window), its limit's kind and reset as given."""
    return [{"type": "system", "subtype": "init"},
            {"type": "rate_limit_event", "rate_limit_info": {
                "status": "rejected", "resetsAt": resets, "rateLimitType": kind, "overageStatus": "rejected",
                "unifiedWindows": {"five_hour": {"utilization": 1.04, "resetsAt": resets}}}},
            {"type": "assistant", "error": "rate_limit", "message": {"model": "<synthetic>", "content": []}},
            {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429, "api_error": None,
             "num_turns": 1, "duration_api_ms": 0, "modelUsage": {},
             "result": "You've hit your session limit · resets 9:20am (UTC)"}]


class Runner:
    def __init__(self, events):
        self.text = "".join(json.dumps(e) + "\n" for e in events)

    def __call__(self, args, stdout_path=None, stderr_path=None, **kw):
        with open(stdout_path, "a") as fh:
            fh.write(self.text)
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")


class ResultTest(unittest.TestCase):
    def run_agent(self, events):
        with tempfile.TemporaryDirectory() as d:
            return ClaudeAgent(model="opus", tools="Read", runner=Runner(events)).run(
                d, "p", transcript=Path(d) / "t.jsonl")

    def test_a_refusal_names_its_limit_and_when_it_resets(self):
        res = self.run_agent(refusal("five_hour"))
        self.assertTrue(res.limited)
        self.assertEqual(res.limit_type, "five_hour")
        self.assertEqual(res.limit_reset, RESET)

    def test_a_call_that_was_not_refused_names_no_limit(self):
        ok = [{"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "rateLimitType": "five_hour",
                                                               "resetsAt": RESET}},
              {"type": "result", "subtype": "success", "is_error": False, "result": "{}", "structured_output": {},
               "num_turns": 3, "duration_api_ms": 900}]
        res = self.run_agent(ok)
        self.assertFalse(res.limited)
        self.assertIsNone(res.limit_type)
        self.assertIsNone(res.limit_reset)


class WaitTest(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def worker(self, calls, refusals):
        """refusals: {model: [kind, ...]}, one refusal of that kind per call of that model until the list runs out."""
        left = {m: list(k) for m, k in refusals.items()}

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            calls.append((phase, model))
            if left.get(model):
                kind = left[model].pop(0)
                return AgentResult(ok=False, error=f"{model} refused ({kind})", limited=True, limit_type=kind,
                                   limit_reset=int(time.time()) + 7200)
            if phase == "propose":
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        return run

    def run_one(self, llm, refusals, wait=lambda s: True):
        camp = self.campaign(llm)
        calls, waits, logs = [], [], []

        def waiter(secs):
            waits.append(secs)
            return wait(secs)
        r = LiveRunner(camp, self.worker(calls, refusals), indexer=lambda ids: None, round_id="iter0001",
                       checker=fixed_checker("novel"), log=logs.append)
        with mock.patch.object(live, "wait_unless_stopping", side_effect=waiter):
            out = r.run_batch([f"{ROOT}0"])
        return camp.tree.get(out[0]["id"]), calls, waits, logs

    def test_an_account_limit_is_waited_out_and_the_attempt_goes_on(self):
        node, calls, waits, logs = self.run_one({"worker_models": ["opus"]}, {"opus": ["five_hour", "five_hour"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(calls, [("propose", "opus")] * 3 + [("implement", "opus")])
        self.assertEqual(len(waits), 2)
        self.assertTrue(all(live.LIMIT_MIN_WAIT_S <= s <= live.LIMIT_POLL_S for s in waits), waits)
        self.assertTrue(any("waiting until" in m and "UTC" in m for m in logs), logs)
        self.assertGreater(node["worker"].get("waited_for_limit_s", 0), 0)

    def test_when_the_fallback_meets_the_account_limit_too_the_attempt_waits_then_tries_its_own_model_first(self):
        node, calls, waits, _ = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                             {"fable": ["five_hour"], "opus": ["five_hour"]})
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(calls[:3], [("propose", "fable"), ("propose", "opus"), ("propose", "fable")])
        self.assertEqual(len(waits), 1)
        self.assertEqual(node["worker"]["model"], "fable")
        self.assertNotIn("fallback_refused", node["worker"])

    def test_a_models_own_limit_without_a_fallback_fails_as_before(self):
        node, calls, waits, _ = self.run_one({"worker_models": ["fable"]}, {"fable": ["seven_day_overage_included"]})
        self.assertFalse(node["valid"])
        self.assertEqual(waits, [])
        self.assertEqual(calls, [("propose", "fable")])

    def test_a_fallback_refused_for_its_own_limit_fails_as_before(self):
        node, calls, waits, _ = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                             {"fable": ["seven_day_overage_included"], "opus": ["seven_day_opus"]})
        self.assertFalse(node["valid"])
        self.assertEqual(waits, [])
        self.assertEqual(node["worker"].get("fallback_refused"), "opus")

    def test_a_stopping_run_ends_the_wait(self):
        node, calls, waits, _ = self.run_one({"worker_models": ["opus"]}, {"opus": ["five_hour"] * 5},
                                             wait=lambda s: False)
        self.assertEqual(len(waits), 1)
        self.assertEqual(calls, [("propose", "opus")])
        self.assertFalse(node["valid"])

    def test_the_wait_is_until_the_reset_and_never_longer_than_a_poll(self):
        now = 1_000_000.0
        with mock.patch.object(live.time, "time", return_value=now):
            self.assertEqual(live._limit_wait(AgentResult(ok=False, limited=True, limit_reset=int(now) + 90)), 95)
            self.assertEqual(live._limit_wait(AgentResult(ok=False, limited=True, limit_reset=int(now) + 99999)),
                             live.LIMIT_POLL_S)
            self.assertEqual(live._limit_wait(AgentResult(ok=False, limited=True, limit_reset=int(now) - 50)),
                             live.LIMIT_MIN_WAIT_S)
            self.assertEqual(live._limit_wait(AgentResult(ok=False, limited=True)), live.LIMIT_POLL_S)


class ProposalTest(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def test_a_failed_calls_text_is_not_recorded_as_the_proposal(self):
        def run(workspace, prompt, system, model=None):
            return AgentResult(ok=False, error="boom", result_text="You've hit your session limit")
        camp = self.campaign({"worker_models": ["opus"]})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        out = r.run_batch([f"{ROOT}0"])
        node = camp.tree.get(out[0]["id"])
        self.assertNotIn("session limit", node.get("proposal") or "")


class BriefTest(unittest.TestCase):
    def test_the_brief_says_a_background_helper_takes_no_timeout_and_paths_stay_in_the_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            from drsi.store import Campaign
            camp = Campaign.create("b", {"workspace": {"repo": "/x", "mutable": ["a"]},
                                         "live": {"offload": {"cmd": "/bin/echo"}}}, home=Path(d))
            text = "\n".join(offload.brief(camp))
        self.assertIn("no timeout", text)
        self.assertIn("stopped at its timeout", text)
        self.assertIn("does not exist there", text)



class PruneDryRunTest(unittest.TestCase):
    """The dry run reports the worlds the run removes and rewrites (round 18's fixture)."""

    def test_a_dry_run_reports_the_worlds_the_run_changes(self):
        from drsi.prune import prune
        from tests.test_round18 import PruneTest
        case = PruneTest("test_a_dry_run_changes_nothing")
        case.setUp()
        self.addCleanup(case.tearDown)
        case.fixture()
        dry = prune(case.camp, error_match="account unavailable", dry_run=True, log=lambda m: None)
        real = prune(case.camp, error_match="account unavailable", log=lambda m: None)
        self.assertTrue(real["worlds_removed"] or real["worlds_rewritten"])
        self.assertEqual(sorted(dry["worlds_removed"]), sorted(real["worlds_removed"]))
        self.assertEqual(sorted(dry["worlds_rewritten"]), sorted(real["worlds_rewritten"]))


if __name__ == "__main__":
    unittest.main()
