"""Round 51: what two days on 0.4.5 showed.

- The account's usage limit was crossed in the middle of seven worker calls (2026-10-02, 09:25 and 18:36 UTC): each had
  worked for some turns when the stream's last usage event read rejected and the call ended with an error result of
  status 429. 0.4.5 waits only for a call refused before the model ran, so these seven checked proposals were recorded
  as failed builds. A call cut off by a usage limit is now waited out like a refusal (its kind and reset time read the
  same way), and before the call is made again what it did is undone: a build's checkout is made afresh at the
  attempt's start, a proposal's checkout too and its proposal file cleared, the dream's sandbox reset to the policy it
  was given. The novelty check's calls are single prompts, so a limit there, before or after work, is waited out the
  same way. The record counts the cut-off calls (worker.cut_off_by_limit).
- A worker started its offload helper with a shell `&` inside a background command; the `&` cut it loose from the
  worker's session. The brief says to start it with the background parameter and no `&` of its own.
- Review (Opus): the cut-off read ignored the result, so a call that failed otherwise after a rejected usage event was
  made again until the window reset; a cut-off needs an error result of status 429 that is not a short-term
  rate_limit_error, and either the rejected event or Claude Code's synthetic rate_limit message. A short-term
  rate_limit_error is again an ordinary error to the novelty check (tried again at once), not a 10-minute wait. A stop
  during the wait committed the cut-off build's half work; the reset now runs first and the record says the attempt
  was stopped in a limit wait. The reset also clears what the cut-off call left in the attempt's proposal directory.
- The scorer's summary was kept to 800 characters, which lost the per-margin lines a reading turns on. It is kept to
  4000.
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import live, offload
from drsi.agent import AgentResult, ClaudeAgent
from drsi.live import LiveRunner
from drsi.llm import ClaudeCLI, LLMLimited
from drsi.question import ROOT
from tests import test_round10 as r10
from tests.test_live import fixed_checker

RESET = 1790971800


def cut_off_stream(turns: int = 9) -> list[dict]:
    """A real cut-off's stream (2026-10-02, a worker's build): work, then the limit, then a 429 result."""
    return [{"type": "system", "subtype": "init"},
            {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "rateLimitType": "five_hour",
                                                             "resetsAt": RESET}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}},
            {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour",
                                                             "resetsAt": RESET}},
            {"type": "assistant", "error": "rate_limit", "message": {"model": "<synthetic>", "content": []}},
            {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429, "num_turns": turns,
             "duration_api_ms": 179839, "modelUsage": {"claude-opus-5-5": {"inputTokens": 10}},
             "result": "You've hit your session limit · resets 3:10pm (UTC)"}]


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

    def test_a_call_cut_off_by_the_limit_says_so_with_its_kind_and_reset(self):
        res = self.run_agent(cut_off_stream())
        self.assertFalse(res.ok)
        self.assertTrue(res.cut_off)
        self.assertFalse(res.limited)  # it did work: not a refusal before the model ran
        self.assertEqual((res.limit_type, res.limit_reset), ("five_hour", RESET))

    def test_a_call_that_failed_otherwise_after_work_is_not_cut_off(self):
        ev = cut_off_stream()
        ev[3]["rate_limit_info"]["status"] = "allowed"
        ev[-1].update(api_error_status=500, result="API Error: 500")
        del ev[4]
        res = self.run_agent(ev)
        self.assertFalse(res.cut_off)
        self.assertFalse(res.limited)


class ShapeTest(ResultTest):
    def test_a_rejected_event_before_another_failure_is_not_a_cut_off(self):
        for result in ({"api_error_status": 500, "result": "API Error: 500"},
                       {"subtype": "error_max_turns", "is_error": True, "api_error_status": None},
                       {"api_error_status": 400, "result": "Prompt is too long"}):
            ev = cut_off_stream()
            del ev[4]  # no synthetic rate_limit message: the call ended for another reason
            ev[-1] = dict(ev[-1], **result)
            self.assertFalse(self.run_agent(ev).cut_off, result)

    def test_a_short_term_rate_limit_is_not_a_cut_off(self):
        ev = cut_off_stream()
        ev[-1] = dict(ev[-1], api_error="rate_limit_error")
        self.assertFalse(self.run_agent(ev).cut_off)

    def test_a_cut_off_without_the_rejected_event_is_still_one(self):
        ev = cut_off_stream()
        del ev[3]  # the synthetic rate_limit message and the 429 result remain
        res = self.run_agent(ev)
        self.assertTrue(res.cut_off)


class Base(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def run_one(self, llm, cut, seen):
        """cut: {(model, phase): count} calls cut off after work; each such call leaves a file in the checkout (and, in
        the propose phase, a stale proposal file). seen collects (phase, model, leftovers present at the call's start)."""
        camp = self.campaign(llm)
        left, waits = dict(cut), []

        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            pfile = camp.root / "work" / "_proposals" / workspace.name / "proposal.txt"
            seen.append((phase, model, workspace.joinpath("partial.txt").exists(), pfile.exists()))
            if left.get((model, phase)):
                left[(model, phase)] -= 1
                workspace.joinpath("partial.txt").write_text("half a build\n")
                if phase == "propose":
                    pfile.parent.mkdir(parents=True, exist_ok=True)
                    pfile.write_text("a stale proposal from a call the limit cut off\n")
                return AgentResult(ok=False, error="cut off", cut_off=True, limit_type="five_hour",
                                   limit_reset=None)
            if phase == "propose":
                return AgentResult(ok=True, structured={"proposal": "the real idea", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        with mock.patch.object(live, "wait_unless_stopping", side_effect=lambda s: waits.append(s) or True):
            out = r.run_batch([f"{ROOT}0"])
        return camp.tree.get(out[0]["id"]), waits


class CutOffTest(Base):
    def test_a_build_cut_off_waits_then_is_made_again_on_a_fresh_checkout(self):
        seen = []
        node, waits = self.run_one({"worker_models": ["opus"]}, {("opus", "implement"): 1}, seen)
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(len(waits), 1)
        builds = [s for s in seen if s[0] == "implement"]
        self.assertEqual(len(builds), 2)
        self.assertFalse(builds[1][2], "the cut-off call's half build survived into the call made again")
        self.assertEqual(node["worker"].get("cut_off_by_limit"), 1)
        self.assertGreater(node["worker"].get("waited_for_limit_s", 0), 0)

    def test_a_proposal_cut_off_is_made_again_with_no_stale_proposal_or_leftovers(self):
        seen = []
        node, waits = self.run_one({"worker_models": ["opus"]}, {("opus", "propose"): 1}, seen)
        self.assertTrue(node["valid"], node.get("fail_class"))
        proposals = [s for s in seen if s[0] == "propose"]
        self.assertEqual(len(proposals), 2)
        self.assertFalse(proposals[1][2], "the cut-off proposal call's files survived")
        self.assertFalse(proposals[1][3], "the cut-off call's proposal file could be judged")
        self.assertNotIn("stale", node.get("proposal") or "")

    def test_a_build_cut_off_goes_to_the_fallback_on_a_fresh_checkout(self):
        seen = []
        node, waits = self.run_one({"worker_models": ["fable"], "worker_fallback": "opus"},
                                   {("fable", "implement"): 1}, seen)
        self.assertTrue(node["valid"], node.get("fail_class"))
        self.assertEqual(waits, [])
        self.assertEqual([s[:2] for s in seen if s[0] == "implement"], [("implement", "fable"), ("implement", "opus")])
        self.assertFalse([s for s in seen if s[0] == "implement"][1][2])

    def test_a_cut_off_calls_notes_in_the_proposal_directory_are_cleared(self):
        camp = self.campaign({"worker_models": ["opus"]})
        state, seen = {"cut": True}, []

        def run(workspace, prompt, system, model=None):
            pdir = camp.root / "work" / "_proposals" / workspace.name
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            seen.append((pdir / "scratch.md").exists())
            if state.pop("cut", False):
                pdir.mkdir(parents=True, exist_ok=True)
                (pdir / "scratch.md").write_text("notes from a build the limit cut off\n")
                return AgentResult(ok=False, error="cut off", cut_off=True, limit_type="five_hour")
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        with mock.patch.object(live, "wait_unless_stopping", return_value=True):
            out = r.run_batch([f"{ROOT}0"])
        self.assertTrue(camp.tree.get(out[0]["id"])["valid"])
        self.assertEqual(seen, [False, False])

    def test_a_stop_during_the_wait_records_the_cut_off_call(self):
        camp = self.campaign({"worker_models": ["opus"]})

        def run(workspace, prompt, system, model=None):
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            workspace.joinpath("value.txt").write_text("half a build\n")
            return AgentResult(ok=False, error="cut off", cut_off=True, limit_type="five_hour")
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        with mock.patch.object(live, "wait_unless_stopping", return_value=False):
            out = r.run_batch([f"{ROOT}0"])
        node = camp.tree.get(out[0]["id"])
        self.assertFalse(node["valid"])
        self.assertEqual(node["worker"].get("cut_off_by_limit"), 1)
        self.assertTrue(node["worker"].get("stopped_in_limit_wait"))
        self.assertEqual(node["artifacts"].get("changed"), [], "the cut-off build's half work was committed")


class CheckTest(unittest.TestCase):
    def test_a_check_cut_off_after_work_is_a_limit(self):
        out = json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                          "num_turns": 4, "duration_api_ms": 51000, "modelUsage": {"opus": {"inputTokens": 3}},
                          "result": "You've hit your session limit"})

        def runner(args, **kw):
            return subprocess.CompletedProcess(args, 1, stdout=out, stderr="")
        with self.assertRaises(LLMLimited):
            ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"})

    def test_a_short_term_rate_limit_is_an_ordinary_error_tried_again_at_once(self):
        outs = [json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                            "api_error": "rate_limit_error", "num_turns": 2, "duration_api_ms": 900,
                            "result": "API Error: 429 rate_limit_error"}),
                json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "{}",
                            "structured_output": {"ok": 1}})]

        def runner(args, **kw):
            o = outs.pop(0)
            return subprocess.CompletedProcess(args, 1 if "is_error\": true" in o else 0, stdout=o, stderr="")
        self.assertEqual(ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"}), {"ok": 1})


class DreamTest(unittest.TestCase):
    def test_a_developer_cut_off_is_made_again_in_a_reset_sandbox(self):
        from drsi import dream
        from tests.test_dream import DREAM_CFG, DreamTest as Base_, Dev
        case = Base_("test_unchanged_file_is_not_deployed")
        case.setUp()
        self.addCleanup(case.tearDown)
        inner, n, seen = Dev(), {"calls": 0}, []

        def dev(sandbox, prompt):
            n["calls"] += 1
            seen.append(((sandbox / "junk.txt").exists(), (sandbox / "method.py").read_text()))
            if n["calls"] == 1:
                (sandbox / "junk.txt").write_text("half a revision\n")
                (sandbox / "method.py").write_text("# half-edited\n")
                return AgentResult(ok=False, error="cut off", cut_off=True, limit_type="five_hour")
            return inner(sandbox, prompt)
        with mock.patch.object(dream, "wait_unless_stopping", return_value=True):
            dream.run_dream(case.pdir, case.worlds, dev, DREAM_CFG, case.logs)
        self.assertGreaterEqual(len(seen), 2)
        self.assertFalse(seen[1][0], "the cut-off call's file survived into the call made again")
        self.assertEqual(seen[1][1], seen[0][1], "the call made again did not start from the policy it was given")


class BriefTest(unittest.TestCase):
    def test_the_brief_says_no_shell_ampersand(self):
        with tempfile.TemporaryDirectory() as d:
            from drsi.store import Campaign
            camp = Campaign.create("b", {"workspace": {"repo": "/x", "mutable": ["a"]},
                                         "live": {"offload": {"cmd": "/bin/echo"}}}, home=Path(d))
            text = "\n".join(offload.brief(camp))
        self.assertIn("no `&`", text)


class SummaryCapTest(Base):
    def test_a_long_scorer_summary_is_kept_by_the_loop_and_by_rescore(self):
        from drsi import rescore as rescore_mod
        long = "score 0.5 after a long run. " + "DETAILS " * 350  # about 3,000 characters
        real = live.run_scorer

        def scorer(*a, **k):
            sc = real(*a, **k)
            sc["summary"] = long
            return sc
        camp = self.campaign({"worker_models": ["opus"]})

        def run(workspace, prompt, system, model=None):
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        with mock.patch.object(live, "run_scorer", side_effect=scorer):
            out = r.run_batch([f"{ROOT}0"])
        nid = out[0]["id"]
        self.assertEqual(camp.tree.get(nid)["artifacts"]["scorer_summary"], long)
        with mock.patch.object(rescore_mod, "run_scorer", side_effect=scorer):
            rescore_mod.rescore(camp, ids={nid})
        self.assertEqual(camp.tree.get(nid)["artifacts"]["scorer_summary"], long)

if __name__ == "__main__":
    unittest.main()
