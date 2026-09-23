"""Scorer contract, process cleanup and isolation of headless Claude runs."""
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from drsi import cli
from drsi.agent import ClaudeAgent, run_group
from drsi.llm import ClaudeCLI
from drsi.scorer import run_scorer
from drsi.store import Campaign


class ScorerContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_crashed_scorer_is_an_error_even_after_printing_json(self):
        r = run_scorer("echo '{\"score\": 9, \"valid\": true}'; exit 1", self.ws, timeout=10)
        self.assertEqual(r["fail_class"], "eval_error")

    def test_only_the_last_line_counts(self):
        r = run_scorer("echo '{\"score\": 9, \"valid\": true}'; echo 'grader crashed after this'", self.ws, timeout=10)
        self.assertEqual(r["fail_class"], "eval_error")

    def test_non_finite_and_boolean_scores_rejected(self):
        for raw in ("NaN", "Infinity", "true"):
            r = run_scorer(f"echo '{{\"score\": {raw}, \"valid\": true}}'", self.ws, timeout=10)
            self.assertEqual(r["fail_class"], "eval_error", raw)

    def test_background_child_does_not_turn_a_result_into_a_timeout(self):
        t0 = time.time()
        r = run_scorer("(sleep 7773 &) ; echo '{\"score\": 1, \"valid\": true}'", self.ws, timeout=5)
        self.assertTrue(r["valid"], r)
        self.assertLess(time.time() - t0, 5)
        time.sleep(0.2)
        self.assertNotEqual(subprocess.run(["pgrep", "-f", "sleep 7773"]).returncode, 0)


class ProcessGroupTest(unittest.TestCase):
    def test_timeout_kills_grandchildren(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_group(["bash", "-c", "sleep 7772 & sleep 7772"], input="", capture_output=True, text=True,
                      timeout=1, cwd=None)
        time.sleep(0.2)
        self.assertNotEqual(subprocess.run(["pgrep", "-f", "sleep 7772"]).returncode, 0)


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("DRSI_HOME")
        os.environ["DRSI_HOME"] = self.tmp.name

    def tearDown(self):
        if self.old is None:
            os.environ.pop("DRSI_HOME", None)
        else:
            os.environ["DRSI_HOME"] = self.old
        self.tmp.cleanup()

    def test_judge_and_classifier_run_from_a_neutral_directory(self):
        seen = {}

        def runner(args, input=None, capture_output=None, text=None, timeout=None, cwd=None, env=None):
            seen["cwd"] = cwd
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(
                {"type": "result", "subtype": "success", "is_error": False, "structured_output": {"x": 1}}), stderr="")
        ClaudeCLI(model="opus", runner=runner).json("p", {"type": "object"})
        self.assertEqual(Path(seen["cwd"]), Path(self.tmp.name) / "_neutral")
        self.assertTrue(Path(seen["cwd"]).is_dir())

    def test_worker_is_sandboxed_to_its_workspace_with_hooks_off(self):
        camp = Campaign.create("w", {"workspace": {"repo": "/x", "mutable": ["a"]}})
        ws = camp.root / "work" / "iter0001-001"
        agent = cli.worker_agent(camp, ws, "system text")
        args = agent.build_args()
        settings = json.loads(args[args.index("--settings") + 1])
        sb = settings["sandbox"]
        self.assertTrue(sb["enabled"])
        self.assertFalse(sb["allowUnsandboxedCommands"])
        self.assertEqual(sorted(sb["filesystem"]["allowWrite"]),
                         sorted([str(ws), str(camp.root / "work" / "_proposals" / "iter0001-001")]))
        self.assertNotIn("excludedCommands", sb)  # workers never run drsi; nothing leaves the sandbox
        self.assertTrue(settings["disableAllHooks"])
        self.assertEqual(args[args.index("--setting-sources") + 1], "local")

    def test_policy_developer_is_confined_to_its_sandbox(self):
        camp = Campaign.create("d", {})
        args = cli.developer_agent(camp).build_args()
        self.assertIn("--restricted", args)
        self.assertEqual(args[args.index("--tools") + 1], "Read,Edit,Write")


class ConfigSafetyTest(unittest.TestCase):
    def test_mutable_string_is_coerced_and_default_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("m", {"workspace": {"mutable": "src/**"}}, home=Path(d))
            self.assertEqual(camp.config["workspace"]["mutable"], ["src/**"])
            other = Campaign.create("n", {}, home=Path(d))
            self.assertEqual(other.config["workspace"]["mutable"], [])


if __name__ == "__main__":
    unittest.main()
