"""Round 52: Claude Code replaced itself under a running call.

- Claude Code updates itself in place: npm replaces its package and the `claude` link. A worker call started in that
  window found no binary (2026-10-03 04:54 UTC: FileNotFoundError: 'claude'), and the attempt was
  recorded as a failure. A `claude` call that finds its binary missing now waits for it to be back (checking every
  2 s, for up to agent.CLAUDE_GONE_WAIT_S) and is started again, in the worker's and the developer's agent and in the
  judge's and classifier's CLI client alike. While npm writes the new binary, starting it can also fail as busy
  (ETXTBSY) or not yet executable (ENOEXEC): those are waited out the same way, and the call is tried again until it
  starts or the wait runs out. Anything else missing, such as the working directory, is the error it was; a binary that
  never comes back, or a run that begins stopping, ends the wait with the same error.
- Review (Opus): npm 11 renames the old link away, links a placeholder script (exec: ENOEXEC), then the new binary;
  the window lasted about 2 s, and the same version was installed again every 15 minutes or so. The wait now looks for
  the binary on the call's own PATH, its deadline is monotonic, and only args[0] as given counts. A worker whose
  `claude` never came back was recorded as the model's failure (agent_error); it is the loop's (orchestrator_error).
  A policy developer that could not be started ended the whole run with a traceback; it fails its revision.
- Review (Grok): between linking npm's placeholder and setting its mode a spawn fails with EACCES, which is waited
  out too; a stop that comes as the binary reappears ends the wait with the original error, not the stopping run's;
  a missing working directory whose path happens to be the binary's (CPython then names that path) is not waited on.
"""
import errno
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import agent, live
from drsi.agent import AgentResult, ClaudeAgent
from drsi.live import LiveRunner
from drsi.llm import ClaudeCLI
from drsi.question import ROOT
from tests import test_round10 as r10
from tests.test_live import fixed_checker

GONE = FileNotFoundError(2, "No such file or directory", "claude")
OK_STREAM = [{"type": "system", "subtype": "init"},
             {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": "done",
              "structured_output": {"proposal": "p"}}]


class Runner:
    """Raises `errors` in order, then writes the stream and exits 0."""
    def __init__(self, *errors):
        self.errors, self.calls = list(errors), []

    def __call__(self, args, stdout_path=None, **kw):
        self.calls.append(args)
        if self.errors:
            raise self.errors.pop(0)
        if stdout_path:
            with open(stdout_path, "a") as fh:
                fh.write("".join(json.dumps(e) + "\n" for e in OK_STREAM))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(OK_STREAM[-1]), stderr="")


class Base(unittest.TestCase):
    def setUp(self):
        self.which = [None, None, "/opt/homebrew/bin/claude", "/opt/homebrew/bin/claude"]
        self.waits = []
        for p in (mock.patch.object(agent.shutil, "which", side_effect=self.which_),
                  mock.patch.object(agent._STOPPING, "wait", side_effect=self.wait_)):
            p.start()
            self.addCleanup(p.stop)

    def which_(self, name, path=None):
        self.paths = getattr(self, "paths", []) + [path]
        return self.which.pop(0) if self.which else "/opt/homebrew/bin/claude"

    def wait_(self, secs):
        self.waits.append(secs)
        return False

    def run_agent(self, runner):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.jsonl"
            res = ClaudeAgent(model="opus", tools="Read", runner=runner).run(d, "p", transcript=t)
            return res, t.read_text().splitlines()


class AgentTest(Base):
    def test_a_call_started_while_claude_replaces_itself_waits_for_it_and_runs(self):
        runner = Runner(GONE)
        res, lines = self.run_agent(runner)
        self.assertTrue(res.ok, res.error)
        self.assertEqual(len(runner.calls), 2)
        self.assertGreaterEqual(len(self.waits), 2)  # checked until the binary was back, and back again 2 s on
        self.assertEqual(json.loads(lines[0])["type"], "drsi_call")  # the transcript keeps the call's own line
        self.assertEqual(sum(1 for l in lines if '"drsi_call"' in l), 1)

    def test_a_binary_still_being_written_is_waited_out_too(self):
        busy = OSError(errno.ETXTBSY, "Text file busy", "claude")
        half = OSError(errno.ENOEXEC, "Exec format error", "claude")
        runner = Runner(GONE, busy, half)
        res, lines = self.run_agent(runner)
        self.assertTrue(res.ok, res.error)
        self.assertEqual(len(runner.calls), 4)

    def test_a_binary_that_stays_broken_fails_once_the_wait_runs_out(self):
        runner = Runner(*[OSError(errno.ENOEXEC, "Exec format error", "claude") for _ in range(1000)])
        with mock.patch.object(agent, "CLAUDE_GONE_WAIT_S", 0), self.assertRaises(OSError):
            self.run_agent(runner)
        self.assertEqual(len(runner.calls), 1)

    def test_a_placeholder_not_yet_executable_is_waited_out_too(self):
        runner = Runner(PermissionError(errno.EACCES, "Permission denied", "claude"))
        res, lines = self.run_agent(runner)
        self.assertTrue(res.ok, res.error)
        self.assertEqual(len(runner.calls), 2)

    def test_a_stop_as_the_binary_reappears_ends_the_wait_with_the_original_error(self):
        runner = Runner(GONE)
        with mock.patch.object(agent._STOPPING, "is_set", return_value=True), \
                self.assertRaises(FileNotFoundError):
            agent.run_claude(runner, ["claude", "-p"])
        self.assertEqual(len(runner.calls), 1)

    def test_a_missing_working_directory_named_like_the_binary_is_not_waited_on(self):
        with tempfile.TemporaryDirectory() as d:
            where = os.path.join(d, "claude")  # neither the binary nor the directory exists
            runner = Runner(FileNotFoundError(2, "No such file or directory", where))
            with self.assertRaises(FileNotFoundError):
                agent.run_claude(runner, [where, "-p"], cwd=where)
        self.assertEqual((len(runner.calls), self.waits), (1, []))

    def test_a_missing_working_directory_is_not_waited_on(self):
        runner = Runner(FileNotFoundError(2, "No such file or directory", "/no/such/checkout"))
        with self.assertRaises(FileNotFoundError):
            self.run_agent(runner)
        self.assertEqual((len(runner.calls), self.waits), (1, []))

    def test_a_binary_that_never_comes_back_fails_as_before(self):
        self.which = [None] * 1000
        runner = Runner(GONE)
        with mock.patch.object(agent, "CLAUDE_GONE_WAIT_S", 0), self.assertRaises(FileNotFoundError):
            self.run_agent(runner)
        self.assertEqual(len(runner.calls), 1)

    def test_a_stop_ends_the_wait(self):
        runner = Runner(GONE)
        with mock.patch.object(agent._STOPPING, "wait", side_effect=lambda s: True), \
                self.assertRaises(FileNotFoundError):
            self.run_agent(runner)
        self.assertEqual(len(runner.calls), 1)


class PathTest(Base):
    def test_the_wait_looks_for_claude_on_the_calls_own_path(self):
        agent.run_claude(Runner(GONE), ["claude", "-p"], env={"PATH": "/only/here"})
        self.assertEqual(set(self.paths), {"/only/here"})


class RecordTest(unittest.TestCase):
    setUp = r10.WorkerModelsTest.setUp
    tearDown = r10.WorkerModelsTest.tearDown
    campaign = r10.WorkerModelsTest.campaign

    def test_a_worker_whose_claude_never_came_back_is_the_loops_failure_not_the_models(self):
        camp = self.campaign({"worker_models": ["opus"]})

        def run(workspace, prompt, system, model=None):
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            raise FileNotFoundError(2, "No such file or directory", "claude")
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        out = r.run_batch([f"{ROOT}0"])
        node = camp.tree.get(out[0]["id"])
        self.assertEqual(node.get("fail_class"), "orchestrator_error")
        self.assertIn("claude", node["text"]["orchestrator_error"])


class DreamTest(unittest.TestCase):
    def test_a_developer_that_cannot_be_started_is_a_failed_revision_not_a_crashed_run(self):
        from drsi import dream
        from tests.test_dream import DREAM_CFG, DreamTest as Base_
        case = Base_("test_unchanged_file_is_not_deployed")
        case.setUp()
        self.addCleanup(case.tearDown)

        def dev(sandbox, prompt):
            raise FileNotFoundError(2, "No such file or directory", "claude")
        rep = dream.run_dream(case.pdir, case.worlds, dev, DREAM_CFG, case.logs)
        self.assertEqual([r["stage"] for r in rep["revisions"]], ["agent"] * DREAM_CFG["dream"]["M"])
        self.assertIn("FileNotFoundError", rep["revisions"][0]["error"])


class RealExecTest(unittest.TestCase):
    def test_npms_sequence_missing_then_a_placeholder_then_the_binary(self):
        # npm 11 renames the old link away, links a placeholder script with no shebang (exec: ENOEXEC), then puts the
        # real binary in its place; CPython names args[0] in each error, an absolute path here
        pause = threading.Event()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(agent._STOPPING, "wait", side_effect=lambda s: pause.wait(0.05)):
            exe = Path(d) / "bin" / "claude"
            exe.parent.mkdir()

            def update():
                time.sleep(0.1)
                exe.write_text("placeholder, not a script\n")
                exe.chmod(0o755)
                time.sleep(0.5)
                new = exe.with_name(".claude-new")
                new.write_text("#!/bin/sh\ncat >/dev/null\n" + "".join(f"echo '{json.dumps(e)}'\n" for e in OK_STREAM))
                new.chmod(0o755)
                os.replace(new, exe)
            th = threading.Thread(target=update)
            th.start()
            res = ClaudeAgent(model="opus", tools="Read", binary=str(exe), timeout=30).run(
                d, "p", transcript=Path(d) / "w" / "t.jsonl")
            th.join()
        self.assertTrue(res.ok, res.error)


class CLITest(Base):
    def test_the_judges_call_waits_for_claude_as_well(self):
        runner = Runner(GONE)
        out = ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"})
        self.assertEqual(out, {"proposal": "p"})
        self.assertEqual(len(runner.calls), 2)


if __name__ == "__main__":
    unittest.main()
