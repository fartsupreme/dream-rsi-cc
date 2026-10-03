"""Round 52: Claude Code replaced itself under a running call.

- Claude Code updates itself in place: npm replaces its package and the `claude` link. A worker call started in that
  window found no binary (2026-10-03 04:54 UTC: FileNotFoundError: 'claude'), and the attempt was
  recorded as a failure. A `claude` call that finds its binary missing now waits for it to be back (checking every
  2 s, for up to agent.CLAUDE_GONE_WAIT_S) and is started again, in the worker's and the developer's agent and in the
  judge's and classifier's CLI client alike. While npm writes the new binary, starting it can also fail as busy
  (ETXTBSY) or not yet executable (ENOEXEC): those are waited out the same way, and the call is tried again until it
  starts or the wait runs out. Anything else missing, such as the working directory, is the error it was; a binary that
  never comes back, or a run that begins stopping, ends the wait with the same error.
"""
import errno
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import agent
from drsi.agent import ClaudeAgent
from drsi.llm import ClaudeCLI

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

    def which_(self, name):
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


class CLITest(Base):
    def test_the_judges_call_waits_for_claude_as_well(self):
        runner = Runner(GONE)
        out = ClaudeCLI(model="opus", runner=runner, cwd="/tmp").json("p", {"type": "object"})
        self.assertEqual(out, {"proposal": "p"})
        self.assertEqual(len(runner.calls), 2)


if __name__ == "__main__":
    unittest.main()
