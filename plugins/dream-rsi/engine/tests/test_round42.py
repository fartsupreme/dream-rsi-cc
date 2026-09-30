"""Round 42: what the first live round on 0.4.2 showed.

- Claude Code's auto memory was on in every call the engine makes. A worker's session loaded the MEMORY.md of its
  checkout's repository at startup and could write notes there (the memory folder is outside the Bash sandbox, and
  Claude Code lets its file tools write it): in one campaign 259 notes had built up, including an index of limits no
  design was said to escape, and every worker started with them. That is a channel between workers the orchestrator
  neither sees nor checks, and a write outside the worker's sandbox. Every call now runs with auto memory off, by
  setting where the call takes settings and by environment everywhere, and a worker's file tools are denied
  ~/.claude (a session may write its own memory folder even with auto memory off).
- A worker started a run on the compute host in the background, could not wait for it (a polling command it wrote
  needed approval a headless session cannot give), ended its session, and the run was stopped half done. The helper
  now attaches to a running request, `offload.py --wait ID`, returning when it ends or after --for seconds (default
  540, inside the shell's 10-minute limit), and the brief says a session has no later turn: stopping ends it.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from drsi import cli, offload
from drsi.agent import ClaudeAgent
from drsi.llm import ClaudeCLI
from tests.test_round40 import Base, FAKE as FAKE40

CLIENT = Path(offload.__file__)
OFF = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
RESULT = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "{}",
                     "structured_output": {}})


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append((args, kw))
        return subprocess.CompletedProcess(args, 0, stdout=RESULT, stderr="")


class MemoryTest(unittest.TestCase):
    def test_every_agent_call_runs_with_auto_memory_off(self):
        for env in (None, {"X": "1"}):
            r = Recorder()
            ClaudeAgent(model="opus", tools="Read", runner=r, env=env).run("/tmp", "p")
            self.assertEqual(r.calls[0][1]["env"].get(OFF), "1", env)
            if env:
                self.assertEqual(r.calls[0][1]["env"]["X"], "1")

    def test_the_worker_settings_turn_auto_memory_off(self):
        import tempfile
        from drsi.store import Campaign
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("m", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(d))
            args = cli.worker_agent(camp, camp.root / "work" / "iter0001-001", "s").build_args()
            settings = json.loads(args[args.index("--settings") + 1])
        self.assertIs(settings.get("autoMemoryEnabled"), False)
        # Claude Code lets a session write its own project memory folder even with auto memory off (checked with a
        # real worker): the file tools are denied everything under ~/.claude
        self.assertIn("Edit(~/.claude/**)", settings["permissions"]["deny"])
        self.assertIn("Write(~/.claude/**)", settings["permissions"]["deny"])

    def test_the_developer_and_the_judge_run_with_auto_memory_off(self):
        import tempfile
        from drsi.store import Campaign
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("m", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(d))
            dev = cli.developer_agent(camp)
            r = Recorder()
            dev.runner = r
            dev.run(d, "p")
            self.assertEqual(r.calls[0][1]["env"].get(OFF), "1")
        r = Recorder()
        ClaudeCLI(model="opus", runner=r, cwd="/tmp").json("p", {"type": "object"})
        self.assertEqual(r.calls[0][1]["env"].get(OFF), "1")


FAKE = FAKE40.replace('case "$1" in', 'case "$1" in\n  slow) echo "part one"; sleep "$2"; echo "part two" ;;')


class WaitTest(Base):
    def setUp(self):
        super().setUp()
        self.fake.write_text(FAKE)

    def start(self, *args):
        return subprocess.Popen([sys.executable, str(CLIENT), "--", *args], cwd=self.ws, env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def rid(self):
        self.assertTrue(self.wait_for(lambda: any(self.d.glob("*.req.json"))))
        return next(self.d.glob("*.req.json")).name.split(".")[0]

    def test_wait_attaches_to_a_running_request_and_returns_its_status(self):
        with self.serve():
            helper = self.start("slow", "3")
            rid = self.rid()
            out = self.client("--wait", rid, "--for", "60")
            helper.wait(timeout=30)
        self.assertEqual(out.returncode, 3)
        self.assertIn("part one", out.stdout)
        self.assertIn("part two", out.stdout)

    def test_wait_returns_while_the_run_goes_on_and_says_so(self):
        with self.serve():
            helper = self.start("slow", "20")
            rid = self.rid()
            t0 = time.monotonic()
            out = self.client("--wait", rid, "--for", "2")
            self.assertLess(time.monotonic() - t0, 10)
            self.assertEqual(out.returncode, offload.STILL_RUNNING)
            self.assertIn("still", out.stderr)
            self.assertIn(f"--wait {rid}", out.stderr)
            self.assertIsNone(helper.poll())  # the run goes on under its own helper
            helper.kill()

    def test_wait_on_an_unknown_request_says_so(self):
        with self.serve():
            out = self.client("--wait", "0123456789ab", "--for", "2")
        self.assertEqual(out.returncode, 2)
        self.assertIn("no request", out.stderr)

    def test_the_brief_says_a_session_has_no_later_turn_and_how_to_wait(self):
        text = "\n".join(offload.brief(self.camp))
        self.assertIn("--wait", text)
        self.assertIn("no later turn", text)


if __name__ == "__main__":
    unittest.main()
