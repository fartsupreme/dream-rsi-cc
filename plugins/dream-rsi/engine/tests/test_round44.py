"""Round 44: the review of round 42 (Opus), its findings not already closed by round 43, each reproduced first.

- A real worker passed Claude Code's background-task id to --wait instead of the request id, was told "no request
  here (a run ends with the session that asked for it)", concluded its run was gone and started a duplicate. --wait
  with no id attaches to the worker's only running request; an unknown id lists the running ones; a command given
  with --wait is refused rather than ignored; the brief says where the id is and to give the Bash tool the 10-minute
  timeout --wait needs.
- The deny rule named ~/.claude, but Claude Code lets a session write its memory folder wherever its config
  directory is (CLAUDE_CONFIG_DIR, inherited by every call): that directory is denied too.
- A worker could still read ~/.claude on purpose with Bash (the Read tool already needs an approval a headless
  session cannot give): the notes earlier workers left, and every Claude Code transcript on the machine. Reading it
  is denied as well (checked with a real call: the Read tool and cat both refused).
- Every call also runs with CLAUDE_CODE_DISABLE_CLAUDE_MDS=1, a second guard beside round 43's empty settings
  source (checked: with it, a CLAUDE.local.md is not read even under the "local" source).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import cli, offload
from drsi.agent import call_env
from drsi.store import Campaign
from tests.test_round42 import FAKE as FAKE42
from tests.test_round40 import Base

CLIENT = Path(offload.__file__)


def deny(env=None):
    with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", env or {}, clear=False):
        if env is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        camp = Campaign.create("s", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(d))
        args = cli.worker_agent(camp, camp.root / "work" / "iter0001-001", "s").build_args()
        return json.loads(args[args.index("--settings") + 1])["permissions"]["deny"]


class DenyTest(unittest.TestCase):
    def test_the_config_directory_is_neither_written_nor_read(self):
        rules = deny()
        self.assertIn("Edit(~/.claude/**)", rules)
        self.assertIn("Read(~/.claude/**)", rules)

    def test_a_moved_config_directory_is_denied_where_it_is(self):
        with tempfile.TemporaryDirectory() as cfg:
            rules = deny({"CLAUDE_CONFIG_DIR": cfg})
            real = os.path.realpath(cfg)
        self.assertIn(f"Edit(/{real}/**)", rules)
        self.assertIn(f"Read(/{real}/**)", rules)
        self.assertIn("Edit(~/.claude/**)", rules)

    def test_every_call_runs_with_claude_mds_off(self):
        self.assertEqual(call_env().get("CLAUDE_CODE_DISABLE_CLAUDE_MDS"), "1")
        self.assertEqual(call_env({"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "0"})["CLAUDE_CODE_DISABLE_CLAUDE_MDS"], "1")


class WaitIdTest(Base):
    def setUp(self):
        super().setUp()
        self.fake.write_text(FAKE42)

    def start(self, *args):
        return subprocess.Popen([sys.executable, str(CLIENT), "--", *args], cwd=self.ws, env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_wait_with_no_id_attaches_to_the_only_running_request(self):
        with self.serve():
            helper = self.start("slow", "3")
            self.assertTrue(self.wait_for(lambda: any(self.d.glob("*.req.json"))))
            out = self.client("--wait", "--for", "60")
            helper.wait(timeout=30)
        self.assertEqual(out.returncode, 3)
        self.assertIn("part two", out.stdout)

    def test_an_unknown_id_lists_the_running_requests(self):
        with self.serve():
            helper = self.start("slow", "20")
            self.assertTrue(self.wait_for(lambda: any(self.d.glob("*.req.json"))))
            rid = next(self.d.glob("*.req.json")).name.split(".")[0]
            out = self.client("--wait", "b5bd1l6ye", "--for", "2")
            helper.kill()
        self.assertEqual(out.returncode, 2)
        self.assertIn(rid, out.stderr)
        self.assertNotIn("ends with the session", out.stderr)

    def test_with_two_running_requests_wait_with_no_id_names_them(self):
        with self.serve():
            a = self.start("slow", "20")
            self.assertTrue(self.wait_for(lambda: len(list(self.d.glob("*.req.json"))) == 1))
            b = self.start("slow", "20")  # queued behind the first
            self.assertTrue(self.wait_for(lambda: len(list(self.d.glob("*.req.json"))) == 2))
            out = self.client("--wait", "--for", "2")
            a.kill()
            b.kill()
        self.assertEqual(out.returncode, 2)
        for p in self.d.glob("*.req.json"):
            self.assertIn(p.name.split(".")[0], out.stderr)

    def test_a_command_given_with_wait_is_refused(self):
        with self.serve():
            out = self.client("--wait", "0123456789ab", "--", "python3", "x.py")
        self.assertEqual(out.returncode, 2)
        self.assertIn("either", out.stderr)
        self.assertFalse((self.bin / "called.txt").exists())

    def test_the_brief_says_where_the_id_is_and_the_timeout_wait_needs(self):
        text = "\n".join(offload.brief(self.camp))
        self.assertIn("600000", text)
        self.assertIn("with no ID", text)


if __name__ == "__main__":
    unittest.main()
