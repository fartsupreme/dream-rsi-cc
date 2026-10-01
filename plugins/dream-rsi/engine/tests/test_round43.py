"""Round 43: the cross-vendor review of round 42 (Grok), each finding reproduced first.

- Workers ran with --setting-sources local, and that source still makes Claude Code load CLAUDE.local.md from the
  working directory and its parents at startup (checked: a canary there was read back with "local" and not with no
  source). A worker could write one into its checkout, the orchestrator commits the checkout whole, and a continuation
  starting from that commit loaded it: notes passed along a lineage behind the map. The judge and classifier ran
  with "project,local" from a directory under the operator's home, so a CLAUDE.md there or above would have reached
  every verdict. Every call now loads no settings source; what a call needs comes in --settings and its flags.
- call_env kept CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD from the orchestrator's environment, which makes the
  proposal directory's CLAUDE.md load; it is removed.
- An Edit(...) rule covers every file-editing tool, so Write(~/.claude/**) was never consulted (and warned): gone.
- --wait's status 75 can also be a command's own; the brief keys on the line it prints.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import cli, offload
from drsi.agent import ClaudeAgent, call_env
from drsi.llm import ClaudeCLI
from drsi.store import Campaign


def sources(args):
    return args[args.index("--setting-sources") + 1]


class SourcesTest(unittest.TestCase):
    def test_no_call_loads_a_settings_source(self):
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("s", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(d))
            self.assertEqual(sources(cli.worker_agent(camp, camp.root / "work" / "iter0001-001", "s").build_args()), "")
            self.assertEqual(sources(cli.developer_agent(camp).build_args()), "")
        self.assertEqual(sources(ClaudeAgent(model="opus", tools="Read").build_args()), "")
        self.assertEqual(sources(ClaudeCLI(model="opus").build_args({"type": "object"})), "")

    def test_the_proposal_directorys_instructions_never_load(self):
        with mock.patch.dict("os.environ", {"CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1"}):
            self.assertNotIn("CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD", call_env())
            self.assertNotIn("CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD",
                             call_env({"CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1"}))

    def test_one_edit_rule_denies_the_memory_folder(self):
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("s", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(d))
            args = cli.worker_agent(camp, camp.root / "work" / "iter0001-001", "s").build_args()
        rules = json.loads(args[args.index("--settings") + 1])["permissions"]["deny"]
        self.assertIn("Edit(~/.claude/**)", rules)
        self.assertNotIn("Write(~/.claude/**)", rules)  # round 44 adds Read rules beside it

    def test_the_brief_keys_waiting_on_the_line_wait_prints(self):
        camp = mock.Mock(config={"live": {"offload": {"cmd": "/bin/sh"}}})
        text = "\n".join(offload.brief(camp))
        self.assertIn("the run is still going", text)


if __name__ == "__main__":
    unittest.main()
