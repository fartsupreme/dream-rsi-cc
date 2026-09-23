import json
import re
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.dream import run_dream, split_evolve
from tests.test_policy import SEED, chain_world

CFG = {"search": {"W": 4, "K1": 6}, "dream": {"M": 2, "betas": [0.0, 0.5, 1.0], "lambda": 0.25,
                                               "beta1": 0.01, "beta2": 0.01}}
SERIAL_BLOCK = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        roots = question.legal_roots()
        if roots:
            return roots[:1]
        leaves = [c for c in question.legal_actions() if not c.startswith("root:")]
        return leaves[:1]
    # EVOLVE-BLOCK-END
"""


def serial_policy() -> str:
    src = SEED.read_text()
    before, _, after = split_evolve(src)
    return before + SERIAL_BLOCK + after


def seed_block() -> str:
    return split_evolve(SEED.read_text())[1]


class Dev:
    """Scripted policy developer: applies edit(src) -> new src, records what it saw."""

    def __init__(self, *edits, ok=True):
        self.edits, self.ok, self.seen, self.prompts = list(edits), ok, [], []

    def __call__(self, sandbox: Path, prompt: str):
        self.prompts.append(prompt)
        f = sandbox / "method.py"
        src = f.read_text()
        self.seen.append(src)
        if not self.ok:
            return AgentResult(ok=False, error="agent crashed")
        f.write_text(self.edits.pop(0)(src) if self.edits else src)
        return AgentResult(ok=True, result_text="edited")


def replace_block(new_block):
    def edit(src):
        before, _, after = split_evolve(src)
        return before + new_block + after
    return edit


class DreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = Path(self.tmp.name) / "policy"
        self.pdir.mkdir()
        (self.pdir / "method.py").write_text(serial_policy())
        self.logs = Path(self.tmp.name) / "logs"
        self.worlds = [chain_world(), chain_world(n_roots=3, depth=8, climb=0.05)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_evolve_roundtrip(self):
        before, block, after = split_evolve(SEED.read_text())
        self.assertIn("EVOLVE-BLOCK-START", block)
        self.assertEqual(before + block + after, SEED.read_text())

    def test_better_revision_is_deployed_and_archived(self):
        dev = Dev(replace_block(seed_block()))
        rep = run_dream(self.pdir, self.worlds, dev, CFG, self.logs)
        self.assertTrue(rep["deployed"], rep)
        self.assertGreater(rep["best_reward"], rep["incumbent_reward"])
        self.assertIn("coverage rule", (self.pdir / "method.py").read_text())
        self.assertTrue((self.pdir / "versions" / "v0000.py").exists())
        self.assertTrue((self.pdir / "versions" / "v0001.py").exists())
        self.assertEqual(len(list(self.logs.glob("dream-*.json"))), 1)

    def test_second_revision_starts_from_best_so_far(self):
        dev = Dev(replace_block(seed_block()), lambda s: s)
        run_dream(self.pdir, self.worlds, dev, CFG, self.logs)
        self.assertIn("coverage rule", dev.seen[1])

    def test_edit_outside_block_rejected(self):
        dev = Dev(lambda s: s.replace("beta = 0.6", "beta = 0.9"), lambda s: s)
        rep = run_dream(self.pdir, self.worlds, dev, CFG, self.logs)
        self.assertFalse(rep["deployed"])
        self.assertEqual(rep["revisions"][0]["stage"], "scope")

    def test_guard_failure_recorded(self):
        bad = SERIAL_BLOCK.replace("roots = question.legal_roots()", "import os\n        roots = question.legal_roots()")
        dev = Dev(replace_block(bad), lambda s: s)
        rep = run_dream(self.pdir, self.worlds, dev, CFG, self.logs)
        self.assertEqual(rep["revisions"][0]["stage"], "guard")
        self.assertFalse(rep["deployed"])
        self.assertEqual((self.pdir / "method.py").read_text(), serial_policy())

    def test_worse_revision_is_not_deployed(self):
        (self.pdir / "method.py").write_text(SEED.read_text())
        dev = Dev(replace_block(SERIAL_BLOCK.replace("roots[:1]", "roots[:1]  # worse")), lambda s: s)
        rep = run_dream(self.pdir, self.worlds, dev, CFG, self.logs)
        self.assertEqual(rep["revisions"][0]["stage"], "scored")
        self.assertLess(rep["revisions"][0]["reward"], rep["incumbent_reward"])
        self.assertFalse(rep["deployed"])
        self.assertEqual((self.pdir / "method.py").read_text(), SEED.read_text())

    def test_unchanged_file_is_not_deployed(self):
        rep = run_dream(self.pdir, self.worlds, Dev(), CFG, self.logs)
        self.assertFalse(rep["deployed"])
        self.assertEqual([r["stage"] for r in rep["revisions"]], ["unchanged", "unchanged"])

    def test_agent_failure_recorded(self):
        rep = run_dream(self.pdir, self.worlds, Dev(ok=False), CFG, self.logs)
        self.assertEqual(rep["revisions"][0]["stage"], "agent")
        self.assertFalse(rep["deployed"])

    def test_prompt_carries_scoring_contract_and_report(self):
        dev = Dev()
        run_dream(self.pdir, self.worlds, dev, CFG, self.logs)
        p = dev.prompts[0]
        self.assertIn("EVOLVE-BLOCK", p)
        self.assertIn("pareto", p.lower())
        self.assertIn("prefix-only", p.lower())

    def test_missing_policy_seeded_from_package(self):
        (self.pdir / "method.py").unlink()
        rep = run_dream(self.pdir, self.worlds, Dev(), CFG, self.logs)
        self.assertTrue((self.pdir / "method.py").exists())
        self.assertTrue(rep["incumbent_reward"] > 0)


if __name__ == "__main__":
    unittest.main()
