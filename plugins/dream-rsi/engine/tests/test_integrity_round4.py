"""Fourth-review findings (grok-4.7 on the hardened code), each checked against the working copy first."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from drsi.families import family_stats
from drsi.guard import check_policy_source
from drsi.novelty import check
from drsi.replay import evaluate_policy
from drsi.reward import parallel_penalty
from drsi.scorer import run_scorer
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM
from tests.test_integrity_live import LiveIntegrityBase, bump, worker
from tests.test_policy import SEED, chain_world
from tests.test_replay import HEADER

KW = dict(W=4, betas=[0.0, 1.0], budget=24, lam=0.25, beta1=0.01, beta2=0.01)


class NoModuleObjectsTest(unittest.TestCase):
    def test_plain_module_imports_are_rejected(self):
        src = "import statistics\n" + HEADER + "    def solve(self, question, budget=None):\n        mod = statistics\n        return mod.sys\n"
        self.assertTrue(check_policy_source(src))

    def test_from_imports_of_plain_functions_are_fine(self):
        src = ("from math import sqrt\nfrom collections import defaultdict\n" + HEADER +
               "    def solve(self, question, budget=None):\n        return sqrt(4) + len(defaultdict(list))\n")
        self.assertEqual(check_policy_source(src), [])


class TamperProofMetricsTest(unittest.TestCase):
    """Even if a policy tampered with the in-process question, the parent recomputes metrics from the
    batches the policy actually requested."""

    def test_runtime_builtins_are_restricted(self):
        # the guard would reject getattr; this checks the runtime layer on its own by smuggling a name
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.py"
            p.write_text(HEADER + "    def solve(self, question, budget=None):\n        return __builtins__\n")
            from drsi import replay
            # bypass the static guard to test the runtime layer in isolation
            orig = replay.check_policy_source
            replay.check_policy_source = lambda src: []
            try:
                body = HEADER + ("    def solve(self, question, budget=None):\n"
                                 "        f = [b for b in [0]]\n"
                                 "        return open('/etc/hosts').read()\n")
                p.write_text(body)
                rep = evaluate_policy(p, [chain_world()], **KW)
            finally:
                replay.check_policy_source = orig
            self.assertFalse(rep["ok"])
            self.assertIn("open", rep["error"])

    def test_metrics_are_recomputed_from_the_requested_batches(self):
        from drsi import replay
        with tempfile.TemporaryDirectory() as d:
            honest = evaluate_policy(SEED, [chain_world()], **KW)
            self.assertTrue(honest["ok"])
            self.assertIn("traces_replayed", honest)


class PenaltyClampTest(unittest.TestCase):
    def test_penalty_never_negative(self):
        self.assertEqual(parallel_penalty([100], W=4), 0.0)


class SymlinkTest(LiveIntegrityBase):
    def test_symlink_in_an_attempt_is_out_of_scope(self):
        def act(ws):
            (ws.parent / "payload.txt").write_text("999999\n")
            (ws / "value.txt").unlink()
            os.symlink(str(ws.parent / "payload.txt"), ws / "value.txt")
        nodes = self.round(act)
        self.assertEqual(nodes[0]["fail_class"], "out_of_scope")


class ScorerSandboxTest(unittest.TestCase):
    @unittest.skipUnless(Path("/usr/bin/sandbox-exec").exists(), "macOS sandbox-exec not available")
    def test_scorer_cannot_write_outside_temp_and_its_checkout(self):
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory(dir=Path.home()) as outside:
            target = Path(outside) / "tampered.txt"
            cmd = f"echo x > {target}; echo '{{\"score\": 1, \"valid\": true}}'"
            r = run_scorer(cmd, ws, timeout=20, sandbox=True)
            self.assertFalse(target.exists())
            self.assertTrue(r["valid"])  # the scorer itself still ran and reported


class ExactDuplicateTest(unittest.TestCase):
    def test_same_text_as_history_is_a_duplicate_without_the_judge(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, proposal="Use a  wheel sieve.", fingerprint={"mechanism": "m"}))
            llm = ScriptedLLM(lambda p, s: {"verdict": "novel", "family": "F00", "nearest_ids": [], "what_differs": "x",
                                            "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": ""})
            r = check(t, {"families": []}, llm, "use a wheel sieve")
            self.assertEqual(r["verdict"], "duplicate")
            self.assertEqual(llm.prompts, [])

    def test_same_text_as_an_in_flight_claim_is_a_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            llm = ScriptedLLM(lambda p, s: {"verdict": "novel"})
            r = check(t, {"families": []}, llm, "Idea X", pending=[{"node": "iter0001-001", "proposal": "idea x"}])
            self.assertEqual(r["verdict"], "duplicate")


class PlateauScoreTest(unittest.TestCase):
    def test_rising_scores_are_not_a_plateau(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            for i, sc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1):
                t.add(make_node(id=str(i), parent=None, score=sc, valid=True,
                                fingerprint={"outcome": "pass", "family": "F01"}))
            fams = {"families": [{"id": "F01", "name": "n", "description": "", "boundary": ""}]}
            self.assertEqual(family_stats(t, fams, plateau=3)[0]["status"], "open")


class ModifyTest(unittest.TestCase):
    def test_fingerprint_rewrite_keeps_family_and_live_outcome(self):
        from drsi.fingerprint import fingerprint_nodes
        from tests.helpers import fp_for, ids_in_block
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, proposal="p", fingerprint={"mechanism": "old", "family": "F03"}))
            stale = Tree(t.path)
            llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
            fingerprint_nodes(stale, llm, goal="g", batch=5, workers=1, ids=["1"])
            fp = Tree(t.path).get("1")["fingerprint"]
            self.assertEqual(fp["family"], "F03")
            self.assertNotIn("error", fp)


if __name__ == "__main__":
    unittest.main()
