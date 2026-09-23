"""Regressions found by the third review (each reproduced here first)."""
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from drsi.agent import kill_all_children, run_group
from drsi.dream import block_problems
from drsi.families import rebuild_families
from drsi.fingerprint import fingerprint_nodes
from drsi.guard import check_policy_source
from drsi.importer import import_jsonl
from drsi.question import ReplayQuestion
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM, fp_for, ids_in_block
from tests.test_policy import chain_world
from tests.test_replay import HEADER


class ImportRepeatTest(unittest.TestCase):
    def test_a_ledger_that_repeats_an_id_keeps_syncing(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "l.jsonl"
            src.write_text("\n".join(json.dumps(r) for r in [
                {"id": 1, "candidate": "a", "verdict": "v"}, {"id": 3, "candidate": "GAMMA", "verdict": "v"},
                {"id": 3, "candidate": "GAMMA (re-measured)", "verdict": "v"}]) + "\n")
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, src)
            self.assertEqual(import_jsonl(t, src), 0)  # no ValueError on the second pass


class GuardStoreContextTest(unittest.TestCase):
    def test_attribute_writes_through_loops_with_and_comprehensions_rejected(self):
        for stmt in ("for question.max_parallelism in [64]:\n            pass",
                     "x = [1 for question.probes in [0]]",
                     "with question as question.probes:\n            pass"):
            src = HEADER + "    def solve(self, question, budget=None):\n        " + stmt + "\n"
            self.assertTrue(check_policy_source(src), stmt)

    def test_max_parallelism_is_read_only(self):
        q = ReplayQuestion(chain_world(1, 2), 2)
        with self.assertRaises(AttributeError):
            q.max_parallelism = 64


class InterruptTest(unittest.TestCase):
    def test_kill_all_children_stops_running_groups(self):
        box = {}

        def run():
            t0 = time.time()
            try:
                run_group(["bash", "-c", "sleep 7774 & sleep 7774"], input="", timeout=60, cwd=None)
            except Exception as e:  # noqa: BLE001
                box["err"] = e
            box["secs"] = time.time() - t0
        th = threading.Thread(target=run)
        th.start()
        time.sleep(0.5)
        kill_all_children()
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertLess(box["secs"], 10)
        time.sleep(0.2)
        self.assertNotEqual(subprocess.run(["pgrep", "-f", "sleep 7774"]).returncode, 0)


class LiveOutcomeStickyTest(unittest.TestCase):
    def test_refingerprinting_keeps_the_scorer_outcome_for_live_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="L1", parent=None, source="live", valid=False, fail_class="gate_fail",
                            artifacts={"outcome": "killed", "killed_by": "gate_fail"},
                            fingerprint={"error": "classifier down", "outcome": "killed", "killed_by": "gate_fail"}))
            llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i, outcome="pass", killed_by="") for i in ids_in_block(p)]})
            fingerprint_nodes(t, llm, goal="g", batch=5, workers=1)
            fp = Tree(t.path).get("L1")["fingerprint"]
            self.assertEqual((fp["outcome"], fp["killed_by"]), ("killed", "gate_fail"))


class RebuildTransactionTest(unittest.TestCase):
    def test_failed_rebuild_leaves_the_old_families_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            for i in range(1, 7):
                t.add(make_node(id=str(i), parent=None, fingerprint={"mechanism": f"m{i}", "family": "F01"}))
            path = Path(d) / "f.json"
            path.write_text(json.dumps({"families": [{"id": "F01", "name": "old", "description": "", "boundary": ""}]}))

            def down(p, s):
                raise RuntimeError("claude -p timed out")
            with self.assertRaises(RuntimeError):
                rebuild_families(t, ScriptedLLM(down), "g", path)
            self.assertEqual({n["fingerprint"].get("family") for n in Tree(t.path).nodes()}, {"F01"})
            self.assertEqual(json.loads(path.read_text())["families"][0]["name"], "old")


class EvolveCommentTest(unittest.TestCase):
    def test_column_zero_comment_in_block_is_fine(self):
        block = "    # EVOLVE-BLOCK-START\n# widen patience at high beta\n    def helper(self):\n        return 1\n    # EVOLVE-BLOCK-END\n"
        self.assertEqual(block_problems(block), [])


class ImportScaleTest(unittest.TestCase):
    def test_long_forward_chain_imports_quickly(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "l.jsonl"
            n = 6000
            src.write_text("".join(json.dumps({"key": str(i), "from": str(i + 1), "idea": f"i{i}"}) + "\n"
                                   for i in range(n)))
            t = Tree(Path(d) / "t.jsonl")
            t0 = time.time()
            import_jsonl(t, src, preset="generic", field_map={"id": "key", "parent": "from", "proposal": "idea"})
            self.assertLess(time.time() - t0, 3.0)
            self.assertEqual(len(Tree(t.path)), n)


if __name__ == "__main__":
    unittest.main()
