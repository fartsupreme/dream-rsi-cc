"""Round 37: the cross-vendor review of rounds 34 and 35 (Grok).

as_recorded checked that each attempt's cell equals its parent but not that the parent is in the world: an attempt
whose parent is missing passed, replay could not reach it, and the world's best dropped to what was left. On four
such worlds the revision that opens one root a batch deployed over the one that opens four at once (replay 0.5278
against 0.5270), as on the pruned worlds of round 35. A cell must also be the parent's id as text, as live writes it
(an integer 5 read as "5"). Live writers never produce either shape (world_from_tree clears a parent outside the
round, prune rewrites a parent and not its cell); the rule now holds by itself.
"""
import tempfile
import unittest
from pathlib import Path

from drsi.dream import SEED_POLICY, run_dream
from drsi.store import DEFAULT_CONFIG
from drsi.worlds import comparable
from tests.helpers import with_block
from tests.test_dream import Dev
from tests.test_round35 import SERIAL, WIDE


def dangling(n):
    nodes = [{"id": f"r{n}_{i}", "parent": None, "cell": f"root:{i}", "score": s + 0.0001 * n, "valid": True,
              "fail_class": "ok"} for i, s in enumerate((0.51, 0.50, 0.50, 0.50))]
    nodes.append({"id": f"x{n}", "parent": f"gone{n}", "cell": f"gone{n}", "score": 1.0, "valid": True,
                  "fail_class": "ok"})
    return {"id": f"iter{n + 1:04d}", "baseline": 0.0, "nodes": nodes}


class ParentTest(unittest.TestCase):
    def test_an_attempt_whose_parent_is_missing_is_not_what_live_records(self):
        self.assertEqual(comparable([dangling(0)]), [])

    def test_a_cell_must_be_the_parents_id_as_text(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "5", "parent": None, "cell": "root:0", "score": 0.1, "valid": True},
            {"id": "c", "parent": "5", "cell": 5, "score": 1.0, "valid": True}]}
        self.assertEqual(comparable([w]), [])
        w["nodes"][1]["cell"] = "5"
        self.assertEqual(len(comparable([w])), 1)

    def test_the_revision_that_gains_only_on_such_worlds_is_not_deployed(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(with_block(WIDE)(SEED_POLICY.read_text()))
            cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"], M=1)}
            rep = run_dream(pdir, [dangling(n) for n in range(4)], Dev(with_block(SERIAL)), cfg, Path(d) / "logs")
        self.assertFalse(rep["deployed"], rep)


if __name__ == "__main__":
    unittest.main()
