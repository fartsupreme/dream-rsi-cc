"""Round 32: the cross-vendor check of rounds 28 to 31 (Grok), each finding reproduced here first.

- A world with no cells at all cannot say which child was opened from which leaf, so replay reveals a leaf's first
  child in file order. Where a prune re-parented an attempt, that is a continuation live never ran: on four such
  worlds a revision that probes the best root's leaf deployed under the default config, replaying at 0.367 against
  0.145 on the same attempts. Live rounds store every attempt's cell and older worlds get theirs from the tree; the
  dream now compares only on worlds whose every attempt has its cell.
- Gate trees could be told from the campaign's own: their random walks left [lo, hi], a valid attempt was always
  "ok" whatever the campaign calls one, and a campaign without failures got failures classed "ok". Walks now reflect
  at the bounds, and valid and failed attempts take the campaign's classes.
- `drsi run --history` counted the history world toward dream.min_worlds, which the dream then left out; and
  `drsi replay --history` printed a reward over it without saying the dream does not compare there.
"""
import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from drsi.dream import SEED_POLICY, gate_worlds, run_dream
from drsi.store import DEFAULT_CONFIG
from drsi.worlds import comparable
from tests.helpers import own_worlds, with_block
from tests.test_dream import Dev
from tests.test_policy import chain_world

SEED = SEED_POLICY.read_text()
# Grok's pair: the incumbent opens four roots and stops; the revision then probes its best leaf once
INC = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        if not question.rounds:
            return question.legal_roots()[:question.max_parallelism]
        return []
    # EVOLVE-BLOCK-END
"""
CAND = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        W = question.max_parallelism
        if not question.rounds:
            return question.legal_roots()[:W]
        if question.rounds >= 2:
            return []
        obs = question.observed()
        leaves = [c for c in question.legal_actions() if not c.startswith("root:")]
        leaves.sort(key=lambda c: (obs[c].score if obs[c].valid and obs[c].score is not None else -1.0, c),
                    reverse=True)
        return leaves[:1]
    # EVOLVE-BLOCK-END
"""


def pruned_world(k):
    """Live opened b (a failed attempt) from a, then c from b; a prune removed b and re-parented c to a. No cells."""
    others = [{"id": f"r{i}", "parent": None, "score": s + 0.001 * k, "valid": True, "fail_class": "ok"}
              for i, s in enumerate((0.3, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02))]
    return {"id": f"iter{k + 1:04d}", "baseline": 0.0, "nodes": [
        {"id": "a", "parent": None, "score": 0.5, "valid": True, "fail_class": "ok"},
        {"id": "c", "parent": "a", "score": 1.0, "valid": True, "fail_class": "ok"}] + others}


class CellTest(unittest.TestCase):
    def test_a_world_without_cells_is_not_compared_on(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(with_block(INC)(SEED))
            cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"], M=1)}
            rep = run_dream(pdir, [pruned_world(k) for k in range(4)], Dev(with_block(CAND)), cfg,
                            Path(d) / "logs")
        self.assertFalse(rep["deployed"], rep)
        self.assertIn("cell", rep["skipped"] or "", rep)

    def test_comparable_worlds_are_live_with_every_cell(self):
        celled = own_worlds(SEED, 4, 24, n=1)[0]
        bare = dict(celled, nodes=[{k: v for k, v in n.items() if k != "cell"} for n in celled["nodes"]])
        partial = dict(celled, nodes=[dict(n, cell=None) if i == 1 else n for i, n in enumerate(celled["nodes"])])
        history = dict(celled, live=False)
        self.assertEqual(comparable([celled, bare, partial, history]), [celled])


class GateTest(unittest.TestCase):
    def test_gate_scores_stay_in_the_campaigns_range(self):
        worlds = gate_worlds(32, 4, 24, lo=0.2, hi=0.5)
        scores = [n["score"] for w in worlds for n in w["nodes"] if n["valid"]]
        self.assertTrue(scores)
        self.assertGreaterEqual(min(scores), 0.2)
        self.assertLessEqual(max(scores), 0.5)

    def test_gate_attempts_take_the_campaigns_classes(self):
        worlds = gate_worlds(8, 4, 24, fail_classes=["agent_error"], valid_classes=["pass"])
        self.assertEqual({n["fail_class"] for w in worlds for n in w["nodes"] if n["valid"]}, {"pass"})
        self.assertEqual({n["fail_class"] for w in worlds for n in w["nodes"] if not n["valid"]}, {"agent_error"})
        fallback = gate_worlds(8, 4, 24)
        self.assertNotIn("ok", {n["fail_class"] for w in fallback for n in w["nodes"] if not n["valid"]})


class HistoryTest(unittest.TestCase):
    def test_drsi_run_does_not_count_the_history_world(self):
        from drsi import live
        self.assertNotIn("history_world", live.run_cycles.__code__.co_varnames)

    def test_drsi_replay_says_the_history_is_not_compared_on(self):
        from drsi import cli
        cfg = {"search": {"W": 2, "K1": 2}, "dream": dict(DEFAULT_CONFIG["dream"])}
        worlds = [chain_world(4, 4), dict(chain_world(4, 4), id="history", live=False)]
        out = io.StringIO()
        with mock.patch.object(cli, "resolve_campaign", return_value=mock.Mock(config=cfg)), \
                mock.patch.object(cli, "_worlds", return_value=worlds), \
                mock.patch.object(cli, "_policy_path", return_value=SEED_POLICY), redirect_stdout(out):
            cli.cmd_replay(Namespace(campaign="x", history=True, policy=None))
        self.assertIn("imported history", out.getvalue())


if __name__ == "__main__":
    unittest.main()
