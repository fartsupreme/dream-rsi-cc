"""Round 29: findings of the last cross-vendor review before 0.4.0 (Grok, on the round-27 tip), each reproduced first.

Round 28 already made a world keep the family each attempt was shown with. Grok's deploy still stands on the worlds
frozen before that: a revision that acted on a family the world holds but live did not show scored 0.906 in replay
and -0.208 on the same attempts. The family a live policy sees depends on when the classifier got to an attempt, so
it is not shown to policies at all now, in any environment (neither the seed nor the deployed policy reads it).
And three ways a world could differ from what live showed:
- a child was taken in file order; it is the one opened from that leaf (its recorded cell) where cells exist;
- one root without a root-slot cell (a pruned attempt's continuation, re-rooted) sent every root back to list order;
  each root with a slot keeps it, and one without is not reachable through any slot;
- the history world (imported ledger rows, some scored by the classifier's outcome) was never a live round: it is
  kept out of the dream's comparison.
Gate trees take the campaign's failure classes, and a world counts as informative only if replay can reveal one of
its valid scores.
"""
import tempfile
import unittest
from pathlib import Path

from drsi.dream import SEED_POLICY, gate_worlds, run_dream
from drsi.question import PolicyQuestion, RecordEnd, ReplayQuestion
from drsi.replay import evaluate_policy
from drsi.worlds import informative
from tests.helpers import own_worlds, with_block
from tests.test_dream import Dev
from tests.test_policy import chain_world
from tests.test_round23 import ALLROOTS

# Grok's revision: roots, then (where families look like the campaign's) deepen, else stop
FAMILY_KEYED = ALLROOTS.replace(
    "        if roots:\n            return roots[:W]\n",
    "        fams = [o.family for o in question.observed().values()]\n"
    "        if fams and all(f is None for f in fams):\n            return []\n"
    "        if roots:\n            return roots[:W]\n")
assert FAMILY_KEYED != ALLROOTS


def write(d, name, src) -> Path:
    p = Path(d) / name
    p.write_text(src)
    return p


class FamilyTest(unittest.TestCase):
    def test_a_policy_sees_no_family_in_any_environment(self):
        w = chain_world(2, 2)
        for n in w["nodes"]:
            n["family"] = "F01"
        v = PolicyQuestion(ReplayQuestion(w, 2, max_probes=6))
        out = v.probe_batch(["root:0", "root:1"])
        self.assertEqual([o.family for o in out], [None, None])
        self.assertEqual(v.meta("a1").tags, ())

    def test_a_family_keyed_revision_scores_the_same_whatever_the_world_says(self):
        tagged, bare = chain_world(4, 4), chain_world(4, 4)
        for n in tagged["nodes"]:
            n["family"] = "F01"
        kw = dict(W=2, betas=[], budget=8, lam=0.25, beta1=0.01, beta2=0.01)
        with tempfile.TemporaryDirectory() as d:
            pol = write(d, "keyed.py", with_block(FAMILY_KEYED)(SEED_POLICY.read_text()))
            self.assertEqual(evaluate_policy(pol, [tagged], **kw)["reward"],
                             evaluate_policy(pol, [bare], **kw)["reward"])


class RecordShapeTest(unittest.TestCase):
    def test_a_leaf_reveals_the_child_opened_from_it(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "p", "parent": None, "score": 0.1, "valid": True, "cell": "root:0"},
            {"id": "x", "parent": "p", "score": 1.0, "valid": True, "cell": "elsewhere"},  # re-parented by a prune
            {"id": "c", "parent": "p", "score": 0.2, "valid": True, "cell": "p"}]}
        q = ReplayQuestion(w, 2, max_probes=6)
        q.probe_batch(["root:0"])
        self.assertEqual(q.probe_batch(["p"])[0].id, "c")

    def test_a_root_without_a_slot_leaves_the_other_roots_in_theirs(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": f"r{j}", "parent": None, "score": 0.1 * j, "valid": True, "cell": f"root:{j}"} for j in range(4)]
            + [{"id": "orphan", "parent": None, "score": 0.9, "valid": True, "cell": "iter0001-003"}]}
        q = ReplayQuestion(w, 4, max_probes=12)
        out = q.probe_batch(["root:3", "root:2", "root:1", "root:0"])
        self.assertEqual([o.id for o in out], ["r3", "r2", "r1", "r0"])
        with self.assertRaises(RecordEnd):  # the orphan was opened from no slot: no slot reaches it
            q.probe_batch(["root:4"])

    def test_the_history_world_is_not_compared_on(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(SEED_POLICY.read_text())
            worlds = [dict(w, live=False) for w in own_worlds(SEED_POLICY.read_text(), 4, 24, n=4)]
            cfg = {"search": {"W": 4, "K1": 6}, "dream": {"M": 1, "betas": [0.0, 1.0], "lambda": 0.25, "beta1": 0.01,
                                                          "beta2": 0.01, "bootstrap": 100, "gate_worlds": 4}}
            rep = run_dream(pdir, worlds, Dev(), cfg, Path(d) / "logs")
        self.assertIn("recorded live", rep["skipped"] or "", rep)


class CountTest(unittest.TestCase):
    def test_a_score_replay_can_never_reveal_does_not_make_a_world_informative(self):
        hidden = {"id": "h", "baseline": 0.0, "nodes": [
            {"id": "r", "parent": None, "score": None, "valid": False},
            {"id": "a", "parent": "r", "score": None, "valid": False},
            {"id": "b", "parent": "r", "score": 0.9, "valid": True}]}  # a second child: never revealed
        self.assertEqual(informative([hidden]), 0)
        self.assertEqual(informative([chain_world(1, 1)]), 1)

    def test_gate_trees_fail_with_the_campaigns_classes(self):
        classes = {n["fail_class"] for w in gate_worlds(4, 2, 4, fail_classes=["eval_error", "not_novel"])
                   for n in w["nodes"] if not n["valid"]}
        self.assertTrue(classes)
        self.assertLessEqual(classes, {"eval_error", "not_novel"})


if __name__ == "__main__":
    unittest.main()
