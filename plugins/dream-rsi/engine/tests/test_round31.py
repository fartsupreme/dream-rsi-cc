"""Round 31: the check of round 30 (Opus). The replay bound held (100k random incumbent, revision and ground-truth
triples; the batch that ends the record, argued and fuzzed with invalid attempts, ties and more unanswered probes
than recorded cells). It found:
- reachable() looped forever on an imported attempt whose id looks like a root slot ("root:0"): the slot resolved to
  that attempt, whose children were looked up as the slot again. Replay renames such an id (recorded ids never reach
  a policy), so a slot and an attempt are never the same cell.
- With a timeout per run and none per evaluation, a revision running just under it cost (runs / 4) x 120 s on each
  of its two replays, while the loop that runs the dream waits. A revision's evaluation now has a limit: four times
  the incumbent's own evaluation, and at least 120 s. The incumbent has none, so a pool growing never fails it.
"""
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import dream
from drsi.dream import SEED_POLICY, run_dream
from drsi.question import ReplayQuestion
from drsi.replay import evaluate_policy, reachable, world_best
from drsi.store import DEFAULT_CONFIG
from tests.helpers import own_worlds, with_block
from tests.test_dream import Dev
from tests.test_policy import chain_world
from tests.test_round28 import write

SEED = SEED_POLICY.read_text()
SLOWER = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        sum(range(10 ** 8))
        return question.legal_roots()[:question.max_parallelism]
    # EVOLVE-BLOCK-END
"""


class Alarm:
    """A hang becomes a failure."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        def fire(*_):
            raise TimeoutError("hung")
        self.old = signal.signal(signal.SIGALRM, fire)
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)


SLOT_IDS = {"id": "history", "baseline": 0.0, "live": False, "nodes": [
    {"id": "root:0", "parent": None, "score": 0.2, "valid": True},
    {"id": "c", "parent": "root:0", "score": 0.7, "valid": True},
    {"id": "root:1", "parent": None, "score": 0.1, "valid": True},
    {"id": "node:root:1", "parent": "root:1", "score": 0.4, "valid": True}]}


class SlotIdTest(unittest.TestCase):
    def test_an_attempt_whose_id_looks_like_a_slot_is_reached_once(self):
        with Alarm(5):
            self.assertEqual(sorted(n["score"] for n in reachable(SLOT_IDS)), [0.1, 0.2, 0.4, 0.7])
            self.assertEqual(world_best(SLOT_IDS), 0.7)

    def test_replay_reveals_it_as_an_attempt_not_a_slot(self):
        q = ReplayQuestion(SLOT_IDS, 2, max_probes=8)
        roots = q.probe_batch(["root:0", "root:1"])
        self.assertEqual([o.score for o in roots], [0.2, 0.1])
        self.assertTrue(all(not o.id.startswith("root:") for o in roots))
        kids = q.probe_batch([o.id for o in roots])
        self.assertEqual([o.score for o in kids], [0.7, 0.4])

    def test_a_world_with_cells_keeps_its_slots_and_children(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "root:5", "parent": None, "score": 0.3, "valid": True, "cell": "root:1"},
            {"id": "k", "parent": "root:5", "score": 0.6, "valid": True, "cell": "root:5"}]}
        self.assertEqual([n["score"] for n in reachable(w)], [0.3, 0.6])
        q = ReplayQuestion(w, 2, max_probes=8)
        self.assertEqual([o.score for o in q.probe_batch(["root:1"])], [0.3])


class RevisionTimeTest(unittest.TestCase):
    def test_an_evaluation_limit_ends_a_slow_policy(self):
        with tempfile.TemporaryDirectory() as d:
            slow = write(d, "slow.py", with_block(SLOWER)(SEED))
            start = time.monotonic()
            rep = evaluate_policy(slow, [chain_world(3, 3) for _ in range(64)], W=2, betas=[], budget=4, lam=0.25,
                                  beta1=0.01, beta2=0.01, timeout=60, total_timeout=2)
        self.assertFalse(rep["ok"])
        self.assertIn("timeout", rep["error"])
        self.assertLess(time.monotonic() - start, 2 + 4)

    def test_a_revision_much_slower_than_the_incumbent_is_refused_in_time(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(SEED)
            worlds = own_worlds(SEED, 4, 24, n=4)
            cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"], M=1)}
            start = time.monotonic()
            with mock.patch.object(dream, "REVISION_FLOOR_S", 1):
                rep = run_dream(pdir, worlds, Dev(with_block(SLOWER)), cfg, Path(d) / "logs")
            took = time.monotonic() - start
        self.assertIsNone(rep["skipped"])
        (rev,) = rep["revisions"]
        self.assertEqual(rev["stage"], "run", rev)
        self.assertIn("timeout", rev["error"])
        self.assertLess(took, 60)


if __name__ == "__main__":
    unittest.main()
