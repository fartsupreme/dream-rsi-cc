import unittest

from drsi.question import IllegalBatch, ReplayQuestion

# roots A(.1), E(.3); A->B(.2)->C(.5); A->D(.9) second child; E->F(invalid)
WORLD = {
    "id": "w1", "baseline": 0.0,
    "nodes": [
        {"id": "A", "parent": None, "score": 0.1, "valid": True},
        {"id": "B", "parent": "A", "score": 0.2, "valid": True},
        {"id": "C", "parent": "B", "score": 0.5, "valid": True},
        {"id": "D", "parent": "A", "score": 0.9, "valid": True},
        {"id": "E", "parent": None, "score": 0.3, "valid": True},
        {"id": "F", "parent": "E", "score": None, "valid": False, "fail_class": "eval_error"},
    ],
}


class ReplayQuestionTest(unittest.TestCase):
    def setUp(self):
        self.q = ReplayQuestion(WORLD, max_parallelism=2)
        self.q.reset()

    def test_initial_state_offers_root_slots_only(self):
        self.assertEqual(self.q.observed(), {})
        self.assertEqual(self.q.legal_actions(), ["root:0", "root:1"])
        self.assertEqual(self.q.legal_roots(), ["root:0", "root:1"])
        self.assertEqual((self.q.probes, self.q.rounds), (0, 0))

    def test_root_slot_reveals_recorded_root_in_order(self):
        out = self.q.probe_batch(["root:0"])
        self.assertEqual(out[0].id, "A")
        self.assertEqual(self.q.legal_actions(), ["root:1", "A"])
        self.assertEqual((self.q.probes, self.q.rounds), (1, 1))

    def test_leaf_reveals_its_first_recorded_child(self):
        self.q.probe_batch(["root:0"])
        out = self.q.probe_batch(["A", "root:1"])
        self.assertEqual([o.id for o in out], ["B", "E"])
        self.assertEqual(set(self.q.observed()), {"A", "B", "E"})
        self.assertNotIn("D", self.q.observed())  # sibling never revealed: A is no longer a leaf
        self.assertEqual(sorted(self.q.legal_actions()), ["B", "E"])

    def test_leaf_without_recorded_child_returns_none_and_is_exhausted(self):
        self.q.probe_batch(["root:0"])
        self.q.probe_batch(["A"])
        self.q.probe_batch(["B"])
        out = self.q.probe_batch(["C"])
        self.assertIsNone(out[0])
        self.assertNotIn("C", self.q.legal_actions())
        self.assertEqual(self.q.probes, 3)  # the empty probe reveals nothing
        self.assertEqual(self.q.rounds, 4)

    def test_invalid_node_observed_with_fail_class(self):
        self.q.probe_batch(["root:0", "root:1"])
        out = self.q.probe_batch(["E"])
        self.assertEqual(out[0].id, "F")
        self.assertFalse(out[0].valid)
        self.assertIsNone(out[0].score)
        self.assertEqual(out[0].fail_class, "eval_error")

    def test_illegal_batches_rejected(self):
        with self.assertRaises(IllegalBatch):
            self.q.probe_batch([])
        with self.assertRaises(IllegalBatch):
            self.q.probe_batch(["root:0", "root:0"])
        with self.assertRaises(IllegalBatch):
            self.q.probe_batch(["root:0", "root:1", "root:2"])  # over max_parallelism
        with self.assertRaises(IllegalBatch):
            self.q.probe_batch(["B"])  # not revealed yet

    def test_meta_for_root_slot_and_node(self):
        m = self.q.meta("root:1")
        self.assertEqual((m.branch, m.attempt, m.parent_id), (1, 0, None))
        self.q.probe_batch(["root:0"])
        self.q.probe_batch(["A"])
        m = self.q.meta("B")
        self.assertEqual((m.branch, m.attempt, m.parent_id), (0, 1, "A"))
        self.assertEqual(self.q.opened_branches(), [0])

    def test_on_reveal_called_per_revealed_node(self):
        seen = []
        self.q.probe_batch(["root:0", "root:1"], on_reveal=lambda o: seen.append(o.id))
        self.assertEqual(seen, ["A", "E"])

    def test_best_score_and_baseline(self):
        self.assertEqual(self.q.baseline_score, 0.0)
        self.q.probe_batch(["root:0", "root:1"])
        self.assertEqual(self.q.best_score(), 0.3)

    def test_reset_before_probing_starts_clean(self):
        self.q.reset()
        self.assertEqual(self.q.observed(), {})
        self.assertEqual((self.q.probes, self.q.rounds), (0, 0))
        self.assertEqual(self.q.batch_sizes, [])

    def test_root_slots_exhaust_with_recorded_roots(self):
        self.q.probe_batch(["root:0", "root:1"])
        self.assertEqual(self.q.legal_roots(), [])


if __name__ == "__main__":
    unittest.main()
