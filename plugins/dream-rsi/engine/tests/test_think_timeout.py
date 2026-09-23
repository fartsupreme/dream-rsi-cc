"""A deployed policy that stops probing and never returns must not hang a live round."""
import json
import time
import unittest

from drsi.live import LiveRunner, live_round, load_policy
from tests.test_integrity_live import LiveIntegrityBase, bump, worker
from tests.test_replay import HEADER


class ThinkTimeTest(LiveIntegrityBase):
    def test_policy_stuck_between_batches_is_stopped(self):
        raw = json.loads(self.camp.config_path.read_text())
        raw["live"]["think_timeout_s"] = 2
        self.camp.save_config(raw)
        from pathlib import Path
        p = Path(self.tmp.name) / "stuck.py"
        p.write_text(HEADER + ("    def solve(self, question, budget=None):\n        question.reset()\n"
                               "        question.probe_batch(question.legal_roots()[:1])\n"
                               "        while True:\n            pass\n"))
        r = LiveRunner(self.camp, worker(bump), indexer=lambda ids: None, round_id="iter0001")
        t0 = time.time()
        summary = live_round(self.camp, load_policy(p), r)
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(summary["attempts"], 1)  # the attempt it did make is kept


if __name__ == "__main__":
    unittest.main()
