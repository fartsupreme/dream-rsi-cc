import tempfile
import unittest
from pathlib import Path

from drsi.replay import evaluate_policy
from tests.test_policy import SEED, chain_world

HEADER = ("from drsi.policy.api import LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result\n"
          "class OptimalPolicy(LLMDesignedMethod):\n")


def write(tmp, body):
    p = Path(tmp) / "policy.py"
    p.write_text(HEADER + body)
    return p


ROOTS_ONLY = """    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        while not _budget_done(question, budget):
            roots = question.legal_roots()
            if not roots:
                break
            question.probe_batch(roots[:1], on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)
"""


ORDERED = """    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        while not _budget_done(question, budget):
            acts = question.legal_actions()
            order = ORDER
            pick = [a for a in order if a in acts][:1] or acts[:1]
            question.probe_batch(pick, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)
"""
# roots A(.9) and B(.1)->B1(.2)->B2(1.0): both orders reveal everything; one reaches .9 at once
EARLY_WORLD = {"id": "early", "baseline": 0.0, "nodes": [
    {"id": "A", "parent": None, "score": 0.9, "valid": True},
    {"id": "B", "parent": None, "score": 0.1, "valid": True},
    {"id": "B1", "parent": "B", "score": 0.2, "valid": True},
    {"id": "B2", "parent": "B1", "score": 1.0, "valid": True}]}


class AnytimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kw = dict(W=2, betas=[0.0, 1.0], budget=None, lam=0.25, beta1=0.01, beta2=0.01)

    def tearDown(self):
        self.tmp.cleanup()

    def policy(self, name, order):
        p = Path(self.tmp.name) / f"{name}.py"
        p.write_text(HEADER.replace("class OptimalPolicy", f"ORDER = {order!r}\nclass OptimalPolicy") + ORDERED)
        return p

    def test_reaching_good_attempts_earlier_scores_higher_even_when_both_explore_everything(self):
        early = evaluate_policy(self.policy("early", ["root:0", "root:1", "B", "B1"]), [EARLY_WORLD], **self.kw)
        late = evaluate_policy(self.policy("late", ["root:1", "B", "B1", "root:0"]), [EARLY_WORLD], **self.kw)
        self.assertTrue(early["ok"] and late["ok"], (early, late))
        self.assertGreater(early["reward"], late["reward"])

    def test_small_world_explored_fully_still_gives_signal(self):
        rep = evaluate_policy(SEED, [EARLY_WORLD], **dict(self.kw, W=2))
        self.assertGreater(rep["auc"], 0.0)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.worlds = [chain_world(), chain_world(n_roots=3, depth=8, climb=0.05)]
        self.kw = dict(W=4, betas=[0.0, 0.5, 1.0], budget=24, lam=0.25, beta1=0.01, beta2=0.01)

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_policy_evaluates_deterministically(self):
        rep = evaluate_policy(SEED, self.worlds, **self.kw)
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(set(rep["per_beta"]), {"0.0", "0.5", "1.0"})
        self.assertGreater(rep["reward"], 0.0)
        self.assertIn("eq1_default_beta", rep)
        for row in rep["per_beta"].values():
            self.assertTrue(0.0 <= row["attainment"] <= 1.0)
            self.assertTrue(0.0 <= row["work"] <= 1.0)

    def test_seed_beats_serial_roots_only_policy(self):
        seed = evaluate_policy(SEED, self.worlds, **self.kw)
        serial = evaluate_policy(write(self.tmp.name, ROOTS_ONLY), self.worlds, **self.kw)
        self.assertTrue(serial["ok"], serial)
        self.assertGreater(seed["reward"], serial["reward"])

    def test_guard_failure_reported(self):
        rep = evaluate_policy(write(self.tmp.name, "    def solve(self, q, budget=None):\n        import os\n"),
                              self.worlds, **self.kw)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["stage"], "guard")

    def test_exception_reported(self):
        rep = evaluate_policy(write(self.tmp.name, "    def solve(self, q, budget=None):\n        raise ValueError('bad')\n"),
                              self.worlds, **self.kw)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["stage"], "run")
        self.assertIn("bad", rep["error"])

    def test_illegal_batch_reported(self):
        body = ("    def solve(self, question, budget=None):\n        question.reset()\n"
                "        question.probe_batch(['nope'])\n")
        rep = evaluate_policy(write(self.tmp.name, body), self.worlds, **self.kw)
        self.assertFalse(rep["ok"])
        self.assertIn("illegal", rep["error"].lower())

    def test_timeout_reported(self):
        body = "    def solve(self, question, budget=None):\n        while True:\n            pass\n"
        rep = evaluate_policy(write(self.tmp.name, body), self.worlds, timeout=2, **self.kw)
        self.assertFalse(rep["ok"])
        self.assertIn("timeout", rep["error"].lower())

    def test_nondeterminism_detected(self):
        body = ("    def solve(self, question, budget=None):\n        question.reset()\n        res = SimResult()\n"
                "        n = min(4, len(list({'aa', 'bbbb', 'c'})[0]))  # set order follows the hash seed\n"
                "        question.probe_batch(question.legal_roots()[:n], on_reveal=lambda _: _record_curve(res, question))\n"
                "        return finalize_result(question, res)\n")
        rep = evaluate_policy(write(self.tmp.name, body), self.worlds, **self.kw)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["stage"], "determinism")


if __name__ == "__main__":
    unittest.main()
