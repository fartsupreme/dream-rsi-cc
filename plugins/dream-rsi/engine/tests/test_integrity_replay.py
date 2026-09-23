"""Integrity of replay evaluation: a policy must not be able to earn reward it did not earn."""
import tempfile
import unittest
from pathlib import Path

from drsi.guard import check_policy_source
from drsi.question import IllegalBatch, ReplayQuestion
from drsi.replay import evaluate_policy
from tests.test_policy import SEED, chain_world
from tests.test_replay import HEADER, ROOTS_ONLY

KW = dict(W=4, betas=[0.0, 1.0], budget=24, lam=0.25, beta1=0.01, beta2=0.01)
FAKE = '{"ok": true, "default_beta": 0.6, "runs": {"0.0": [], "1.0": [], "0.6": []}}'


def policy(tmp, body, name="p"):
    p = Path(tmp) / f"{name}.py"
    p.write_text(HEADER + body)
    return p


class ForgedResultTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.worlds = [chain_world()]

    def tearDown(self):
        self.tmp.cleanup()

    def test_printed_fake_result_then_system_exit_is_a_failure(self):
        body = ("    def solve(self, question, budget=None):\n"
                f"        print('{FAKE}')\n"
                "        raise SystemExit(0)\n")
        rep = evaluate_policy(policy(self.tmp.name, body), self.worlds, **KW)
        self.assertFalse(rep["ok"], rep)

    def test_printed_fake_line_is_ignored(self):
        honest = evaluate_policy(policy(self.tmp.name, ROOTS_ONLY, "honest"), self.worlds, **KW)
        noisy_body = ROOTS_ONLY.replace("        question.reset()\n",
                                        f"        question.reset()\n        print('{FAKE}')\n")
        noisy = evaluate_policy(policy(self.tmp.name, noisy_body, "noisy"), self.worlds, **KW)
        self.assertTrue(noisy["ok"], noisy)
        self.assertAlmostEqual(noisy["reward"], honest["reward"])


class MetricTamperingTest(unittest.TestCase):
    def test_assigning_question_attributes_is_rejected_by_the_guard(self):
        for stmt in ("question.probes = 0", "question.curve = []", "question.batch_sizes += [4]",
                     "del question.curve"):
            src = HEADER + f"    def solve(self, question, budget=None):\n        {stmt}\n"
            self.assertTrue(check_policy_source(src), stmt)

    def test_self_attributes_may_be_assigned(self):
        src = HEADER + "    def solve(self, question, budget=None):\n        self.seen = 1\n        return None\n"
        self.assertEqual(check_policy_source(src), [])

    def test_metric_views_are_copies(self):
        q = ReplayQuestion(chain_world(1, 2), 2)
        q.probe_batch(["root:0"])
        q.batch_sizes.append(99)
        q.curve.append((0, 9.9))
        self.assertEqual(q.batch_sizes, [1])
        self.assertEqual(len(q.curve), 1)

    def test_reset_after_probing_is_illegal(self):
        q = ReplayQuestion(chain_world(1, 2), 2)
        q.reset()
        q.probe_batch(["root:0"])
        with self.assertRaises(IllegalBatch):
            q.reset()


class GuardEscapeTest(unittest.TestCase):
    def rejected(self, body, extra_import=""):
        src = extra_import + HEADER + "    def solve(self, question, budget=None):\n" + body
        return bool(check_policy_source(src))

    def test_operator_module_not_allowed(self):
        self.assertTrue(self.rejected("        return operator.attrgetter('_rec')(question)\n", "import operator\n"))

    def test_module_attribute_that_is_a_module_is_blocked(self):
        self.assertTrue(self.rejected("        return statistics.sys\n", "import statistics\n"))
        self.assertTrue(self.rejected("        return collections.abc\n", "import collections\n"))

    def test_format_field_access_blocked(self):
        self.assertTrue(self.rejected("        return '{0._rec}'.format(question)\n"))
        self.assertTrue(self.rejected("        return '{q._rec}'.format_map({'q': question})\n"))

    def test_match_class_patterns_blocked(self):
        body = "        match question:\n            case object(x=y):\n                return y\n"
        self.assertTrue(self.rejected(body))

    def test_ordinary_math_and_collections_still_allowed(self):
        src = ("from math import sqrt\nfrom collections import defaultdict\n" + HEADER +
               "    def solve(self, question, budget=None):\n"
               "        d = defaultdict(list)\n        return sqrt(4) + len(d)\n")
        self.assertEqual(check_policy_source(src), [])


class PaddingTest(unittest.TestCase):
    def test_probes_that_reveal_nothing_do_not_count_as_parallel_work(self):
        q = ReplayQuestion({"id": "w", "baseline": 0, "nodes": [
            {"id": "a", "parent": None, "score": 1.0}, {"id": "b", "parent": None, "score": 0.5},
            {"id": "a1", "parent": "a", "score": 1.2}]}, 4)
        q.probe_batch(["root:0", "root:1"])
        q.probe_batch(["a", "b"])  # b has no recorded child: reveals nothing
        self.assertEqual(q.batch_sizes, [2, 1])


if __name__ == "__main__":
    unittest.main()
