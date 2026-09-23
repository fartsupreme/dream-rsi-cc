import unittest
from pathlib import Path

from drsi.guard import check_policy_source
from drsi.policy.api import LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result
from drsi.question import ReplayQuestion

SEED = Path(__file__).resolve().parents[1] / "drsi" / "policy" / "method.py"


def chain_world(n_roots=6, depth=6, climb=0.1):
    """Each root starts at r*0.05 and climbs by `climb` per step until step 3, then plateaus."""
    nodes = []
    for r in range(n_roots):
        prev = None
        for d in range(depth):
            nid = f"r{r}d{d}"
            score = r * 0.05 + min(d, 3) * climb
            nodes.append({"id": nid, "parent": prev, "score": score, "valid": True})
            prev = nid
    return {"id": "chains", "baseline": 0.0, "nodes": nodes}


def load_seed():
    ns: dict = {}
    exec(compile(SEED.read_text(), str(SEED), "exec"), ns)
    return ns["OptimalPolicy"]


class ApiTest(unittest.TestCase):
    def test_budget_done_on_budget_or_no_actions(self):
        q = ReplayQuestion({"id": "e", "baseline": 0, "nodes": []}, 2)
        self.assertTrue(_budget_done(q, None))  # nothing legal
        q2 = ReplayQuestion(chain_world(1, 2), 2)
        self.assertFalse(_budget_done(q2, 1))
        q2.probe_batch(["root:0"])
        self.assertTrue(_budget_done(q2, 1))

    def test_record_and_finalize(self):
        q = ReplayQuestion(chain_world(1, 2), 2)
        res = SimResult()
        q.probe_batch(["root:0"], on_reveal=lambda _: _record_curve(res, q))
        out = finalize_result(q, res)
        self.assertEqual(out["curve"], [(1, 0.0)])
        self.assertEqual(out["probes"], 1)
        self.assertEqual(out["batch_sizes"], [1])

    def test_beta_knob(self):
        self.assertEqual(LLMDesignedMethod(beta=0.2).beta, 0.2)


class GuardTest(unittest.TestCase):
    def ok(self, body):
        return check_policy_source(
            "from drsi.policy.api import LLMDesignedMethod\n"
            "class OptimalPolicy(LLMDesignedMethod):\n"
            "    def solve(self, question, budget=None):\n" + body)

    def test_seed_policy_passes(self):
        self.assertEqual(check_policy_source(SEED.read_text()), [])

    def test_minimal_policy_passes(self):
        self.assertEqual(self.ok("        return None\n"), [])

    def test_rejects_os_import(self):
        self.assertTrue(self.ok("        import os\n        return None\n"))

    def test_rejects_private_attribute_access(self):
        self.assertTrue(self.ok("        return question._rec\n"))
        self.assertTrue(self.ok("        return question.__dict__\n"))

    def test_rejects_dangerous_builtins(self):
        for bad in ("open('x')", "eval('1')", "exec('1')", "getattr(question, 'world')", "__import__('os')",
                    "vars(question)", "globals()", "compile('1','x','exec')"):
            self.assertTrue(self.ok(f"        return {bad}\n"), bad)

    def test_rejects_world_attribute(self):
        self.assertTrue(self.ok("        return question.world\n"))

    def test_requires_optimal_policy_solve(self):
        self.assertTrue(check_policy_source("x = 1\n"))

    def test_syntax_error_reported(self):
        self.assertTrue(check_policy_source("def (:\n"))


class SeedPolicyTest(unittest.TestCase):
    def test_seed_uses_parallelism_and_reaches_best_region(self):
        Policy = load_seed()
        q = ReplayQuestion(chain_world(), max_parallelism=4)
        out = Policy(beta=0.6).solve(q, budget=30)
        self.assertGreaterEqual(max(out["batch_sizes"]), 4)
        self.assertGreaterEqual(out["best"], 0.3)

    def test_low_beta_spends_less_than_high_beta(self):
        Policy = load_seed()
        lo = Policy(beta=0.0).solve(ReplayQuestion(chain_world(), 4), budget=None)
        hi = Policy(beta=1.0).solve(ReplayQuestion(chain_world(), 4), budget=None)
        self.assertLess(lo["probes"], hi["probes"])

    def test_seed_is_deterministic(self):
        Policy = load_seed()
        a = Policy(beta=0.6).solve(ReplayQuestion(chain_world(), 4), budget=40)
        b = Policy(beta=0.6).solve(ReplayQuestion(chain_world(), 4), budget=40)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
