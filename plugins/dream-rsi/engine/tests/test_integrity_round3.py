"""Third-review findings on replay integrity and ranking (each checked against the code first)."""
import tempfile
import unittest
from pathlib import Path

from drsi.dream import block_problems, build_prompt
from drsi.guard import ALLOWED_MODULES, check_policy_source
from drsi.question import IllegalBatch, ReplayQuestion
from drsi.replay import evaluate_policy
from tests.test_dream import CFG
from tests.test_policy import chain_world
from tests.test_replay import HEADER


def world_with_best_under_root(k, n_roots=4, depth=4):
    nodes = []
    for r in range(n_roots):
        prev = None
        for d in range(depth):
            nid = f"r{r}d{d}"
            nodes.append({"id": nid, "parent": prev, "score": 1.0 if (r == k and d == 1) else 0.1})
            prev = nid
    return {"id": f"w{k}", "baseline": 0.0, "nodes": nodes}


class ApiImportTest(unittest.TestCase):
    def test_only_the_documented_api_names_can_be_imported(self):
        src = ("from drsi.policy.api import __builtins__ as bb\n" + HEADER +
               "    def solve(self, question, budget=None):\n        return bb\n")
        self.assertTrue(check_policy_source(src))

    def test_walrus_and_decorators_rejected(self):
        walrus = HEADER + "    def solve(self, question, budget=None):\n        return (x := 1)\n"
        deco = HEADER + "    @staticmethod\n    def helper():\n        return 1\n    def solve(self, question, budget=None):\n        return None\n"
        self.assertTrue(check_policy_source(walrus))
        self.assertTrue(check_policy_source(deco))
        self.assertTrue(block_problems("    def helper(self, x=(solve := 1)):\n        return x\n"))


class BudgetExactTest(unittest.TestCase):
    def test_last_batch_is_trimmed_to_the_budget(self):
        q = ReplayQuestion(chain_world(8, 2), 4, max_probes=5)
        q.probe_batch(["root:0", "root:1", "root:2", "root:3"])
        q.probe_batch(q.legal_actions()[:4])
        self.assertEqual(q.probes, 5)

    def test_reentrant_probe_from_on_reveal_is_illegal(self):
        q = ReplayQuestion(chain_world(8, 2), 4, max_probes=100)
        seen = []

        def again(_):
            seen.append(1)
            q.probe_batch([q.legal_roots()[0]])
        with self.assertRaises(IllegalBatch):
            q.probe_batch(["root:0"], on_reveal=again)

    def test_aborted_batch_gets_credit_only_for_what_it_revealed(self):
        q = ReplayQuestion(chain_world(8, 2), 4)

        def boom(_):
            raise RuntimeError("stop after the first reveal")
        with self.assertRaises(RuntimeError):
            q.probe_batch(["root:0", "root:1", "root:2", "root:3"], on_reveal=boom)
        self.assertEqual(q.batch_sizes, [1])


class RankingTest(unittest.TestCase):
    kw = dict(W=1, betas=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], budget=16, lam=0.0, beta1=0.01, beta2=0.01)

    def policy(self, d, name, branch_expr):
        # open every root first, then deepen one branch chosen by `branch_expr`, then the rest
        body = ("    def solve(self, question, budget=None):\n        question.reset()\n        res = SimResult()\n"
                f"        target = {branch_expr}\n"
                "        while not _budget_done(question, budget):\n"
                "            roots = question.legal_roots()\n"
                "            leaves = [a for a in question.legal_actions() if not a.startswith('root:')]\n"
                "            if roots:\n                pick = roots[:1]\n"
                "            else:\n"
                "                pref = [c for c in leaves if question.meta(c).branch == target]\n"
                "                pick = (pref or leaves)[:1]\n"
                "            if not pick:\n                break\n"
                "            question.probe_batch(pick, on_reveal=lambda _: _record_curve(res, question))\n"
                "        return finalize_result(question, res)\n")
        p = Path(d) / f"{name}.py"
        p.write_text(HEADER + body)
        return p

    def test_varying_search_order_with_beta_does_not_buy_an_oracle(self):
        worlds = [world_with_best_under_root(k) for k in range(4)]
        with tempfile.TemporaryDirectory() as d:
            varied = evaluate_policy(self.policy(d, "varied", "int(round(self.beta * 5)) % 4"), worlds, **self.kw)
            fixed = evaluate_policy(self.policy(d, "fixed", "0"), worlds, **self.kw)
            self.assertTrue(varied["ok"] and fixed["ok"], (varied, fixed))
            # neither knows where the good attempt is; varying the order by beta must not look like knowing
            self.assertLessEqual(varied["auc"], fixed["auc"] + 0.05)

    def test_worlds_without_signal_do_not_change_the_ranking(self):
        good = chain_world()
        dead = {"id": "dead", "baseline": 0.0, "nodes": [dict(n, score=None, valid=False) for n in good["nodes"]]}
        kw = dict(W=4, betas=[0.0, 1.0], budget=24, lam=0.25, beta1=0.01, beta2=0.01)
        from tests.test_policy import SEED
        a = evaluate_policy(SEED, [good], **kw)
        b = evaluate_policy(SEED, [good, dead, dead], **kw)
        self.assertAlmostEqual(a["reward"], b["reward"])


class PromptTest(unittest.TestCase):
    def test_dream_prompt_lists_exactly_the_allowed_imports(self):
        prompt = build_prompt(CFG)
        for m in ALLOWED_MODULES:
            self.assertIn(m, prompt)
        for bad in ("functools", "operator", "hasattr"):
            self.assertNotIn(bad + ",", prompt.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
