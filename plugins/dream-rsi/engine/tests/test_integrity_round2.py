"""Second-review integrity fixes (grok-4.7 findings, each checked against the code first)."""
import json
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.dream import run_dream, split_evolve
from drsi.families import assign_families
from drsi.guard import check_policy_source
from drsi.importer import import_jsonl
from drsi.novelty import check
from drsi.question import IllegalBatch, ReplayQuestion
from drsi.replay import evaluate_policy, world_best
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM
from tests.test_dream import CFG, Dev, replace_block, seed_block
from tests.test_policy import SEED, chain_world
from tests.test_replay import HEADER

KW = dict(W=4, betas=[0.0, 0.5, 1.0], budget=24, lam=0.25, beta1=0.01, beta2=0.01)


class FreshPolicyPerRunTest(unittest.TestCase):
    def test_module_state_does_not_carry_between_runs(self):
        src = ("RUNS = []\n" + HEADER +
               "    def solve(self, question, budget=None):\n"
               "        question.reset()\n        res = SimResult()\n        RUNS.append(1)\n"
               "        n = min(len(RUNS), question.max_parallelism)\n"
               "        question.probe_batch(question.legal_roots()[:n], on_reveal=lambda _: _record_curve(res, question))\n"
               "        return finalize_result(question, res)\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.py"
            p.write_text(src)
            rep = evaluate_policy(p, [chain_world(), chain_world(3, 4)], **KW)
            self.assertTrue(rep["ok"], rep)
            works = {round(r["work"], 6) for r in rep["per_beta"].values()}
            self.assertEqual(len(works), 1)  # every run started from a fresh module


class GuardRound2Test(unittest.TestCase):
    def test_extra_classes_and_dunder_methods_rejected(self):
        extra = "class Evil(str):\n    pass\n" + HEADER + "    def solve(self, question, budget=None):\n        return None\n"
        self.assertTrue(check_policy_source(extra))
        dunder = HEADER + "    def __getattribute__(self, n):\n        return 1\n    def solve(self, question, budget=None):\n        return None\n"
        self.assertTrue(check_policy_source(dunder))
        fmt = HEADER + "    def solve(self, question, budget=None):\n        return format(question, '')\n"
        self.assertTrue(check_policy_source(fmt))

    def test_init_is_still_allowed(self):
        ok = HEADER + ("    def __init__(self, beta=None):\n        super().__init__(beta)\n"
                       "    def solve(self, question, budget=None):\n        return None\n")
        self.assertEqual(check_policy_source(ok), [])

    def test_cells_must_be_plain_strings(self):
        class Sneaky(str):
            pass
        q = ReplayQuestion(chain_world(1, 2), 2)
        with self.assertRaises(IllegalBatch):
            q.probe_batch([Sneaky("root:0")])


class BudgetTest(unittest.TestCase):
    def test_no_new_batch_once_the_budget_is_spent(self):
        q = ReplayQuestion(chain_world(4, 4), 4, max_probes=4)
        q.probe_batch(["root:0", "root:1", "root:2", "root:3"])
        with self.assertRaises(IllegalBatch):
            q.probe_batch(["root:0"] if False else [q.legal_actions()[0]])

    def test_evolve_block_may_not_redefine_solve(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(SEED.read_text())
            evil = seed_block().replace("    # EVOLVE-BLOCK-END",
                                        "    def solve(self, question, budget=None):\n        return None\n    # EVOLVE-BLOCK-END")
            rep = run_dream(pdir, [chain_world()], Dev(replace_block(evil)), dict(CFG, dream=dict(CFG["dream"], M=1)),
                            Path(d) / "logs")
            self.assertEqual(rep["revisions"][0]["stage"], "scope")


class ReachableWorldTest(unittest.TestCase):
    def test_world_best_ignores_attempts_replay_can_never_reveal(self):
        world = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "a", "parent": None, "score": 0.2}, {"id": "a1", "parent": "a", "score": 0.3},
            {"id": "a2", "parent": "a", "score": 0.9}]}  # a2 is a second child: unreachable
        self.assertEqual(world_best(world), 0.3)


class MergeTest(unittest.TestCase):
    def test_family_assignment_does_not_clobber_a_newer_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, fingerprint={"mechanism": "old", "outcome": "refuted"}))
            stale = Tree(t.path)
            Tree(t.path).update("1", fingerprint={"mechanism": "new", "outcome": "partial"})
            assign_families(stale, {"families": [{"id": "F01", "name": "n", "description": "", "boundary": ""}]},
                            ScriptedLLM(lambda p, s: {"items": [{"id": "1", "family": "F01"}]}))
            fp = Tree(t.path).get("1")["fingerprint"]
            self.assertEqual((fp["mechanism"], fp["family"]), ("new", "F01"))


class RecallTest(unittest.TestCase):
    def test_proposal_without_search_hits_still_shows_recent_history(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, proposal="alpha", fingerprint={"mechanism": "alpha sieve"}))
            llm = ScriptedLLM(lambda p, s: {"verdict": "novel", "family": "F00", "nearest_ids": [], "what_differs": "",
                                            "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": ""})
            check(t, {"families": []}, llm, "the of and")
            self.assertIn("alpha sieve", llm.prompts[0])


class ImportOrderTest(unittest.TestCase):
    def test_generic_parent_that_appears_later_is_linked(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "l.jsonl"
            src.write_text(json.dumps({"key": "b", "from": "a", "idea": "child"}) + "\n" +
                           json.dumps({"key": "a", "idea": "parent"}) + "\n")
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, src, preset="generic", field_map={"id": "key", "parent": "from", "proposal": "idea"})
            self.assertEqual(t.get("b")["parent"], "a")


if __name__ == "__main__":
    unittest.main()
