"""Guarantees the review found untested."""
import tempfile
import unittest
from pathlib import Path

from drsi.families import build_taxonomy, load_families
from drsi.live import load_policy
from drsi.replay import evaluate_policy
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM
from tests.test_policy import chain_world
from tests.test_replay import HEADER, ROOTS_ONLY


class GapTest(unittest.TestCase):
    def test_load_policy_refuses_guard_failures(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.py"
            p.write_text(HEADER + "    def solve(self, question, budget=None):\n        import os\n")
            with self.assertRaises(ValueError):
                load_policy(p)

    def test_reported_metrics_come_from_the_question_not_the_return_value(self):
        liar = ROOTS_ONLY.replace("        return finalize_result(question, res)",
                                  "        return {'probes': 0, 'best': 1e9, 'batch_sizes': [4] * 99, 'curve': [(0, 1e9)]}")
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "honest.py", Path(d) / "liar.py"
            a.write_text(HEADER + ROOTS_ONLY)
            b.write_text(HEADER + liar)
            kw = dict(W=4, betas=[0.0, 1.0], budget=24, lam=0.25, beta1=0.01, beta2=0.01)
            ra, rb = evaluate_policy(a, [chain_world()], **kw), evaluate_policy(b, [chain_world()], **kw)
            self.assertAlmostEqual(ra["reward"], rb["reward"])

    def test_every_swept_beta_contributes_to_the_auc(self):
        # the policy stops after beta-dependent work; a sweep over two betas beats either beta alone
        body = ("    def solve(self, question, budget=None):\n        question.reset()\n        res = SimResult()\n"
                "        limit = 1 if self.beta < 0.5 else 6\n"
                "        while not _budget_done(question, budget) and question.probes < limit:\n"
                "            acts = [a for a in question.legal_actions() if not a.startswith('root:')] or question.legal_roots()\n"
                "            question.probe_batch(acts[:1], on_reveal=lambda _: _record_curve(res, question))\n"
                "        return finalize_result(question, res)\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.py"
            p.write_text(HEADER + body)
            w = [chain_world(1, 6)]
            base = dict(W=1, budget=24, lam=0.0, beta1=0.01, beta2=0.01)
            both = evaluate_policy(p, w, betas=[0.0, 1.0], **base)
            only_low = evaluate_policy(p, w, betas=[0.0], **base)
            self.assertGreater(both["auc"], only_low["auc"])

    def test_update_many_refuses_to_change_id_or_parent(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None))
            for bad in ({"id": "2"}, {"parent": "x"}):
                with self.assertRaises(TypeError):
                    t.update_many({"1": bad})

    def test_rebuild_clears_old_family_ids(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            for i in range(1, 7):
                t.add(make_node(id=str(i), parent=None,
                                fingerprint={"mechanism": f"m{i}", "outcome": "refuted", "family": "F09"}))
            from drsi.families import rebuild_families
            llm = ScriptedLLM(lambda p, s: {"families": [{"id": "F01", "name": "n", "description": "d", "boundary": "b"}]}
                              if "families" in s.get("properties", {}) else {"items": []})
            rebuild_families(t, llm, "g", Path(d) / "f.json")
            fams = {(n.get("fingerprint") or {}).get("family") for n in Tree(t.path).nodes()}
            self.assertNotIn("F09", fams)  # stale ids from the discarded taxonomy are gone


if __name__ == "__main__":
    unittest.main()
