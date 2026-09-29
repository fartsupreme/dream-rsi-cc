"""Round 27: findings of the cross-vendor review of round 25 (Grok) that round 26 left open, each reproduced first.

Dream step. Past the record, replay made up a failed attempt and let the policy go on. What it made up could not match
a live failure in every field (no family on a root, the world's usual failure class read off attempts the policy had
not revealed), so a revision that noticed one switched to behaviour it never shows live and deployed (0.8796 against
0.5203); and continuing made-up failures ranked two policies opposite to their live order. Replay cannot know what
work past the record would have found, so it no longer pretends to: the first probe the record cannot answer ends
that run. What the batch's recorded cells reveal counts, the probes past the record cost budget and reveal nothing,
and the rest of the budget counts as empty batches. A policy cannot catch the end (it is not an Exception), and
nothing it does afterwards counts. So a candidate's replay never scores above what the same policy does live on the
same attempts, and the incumbent, compared on its own rounds, never reaches the end.

Novelty check: a citation of an attempt's own id (not the label it was shown under) resolved through the flattened
labels, so the fullwidth "Ａ7" reached "A7", and an id longer than its label reached nothing; an exact id is now
matched before the labels, and the judge is told to cite by the label.
"""
import tempfile
import unittest
from pathlib import Path

from drsi.dream import SEED_POLICY, run_dream
from drsi.novelty import JUDGE_RULES, check
from drsi.question import PolicyQuestion, RecordEnd, ReplayQuestion
from drsi.replay import evaluate_policy
from drsi.reward import live_penalty
from drsi.store import Tree, make_node
from tests.helpers import BEST_FIRST, WORST_FIRST, ScriptedLLM, own_worlds, record, truth_world, with_block
from tests.test_dream import Dev
from tests.test_policy import chain_world
from tests.test_round20 import FAMS
from tests.test_round23 import ALLROOTS

# Grok's revision: open roots until an invalid, scoreless, family-less attempt appears, then deepen valid leaves
ORACLE = ALLROOTS.replace(
    "        if roots:\n            return roots[:W]\n",
    "        seen_end = any((not o.valid) and o.score is None and o.family is None\n"
    "                       for o in question.observed().values())\n"
    "        if roots and not seen_end:\n            return roots[:W]\n")
assert ORACLE != ALLROOTS

# keeps probing after any refusal it can catch
STUBBORN = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        W = question.max_parallelism
        state["asked"] = state.get("asked", 0) + 1
        if state["asked"] > 1:
            try:
                question.probe_batch(question.legal_roots()[:W])
            except Exception:
                pass
        return question.legal_roots()[:W]
    # EVOLVE-BLOCK-END
"""

KW = dict(W=4, betas=[], budget=12, lam=0.25, beta1=0.01, beta2=0.01)


def write(d, name, src) -> Path:
    p = Path(d) / name
    p.write_text(src)
    return p


class EndOfRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_probe_past_the_recorded_roots_ends_the_run(self):
        q = ReplayQuestion(chain_world(2, 2), 4, max_probes=12)
        with self.assertRaises(RecordEnd):
            q.probe_batch(["root:0", "root:1", "root:2", "root:3"])
        self.assertEqual(q.probes, 4)  # the probes past the record cost budget
        self.assertEqual(set(q.observed()), {"r0d0", "r1d0"})  # and reveal nothing
        with self.assertRaises(RecordEnd):
            q.probe_batch(["r0d0"])

    def test_a_leaf_past_its_branch_end_ends_the_run(self):
        q = ReplayQuestion(chain_world(1, 2), 4, max_probes=12)
        q.probe_batch(["root:0"])
        q.probe_batch(["r0d0"])
        with self.assertRaises(RecordEnd):
            q.probe_batch(["r0d1"])
        self.assertEqual(set(q.observed()), {"r0d0", "r0d1"})

    def test_the_view_ends_the_same_way_and_a_policy_cannot_catch_it(self):
        self.assertFalse(issubclass(RecordEnd, Exception))
        v = PolicyQuestion(ReplayQuestion(chain_world(2, 2), 4, max_probes=12))
        with self.assertRaises(RecordEnd):
            v.probe_batch(["root:0", "root:1", "root:2"])
        rep = evaluate_policy(write(self.tmp.name, "stubborn.py", with_block(STUBBORN)(SEED_POLICY.read_text())),
                              [chain_world(2, 2)], **KW)
        self.assertTrue(rep["ok"], rep)
        trace = rep["traces"]["runs"][str(float(rep["default_beta"]))][0]["trace"]
        self.assertEqual(len(trace), 1)  # nothing after the end is traced

    def test_the_rest_of_the_budget_counts_as_empty_batches(self):
        rep = evaluate_policy(SEED_POLICY, [chain_world(2, 2)], **KW)
        row = rep["measured"]["runs"][str(float(rep["default_beta"]))][0]
        self.assertEqual((row["probes"], row["unspent"]), (4, 2))
        self.assertAlmostEqual(rep["parallel_penalty"], live_penalty([4], 4, 2))

    def test_the_oracle_cannot_switch_on_what_replay_made_up(self):
        worlds = [chain_world(3, 6) for _ in range(4)]
        oracle = evaluate_policy(write(self.tmp.name, "oracle.py", with_block(ORACLE)(SEED_POLICY.read_text())),
                                 worlds, **KW)
        twin = evaluate_policy(write(self.tmp.name, "twin.py", with_block(ALLROOTS)(SEED_POLICY.read_text())),
                               worlds, **KW)
        self.assertEqual(oracle["reward"], twin["reward"])

    def test_the_oracle_is_not_deployed_over_the_seed_on_the_seeds_own_rounds(self):
        pdir = Path(self.tmp.name) / "policy"
        pdir.mkdir()
        (pdir / "method.py").write_text(SEED_POLICY.read_text())
        cfg = {"search": {"W": 4, "K1": 6}, "dream": {"M": 1, "betas": [0.0, 0.6, 1.0], "lambda": 0.25,
                                                      "beta1": 0.01, "beta2": 0.01, "bootstrap": 200, "gate_worlds": 8}}
        rep = run_dream(pdir, own_worlds(SEED_POLICY.read_text(), 4, 24, plateau=True), Dev(with_block(ORACLE)), cfg,
                        Path(self.tmp.name) / "logs")
        self.assertIsNone(rep["skipped"], rep)
        self.assertFalse(rep["deployed"], rep["revisions"])

    def test_replay_never_scores_a_policy_above_what_it_does_live_on_the_same_attempts(self):
        # the recorded round is part of the ground truth; the ground truth plays the live round
        rec = write(self.tmp.name, "worst.py", with_block(WORST_FIRST)(SEED_POLICY.read_text()))
        for block in (BEST_FIRST, ALLROOTS, ORACLE, WORST_FIRST):
            pol = write(self.tmp.name, "p.py", with_block(block)(SEED_POLICY.read_text()))
            for i in range(3):
                truth = truth_world(i)
                recorded = record(rec, truth, 2, 8, "iter0001")
                kw = dict(W=2, betas=[], budget=8, lam=0.25, beta1=0.01, beta2=0.01)
                replayed = evaluate_policy(pol, [recorded], **kw)["measured"]["runs"]
                live = evaluate_policy(pol, [truth], **kw)["measured"]["runs"]
                (key,) = replayed
                r_curve = dict(replayed[key][0]["curve"])
                l_curve = dict(live[key][0]["curve"])
                for p, best in r_curve.items():
                    self.assertLessEqual(best if best is not None else float("-inf"),
                                         l_curve.get(p, float("-inf")) if l_curve.get(p) is not None
                                         else float("-inf"), (block[:60], i, p))
                self.assertLessEqual(replayed[key][0]["probes"], live[key][0]["probes"])


class CiteByIdTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Tree(Path(self.tmp.name) / "tree.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, nid, mech, outcome, killed):
        self.t.add(make_node(id=nid, parent=None, proposal=mech,
                             fingerprint={"mechanism": mech, "object": "o", "key_move": "k", "kind": "construction",
                                          "outcome": outcome, "killed_by": killed, "why": "w", "family": "F03"}))

    def judge(self, cite):
        def fn(prompt, schema):
            return {"verdict": "retry", "retry_of": cite, "nearest_ids": [cite], "family": "F03",
                    "what_differs": "fixes the crash", "addresses_stopper": True, "targets_gate": "",
                    "doubts": "", "rationale": "r"}
        return ScriptedLLM(fn)

    def test_an_attempts_own_id_reaches_it_before_any_label(self):
        rows = [("A7", "widget merge variant one", "inconclusive", "an index bug crashed it before any measurement"),
                ("Ａ7", "widget merge variant two", "refuted", "speed")]
        for order in (rows, rows[::-1]):  # whichever of the two is shown first takes the plain label
            with tempfile.TemporaryDirectory() as d:
                self.t = Tree(Path(d) / "tree.jsonl")
                for row in order:
                    self.add(*row)
                # the proposal reads like "A7", so the search shows it first and it takes the plain label "A7"
                r = check(self.t, FAMS, self.judge("Ａ7"), "widget merge variant one, with the crash fixed")
                self.assertEqual([b["id"] for b in r["nearest"]], ["Ａ7"], order[0][0])
                self.assertEqual(r["verdict"], "duplicate")  # a retry of the measured attempt is a repeat

    def test_an_id_longer_than_its_label_is_still_found(self):
        base = "x" * 76
        self.add(base + "one", "gear train one", "refuted", "speed")
        self.add(base + "two", "gear train two", "inconclusive", "an index bug crashed it before any measurement")
        r = check(self.t, FAMS, self.judge(base + "two"), "gear train two, with the crash fixed")
        self.assertEqual([b["id"] for b in r["nearest"]], [base + "two"])
        self.assertEqual((r["verdict"], r["retry_of"]), ("retry", base + "two"))

    def test_the_judge_is_told_to_cite_by_label(self):
        self.assertIn("label", JUDGE_RULES)


if __name__ == "__main__":
    unittest.main()
