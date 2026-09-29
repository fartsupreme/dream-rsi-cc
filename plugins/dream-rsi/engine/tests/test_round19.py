"""Round 19: replay artifacts must not buy a deployment.

Found in audit (2026-09-28): on the campaign's four frozen worlds a revision that stops after opening the roots scored
0.6505 against the deployed policy's 0.6272, so the dream step would have deployed it; the one deployment so far
(v0001) changed nothing live, because it acted only when recorded roots ran out. The replay reward now charges only
batches the record can answer, scores the beta that runs live, credits a batch's cells in a fixed order, and a
revision is deployed only on a paired-bootstrap gain that also changes live behaviour. (Port of a patch tested on
synthetic landscapes by the research sweep of 2026-09-28.)"""
import tempfile
import unittest
from pathlib import Path

from drsi.dream import behaviour_differs, deploy_checks, run_dream, split_evolve
from drsi.question import ReplayQuestion
from drsi.replay import evaluate_policy
from drsi.reward import support_penalty
from tests.test_dream import Dev, replace_block, serial_policy
from tests.test_policy import SEED, chain_world

CFG = {"search": {"W": 6, "K1": 4}, "dream": {"M": 1, "betas": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "lambda": 0.25,
                                               "beta1": 0.01, "beta2": 0.01, "bootstrap": 200, "gate_worlds": 16}}
OLD = dict(CFG, dream=dict(CFG["dream"], score="sweep", penalty="realized", bootstrap=0, behaviour_gate=False))
TOPUP = """        if len(batch) < W:
            spare = []
            for cell in question.legal_actions():
                if cell in closed and cell not in batch:
                    hist = by_branch.get(question.meta(cell).branch, [])
                    scores = [o.score for o in hist if o.valid and o.score is not None]
                    spare.append((max(scores) if scores else float("-inf"), -question.meta(cell).branch, cell))
            spare.sort(reverse=True)
            batch.extend(c for _, _, c in spare[:W - len(batch)])
        return batch
"""


def seed_with_topup(src):
    before, block, after = split_evolve(src)
    assert block.count("        return batch\n") == 1
    return before + block.replace("        return batch\n", TOPUP) + after


def stop_after_roots(src):
    before, block, after = split_evolve(src)
    anchor = "        W = question.max_parallelism\n"
    return before + block.replace(anchor, anchor + "        if question.rounds >= 1:\n            return []\n") + after


def short_records():
    """Plateau records shorter than the replay budget (every valid attempt scores about the same), shaped as
    rounds recorded at a smaller width or cut short leave them: 12, 12, 15 and 6 nodes."""
    shapes = [[4, 4, 4], [4, 4, 4], [3, 3, 3, 2, 2, 2], [1] * 6]
    out = []
    for i, shape in enumerate(shapes):
        nodes, k = [], 0
        for d in range(max(shape)):
            for r, length in enumerate(shape):
                if d < length:
                    k += 1
                    valid = (k + i) % 3 != 0
                    nodes.append({"id": f"s{i}r{r}d{d}", "parent": None if d == 0 else f"s{i}r{r}d{d - 1}",
                                  "score": 1.0 + 0.001 * ((7 * k + i) % 5) if valid else None, "valid": valid})
        out.append({"id": f"short{i}", "baseline": 0.0, "nodes": nodes})
    return out


class SupportPenaltyTest(unittest.TestCase):
    def test_full_when_every_answerable_cell_is_probed(self):
        self.assertEqual(support_penalty([(3, 0, 3), (3, 0, 3)], 6), 0.0)

    def test_rounds_the_record_cannot_answer_are_skipped(self):
        self.assertEqual(support_penalty([(6, 0, 6), (6, 6, 0)], 6), 0.0)

    def test_padding_with_dead_cells_earns_nothing(self):
        self.assertAlmostEqual(support_penalty([(6, 3, 6)], 6), 0.5)

    def test_answerable_counts_what_the_record_can_answer(self):
        q = ReplayQuestion(chain_world(2, 2), 6)
        self.assertEqual(q.answerable(q.legal_actions()), 2)
        q.probe_batch(["root:0", "root:1"])
        q.probe_batch(q.legal_actions())
        self.assertEqual(q.answerable(q.legal_actions()), 0)


class DeploySupportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = Path(self.tmp.name) / "policy"
        self.pdir.mkdir()
        self.logs = Path(self.tmp.name) / "logs"
        self.worlds = short_records()

    def tearDown(self):
        self.tmp.cleanup()

    def test_old_rule_deploys_a_stop_after_roots_revision(self):
        (self.pdir / "method.py").write_text(seed_with_topup(SEED.read_text()))
        rep = run_dream(self.pdir, self.worlds, Dev(stop_after_roots), OLD, self.logs)
        self.assertTrue(rep["deployed"], rep["revisions"])

    def test_stop_after_roots_is_not_deployed(self):
        (self.pdir / "method.py").write_text(seed_with_topup(SEED.read_text()))
        rep = run_dream(self.pdir, self.worlds, Dev(stop_after_roots), CFG, self.logs)
        self.assertFalse(rep["deployed"], rep["revisions"])

    def test_old_rule_deploys_a_change_that_only_acts_when_roots_run_out(self):
        (self.pdir / "method.py").write_text(SEED.read_text())
        rep = run_dream(self.pdir, self.worlds, Dev(seed_with_topup), OLD, self.logs)
        self.assertTrue(rep["deployed"], rep["revisions"])

    def test_change_that_only_acts_when_roots_run_out_is_not_deployed(self):
        (self.pdir / "method.py").write_text(SEED.read_text())
        rep = run_dream(self.pdir, self.worlds, Dev(seed_with_topup), dict(CFG, dream=dict(CFG["dream"], bootstrap=0)),
                        self.logs)
        self.assertFalse(rep["deployed"], rep)

    def test_gate_sees_no_live_difference_for_the_topup(self):
        a, b = Path(self.tmp.name) / "a.py", Path(self.tmp.name) / "b.py"
        a.write_text(SEED.read_text())
        b.write_text(seed_with_topup(SEED.read_text()))
        g = behaviour_differs(b, a, 6, 24, n=16)
        self.assertTrue(g["ok"], g)
        self.assertFalse(g["differs"], g)
        c = Path(self.tmp.name) / "c.py"
        c.write_text(stop_after_roots(SEED.read_text()))
        self.assertTrue(behaviour_differs(c, a, 6, 24, n=4)["differs"])

    def test_the_gate_refuses_a_revision_that_changes_nothing_live(self):
        a, b = Path(self.tmp.name) / "a.py", Path(self.tmp.name) / "b.py"
        a.write_text(SEED.read_text())
        b.write_text(seed_with_topup(SEED.read_text()))
        params = {"W": 6, "budget": 24, "lam": 0.25, "beta1": 0.01, "beta2": 0.01, "score": "default",
                  "penalty": "support", "curve": "canonical"}
        out = deploy_checks(b, a, {}, {}, self.worlds, params, {"bootstrap": 0, "gate_worlds": 8})
        self.assertFalse(out["ok"], out)
        self.assertIn("no change in live behaviour", out["why"])
        c = Path(self.tmp.name) / "c.py"  # a live change that spends the whole round passes (round 22: stopping early
        c.write_text(serial_policy())      # does not, since it does less work live)
        self.assertTrue(deploy_checks(c, a, {}, {}, self.worlds, params, {"bootstrap": 0, "gate_worlds": 4})["ok"])

    def test_a_gain_within_resampling_noise_is_refused(self):
        a, b = Path(self.tmp.name) / "a.py", Path(self.tmp.name) / "b.py"
        a.write_text(SEED.read_text())
        b.write_text(seed_with_topup(SEED.read_text()))
        kw = dict(W=6, betas=CFG["dream"]["betas"], budget=24, lam=0.25, beta1=0.01, beta2=0.01)
        ra, rb = evaluate_policy(a, self.worlds, **kw), evaluate_policy(b, self.worlds, **kw)
        params = dict(kw, score="default", penalty="support", curve="canonical")
        out = deploy_checks(b, a, rb, ra, self.worlds, params, {"bootstrap": 100, "gate_worlds": 4})
        self.assertFalse(out["ok"], out)
        self.assertIn("resampling noise", out["why"])

    def test_live_twins_score_the_same(self):
        a, b = Path(self.tmp.name) / "a.py", Path(self.tmp.name) / "b.py"
        a.write_text(SEED.read_text())
        b.write_text(seed_with_topup(SEED.read_text()))
        kw = dict(W=6, betas=CFG["dream"]["betas"], budget=24, lam=0.25, beta1=0.01, beta2=0.01)
        ra, rb = evaluate_policy(a, self.worlds, **kw), evaluate_policy(b, self.worlds, **kw)
        self.assertAlmostEqual(ra["reward"], rb["reward"])

    def test_better_live_policy_still_deploys(self):
        from tests.test_dream import serial_policy, seed_block
        (self.pdir / "method.py").write_text(serial_policy())
        worlds = [chain_world(), chain_world(n_roots=3, depth=8, climb=0.05), chain_world(8, 5, 0.08)]
        rep = run_dream(self.pdir, worlds, Dev(replace_block(seed_block())), CFG, self.logs)
        self.assertTrue(rep["deployed"], rep)


class MinWorldsTest(unittest.TestCase):
    """A dream phase needs enough informative worlds (a valid score and at least one continuation) to separate
    policies; below dream.min_worlds it is skipped and no developer call is made."""

    def setUp(self):
        from drsi.store import Campaign
        from tests.test_live import SCORER
        from tests.test_scorer_workspace import make_repo
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = {"goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
                    "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]},
                    "search": {"W": 2, "K1": 2, "plateau": 3}, "live": {"require_check": False},
                    "llm": {"model": "opus", "classifier_model": "opus", "worker_model": "opus"}}
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def cycles(self, min_worlds, n):
        from drsi.agent import AgentResult
        from drsi.live import run_cycles
        from drsi.store import Campaign
        from tests.test_live import stub_worker
        cfg = dict(self.cfg, dream={"min_worlds": min_worlds})
        camp = Campaign.create(f"t{min_worlds}", cfg, home=self.root / "home")
        calls = []

        def developer(sb, prompt):
            calls.append(prompt)
            return AgentResult(ok=True)
        rep = run_cycles(camp, n, worker_fn=stub_worker(), developer=developer, indexer=lambda ids: None)
        return rep, calls, camp

    def test_below_min_worlds_no_dream_runs(self):
        rep, calls, camp = self.cycles(min_worlds=3, n=2)
        self.assertEqual(calls, [])
        self.assertTrue(all(r["dream"].get("skipped") for r in rep["rounds"]))
        self.assertEqual(list((camp.root / "logs").glob("dream-*.json")), [])

    def test_the_dream_starts_once_enough_worlds_inform_it(self):
        rep, calls, camp = self.cycles(min_worlds=2, n=2)
        self.assertTrue(rep["rounds"][0]["dream"].get("skipped"))
        self.assertFalse(rep["rounds"][1]["dream"].get("skipped"))
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
