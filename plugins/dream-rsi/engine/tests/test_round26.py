"""Round 26: findings of the final review before 0.4.0 (Opus, on round 25), each reproduced here first.

Dream step:
- A policy could still tell replay, the behaviour gate and a live round apart: the ids it was shown (recorded ids in
  completion order, ids made past the record, gate-tree ids, against live's round-NNN in reveal order), str() of the
  question (TracingQuestion against RemoteQuestion), and `self.beta is OptimalPolicy.beta` (replay passed a float,
  live passed nothing). Revisions keyed to any of them deployed under the default checks. A policy is now handed one
  view in every environment: attempts are shown under ids that say only when they were revealed ("a1", "a2", ...),
  the view prints as "<question>", and the policy is built with no argument at its default beta everywhere; the gate's
  trees share the campaign's baseline.
- Replay treats unrecorded work as failure, so it favoured policies that resemble whichever policy recorded the
  worlds: a patient incumbent was replaced by the recorder's twin. The incumbent's replay is exact only where it stays
  on the record, so the dream now compares on those worlds alone (every world an incumbent recorded itself, once the
  root slots map as they did live); a candidate's unrecorded probes still count as failures. dream.min_worlds counts
  those worlds, in `drsi dream` as in the loop, and the skip reason is printed.
- Recorded roots are replayed in the slots they were opened in live (their recorded cell), not in completion order.
- Unknown dream.score or dream.curve values ranked by a fallback silently; they now rank by the defaults and warn.
- The canonical curve credited attempts made past the record among recorded ones by id, so renaming a world moved
  rewards; they are credited last.
Round 19's "a better live policy still deploys" (the seed over a serial incumbent) is gone: a serial policy opens a
fresh root every batch live, so its own rounds hold no continuation and cannot inform a dream; the positive case
is OnRecordTest's, a revision the incumbent's own record vouches for.
Novelty check: ids starting with "#" or "[" (or ending with "]") could not be cited as themselves.
Prune: `--ids ","` crashed.
"""
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from drsi import cli
from drsi.dream import SEED_POLICY, _params, behaviour_differs, config_warnings, run_dream, split_evolve
from drsi.live import LiveRunner, live_round, load_policy
from drsi.novelty import check
from drsi.question import PolicyQuestion, ReplayQuestion
from drsi.replay import evaluate_policy
from drsi.store import Campaign, Tree, make_node
from tests.helpers import BEST_FIRST, WORST_FIRST, ScriptedLLM, record, truth_world, with_block
from tests.test_dream import Dev
from tests.test_live import SCORER, stub_worker
from tests.test_round20 import FAMS
from tests.test_round23 import ALLROOTS
from tests.test_scorer_workspace import make_repo

# All roots while the view is the one every environment shares; otherwise continue the best leaves.
PROBE = ALLROOTS.replace(
    "        if roots:\n            return roots[:W]\n",
    "        same = (str(question) == \"<question>\" and self.beta is OptimalPolicy.beta\n"
    "                and all(i.startswith(\"a\") for i in question.observed()))\n"
    "        if roots and same:\n            return roots[:W]\n")
assert PROBE != ALLROOTS


def write(d, name, src) -> Path:
    p = Path(d) / name
    p.write_text(src)
    return p


CFG = {"search": {"W": 2, "K1": 4}, "dream": {"M": 1, "betas": [0.0, 1.0], "lambda": 0.25, "beta1": 0.01,
                                              "beta2": 0.01, "bootstrap": 200, "gate_worlds": 8, "min_worlds": 4}}


class OneViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.probe = write(self.root, "probe.py", with_block(PROBE)(SEED_POLICY.read_text()))
        self.twin = write(self.root, "twin.py", with_block(ALLROOTS)(SEED_POLICY.read_text()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_view_prints_the_same_and_shows_reveal_order_ids(self):
        world = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "zeta", "parent": None, "score": 0.5, "valid": True},
            {"id": "alpha", "parent": None, "score": 0.1, "valid": True},
            {"id": "mid", "parent": "zeta", "score": 0.7, "valid": True}]}
        v = PolicyQuestion(ReplayQuestion(world, 2, max_probes=6))
        self.assertEqual(str(v), "<question>")
        self.assertEqual(repr(v), "<question>")
        self.assertNotIn("Replay", str(v.probe_batch))
        out = v.probe_batch(["root:0", "root:1"])
        self.assertEqual([o.id for o in out], ["a1", "a2"])
        self.assertEqual(sorted(v.legal_actions()), sorted(["root:2", "root:3", "a1", "a2"]))
        child = v.probe_batch(["a1"])[0]
        self.assertEqual((child.id, child.parent_id, child.score), ("a3", "a1", 0.7))
        self.assertEqual(v.meta("a3").parent_id, "a1")
        self.assertEqual(set(v.observed()), {"a1", "a2", "a3"})

    def test_the_view_refuses_what_the_question_refuses_without_naming_recorded_ids(self):
        world = {"id": "w", "baseline": 0.0, "nodes": [{"id": "secret-7", "parent": None, "score": 0.5}]}
        v = PolicyQuestion(ReplayQuestion(world, 2, max_probes=6))
        v.probe_batch(["root:0"])
        for bad in (["a1", "a1"], ["nope"], [], ["root:1", "root:2", "root:3"]):
            with self.assertRaises(ValueError) as e:
                v.probe_batch(bad)
            self.assertNotIn("secret", str(e.exception))

    def test_replay_sees_what_live_sees(self):
        worlds = [truth_world(0), truth_world(1)]
        # the shipped sweep holds the default beta itself, read from JSON: a float that is not the class's own
        kw = dict(W=2, betas=json.loads("[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]"), budget=8, lam=0.25, beta1=0.01, beta2=0.01)
        probe, twin = evaluate_policy(self.probe, worlds, **kw), evaluate_policy(self.twin, worlds, **kw)
        self.assertTrue(probe["ok"] and twin["ok"], (probe, twin))
        key = str(float(probe["default_beta"]))
        self.assertEqual(probe["traces"]["runs"][key], twin["traces"]["runs"][key])

    def test_the_gate_sees_what_live_sees(self):
        g = behaviour_differs(self.probe, self.twin, W=2, budget=8, n=4)
        self.assertTrue(g["ok"], g)
        self.assertFalse(g["differs"], g)

    def test_a_live_round_sees_what_replay_sees(self):
        repo = make_repo(self.root)
        camp = Campaign.create("toy", {
            "goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
            "workspace": {"repo": str(repo), "mutable": ["value.txt"]},
            "search": {"W": 2, "K1": 2, "plateau": 3}, "live": {"require_check": False}}, home=self.root / "home")
        runner = LiveRunner(camp, stub_worker(), indexer=lambda ids: None, round_id="iter0001")
        live_round(camp, load_policy(self.probe), runner)
        cells = [camp.tree.get(i)["ext"].get("cell") for i in runner.ids]
        self.assertEqual(cells, ["root:0", "root:1", "root:2", "root:3"])


class OnRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        seed = SEED_POLICY.read_text()
        self.worst = write(self.root, "worst.py", with_block(WORST_FIRST)(seed))
        self.worlds = [record(self.worst, truth_world(i), 2, 8, f"iter{i:04d}") for i in range(8)]

    def tearDown(self):
        self.tmp.cleanup()

    def dream(self, incumbent_src, dev, worlds=None, cfg=None):
        pdir = self.root / "policy"
        pdir.mkdir(exist_ok=True)
        (pdir / "method.py").write_text(incumbent_src)
        return run_dream(pdir, worlds or self.worlds, dev, cfg or CFG, self.root / "logs")

    def test_an_incumbent_replays_its_own_rounds_without_leaving_the_record(self):
        rep = evaluate_policy(self.worst, self.worlds, W=2, betas=[], budget=8, lam=0.25, beta1=0.01, beta2=0.01)
        rows = rep["measured"]["runs"][str(float(rep["default_beta"]))]
        self.assertEqual([r["off_record"] for r in rows], [0] * 8)

    def test_recorded_roots_open_in_the_slots_they_were_opened_in(self):
        w = self.worlds[0]
        q = ReplayQuestion(w, 2, max_probes=8)
        for j in range(4):
            node = next(n for n in w["nodes"] if n.get("cell") == f"root:{j}")
            self.assertEqual(q.probe_batch([f"root:{j}"])[0].id, node["id"])

    def test_a_candidate_that_reaches_the_recorded_best_sooner_deploys(self):
        rep = self.dream(self.worst.read_text(), Dev(with_block(BEST_FIRST)))
        self.assertTrue(rep["deployed"], (rep["revisions"], rep.get("deploy_checks"), rep.get("skipped")))
        self.assertEqual(rep["worlds_on_record"], 8)

    def test_a_patient_incumbent_is_not_replaced_on_worlds_it_did_not_record(self):
        # the review's case: worlds recorded by a patience-1 policy, a patient incumbent, the recorder's twin proposed
        seed, line = SEED_POLICY.read_text(), "patience = 1 + int(round(self.beta * 4))"
        recorder = write(self.root, "impatient.py", seed.replace(line, "patience = 1"))
        patient = seed.replace(line, "patience = 5")
        worlds = [record(recorder, truth_world(i, plateau=True), 2, 16, f"iter{i:04d}") for i in range(8)]
        cfg = CFG | {"search": {"W": 2, "K1": 8}}
        rep = self.dream(patient, Dev(lambda src: recorder.read_text()), worlds=worlds, cfg=cfg)
        self.assertFalse(rep["deployed"], rep)
        self.assertIn("on the record", rep["skipped"] or "", rep)

    def test_drsi_dream_respects_min_worlds_and_says_why(self):
        camp = type("Camp", (), {})()
        camp.root = self.root
        camp.config = CFG | {"dream": CFG["dream"] | {"min_worlds": 9}}
        (self.root / "policy").mkdir(exist_ok=True)
        (self.root / "policy" / "method.py").write_text(self.worst.read_text())
        dev = Dev(with_block(BEST_FIRST))
        out = io.StringIO()
        from unittest import mock
        with mock.patch.object(cli, "resolve_campaign", return_value=camp), \
                mock.patch.object(cli, "_worlds", return_value=self.worlds), \
                mock.patch.object(cli, "make_developer", return_value=dev), redirect_stdout(out):
            cli.cmd_dream(Namespace(campaign="x", history=False))
        self.assertEqual(dev.prompts, [])
        self.assertIn("min_worlds", out.getvalue())


class ConfigTest(unittest.TestCase):
    def test_unknown_score_or_curve_ranks_by_the_defaults_and_warns(self):
        cfg = {"search": {"W": 2, "K1": 4}, "dream": CFG["dream"] | {"score": "sweeep", "curve": "canonicle"}}
        p = _params(cfg)
        self.assertEqual((p["score"], p["curve"]), ("default", "canonical"))
        ws = " ".join(config_warnings(cfg))
        self.assertIn("dream.score", ws)
        self.assertIn("dream.curve", ws)

    def test_the_ranking_refuses_an_unknown_score_or_curve(self):
        w = [truth_world(0)]
        kw = dict(W=2, betas=[], budget=8, lam=0.25, beta1=0.01, beta2=0.01)
        for bad in ({"score": "sweeep"}, {"curve": "canonicle"}):
            with self.assertRaises(ValueError):
                evaluate_policy(SEED_POLICY, w, **kw, **bad)

    def test_renaming_a_world_does_not_move_its_reward(self):
        w = {"id": "aaa", "baseline": 0.0, "nodes": [
            {"id": "m1", "parent": None, "score": 0.2, "valid": True},
            {"id": "m2", "parent": "m1", "score": 0.9, "valid": True}]}
        kw = dict(W=2, betas=[], budget=8, lam=0.25, beta1=0.01, beta2=0.01)
        a = evaluate_policy(SEED_POLICY, [w], **kw)["reward"]
        b = evaluate_policy(SEED_POLICY, [dict(w, id="zzz")], **kw)["reward"]
        self.assertEqual(a, b)


class LabelEdgeTest(unittest.TestCase):
    def test_ids_with_hash_or_brackets_are_cited_as_themselves(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "tree.jsonl")
            for nid, outcome, killed in (("3", "inconclusive", "an index bug crashed it before any measurement"),
                                         ("#3", "refuted", "speed"), ("[x]", "refuted", "speed")):
                mech = {"3": "gear train one", "#3": "gear train two", "[x]": "gear train three"}[nid]
                t.add(make_node(id=nid, parent=None, proposal=mech,
                                fingerprint={"mechanism": mech, "object": "o", "key_move": "k", "kind": "construction",
                                             "outcome": outcome, "killed_by": killed, "why": "w", "family": "F03"}))

            def judge_for(needle):
                def fn(prompt, schema):
                    line = next(ln for ln in prompt.splitlines() if needle in ln and ln.startswith("#"))
                    label = line.split()[0]
                    return {"verdict": "retry", "retry_of": label, "nearest_ids": [label], "family": "F03",
                            "what_differs": "fixes the crash", "addresses_stopper": True, "targets_gate": "",
                            "doubts": "", "rationale": "r"}
                return fn
            for needle, nid in (("gear train two", "#3"), ("gear train three", "[x]"), ("gear train one", "3")):
                r = check(t, FAMS, ScriptedLLM(judge_for(needle)), f"{needle}, with the crash fixed")
                self.assertEqual([b["id"] for b in r["nearest"]], [nid], needle)
                self.assertEqual(r["verdict"], "retry" if nid == "3" else "duplicate", needle)


class PruneArgsTest(unittest.TestCase):
    def test_an_empty_id_list_is_an_error_not_a_crash(self):
        from contextlib import redirect_stderr
        from unittest import mock
        err = io.StringIO()
        with mock.patch.object(cli, "resolve_campaign", return_value=object()), redirect_stderr(err):
            rc = cli.cmd_prune(Namespace(campaign="x", ids=" , ", error_match=None, dry_run=True, reason=""))
        self.assertEqual(rc, 2)
        self.assertIn("--ids", err.getvalue())


if __name__ == "__main__":
    unittest.main()
