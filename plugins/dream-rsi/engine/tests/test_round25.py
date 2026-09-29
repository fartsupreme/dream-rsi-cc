"""Round 25: findings of the fourth review (Opus, on round 24), each reproduced here first.

Dream step:
- Replay still differed from live in what a policy can see. A probe past the record came back as nothing: it cost a
  probe but added no attempt (so probes exceeded the attempts observed), and a leaf past its branch end stopped being
  an action. Live, every probe is an attempt. A revision that opened roots until probes and attempts parted, then
  continued its best leaves, outscored a policy with the same live batches (0.8535 against 0.7331) and deployed under
  the default checks. Under a budget a probe past the record now reveals a failed attempt, as live shows one: no
  score, the tree's usual failure class, and a leaf that can be continued like any other.
- The question gave policies evaluator-only state (answerable(), empty_counts); a policy now sees only what a live
  question has.
- Campaigns created with dream.penalty "support" kept ranking by it, and that penalty (like "realized") paid for
  gains that do not exist live. Both are gone: such a campaign ranks by "live", with a warning. `drsi replay` ranks
  as the dream step does.
- The clock curve skipped batches the record could not answer; every batch is a live batch.
- evaluate_policy with no budget capped the roots again, where stopping after the roots paid; a budget is required.
(Round 27 replaced the made-up failed attempt past the record with the end of the run: the tests of it
moved there.)
Novelty check:
- flattened display labels could collide (a width variant, a format character, a long id), so a citation reached
  another record; every shown attempt and in-flight proposal now has its own label, mapped back to its own id;
- a difference made only of blank filler characters counted as a stated difference.
"""
import io
import random
import re
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from drsi import cli
from drsi.dream import SEED_POLICY, _params, run_dream, split_evolve
from drsi.live import LiveQuestion
from drsi.novelty import check
from drsi.question import ReplayQuestion
from drsi.replay import evaluate_policy
from drsi.store import DEFAULT_CONFIG, Tree, make_node
from tests.helpers import ScriptedLLM, own_worlds
from tests.test_dream import Dev
from tests.test_policy import chain_world
from tests.test_round20 import FAMS, tree
from tests.test_round22 import CFG, KW, confirm_capture
from tests.test_round23 import ALLROOTS, all_roots
from tests.test_round24 import chain_worlds, many_root_worlds

# the fourth review's revision: the all-roots block, but it continues its best leaves once probes outnumber the
# attempts it has seen, which in replay meant a probe had gone past the record
DETECTOR = ALLROOTS.replace("        if roots:\n            return roots[:W]\n",
                            "        if roots and question.probes == len(question.observed()):\n"
                            "            return roots[:W]\n")
assert DETECTOR != ALLROOTS


def detector(src):
    before, _, after = split_evolve(src)
    return before + DETECTOR + after


def write(d, name, src) -> Path:
    p = Path(d) / name
    p.write_text(src)
    return p


class ReplayAsLiveTest(unittest.TestCase):
    def test_the_question_shows_a_policy_only_what_a_live_question_has(self):
        def public(cls):
            return {n for n in dir(cls) if not n.startswith("_")}
        self.assertEqual(public(ReplayQuestion), public(LiveQuestion))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.worlds = chain_worlds() + many_root_worlds(n=2)
        self.kw = KW | {"budget": 24}

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_revision_that_waits_for_the_record_to_run_out_scores_as_its_live_twin(self):
        det = evaluate_policy(write(self.tmp.name, "det.py", detector(SEED_POLICY.read_text())), self.worlds,
                              **self.kw)
        twin = evaluate_policy(write(self.tmp.name, "twin.py", all_roots(SEED_POLICY.read_text())), self.worlds,
                               **self.kw)
        self.assertTrue(det["ok"] and twin["ok"], (det, twin))
        self.assertAlmostEqual(det["reward"], twin["reward"], places=9)

    def test_it_is_not_deployed_under_the_default_checks(self):
        pdir = Path(self.tmp.name) / "policy"
        pdir.mkdir()
        (pdir / "method.py").write_text(SEED_POLICY.read_text())
        cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"]) | {"M": 1}}
        rep = run_dream(pdir, own_worlds(SEED_POLICY.read_text(), 4, 24), Dev(detector), cfg, Path(self.tmp.name) / "logs")
        self.assertIsNone(rep["skipped"], rep)
        self.assertFalse(rep["deployed"], rep["revisions"])

    def test_the_clock_advances_every_batch(self):
        rep = evaluate_policy(SEED_POLICY, [chain_world(2, 2)], **self.kw)
        row = rep["measured"]["runs"][str(float(rep["default_beta"]))][0]
        self.assertEqual([t for t, _ in row["curve_clock"]], list(range(1, row["rounds"] + 1)))

    def test_a_budget_is_required(self):
        with self.assertRaises(ValueError):
            evaluate_policy(SEED_POLICY, [chain_world(2, 2)], **(self.kw | {"budget": None}))


class PenaltyTest(unittest.TestCase):
    def test_an_old_penalty_in_a_campaign_ranks_live_and_warns(self):
        for old in ("support", "realized"):
            cfg = {"search": dict(CFG["search"]), "dream": dict(CFG["dream"]) | {"penalty": old}}
            self.assertEqual(_params(cfg)["penalty"], "live")
            with tempfile.TemporaryDirectory() as d:
                pdir = Path(d) / "policy"
                pdir.mkdir()
                (pdir / "method.py").write_text(SEED_POLICY.read_text())
                rep = run_dream(pdir, chain_worlds(), Dev(), cfg | {"dream": cfg["dream"] | {"M": 0}},
                                Path(d) / "logs")
            self.assertTrue(any("dream.penalty" in w for w in rep["warnings"]), rep.get("warnings"))

    def test_the_ranking_refuses_an_old_penalty(self):
        for old in ("support", "realized"):
            with self.assertRaises(ValueError):
                evaluate_policy(SEED_POLICY, chain_worlds(), **KW, penalty=old)

    def test_drsi_replay_ranks_as_the_dream_step(self):
        cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"]) | {"score": "sweep",
                                                                                       "curve": "batch"}}
        camp = mock.Mock(config=cfg)
        rep = {"ok": True, "reward": 0.5, "auc": 0.6, "parallel_penalty": 0.4, "per_beta": {}, "default_beta": 0.6,
               "measured": {"runs": {"0.6": [{"off_record": 0}]}}}
        with mock.patch.object(cli, "resolve_campaign", return_value=camp), \
                mock.patch.object(cli, "_worlds", return_value=[chain_world()]), \
                mock.patch.object(cli, "_policy_path", return_value=SEED_POLICY), \
                mock.patch.object(cli, "evaluate_policy", return_value=rep) as ev, redirect_stdout(io.StringIO()):
            cli.cmd_replay(Namespace(campaign="x", history=False, policy=None))
        self.assertEqual([c.kwargs for c in ev.call_args_list], [_params(cfg)] * ev.call_count)


class LabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_colliding_in_flight_labels_stay_distinct(self):
        t = tree(self.tmp.name)
        pending = [{"node": "iter0001-002", "ticket": "t", "proposal": "a merge of buckets by stride"},
                   {"node": "iter0001-002​", "ticket": "u", "proposal": "an unrelated widget cache"}]
        s = confirm_capture({"verdict": "duplicate", "nearest_ids": ["pending:iter0001-002"]}, {})
        check(t, FAMS, s, "buckets merged by their stride", pending=pending, confirm=True)
        judge = s.prompts[0]
        labels = re.findall(r"^(pending:\S+): ", judge.split("IN-FLIGHT PROPOSALS", 1)[1], re.M)
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(set(labels)), 2)
        cited = s.confirm_prompts[0].split("CITED RECORDS\n", 1)[1].split("OTHER ATTEMPTS THE FIRST JUDGE", 1)[0]
        self.assertIn("a merge of buckets by stride", cited)
        self.assertNotIn("an unrelated widget cache", cited)

    def test_a_citation_reaches_the_attempt_it_names_when_ids_fold_alike(self):
        # "A7" and fullwidth "Ａ7" both flattened to "A7": a retry the judge aimed at the measured twin was granted
        # on the unmeasured one
        t = Tree(Path(self.tmp.name) / "tree.jsonl")
        for nid, outcome, killed in (("A7", "inconclusive", "an index bug crashed it before any measurement"),
                                     ("Ａ7", "refuted", "speed")):
            mech = f"widget merge variant {'one' if nid == 'A7' else 'two'}"
            t.add(make_node(id=nid, parent=None, proposal=mech,
                            fingerprint={"mechanism": mech, "object": "o", "key_move": "k", "kind": "construction",
                                         "outcome": outcome, "killed_by": killed, "why": "w", "family": "F03"}))

        def fn(prompt, schema):
            line = next(ln for ln in prompt.splitlines() if "variant two" in ln)
            label = re.match(r"#(\S+)", line).group(1)
            return {"verdict": "retry", "retry_of": label, "nearest_ids": [label], "family": "F03",
                    "what_differs": "fixes the crash", "addresses_stopper": True, "targets_gate": "",
                    "doubts": "", "rationale": "r"}
        r = check(t, FAMS, ScriptedLLM(fn), "widget merge variant two, with the crash fixed")
        self.assertEqual(r["verdict"], "duplicate", r)  # a retry of the measured attempt is a repeat
        self.assertEqual([b["id"] for b in r["nearest"]], ["Ａ7"])


class BlankDifferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = tree(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_filler_only_difference_does_not_overturn_a_duplicate(self):
        for blank in ("ㅤㅤ", "ﾠ ᅟ", "⠀", "​ᅠ"):
            s = confirm_capture({"verdict": "duplicate", "nearest_ids": ["3"]},
                                {"verdict": "novel", "what_differs": blank})
            r = check(self.t, FAMS, s, "radix bucket merge with cache line density, merged again", confirm=True)
            self.assertEqual(r["verdict"], "duplicate", repr(blank))

    def test_a_filler_only_difference_is_no_variant(self):
        for blank in ("ㅤ", "​", "⠀⠀"):
            s = confirm_capture({"verdict": "variant", "nearest_ids": ["3"], "what_differs": blank,
                                 "addresses_stopper": True}, {})
            r = check(self.t, FAMS, s, "radix bucket merge with cache line density, merged again")
            self.assertEqual(r["verdict"], "duplicate", repr(blank))


if __name__ == "__main__":
    unittest.main()
