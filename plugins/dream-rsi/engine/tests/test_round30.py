"""Round 30: findings of the final review before 0.4.0 (Opus, on the round-29 tip), each reproduced here first.

- A live round numbers the attempts of a batch in the order the policy lists them, and the canonical curve credited a
  batch by id: live, in the policy's own listing order; replayed, in the incumbent's. A revision that listed its
  batches worst first and continued its first roots sooner replayed at 0.713 against the incumbent's 0.695, and ran at
  0.622 on the same 24 attempts. A batch is now credited by score, worst first and best last, the same way everywhere:
  listing order earns nothing, live or replayed. The lower bound still holds, since the probes the record cannot
  answer move each recorded cell of the ending batch at most that many places.
- One deadline covered every run of an evaluation, so a policy taking 1.2 s a world passed at 120 worlds and failed at
  300, and then no dream ran again. Each run has its own timeout again; the first failure still ends the evaluation.
- A world's best score and size counted every root and each first child; they now count what replay can reveal (a root
  through its slot, a child opened from its parent), so a world whose roots have no slot informs nothing.
- A child re-parented by a prune (opened from another attempt) was still revealed from its new parent.
- An interrupt during the run that reads the policy's default beta left that process running.
- `drsi replay --history` counted the history world among the worlds the dream compares on.
- dream.min_worlds of 0 with no world recorded live crashed the report.
"""
import io
import os
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from drsi.dream import SEED_POLICY, run_dream
from drsi.question import RecordEnd, ReplayQuestion
from drsi.replay import ENGINE_DIR, evaluate_policy, reachable, world_best
from drsi.store import DEFAULT_CONFIG
from drsi.worlds import informative
from tests.helpers import record, truth_world, with_block
from tests.test_dream import Dev
from tests.test_policy import chain_world
from tests.test_round28 import our_runners, write

SEED = SEED_POLICY.read_text()
REVERSED = SEED.replace("        return batch\n    # EVOLVE-BLOCK-END", "        return list(reversed(batch))\n    # EVOLVE-BLOCK-END")
assert REVERSED != SEED
KW = dict(W=4, betas=[], budget=24, lam=0.25, beta1=0.01, beta2=0.01)

# Opus's pair: the incumbent opens eight roots, then continues its best leaves; the revision continues its first four
# roots before opening the next four, and lists every batch worst first. Both reveal the same 24 attempts.
INC = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        W = question.max_parallelism
        obs = question.observed()
        roots = question.legal_roots()
        nroots = len([o for o in obs.values() if o.parent_id is None])
        if nroots < 2 * W:
            return roots[:W]
        leaves = [c for c in question.legal_actions() if not c.startswith("root:")]
        leaves.sort(key=lambda c: (obs[c].score if obs[c].valid and obs[c].score is not None else -1.0, c),
                    reverse=True)
        batch = leaves[:W]
        for r in roots:
            if len(batch) >= W:
                break
            batch.append(r)
        return batch
    # EVOLVE-BLOCK-END
"""
CAND = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        W = question.max_parallelism
        obs = question.observed()
        roots = question.legal_roots()
        nroots = len([o for o in obs.values() if o.parent_id is None])
        leaves = [c for c in question.legal_actions() if not c.startswith("root:")]
        leaves.sort(key=lambda c: (obs[c].score if obs[c].valid and obs[c].score is not None else -1.0, c),
                    reverse=True)
        if nroots == 0 or (nroots < 2 * W and question.rounds != 1):
            batch = roots[:W]
        else:
            batch = leaves[:W]
        for r in roots:
            if len(batch) >= W:
                break
            batch.append(r)
        return list(reversed(batch))
    # EVOLVE-BLOCK-END
"""


def truth(i, roots=30, depth=30):
    """One strong root among the first two, weaker ones after; each continuation 0.1 more."""
    nodes = []
    for r in range(roots):
        s = 0.5 + 0.02 * (i % 3) if r == i % 2 else ([0.1, 0.12, 0.08, 0.11][r] if r < 4 else 0.05 if r < 8 else 0.0)
        prev = None
        for d in range(depth):
            nid = f"t{i}r{r}d{d}"
            nodes.append({"id": nid, "parent": prev, "score": round(s + 0.1 * d, 6), "valid": True, "fail_class": "ok"})
            prev = nid
    return {"id": f"truth{i}", "baseline": 0.0, "nodes": nodes}


def rows(rep):
    return rep["measured"]["runs"][str(float(rep["default_beta"]))]


class OrderTest(unittest.TestCase):
    def test_the_order_a_batch_is_listed_in_earns_nothing_live_or_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            seed, rev = write(d, "seed.py", SEED), write(d, "rev.py", REVERSED)
            for i in range(4):
                t = truth_world(i)
                own_seed, own_rev = record(seed, t, 4, 24, "iter0001"), record(rev, t, 4, 24, "iter0001")
                live_seed = evaluate_policy(seed, [own_seed], **KW)["reward"]
                live_rev = evaluate_policy(rev, [own_rev], **KW)["reward"]
                replayed_rev = evaluate_policy(rev, [own_seed], **KW)["reward"]
                self.assertAlmostEqual(live_rev, live_seed, places=12)
                self.assertAlmostEqual(replayed_rev, live_rev, places=12)

    def test_a_revision_replays_no_better_than_it_runs_on_the_same_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            inc_src, cand_src = with_block(INC)(SEED), with_block(CAND)(SEED)
            inc, cand = write(d, "inc.py", inc_src), write(d, "cand.py", cand_src)
            truths = [truth(i) for i in range(6)]
            inc_worlds = [record(inc, t, 4, 24, f"iter{i + 1:04d}") for i, t in enumerate(truths)]
            cand_worlds = [record(cand, t, 4, 24, f"iter{i + 1:04d}") for i, t in enumerate(truths)]
            for a, b in zip(inc_worlds, cand_worlds):  # the same attempts: the same normalisers
                self.assertEqual(sorted(n["score"] for n in a["nodes"]), sorted(n["score"] for n in b["nodes"]))
            replayed = evaluate_policy(cand, inc_worlds, **KW)
            self.assertEqual({r["off_record"] for r in rows(replayed)}, {0})  # the incumbent's record answers it all
            live = evaluate_policy(cand, cand_worlds, **KW)["reward"]
            self.assertAlmostEqual(replayed["reward"], live, places=12)  # so its replay is what it does live
            inc_live = evaluate_policy(inc, inc_worlds, **KW)["reward"]
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(inc_src)
            cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"], M=1)}
            rep = run_dream(pdir, inc_worlds, Dev(lambda src: cand_src), cfg, Path(d) / "logs")
        self.assertIsNone(rep["skipped"])
        if rep["deployed"]:  # a deploy is a gain live, on the same attempts
            self.assertGreater(live, inc_live)

    def test_a_batch_is_credited_worst_first(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "z-001", "parent": None, "score": 0.9, "valid": True, "cell": "root:0"},
            {"id": "z-002", "parent": None, "score": None, "valid": False, "cell": "root:1"},
            {"id": "z-003", "parent": None, "score": 0.2, "valid": True, "cell": "root:2"}]}
        with tempfile.TemporaryDirectory() as d:
            rep = evaluate_policy(write(d, "seed.py", SEED), [w], W=3, betas=[], budget=3, lam=0.25, beta1=0.01,
                                  beta2=0.01)
        self.assertEqual(rows(rep)[0]["curve_canonical"], [[1, None], [2, 0.2], [3, 0.9]])


SLOW = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        sum(range(10 ** 7))
        return question.legal_roots()[:question.max_parallelism]
    # EVOLVE-BLOCK-END
"""


class TimeoutTest(unittest.TestCase):
    def test_each_run_has_its_own_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            slow = write(d, "slow.py", with_block(SLOW)(SEED))
            start = time.monotonic()
            one = evaluate_policy(slow, [chain_world(3, 3)], W=2, betas=[], budget=4, lam=0.25, beta1=0.01,
                                  beta2=0.01, timeout=3)
            self.assertTrue(one["ok"], one)
            per_world = (time.monotonic() - start) / 2  # two replays of one run each, plus the default-beta reads
            n = max(8, int(4 * 2 * 3 / per_world))  # four at a time: the runs of one replay take well over 3 s
            rep = evaluate_policy(slow, [chain_world(3, 3) for _ in range(n)], W=2, betas=[], budget=4, lam=0.25,
                                  beta1=0.01, beta2=0.01, timeout=3)
        self.assertTrue(rep["ok"], rep)


class ReachTest(unittest.TestCase):
    def test_a_world_whose_roots_have_no_slot_informs_nothing(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "a", "parent": None, "score": 0.5, "valid": True, "cell": "iter0003-002"},
            {"id": "b", "parent": "a", "score": 0.9, "valid": True, "cell": "a"}]}
        self.assertEqual(reachable(w), [])
        self.assertIsNone(world_best(w))
        self.assertEqual(informative([w]), 0)

    def test_the_best_and_the_size_count_only_what_replay_can_reveal(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "r0", "parent": None, "score": 0.2, "valid": True, "cell": "root:0"},
            {"id": "c0", "parent": "r0", "score": 0.3, "valid": True, "cell": "r0"},
            {"id": "x", "parent": "r0", "score": 0.95, "valid": True, "cell": "gone"},  # re-parented by a prune
            {"id": "o", "parent": None, "score": 0.9, "valid": True, "cell": "iter0002-004"}]}  # re-rooted
        self.assertEqual([n["id"] for n in reachable(w)], ["r0", "c0"])
        self.assertEqual(world_best(w), 0.3)

    def test_a_child_opened_from_another_attempt_is_not_revealed_from_its_new_parent(self):
        w = {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "p", "parent": None, "score": 0.1, "valid": True, "cell": "root:0"},
            {"id": "x", "parent": "p", "score": 1.0, "valid": True, "cell": "gone"}]}
        q = ReplayQuestion(w, 2, max_probes=6)
        q.probe_batch(["root:0"])
        with self.assertRaises(RecordEnd):
            q.probe_batch(["p"])
        self.assertEqual([n["id"] for n in reachable(w)], ["p"])

    def test_a_world_without_cells_opens_every_root_and_first_child(self):
        w = chain_world(2, 3)
        self.assertEqual(len(reachable(w)), len(w["nodes"]))


SLOW_CLASS = SEED.replace("class OptimalPolicy(LLMDesignedMethod):\n    beta = 0.6",
                          "class OptimalPolicy(LLMDesignedMethod):\n    beta = 0.6 + 0 * sum(range(6 * 10 ** 9))")
assert SLOW_CLASS != SEED


class InterruptTest(unittest.TestCase):
    def tearDown(self):
        subprocess.run(["pkill", "-9", "-f", str(ENGINE_DIR / "drsi" / "replay_runner.py")], capture_output=True)

    def test_an_interrupt_while_the_default_beta_is_read_stops_that_process(self):
        with tempfile.TemporaryDirectory() as d:
            pol = write(d, "slowclass.py", SLOW_CLASS)
            threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
            with self.assertRaises(KeyboardInterrupt):
                evaluate_policy(pol, [chain_world(2, 2)], W=2, betas=[], budget=4, lam=0.25, beta1=0.01, beta2=0.01,
                                timeout=60)
            time.sleep(0.5)
            self.assertEqual(our_runners(), 0)


class CommandTest(unittest.TestCase):
    def test_drsi_replay_counts_only_worlds_recorded_live(self):
        from drsi import cli
        cfg = {"search": {"W": 2, "K1": 2}, "dream": dict(DEFAULT_CONFIG["dream"])}
        worlds = [chain_world(4, 4), dict(chain_world(4, 4), id="history", live=False)]
        out = io.StringIO()
        with mock.patch.object(cli, "resolve_campaign", return_value=mock.Mock(config=cfg)), \
                mock.patch.object(cli, "_worlds", return_value=worlds), \
                mock.patch.object(cli, "_policy_path", return_value=SEED_POLICY), \
                mock.patch.object(cli, "evaluate_policy", wraps=cli.evaluate_policy) as ev, redirect_stdout(out):
            cli.cmd_replay(Namespace(campaign="x", history=True, policy=None))
        self.assertIn("1 of 1", out.getvalue())
        self.assertEqual(ev.call_count, 1)  # the deployed policy is evaluated once

    def test_min_worlds_of_zero_without_a_live_world_skips_the_dream(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(SEED)
            cfg = {"search": {"W": 2, "K1": 2}, "dream": dict(DEFAULT_CONFIG["dream"], M=1, min_worlds=0)}
            rep = run_dream(pdir, [dict(chain_world(3, 3), live=False)], Dev(), cfg, Path(d) / "logs")
        self.assertIn("recorded live", rep["skipped"] or "", rep)


if __name__ == "__main__":
    unittest.main()
