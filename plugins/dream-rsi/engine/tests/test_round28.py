"""Round 28: findings of the last review before 0.4.0 (Opus, on the round-27 tip), each reproduced here first.

It found no revision that deploys with a gain that does not hold live, and no counterexample to the reward bound. It
found two regressions of giving each replay run its own process, and four smaller gaps:
- `drsi stop` no longer stopped a replay at once, and replay processes could outlive the run: they now run in their
  own sessions, registered with the run's children like workers and scorers, and an interrupt kills every one;
- a hanging revision cost (runs / 4) x the timeout, since the other runs went on after one failed: the first failure
  now ends the evaluation, and one timeout covers all of its runs, as it did when they shared a process;
- a live policy sees each attempt's family as it is when revealed, but the world took it from the tree at the end of
  the round, after the families could be assigned again; the world now keeps the family the policy was shown;
- the headline bound test passed with the ending batch's recorded cells credited first: it now compares the curve
  the reward uses, on a round where that order matters;
- gate trees used family names "A" to "F", which no live round shows: they now take the campaign's own names;
- `drsi replay` ranks with the dream step's formula over every world, while the dream compares on the incumbent's
  on-record worlds: it now reports both.
"""
import os
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from drsi.dream import SEED_POLICY, gate_worlds
from drsi.replay import ENGINE_DIR, _aggregate, evaluate_policy
from tests.helpers import with_block
from tests.test_policy import chain_world
from tests.test_round23 import ALLROOTS
from tests.test_round27 import LATE_BEST, hundred_roots

HANG = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        while True:
            pass
    # EVOLVE-BLOCK-END
"""


def write(d, name, src) -> Path:
    p = Path(d) / name
    p.write_text(src)
    return p


def our_runners() -> int:
    out = subprocess.run(["pgrep", "-f", str(ENGINE_DIR / "drsi" / "replay_runner.py")], capture_output=True,
                         text=True).stdout
    return len(out.split())


class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hang = write(self.tmp.name, "hang.py", with_block(HANG)(SEED_POLICY.read_text()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_hanging_revision_fails_within_one_timeout(self):
        start = time.monotonic()
        rep = evaluate_policy(self.hang, [chain_world(3, 3) for _ in range(8)], W=2, betas=[0.0, 0.5, 1.0], budget=4,
                              lam=0.25, beta1=0.01, beta2=0.01, timeout=3)
        self.assertFalse(rep["ok"])
        self.assertIn("timeout", rep["error"])
        self.assertLess(time.monotonic() - start, 3 + 4)
        self.assertEqual(our_runners(), 0)

    def test_an_interrupt_stops_every_replay_process_at_once(self):
        # a real signal, as `drsi stop` and Ctrl-C send (a simulated one cannot wake a blocked wait)
        threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        start = time.monotonic()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_policy(self.hang, [chain_world(3, 3) for _ in range(8)], W=2, betas=[0.0, 1.0], budget=4,
                            lam=0.25, beta1=0.01, beta2=0.01, timeout=60)
        self.assertLess(time.monotonic() - start, 10)
        self.assertEqual(our_runners(), 0)


class RecordTest(unittest.TestCase):
    def test_a_world_keeps_the_family_each_attempt_was_shown(self):
        from drsi.live import LiveRunner, live_round, load_policy
        from drsi.store import Campaign
        from drsi.worlds import load_worlds
        from tests.test_live import SCORER, stub_worker
        from tests.test_scorer_workspace import make_repo
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            camp = Campaign.create("toy", {
                "goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
                "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]},
                "search": {"W": 2, "K1": 2, "plateau": 3}, "live": {"require_check": False}}, home=root / "home")
            calls = []

            def indexer(ids):  # the first call fails; later ones assign every attempt so far to F09
                calls.append(list(ids))
                if len(calls) == 1:
                    raise RuntimeError("classifier unavailable")
                for i in runner.ids:
                    camp.tree.update(i, fingerprint={"family": "F09"})
            runner = LiveRunner(camp, stub_worker(), indexer=indexer, round_id="iter0001")
            live_round(camp, load_policy(SEED_POLICY), runner)
            world = load_worlds(camp.root / "trace_pool")[0]
            fam = {n["id"]: n["family"] for n in world["nodes"]}
            first, later = calls[0], [i for batch in calls[1:] for i in batch]
            self.assertTrue(first and later, calls)
            self.assertEqual({fam[i] for i in first}, {None})  # shown with no family: its first index failed
            self.assertEqual({fam[i] for i in later}, {"F09"})
            self.assertEqual(camp.tree.get(first[0])["fingerprint"]["family"], "F09")  # the tree moved on


class BoundTest(unittest.TestCase):
    def test_the_bound_holds_on_the_curve_the_reward_uses_where_batch_order_matters(self):
        recorded = hundred_roots()
        truth = hundred_roots()  # live, the three continuations the record lacks are attempts that fail
        truth["nodes"] += [{"id": f"r{r:03d}c", "parent": f"r{r:03d}", "score": None, "valid": False,
                            "fail_class": "eval_error"} for r in range(3)]
        kw = dict(W=4, betas=[], budget=24, lam=0.25, beta1=0.01, beta2=0.01)
        with tempfile.TemporaryDirectory() as d:
            pol = write(d, "late.py", with_block(LATE_BEST)(SEED_POLICY.read_text()))
            replayed = evaluate_policy(pol, [recorded], **kw)
            live = evaluate_policy(pol, [truth], **kw)
        key = str(float(replayed["default_beta"]))
        r_row, l_row = replayed["measured"]["runs"][key][0], live["measured"]["runs"][key][0]
        self.assertEqual(r_row["curve_canonical"][-1], [24, 1.0])  # the best credited after the unanswered probes
        self.assertEqual(l_row["curve_canonical"][-1], [24, 1.0])
        self.assertLessEqual(replayed["reward"], _aggregate(live["measured"], [recorded], 4, 0.25, 0.01, 0.01)["reward"])


class GateTest(unittest.TestCase):
    def test_gate_trees_use_the_campaigns_family_names(self):
        # (round 30: policies see no family since round 29, so no policy can tell the gate's names apart)
        fams = {n["family"] for w in gate_worlds(4, 2, 4, families=["F01", "F02"]) for n in w["nodes"]}
        self.assertLessEqual(fams, {"F01", "F02", None})


class ReplayCommandTest(unittest.TestCase):
    def test_drsi_replay_reports_the_worlds_the_dream_compares_on(self):
        import io
        from argparse import Namespace
        from contextlib import redirect_stdout
        from unittest import mock
        from drsi import cli
        from drsi.store import DEFAULT_CONFIG
        cfg = {"search": {"W": 2, "K1": 2}, "dream": dict(DEFAULT_CONFIG["dream"])}
        # a world the seed stays on the record in, and one it leaves at once (two roots, a batch of two slots past)
        stay = chain_world(4, 4)
        leave = {"id": "short", "baseline": 0.0, "nodes": [{"id": "x", "parent": None, "score": 0.5, "valid": True}]}
        camp = mock.Mock(config=cfg)
        out = io.StringIO()
        with mock.patch.object(cli, "resolve_campaign", return_value=camp), \
                mock.patch.object(cli, "_worlds", return_value=[stay, leave]), \
                mock.patch.object(cli, "_policy_path", return_value=SEED_POLICY), redirect_stdout(out):
            cli.cmd_replay(Namespace(campaign="x", history=False, policy=None))
        self.assertIn("on the record", out.getvalue())
        self.assertIn("1 of 2", out.getvalue())


if __name__ == "__main__":
    unittest.main()
