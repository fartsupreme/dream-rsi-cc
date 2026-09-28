"""Round 11: rescoring recorded attempts with the campaign's current scorer.

Found in use (2026-09-28): the campaign's scorer was corrected several times while the live loop ran, and the attempts
scored by earlier versions kept readings that no longer meant what the map and the frozen round worlds took them to
mean (an attempt read by an earlier version topped the record). `drsi rescore` runs the current scorer on each
attempt's own commit, exactly as the loop scores it, keeps the old reading on the node, and recomputes the outcomes the
map shows and the scores the frozen worlds hold. Attempts that never reached the scorer are left alone.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from drsi import cli
from drsi.agent import AgentResult
from drsi.live import LiveRunner
from drsi.question import ROOT
from drsi.rescore import rescore
from drsi.store import Campaign
from drsi.worlds import freeze_world, world_from_tree
from tests.test_scorer_workspace import make_repo


def scorer(times: int) -> str:
    return f"python3 -c \"import json;print(json.dumps({{'score': {times} * int(open('value.txt').read()), 'valid': True}}))\""


def worker(action):
    def run(workspace: Path, prompt: str, system: str) -> AgentResult:
        action(workspace)
        return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                "notes": ""})
    return run


def bump(ws: Path):
    ws.joinpath("value.txt").write_text(f"{int(ws.joinpath('value.txt').read_text()) + 1}\n")


class RescoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.camp = Campaign.create("t", {
            "goal": "maximise value.txt", "scorer": {"cmd": scorer(1), "timeout_s": 30, "serial": False},
            "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]},
            "search": {"W": 1, "K1": 2, "plateau": 3}, "live": {"require_check": False}, "baseline": 1.0,
        }, home=root / "home")

    def tearDown(self):
        self.tmp.cleanup()

    def batch(self, rid, cells, action=bump):
        r = LiveRunner(self.camp, worker(action), indexer=lambda ids: None, round_id=rid)
        return [self.camp.tree.get(o["id"]) for o in r.run_batch(cells)]

    def set_scorer(self, times):
        self.camp.update_config(lambda raw: raw["scorer"].__setitem__("cmd", scorer(times)))

    def test_the_current_scorer_replaces_the_reading_and_the_old_one_is_kept(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"])[0]
        self.assertEqual((n1["score"], n1["artifacts"]["outcome"]), (2.0, "pass"))
        self.set_scorer(10)
        rep = rescore(self.camp, ids={n1["id"]}, parallel=2)
        n1 = self.camp.tree.get(n1["id"])
        self.assertEqual(n1["score"], 20.0)
        self.assertEqual(n1["artifacts"]["rescored"][0]["score"], 2.0)
        self.assertEqual(rep["rescored"][n1["id"]], (2.0, 20.0))

    def test_attempts_that_never_reached_the_scorer_are_left_alone(self):
        def sneaky(ws):
            ws.joinpath("README").write_text("not the attempt's to write\n")
            bump(ws)
        n1 = self.batch("iter0001", [f"{ROOT}0"], sneaky)[0]
        self.assertEqual(n1["fail_class"], "out_of_scope")
        self.set_scorer(10)
        rep = rescore(self.camp, ids={n1["id"]})
        self.assertEqual(rep["skipped"], [n1["id"]])
        n1 = self.camp.tree.get(n1["id"])
        self.assertEqual((n1["score"], n1["fail_class"], "rescored" in n1["artifacts"]), (None, "out_of_scope", False))

    def test_a_childs_outcome_is_judged_again_against_its_rescored_parent(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"])[0]
        n2 = self.batch("iter0002", [n1["id"]])[0]
        self.assertEqual((n2["score"], n2["artifacts"]["outcome"]), (3.0, "pass"))
        self.set_scorer(10)
        rescore(self.camp, ids={n1["id"]})
        n2 = self.camp.tree.get(n2["id"])
        self.assertEqual((n2["score"], n2["artifacts"]["outcome"], n2["fingerprint"]["outcome"]),
                         (3.0, "measured", "measured"))

    def test_frozen_round_worlds_carry_the_new_readings(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"])[0]
        pool = self.camp.root / "trace_pool"
        freeze_world(pool, world_from_tree(self.camp.tree, "iter0001", 1.0, ids={n1["id"]}))
        self.set_scorer(10)
        rescore(self.camp, ids={n1["id"]})
        world = json.loads((pool / "iter0001" / "world.json").read_text())
        self.assertEqual([n["score"] for n in world["nodes"] if n["id"] == n1["id"]], [20.0])

    def test_the_cli_needs_ids_or_all_and_reports_each_attempt(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"])[0]
        self.set_scorer(10)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["rescore", "-c", str(self.camp.root)]), 2)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["rescore", "-c", str(self.camp.root), "--all"]), 0)
        self.assertIn(f"{n1['id']}: 2.0 -> 20.0", out.getvalue())


if __name__ == "__main__":
    unittest.main()
