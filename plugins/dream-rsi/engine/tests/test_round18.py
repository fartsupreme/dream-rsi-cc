"""Round 18: `drsi prune`, removing recorded attempts that did no work.

Found in use (2026-09-28): workers that failed before doing anything (every call refused for hours) were recorded as
attempts, over a thousand of them, with their frozen round worlds, branches and proposal directories. The counts, the
map and the replay pool then carried them as if they were failed ideas. They were removed by hand, and nothing
recorded that. `drsi prune` removes such attempts by id or by the error text they carry, but only ones that did no
work (a worker or orchestration failure with no score and no changed files), re-roots anything that continued from
them, withdraws their novelty claims, and logs what it removed.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from drsi import cli, guardian
from drsi.agent import AgentResult
from drsi.checks import pending_checks
from drsi.live import LiveRunner
from drsi.prune import prune
from drsi.question import ROOT
from drsi.store import Campaign
from drsi.worlds import freeze_world, load_worlds, world_from_tree
from tests.test_round11 import bump, scorer
from tests.test_scorer_workspace import make_repo

REFUSED = "account unavailable: refused before any work"


def worker(ok_for: set):
    """Fails with REFUSED unless the workspace name is in ok_for."""
    def run(workspace: Path, prompt: str, system: str) -> AgentResult:
        if workspace.name not in ok_for:
            return AgentResult(ok=False, error=REFUSED)
        bump(workspace)
        return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                "notes": ""})
    return run


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.camp = Campaign.create("t", {
            "goal": "maximise value.txt", "scorer": {"cmd": scorer(1), "timeout_s": 30, "serial": False},
            "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]},
            "search": {"W": 2, "K1": 2, "plateau": 3}, "live": {"require_check": False}, "baseline": 1.0,
        }, home=root / "home")

    def tearDown(self):
        self.tmp.cleanup()

    def round(self, rid, batches, ok_for, freeze=True):
        r = LiveRunner(self.camp, worker(ok_for), indexer=lambda ids: None, round_id=rid)
        for cells in batches:
            r.run_batch(cells)
        if freeze:
            freeze_world(self.camp.root / "trace_pool", world_from_tree(self.camp.tree, rid, 1.0, ids=set(r.ids)))
        return r.ids

    def branches(self):
        from drsi.workspace import _git
        out = _git(self.camp.root / "repo", "branch", "--list", "drsi/*", "--format=%(refname:short)", check=False)
        return set(out.split())

    def fixture(self):
        # round 1: one good root, one refused root; a continuation of the refused one that did work
        ids1 = self.round("iter0001", [[f"{ROOT}0", f"{ROOT}1"], ["iter0001-002"]],
                          ok_for={"iter0001-001", "iter0001-003"})
        # round 2: everything refused, so its world holds only refused attempts
        ids2 = self.round("iter0002", [[f"{ROOT}0", f"{ROOT}1"]], ok_for=set())
        return ids1, ids2

    def test_attempts_that_did_no_work_go_with_their_worlds_branches_and_proposals(self):
        ids1, ids2 = self.fixture()
        t = self.camp.tree
        self.assertEqual(t.get("iter0001-002")["fail_class"], "agent_error")
        self.assertEqual(t.get("iter0001-003")["parent"], "iter0001-002")
        for nid in ("iter0001-002", "iter0002-001"):
            (self.camp.root / "work" / "_proposals" / nid).mkdir(parents=True, exist_ok=True)
        before = self.branches()

        rep = prune(self.camp, error_match="account unavailable", reason="test", log=lambda m: None)

        t = self.camp.tree
        self.assertEqual(sorted(rep["pruned"]), ["iter0001-002", "iter0002-001", "iter0002-002"])
        for nid in rep["pruned"]:
            self.assertNotIn(nid, t)
            self.assertNotIn(f"drsi/{nid}", self.branches())
            self.assertFalse((self.camp.root / "work" / "_proposals" / nid).exists())
        self.assertIn("iter0001-001", t)
        child = t.get("iter0001-003")  # its parent did no work, so it starts from the base: a root now
        self.assertEqual((child["parent"], child["seq"]), (None, 0))
        self.assertEqual(rep["reparented"], ["iter0001-003"])
        worlds = {w["id"]: w for w in load_worlds(self.camp.root / "trace_pool")}
        self.assertEqual(sorted(worlds), ["iter0001"])  # iter0002 held nothing but refused attempts
        w1 = {n["id"]: n for n in worlds["iter0001"]["nodes"]}
        self.assertEqual(sorted(w1), ["iter0001-001", "iter0001-003"])
        self.assertIsNone(w1["iter0001-003"]["parent"])
        self.assertTrue({f"drsi/{n}" for n in rep["pruned"]} <= before)  # they had branches, now gone
        log = [json.loads(line) for line in (self.camp.root / "logs" / "prune.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(log[-1]["ids"]), sorted(rep["pruned"]))
        self.assertEqual(log[-1]["reason"], "test")
        self.assertIn(f"{len(t)} attempts", self.camp.map_path.read_text().splitlines()[0])

    def test_an_attempt_that_did_work_is_refused_whatever_it_matches(self):
        self.fixture()
        self.camp.tree.update("iter0002-001", artifacts=dict(self.camp.tree.get("iter0002-001")["artifacts"],
                                                            changed=["value.txt"]))
        rep = prune(self.camp, ids={"iter0001-001", "iter0002-001", "iter0002-002"}, log=lambda m: None)
        self.assertEqual(rep["pruned"], ["iter0002-002"])
        self.assertEqual(sorted(rep["refused"]), ["iter0001-001", "iter0002-001"])
        self.assertIn("iter0001-001", self.camp.tree)
        self.assertIn("iter0002-001", self.camp.tree)

    def test_a_dry_run_changes_nothing(self):
        self.fixture()
        tree_before = (self.camp.root / "tree.jsonl").read_text()
        worlds_before = sorted(p.name for p in (self.camp.root / "trace_pool").iterdir())
        branches_before = self.branches()
        rep = prune(self.camp, error_match="account unavailable", dry_run=True, log=lambda m: None)
        self.assertEqual(len(rep["pruned"]), 3)
        self.assertEqual((self.camp.root / "tree.jsonl").read_text(), tree_before)
        self.assertEqual(sorted(p.name for p in (self.camp.root / "trace_pool").iterdir()), worlds_before)
        self.assertEqual(self.branches(), branches_before)
        self.assertFalse((self.camp.root / "logs" / "prune.jsonl").exists())

    def test_a_round_still_running_is_left_alone(self):
        self.fixture()
        self.round("iter0003", [[f"{ROOT}0"]], ok_for=set(), freeze=False)  # no world yet: in progress
        lock = open(self.camp.root / "logs" / guardian.LOCK, "w")
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            rep = prune(self.camp, error_match="account unavailable", log=lambda m: None)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        self.assertIn("iter0003-001", rep["in_progress"])
        self.assertIn("iter0003-001", self.camp.tree)
        self.assertNotIn("iter0002-001", self.camp.tree)

    def test_a_pruned_attempts_claim_is_withdrawn(self):
        self.fixture()
        with open(self.camp.checks_path, "a") as fh:
            fh.write(json.dumps({"node": "iter0002-001", "verdict": "claim", "proposal": "an idea",
                                 "checked": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                        __import__("time").gmtime())}) + "\n")
        prune(self.camp, error_match="account unavailable", log=lambda m: None)
        self.assertNotIn("iter0002-001", [p["node"] for p in pending_checks(self.camp)])

    def test_the_command_needs_a_selection_and_reports(self):
        self.fixture()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["prune", "-c", str(self.camp.root)]), 2)
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["prune", "-c", str(self.camp.root), "--error-match", "account unavailable",
                             "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("3 attempts would be removed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
