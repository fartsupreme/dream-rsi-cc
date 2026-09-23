"""Integrity of the live loop: a worker gets credit only for what the scorer measures on what it committed."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.live import LiveRunner, live_round, load_policy, next_round_id
from drsi.store import Campaign, make_node
from tests.test_policy import SEED
from tests.test_scorer_workspace import git, make_repo

SCORER = "python3 -c \"import json;print(json.dumps({'score': int(open('value.txt').read()), 'valid': True}))\""
PLANT_AWARE = ("python3 -c \"import os,json; b=os.path.exists('__pycache__/boost'); "
               "print(json.dumps({'score': 1000 if b else int(open('value.txt').read()), 'valid': True}))\"")


def worker(action):
    def run(workspace: Path, prompt: str, system: str) -> AgentResult:
        if "PHASE: PROPOSE" in system:
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "", "self_reported_score": None,
                                                    "notes": ""})
        action(workspace)
        return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": 999,
                                                "notes": ""})
    return run


def bump(ws: Path):
    ws.joinpath("value.txt").write_text(f"{int(ws.joinpath('value.txt').read_text()) + 1}\n")


class LiveIntegrityBase(unittest.TestCase):
    scorer = SCORER

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src = make_repo(root)
        self.camp = Campaign.create("t", {
            "goal": "maximise value.txt", "scorer": {"cmd": self.scorer, "timeout_s": 30},
            "workspace": {"repo": str(self.src), "mutable": ["value.txt"]},
            "search": {"W": 1, "K1": 2, "plateau": 3}, "live": {"require_check": False},
        }, home=root / "home")
        self.policy = load_policy(SEED)

    def tearDown(self):
        self.tmp.cleanup()

    def round(self, fn, rid="iter0001"):
        r = LiveRunner(self.camp, worker(fn), indexer=lambda ids: None, round_id=rid)
        live_round(self.camp, self.policy, r)
        return [self.camp.tree.get(i) for i in r.ids]


class ScopeTest(LiveIntegrityBase):
    def test_worker_that_commits_its_own_out_of_scope_edit_is_caught(self):
        def act(ws):
            ws.joinpath("README").write_text("hijacked\n")
            git(ws, "add", "-A")
            git(ws, "-c", "user.email=w@w", "-c", "user.name=w", "commit", "-qm", "sneaky")
            bump(ws)
        nodes = self.round(act)
        self.assertEqual(nodes[0]["fail_class"], "out_of_scope")
        self.assertIn("README", nodes[0]["artifacts"]["out_of_scope"])

    def test_rename_of_out_of_scope_file_is_caught(self):
        def act(ws):
            ws.joinpath("value.txt").unlink()
            git(ws, "mv", "README", "value.txt")
            ws.joinpath("value.txt").write_text("5\n")
        nodes = self.round(act)
        self.assertEqual(nodes[0]["fail_class"], "out_of_scope")

    def test_child_of_out_of_scope_parent_is_invalid_too(self):
        calls = {"n": 0}

        def act(ws):
            calls["n"] += 1
            if calls["n"] == 1:
                ws.joinpath("README").write_text("bad\n")
            bump(ws)
        nodes = self.round(act)
        self.assertGreaterEqual(len(nodes), 2)
        self.assertTrue(all(n["fail_class"] == "out_of_scope" for n in nodes), [n["fail_class"] for n in nodes])


class CleanCheckoutTest(LiveIntegrityBase):
    scorer = PLANT_AWARE

    def test_ignored_planted_file_does_not_reach_the_scorer(self):
        def act(ws):
            ws.joinpath("__pycache__").mkdir(exist_ok=True)
            ws.joinpath("__pycache__", "boost").write_text("x")
            bump(ws)
        nodes = self.round(act)
        self.assertEqual(nodes[0]["score"], 2.0)


class CheckFlowTest(LiveIntegrityBase):
    def setUp(self):
        super().setUp()
        raw = json.loads(self.camp.config_path.read_text())
        raw["live"]["require_check"] = True
        raw["search"]["K1"] = 1
        self.camp.save_config(raw)

    def test_recorded_proposal_is_the_judged_one_not_the_workers_later_claim(self):
        from tests.test_live import fixed_checker, stub_worker
        chk = fixed_checker("variant")
        r = LiveRunner(self.camp, stub_worker(proposals=["the judged proposal"]), indexer=lambda ids: None,
                       round_id="iter0001", checker=chk)
        live_round(self.camp, self.policy, r)
        n = self.camp.tree.get(r.ids[0])
        self.assertEqual(n["proposal"], "the judged proposal")
        self.assertEqual(chk.seen[0], ("iter0001-001", "the judged proposal"))

    def test_the_implement_brief_carries_the_accepted_proposal_and_the_checks_notes(self):
        from tests.test_live import fixed_checker, stub_worker
        w = stub_worker(proposals=["a wheel sieve"])
        r = LiveRunner(self.camp, w, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        live_round(self.camp, self.policy, r)
        impl = [c["system"] for c in w.calls if "PHASE: IMPLEMENT" in c["system"]][0]
        self.assertIn("a wheel sieve", impl)
        self.assertIn("accepted (novel)", impl)

    def test_worker_that_writes_no_proposal_is_an_agent_error(self):
        def silent(workspace, prompt, system):
            return AgentResult(ok=True, structured={"proposal": "", "summary": "", "self_reported_score": None,
                                                    "notes": ""})
        from tests.test_live import fixed_checker
        r = LiveRunner(self.camp, silent, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        live_round(self.camp, self.policy, r)
        self.assertEqual(self.camp.tree.get(r.ids[0])["fail_class"], "agent_error")


class RobustnessTest(LiveIntegrityBase):
    def test_one_crashing_worker_does_not_lose_the_round(self):
        def run(workspace, prompt, system):
            raise RuntimeError("worker process exploded")
        r = LiveRunner(self.camp, run, indexer=lambda ids: None, round_id="iter0001")
        live_round(self.camp, self.policy, r)
        n = self.camp.tree.get(r.ids[0])
        self.assertEqual(n["fail_class"], "agent_error")
        self.assertIn("exploded", n["text"]["worker_error"])

    def test_round_id_skips_ids_already_used_by_an_interrupted_round(self):
        self.camp.tree.add(make_node(id="iter0001-001", parent=None, source="live", ext={"round": "iter0001"}))
        self.assertEqual(next_round_id(self.camp), "iter0002")

    def test_worktrees_are_removed_after_scoring_but_children_still_build_on_commits(self):
        nodes = self.round(bump)
        self.assertGreaterEqual(len(nodes), 2)
        for n in nodes:
            self.assertFalse(Path(n["artifacts"]["workspace"]).exists())
        self.assertEqual(nodes[1]["score"], nodes[0]["score"] + 1)

    def test_live_outcome_comes_from_the_scorer(self):
        nodes = self.round(bump)
        self.assertEqual(nodes[0]["fingerprint"]["outcome"], "pass")   # 2 > baseline 1

    def test_propose_brief_names_the_proposal_file_and_forbids_building(self):
        seen = []
        raw = json.loads(self.camp.config_path.read_text())
        raw["live"]["require_check"] = True
        self.camp.save_config(raw)

        def run(workspace, prompt, system):
            seen.append(system)
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": "x", "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            bump(workspace)
            return AgentResult(ok=True, structured={})
        from tests.test_live import fixed_checker
        r = LiveRunner(self.camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        live_round(self.camp, self.policy, r)
        self.assertIn(str(self.camp.root / "work" / "_proposals" / "iter0001-001" / "proposal.txt"), seen[0])
        self.assertIn("Do not implement anything yet", seen[0])


if __name__ == "__main__":
    unittest.main()
