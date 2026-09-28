"""Round 17: new branches get distinct suggestions, and workers know what the workspace can score.

Found in use (2026-09-28): the frontier held five suggested directions and a round opened six new branches. Branch b
took suggestion b % 5, so branches 1 and 4 of one batch were handed the same direction and built the same idea side
by side. The frontier is also generated from the campaign goal alone. When a campaign's workspace can build and score
only part of that goal, a suggestion outside it sends a worker to build what its scorer then records as invalid. Two
of a round's first three attempts did this. `live.objective` states what the workspace scores. It goes into every
worker's brief and into the frontier's prompt.
"""
import json
import tempfile
import unittest
from pathlib import Path

from drsi.families import build_frontier, load_families, refresh_frontier
from drsi.live import LiveRunner
from drsi.question import ROOT
from drsi.store import Campaign, Tree
from tests.helpers import ScriptedLLM
from tests.test_families import TAXO, _tree
from tests.test_live import fixed_checker
from tests.test_round10 import SCORER, WorkerModelsTest
from tests.test_scorer_workspace import make_repo

OBJECTIVE = "a new sorter in sorter.py; the scorer times it and checks its output"


class LiveBriefTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = make_repo(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def campaign(self, live: dict | None = None, frontier: list | None = None, name: str = "t") -> Campaign:
        cfg = {"goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
               "workspace": {"repo": str(self.repo), "mutable": ["value.txt"]},
               "search": {"W": 6, "K1": 1, "plateau": 3}, "live": dict({"require_check": True}, **(live or {})),
               "llm": {"model": "opus", "classifier_model": "opus", "worker_model": "opus"}}
        camp = Campaign.create(name, cfg, home=self.root / "home")
        if frontier is not None:
            camp.families_path.write_text(json.dumps({"families": TAXO["families"], "frontier": [
                {"direction": d, "rationale": "r", "avoids": []} for d in frontier]}))
        return camp

    @staticmethod
    def briefs(camp: Campaign, n: int) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}
        worker = WorkerModelsTest.worker([])

        def run(workspace, prompt, system, model=None):
            seen.setdefault(workspace.name, []).append(system)  # the brief travels as the system prompt
            return worker(workspace, prompt, system, model)
        r = LiveRunner(camp, run, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        r.run_batch([f"{ROOT}{i}" for i in range(n)])
        return seen

    def test_no_two_new_branches_of_a_round_are_handed_the_same_suggestion(self):
        camp = self.campaign(frontier=["try A", "try B", "try C", "try D", "try E"])
        seen = self.briefs(camp, 6)
        handed = []
        for prompts in seen.values():  # the map lists every suggestion; the branch's own is on its own line
            got = [line.rsplit(": ", 1)[1] for line in prompts[0].splitlines()
                   if line.startswith("Suggested untried direction for this branch")]
            self.assertLessEqual(len(got), 1)
            handed += got
        self.assertEqual(sorted(handed), ["try A", "try B", "try C", "try D", "try E"])  # each once; one branch free

    def test_the_workspace_objective_is_in_every_brief(self):
        camp = self.campaign(live={"objective": OBJECTIVE}, frontier=["try A"])
        seen = self.briefs(camp, 2)
        self.assertEqual(len(seen), 2)
        for prompts in seen.values():
            self.assertEqual(len(prompts), 2)  # propose, then implement
            for p in prompts:
                self.assertIn(OBJECTIVE, p)
                self.assertLess(p.index("maximise value.txt"), p.index(OBJECTIVE))

    def test_without_an_objective_the_brief_is_as_before(self):
        camp = self.campaign()
        for prompts in self.briefs(camp, 1).values():
            self.assertNotIn("WHAT THIS WORKSPACE SCORES", prompts[0])

    def test_an_objective_that_is_not_text_is_refused(self):
        for bad in (["a list"], "", "   ", 3):
            camp = self.campaign(live={"objective": bad}, name=f"t{abs(hash(str(bad))) % 10**6}")
            with self.assertRaises(ValueError):
                LiveRunner(camp, WorkerModelsTest.worker([]), indexer=lambda ids: None, round_id="iter0001",
                           checker=fixed_checker("novel"))


class FrontierObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.tree = _tree(self.d, [("1", None, "", "refuted", "speed")])
        self.tree.update("1", fingerprint=dict(self.tree.get("1")["fingerprint"], family="F01"))
        self.path = self.d / "families.json"
        self.path.write_text(json.dumps({"families": TAXO["families"]}))
        self.llm = ScriptedLLM(lambda p, s: {"directions": [{"direction": "try Y", "rationale": "r", "avoids": []}]})

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_frontier_asks_for_directions_the_workspace_can_score(self):
        build_frontier(self.tree, load_families(self.path), self.llm, goal="g", path=self.path, objective=OBJECTIVE)
        self.assertIn(OBJECTIVE, self.llm.prompts[0])
        self.assertEqual(load_families(self.path)["frontier"][0]["direction"], "try Y")

    def test_a_refreshed_frontier_carries_the_objective(self):
        refresh_frontier(Tree(self.tree.path), self.llm, "g", self.path, objective=OBJECTIVE)
        self.assertIn(OBJECTIVE, self.llm.prompts[0])

    def test_without_an_objective_the_frontier_prompt_is_as_before(self):
        build_frontier(self.tree, load_families(self.path), self.llm, goal="g", path=self.path)
        self.assertNotIn("workspace", self.llm.prompts[0].lower())


if __name__ == "__main__":
    unittest.main()
