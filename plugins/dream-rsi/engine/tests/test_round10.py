"""Round 10: a model for each worker slot.

Found in use (2026-09-28): a campaign ran every worker on one model. `llm.worker_models` gives each slot of a batch its
own model, in order and cycling (["opus", "fable"] with six workers: three of each), so a campaign can draw ideas from
more than one model at once. The same model proposes and builds an attempt, and every attempt records its model.
"""
import tempfile
import unittest
from pathlib import Path

from drsi import cli
from drsi.agent import AgentResult
from drsi.live import LiveRunner
from drsi.question import ROOT
from drsi.store import Campaign
from tests.test_live import fixed_checker
from tests.test_scorer_workspace import make_repo

SCORER = "python3 -c \"import json;print(json.dumps({'score': int(open('value.txt').read()), 'valid': True}))\""


class WorkerModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = {"goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
                    "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]},
                    "search": {"W": 4, "K1": 1, "plateau": 3}, "live": {"require_check": True}}
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def campaign(self, llm: dict, name: str = "t") -> Campaign:
        cfg = dict(self.cfg, llm=dict({"model": "opus", "classifier_model": "opus", "worker_model": "opus"}, **llm))
        return Campaign.create(name, cfg, home=self.root / "home")

    @staticmethod
    def worker(calls: list, accepts_model: bool = True):
        def run(workspace: Path, prompt: str, system: str, model=None) -> AgentResult:
            calls.append((workspace.name, "propose" if "PHASE: PROPOSE" in system else "implement", model))
            if "PHASE: PROPOSE" in system:
                return AgentResult(ok=True, structured={"proposal": f"idea {workspace.name}", "summary": "",
                                                        "self_reported_score": None, "notes": ""})
            workspace.joinpath("value.txt").write_text("2\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        if accepts_model:
            return run
        return lambda workspace, prompt, system: run(workspace, prompt, system)

    def test_each_slot_gets_its_model_in_order_and_cycling(self):
        camp = self.campaign({"worker_models": ["opus", "fable"]})
        calls = []
        r = LiveRunner(camp, self.worker(calls), indexer=lambda ids: None, round_id="iter0001",
                       checker=fixed_checker("novel"))
        out = r.run_batch([f"{ROOT}{i}" for i in range(4)])
        models = [camp.tree.get(o["id"])["worker"]["model"] for o in out]
        self.assertEqual(models, ["opus", "fable", "opus", "fable"])
        by_node = {}
        for nid, phase, model in calls:
            by_node.setdefault(nid, set()).add(model)
        self.assertTrue(all(len(ms) == 1 for ms in by_node.values()))  # one model proposes and builds an attempt
        self.assertEqual(sorted(m for ms in by_node.values() for m in ms), ["fable", "fable", "opus", "opus"])

    def test_without_worker_models_workers_are_called_as_before_and_the_model_is_recorded(self):
        camp = self.campaign({})
        calls = []
        r = LiveRunner(camp, self.worker(calls, accepts_model=False), indexer=lambda ids: None, round_id="iter0001",
                       checker=fixed_checker("novel"))
        out = r.run_batch([f"{ROOT}0"])
        self.assertEqual(camp.tree.get(out[0]["id"])["worker"]["model"], "opus")
        self.assertEqual({m for _, _, m in calls}, {None})

    def test_the_cli_worker_runs_the_slot_model(self):
        camp = self.campaign({"worker_models": ["fable"]})
        ws = self.root / "ws"
        ws.mkdir()
        self.assertEqual(cli.worker_agent(camp, ws, "system", model="fable").model, "fable")
        self.assertEqual(cli.worker_agent(camp, ws, "system").model, "opus")

    def test_a_bad_worker_models_setting_is_refused(self):
        for i, bad in enumerate(("fable", [], ["opus", ""], [3])):
            camp = self.campaign({"worker_models": bad}, name=f"bad{i}")
            with self.assertRaises(ValueError):
                LiveRunner(camp, self.worker([]), indexer=lambda ids: None, round_id="iter0001")


if __name__ == "__main__":
    unittest.main()
