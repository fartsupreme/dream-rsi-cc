"""Round 21: three small corrections from the audit of 2026-09-28.

- Worker models by slot: slot i ran worker_models[i % n] in every batch, and the policy lists a batch best cell
  first, so the last model always got the least promising cell and model was confounded with branch quality. The
  assignment now rotates by one slot each batch, and every frozen world records each attempt's model.
- A live "pass" meant any score above the parent's, so re-measuring the same code passed about half the time and kept
  a stalled family marked open. `live.pass_margin` sets how far above the reference a pass must be.
- Frontier suggestions came from family names alone, and 3 of 5 had already been tried. Each suggestion is now
  checked against the record: a duplicate or off-target one is dropped, and a variant carries its nearest attempts.
"""
import json
import tempfile
import unittest
from pathlib import Path

from drsi.families import build_frontier, load_families
from drsi.live import LiveRunner, live_outcome
from drsi.question import ROOT
from drsi.worlds import world_from_tree
from tests.helpers import ScriptedLLM
from tests.test_families import TAXO, _tree
from tests.test_live import fixed_checker
from tests.test_round10 import WorkerModelsTest


class ModelRotationTest(unittest.TestCase):
    def setUp(self):
        self.t = WorkerModelsTest()
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def test_the_slot_each_model_gets_rotates_every_batch(self):
        camp = self.t.campaign({"worker_models": ["a", "b", "c"]})
        r = LiveRunner(camp, WorkerModelsTest.worker([]), indexer=lambda ids: None, round_id="iter0001",
                       checker=fixed_checker("novel"))
        first = r.run_batch([f"{ROOT}{i}" for i in range(3)])
        second = r.run_batch([o["id"] for o in first])
        models = lambda out: [camp.tree.get(o["id"])["worker"]["model"] for o in out]  # noqa: E731
        self.assertEqual(models(first), ["a", "b", "c"])
        self.assertEqual(models(second), ["b", "c", "a"])

    def test_a_frozen_world_records_each_attempts_model(self):
        camp = self.t.campaign({"worker_models": ["a", "b"]})
        r = LiveRunner(camp, WorkerModelsTest.worker([]), indexer=lambda ids: None, round_id="iter0001",
                       checker=fixed_checker("novel"))
        r.run_batch([f"{ROOT}0", f"{ROOT}1"])
        w = world_from_tree(camp.tree, "iter0001", ids=set(r.ids))
        self.assertEqual(sorted(n["model"] for n in w["nodes"]), ["a", "b"])


class PassMarginTest(unittest.TestCase):
    def test_a_pass_must_clear_the_margin(self):
        node = {"valid": True, "score": 1.3, "fail_class": "ok"}
        self.assertEqual(live_outcome(node, 1.0)[0], "pass")
        self.assertEqual(live_outcome(node, 1.0, margin=0.5)[0], "measured")
        self.assertEqual(live_outcome(dict(node, score=1.6), 1.0, margin=0.5)[0], "pass")

    def test_the_campaign_margin_reaches_recorded_attempts(self):
        t = WorkerModelsTest()
        t.setUp()
        try:
            camp = t.campaign({})
            camp.update_config(lambda raw: raw.setdefault("live", {}).__setitem__("pass_margin", 5.0))
            camp.update_config(lambda raw: raw.__setitem__("baseline", 1.0))
            r = LiveRunner(camp, WorkerModelsTest.worker([]), indexer=lambda ids: None, round_id="iter0001",
                           checker=fixed_checker("novel"))
            out = r.run_batch([f"{ROOT}0"])
            node = camp.tree.get(out[0]["id"])
            self.assertEqual(node["score"], 2.0)  # the worker writes 2 against a baseline of 1, inside the margin
            self.assertEqual(node["artifacts"]["outcome"], "measured")
        finally:
            t.tearDown()


class FrontierCheckTest(unittest.TestCase):
    def test_suggestions_already_tried_are_dropped_and_variants_carry_their_nearest(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "", "refuted", "speed")])
            t.update("1", fingerprint=dict(t.get("1")["fingerprint"], family="F01"))
            path = Path(d) / "families.json"
            path.write_text(json.dumps({"families": TAXO["families"]}))
            llm = ScriptedLLM(lambda p, s: {"directions": [
                {"direction": "try the old thing again", "rationale": "r", "avoids": []},
                {"direction": "try a thing aimed at nothing", "rationale": "r", "avoids": []},
                {"direction": "try a new twist on it", "rationale": "r", "avoids": []}]})
            verdicts = {"try the old thing again": ("duplicate", ["1"]), "try a thing aimed at nothing": ("off_target", []),
                        "try a new twist on it": ("variant", ["1"])}

            def checker(text):
                v, near = verdicts[text]
                return {"verdict": v, "nearest": [{"id": i} for i in near]}
            out = build_frontier(t, load_families(path), llm, goal="g", path=path, checker=checker)
            self.assertEqual([d["direction"] for d in out], ["try a new twist on it"])
            self.assertEqual(out[0]["near"], ["1"])
            dropped = load_families(path)["frontier_dropped"]
            self.assertEqual(sorted(x["verdict"] for x in dropped), ["duplicate", "off_target"])


if __name__ == "__main__":
    unittest.main()
