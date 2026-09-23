import json
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.live import LiveRunner, live_round, load_policy, run_cycles
from drsi.store import Campaign
from drsi.worlds import load_worlds
from tests.test_policy import SEED
from tests.test_scorer_workspace import make_repo

SCORER = "python3 -c \"import json;print(json.dumps({'score': int(open('value.txt').read()), 'valid': True}))\""


def stub_worker(delta_fn=lambda ws: 1, touch=None, ok=True, proposals=None):
    """Phase-aware stub. PROPOSE: returns the next proposal text. IMPLEMENT: adds delta to value.txt and
    self-reports an absurd score (which must be ignored)."""
    calls = []
    queue = list(proposals or [])

    def run(workspace: Path, prompt: str, system: str) -> AgentResult:
        calls.append({"ws": workspace, "prompt": prompt, "system": system})
        if not ok:
            return AgentResult(ok=False, error="worker crashed")
        if "PHASE: PROPOSE" in system:
            text = queue.pop(0) if queue else "add one to value.txt"
            return AgentResult(ok=True, structured={"proposal": text, "summary": "", "self_reported_score": None,
                                                    "notes": ""})
        v = int((workspace / "value.txt").read_text())
        (workspace / "value.txt").write_text(f"{v + delta_fn(workspace)}\n")
        if touch:
            (workspace / touch).write_text("sneaky\n")
        report = {"proposal": f"add to value from {v}", "summary": "did it", "self_reported_score": 999, "notes": ""}
        return AgentResult(ok=True, structured=report, session_id="s", secs=0.1)
    run.calls = calls
    return run


def fixed_checker(*verdicts):
    """Returns the given verdicts in order (then the last one forever); records what it judged."""
    seen = []

    def check(proposal, node):
        v = verdicts[min(len(seen), len(verdicts) - 1)]
        seen.append((node, proposal))
        return {"verdict": v, "proposal": proposal, "ticket": f"t{len(seen)}", "rule": "", "family": "F00",
                "rationale": "r", "what_differs": "d", "warnings": [], "doubts": "", "nearest": [],
                "targets_gate": "", "addresses_stopper": True, "exit_code": 0}
    check.seen = seen
    return check


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src = make_repo(root)
        self.camp = Campaign.create("toy", {
            "goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
            "workspace": {"repo": str(self.src), "mutable": ["value.txt"]},
            "search": {"W": 2, "K1": 3, "plateau": 3},
            "dream": {"M": 1, "betas": [0.0, 1.0], "lambda": 0.25, "beta1": 0.01, "beta2": 0.01},
            "live": {"require_check": False},
        }, home=root / "home")
        self.policy = load_policy(SEED)

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, worker, **kw):
        return LiveRunner(self.camp, worker, indexer=lambda ids: None, round_id="iter0001", **kw)

    def test_round_grows_tree_scored_by_scorer_not_worker(self):
        r = self.runner(stub_worker())
        summary = live_round(self.camp, self.policy, r)
        tree = self.camp.tree
        nodes = [tree.get(i) for i in r.ids]
        self.assertGreater(len(nodes), 0)
        self.assertLessEqual(len(nodes), 2 * 3)
        self.assertTrue(all(n["score"] != 999 for n in nodes))
        self.assertTrue(all(n["source"] == "live" for n in nodes))
        self.assertEqual(max(n["score"] for n in nodes), summary["best_score"])
        children = [n for n in nodes if n["parent"]]
        for c in children:  # a child starts from its parent's committed state: its value is parent + 1
            self.assertEqual(c["score"], tree.get(c["parent"])["score"] + 1)
        worlds = load_worlds(self.camp.root / "trace_pool")
        self.assertEqual([w["id"] for w in worlds], ["iter0001"])
        self.assertEqual(len(worlds[0]["nodes"]), len(nodes))

    def test_baseline_measured_on_untouched_base(self):
        r = self.runner(stub_worker())
        live_round(self.camp, self.policy, r)
        self.assertEqual(self.camp.config["baseline"], 1.0)

    def test_out_of_scope_edit_invalidates_attempt(self):
        r = self.runner(stub_worker(touch="README"))
        live_round(self.camp, self.policy, r)
        n = self.camp.tree.get(r.ids[0])
        self.assertFalse(n["valid"])
        self.assertEqual(n["fail_class"], "out_of_scope")
        self.assertIn("README", n["artifacts"]["out_of_scope"])

    def test_worker_failure_recorded_as_agent_error(self):
        r = self.runner(stub_worker(ok=False))
        live_round(self.camp, self.policy, r)
        n = self.camp.tree.get(r.ids[0])
        self.assertEqual(n["fail_class"], "agent_error")
        self.assertFalse(n["valid"])

    def _require_check(self):
        cfg = json.loads(self.camp.config_path.read_text())
        cfg["live"]["require_check"] = True
        cfg["search"]["W"] = 1  # one attempt at a time: the stub's proposal queue is shared
        cfg["search"]["K1"] = 1
        self.camp.save_config(cfg)

    def test_duplicate_proposal_is_sent_back_then_the_accepted_one_is_built(self):
        self._require_check()
        w = stub_worker(proposals=["same old idea", "a new idea"])
        chk = fixed_checker("duplicate", "novel")
        r = self.runner(w, checker=chk)
        live_round(self.camp, self.policy, r)
        n = self.camp.tree.get(r.ids[0])
        self.assertEqual(n["fail_class"], "ok")
        self.assertEqual(n["proposal"], "a new idea")
        self.assertEqual([c["verdict"] for c in n["artifacts"]["checks"]], ["duplicate", "novel"])
        second_propose = [c for c in w.calls if "PHASE: PROPOSE" in c["system"]][1]["system"]
        self.assertIn("REPEATS HISTORY", second_propose)

    def test_always_duplicate_is_not_novel_and_never_built(self):
        self._require_check()
        w = stub_worker()
        r = self.runner(w, checker=fixed_checker("duplicate"))
        live_round(self.camp, self.policy, r)
        n = self.camp.tree.get(r.ids[0])
        self.assertEqual(n["fail_class"], "not_novel")
        self.assertEqual(n["artifacts"]["changed"], [])
        self.assertFalse(any("PHASE: IMPLEMENT" in c["system"] for c in w.calls if c["ws"].name == r.ids[0]))

    def test_brief_carries_map_protocol_and_parent(self):
        w = stub_worker()
        r = self.runner(w)
        live_round(self.camp, self.policy, r)
        first = w.calls[0]["system"]
        self.assertIn("MAP OF EVERYTHING TRIED", first)
        self.assertIn("maximise value.txt", first)
        child_calls = [c for c in w.calls if "PARENT ATTEMPT" in c["system"]]
        self.assertTrue(child_calls)

    def test_run_cycles_alternates_live_and_dream(self):
        def developer(sb, prompt):
            return AgentResult(ok=True)  # leaves the policy unchanged
        rep = run_cycles(self.camp, 2, worker_fn=stub_worker(), developer=developer,
                         indexer=lambda ids: None)
        self.assertEqual(len(rep["rounds"]), 2)
        self.assertEqual([w["id"] for w in load_worlds(self.camp.root / "trace_pool")], ["iter0001", "iter0002"])
        self.assertEqual(len(list((self.camp.root / "logs").glob("dream-*.json"))), 2)
        self.assertTrue((self.camp.root / "policy" / "method.py").exists())


if __name__ == "__main__":
    unittest.main()
