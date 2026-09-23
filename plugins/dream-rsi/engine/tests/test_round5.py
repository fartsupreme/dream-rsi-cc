"""Fifth-review findings on the newest code, each reproduced here first."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.families import OTHER, assign_families
from drsi.importer import import_jsonl
from drsi.live import LiveRunner, live_round, next_round_id
from drsi.novelty import check, record_check
from drsi.scorer import run_scorer
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM, fp_for
from tests.test_integrity_live import LiveIntegrityBase, bump
from tests.test_live import fixed_checker


def phase_worker(propose_action=None, proposals=None):
    queue = list(proposals or [])

    def run(ws, prompt, system):
        if "PHASE: PROPOSE" in system:
            if propose_action:
                propose_action(ws)
            text = queue.pop(0) if queue else "idea"
            return AgentResult(ok=True, structured={"proposal": text, "summary": "", "self_reported_score": None,
                                                    "notes": ""})
        bump(ws)
        return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                "notes": ""})
    return run


class LiveRound5Test(LiveIntegrityBase):
    def setUp(self):
        super().setUp()
        raw = json.loads(self.camp.config_path.read_text())
        raw["live"]["require_check"] = True
        raw["search"]["K1"] = 1
        self.camp.save_config(raw)

    def run_round(self, worker, checker, rid="iter0001"):
        r = LiveRunner(self.camp, worker, indexer=lambda ids: None, round_id=rid, checker=checker)
        live_round(self.camp, self.policy, r)
        return [self.camp.tree.get(i) for i in r.ids]

    def test_propose_phase_edits_are_discarded(self):
        def sneaky(ws):
            (ws / "value.txt").write_text("100\n")
        nodes = self.run_round(phase_worker(sneaky), fixed_checker("novel"))
        self.assertEqual(nodes[0]["score"], 2.0)  # built from the clean parent state (1) + 1, not from 100

    def test_rejected_attempt_commits_nothing_of_its_own(self):
        def sneaky(ws):
            (ws / "value.txt").write_text("100\n")
        nodes = self.run_round(phase_worker(sneaky), fixed_checker("duplicate"))
        self.assertEqual(nodes[0]["fail_class"], "not_novel")
        self.assertEqual(nodes[0]["artifacts"]["changed"], [])

    def test_stale_proposal_file_is_not_rejudged(self):
        seen = []

        def chk(proposal, node):
            seen.append(proposal)
            return fixed_checker("duplicate" if len(seen) == 1 else "novel")(proposal, node)

        def writes_then_structured(ws, prompt, system):
            if "PHASE: PROPOSE" in system:
                pf = self.camp.root / "work" / "_proposals" / ws.name / "proposal.txt"
                if not seen:
                    pf.write_text("first idea")
                    text = "first idea"
                else:
                    text = "second idea"  # only in the report this time
                return AgentResult(ok=True, structured={"proposal": text, "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            bump(ws)
            return AgentResult(ok=True, structured={})
        self.run_round(writes_then_structured, chk)
        self.assertEqual(seen, ["first idea", "second idea"])

    def test_judge_outage_is_procedural_not_a_killed_idea(self):
        def down(proposal, node):
            raise RuntimeError("claude CLI timed out")
        nodes = self.run_round(phase_worker(), down)
        self.assertEqual(nodes[0]["fail_class"], "orchestrator_error")
        self.assertEqual(nodes[0]["fingerprint"]["outcome"], "inconclusive")

    def test_outcome_is_recorded_with_the_node(self):
        nodes = self.run_round(phase_worker(), fixed_checker("novel"))
        self.assertEqual(nodes[0]["artifacts"]["outcome"], "pass")

    def test_round_id_skips_ids_that_only_left_branches_behind(self):
        from drsi.workspace import _git
        self.run_round(phase_worker(), fixed_checker("novel"))
        # simulate an interrupted round 2 that created a branch but recorded nothing
        _git(self.camp.root / "repo", "branch", "drsi/iter0002-001")
        self.assertEqual(next_round_id(self.camp), "iter0003")


class ScorerTempTest(unittest.TestCase):
    @unittest.skipUnless(Path("/usr/bin/sandbox-exec").exists(), "macOS only")
    def test_scorer_cannot_write_shared_temp_but_has_its_own(self):
        with tempfile.TemporaryDirectory() as ws:
            marker = Path("/private/tmp") / "drsi_round5_escape_marker.txt"
            if marker.exists():
                marker.unlink()
            cmd = (f"echo x > {marker}; python3 -c \"import tempfile; tempfile.NamedTemporaryFile().write(b'ok')\" "
                   "&& echo '{\"score\": 1, \"valid\": true}'")
            r = run_scorer(cmd, ws, timeout=30, sandbox=True)
            self.assertFalse(marker.exists())
            self.assertTrue(r["valid"], r)


class DataRound5Test(unittest.TestCase):
    def test_error_fingerprints_are_not_filed_under_other(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, fingerprint={"error": "classifier down"}))
            assign_families(t, {"families": [{"id": "F01", "name": "n", "description": "", "boundary": ""}]},
                            ScriptedLLM(lambda p, s: {"items": []}))
            self.assertNotIn("family", Tree(t.path).get("1")["fingerprint"])

    def test_signs_and_operators_are_not_normalised_away(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, proposal="Set the damping exponent to -1/2.", fingerprint={"mechanism": "m"}))
            llm = ScriptedLLM(lambda p, s: {"verdict": "variant", "family": OTHER, "nearest_ids": ["1"], "what_differs": "sign",
                                            "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": ""})
            r = check(t, {"families": []}, llm, "Set the damping exponent to 1/2.")
            self.assertEqual(r["verdict"], "variant")
            self.assertEqual(len(llm.prompts), 1)

    def test_truncated_long_proposal_resubmitted_verbatim_is_a_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            long = "a careful long proposal " * 80
            from drsi.novelty import text_hash
            t.add(make_node(id="1", parent=None, proposal=long[:1200] + "…", ext={"proposal_sha": text_hash(long)},
                            fingerprint={"mechanism": "m"}))
            r = check(t, {"families": []}, ScriptedLLM(lambda p, s: {"verdict": "novel"}), long)
            self.assertEqual(r["verdict"], "duplicate")

    def test_non_string_ledger_fields_do_not_abort_import(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "l.jsonl"
            src.write_text("\n".join(json.dumps(r) for r in [
                {"id": 1, "candidate": 42, "falsifiable": True, "verdict": "v"},
                {"id": 2, "candidate": ["a", "b"], "construction": {"name": 7}, "verdict": "v"}]) + "\n")
            t = Tree(Path(d) / "t.jsonl")
            self.assertEqual(import_jsonl(t, src), 2)

    def test_checks_log_torn_tail_does_not_eat_the_next_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checks.jsonl"
            path.write_bytes(b'{"node": "a", "verdict": "cl\xe2\x80')  # torn inside a multi-byte character
            record_check(path, {"node": "b", "verdict": "claim", "proposal": "Idea Y", "checked": "2026-01-01T00:00:00Z"})
            lines = path.read_text(errors="replace").splitlines()
            self.assertEqual(json.loads(lines[-1])["node"], "b")

    def test_cycle_is_broken_on_the_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "l.jsonl"
            src.write_text("\n".join(json.dumps(r) for r in [
                {"key": "c", "from": "a", "idea": "c"}, {"key": "a", "from": "b", "idea": "a"},
                {"key": "b", "from": "a", "idea": "b"}]) + "\n")
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, src, preset="generic", field_map={"id": "key", "parent": "from", "proposal": "idea"})
            self.assertEqual(t.get("c")["parent"], "a")


if __name__ == "__main__":
    unittest.main()
