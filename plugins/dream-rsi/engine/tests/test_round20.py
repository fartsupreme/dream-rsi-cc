"""Round 20: the novelty gate's measured errors.

Found by a planted test set (2026-09-28: 88 attempts, 65 labelled probes, the plugin's own check() with a real
judge): the gate never called a repeat new (0 of 70), but called about 10% of non-repeats duplicates, and a duplicate
verdict is final. Causes, each fixed here:
- a variant the judge found new but not aimed at its family's stopper was converted to a final duplicate; it is
  now its own verdict, off_target, which asks for a revision and is not final;
- a located fix to an attempt a bug stopped before it was measured was judged a repeat (Dream-RSI's own prompt
  allows that retry); it is now a retry verdict, guarded;
- retrieval matched the proposal's raw words only, so renamed repeats missed their target; the query now carries
  the proposal's own fingerprint, and ids the proposal cites come first;
- a citation counted if it resolved, even when the judge had not been shown it; now only shown ones count;
- a duplicate is confirmed by a second pass over the full cited records before it is final.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from drsi.families import OTHER
from drsi.novelty import EXIT, check
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM

FAMS = {"families": [
    {"id": "F01", "name": "gap sequences", "description": "d", "boundary": "b"},
    {"id": "F02", "name": "bucket merges", "description": "d", "boundary": "b"},
    {"id": "F03", "name": "widget tables", "description": "d", "boundary": "b"},
    {"id": OTHER, "name": "other", "description": "", "boundary": ""},
]}


def tree(d) -> Tree:
    t = Tree(Path(d) / "tree.jsonl")
    rows = [  # id, mechanism, outcome, killed_by, family
        ("1", "shellsort gap sequence with short tail gaps", "refuted", "speed", "F01"),
        ("2", "shellsort gap sequence decoupled from tail", "refuted", "speed", "F01"),
        ("3", "radix bucket merge with cache line density", "partial", "memory", "F02"),
        ("4", "cache oblivious bucket merge", "inconclusive", "an index bug crashed it before any measurement",
         "F02"),
        ("5", "alpha widget table lookup", "partial", "memory", "F03"),
    ] + [(str(i), f"filler scheme number {i}", "refuted", "speed", OTHER) for i in range(6, 13)]
    for i, mech, outcome, killed, fam in rows:
        t.add(make_node(id=i, parent=None, proposal=mech,
                        fingerprint={"mechanism": mech, "object": "o", "key_move": "k", "kind": "construction",
                                     "outcome": outcome, "killed_by": killed, "why": "w", "family_hint": "h",
                                     "family": fam}))
    return t


def llm(judge: dict, confirm: dict | None = None, fp: str = ""):
    """Scripted calls told apart by schema: the query fingerprint, the judge, the duplicate confirmation."""
    def fn(prompt, schema):
        props = schema.get("properties", {})
        if "items" in props:
            return {"items": [{"id": "proposal", "mechanism": fp, "object": "", "key_move": "", "kind": "other",
                               "outcome": "built", "killed_by": "", "why": "", "family_hint": ""}]}
        if "same_mechanism_as" in props:
            return confirm or {"same_mechanism_as": "", "cited_was_measured": True,
                               "proposal_names_located_fix": False, "addresses_recorded_stopper": False,
                               "verdict": "confirm_duplicate", "rationale": "r"}
        return {"family": "F02", "nearest_ids": [], "what_differs": "", "targets_gate": "memory",
                "addresses_stopper": True, "doubts": "", "rationale": "r", "retry_of": ""} | judge
    return ScriptedLLM(fn)


def nearest_section(prompt: str) -> list[str]:
    body = prompt.split("NEAREST PRIOR ATTEMPTS\n", 1)[1].split("\n\n", 1)[0]
    return [line.split()[0] for line in body.splitlines() if line.startswith("#")]


def judge_prompt(scripted) -> str:
    return next(p for p in scripted.prompts if "NEAREST PRIOR ATTEMPTS" in p)


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = tree(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_new_variant_not_aimed_at_the_stopper_is_off_target_not_a_duplicate(self):
        r = check(self.t, FAMS, llm({"verdict": "variant", "what_differs": "a new merge order",
                                     "addresses_stopper": False, "nearest_ids": ["3"]}), "a bucket merge, reordered")
        self.assertEqual(r["verdict"], "off_target")
        self.assertEqual(r["exit_code"], EXIT["off_target"])
        self.assertNotEqual(EXIT["off_target"], EXIT["duplicate"])
        self.assertIn("stopper", r["rule"])

    def test_a_located_fix_to_an_unmeasured_attempt_is_a_retry(self):
        r = check(self.t, FAMS, llm({"verdict": "retry", "retry_of": "4", "nearest_ids": ["4"],
                                     "what_differs": "fixes the off-by-one index at merge step 3"}),
                  "cache oblivious bucket merge again, with the index bug at merge step 3 fixed")
        self.assertEqual(r["verdict"], "retry")
        self.assertEqual(r["exit_code"], EXIT["retry"])
        self.assertEqual(r["retry_of"], "4")

    def test_a_retry_of_an_attempt_the_judge_was_not_shown_is_not_accepted(self):
        r = check(self.t, FAMS, llm({"verdict": "retry", "retry_of": "5", "nearest_ids": ["4"],
                                     "what_differs": "fixes a bug"}),
                  "cache oblivious bucket merge again, with the index bug fixed", k=1, confirm=True)
        self.assertNotEqual(r["verdict"], "retry")

    def test_a_retry_in_a_dead_family_is_not_a_retry(self):
        # canary bff1: a bug fixed in one attempt of a family whose mechanism other attempts measured and refuted;
        # the judge counted the bug as the stopper it addresses
        r = check(self.t, FAMS, llm({"verdict": "retry", "retry_of": "1", "nearest_ids": ["1"], "family": "F01",
                                     "what_differs": "fixes an off-by-one", "addresses_stopper": True}),
                  "shellsort gap sequence with short tail gaps, off-by-one fixed", confirm=True)
        self.assertEqual(r["verdict"], "duplicate")

    def test_a_retry_of_a_measured_attempt_is_not_a_retry(self):
        r = check(self.t, FAMS, llm({"verdict": "retry", "retry_of": "3", "nearest_ids": ["3"],
                                     "what_differs": "fixes a rounding slip"}),
                  "radix bucket merge with cache line density, rounding slip fixed", confirm=True)
        self.assertEqual(r["verdict"], "duplicate")
        self.assertIn("measured", r["rule"])

    def test_the_confirmation_reads_a_cited_in_flight_proposal(self):
        # canary pd2-pd4: the judge cited the in-flight proposal, the confirmation saw only recorded attempts
        seen = {}

        def confirm_fn(prompt, schema):
            seen["prompt"] = prompt
            return {"same_mechanism_as": "pending:iter0001-002", "cited_was_measured": False,
                    "proposal_names_located_fix": False, "addresses_recorded_stopper": False,
                    "verdict": "confirm_duplicate", "rationale": "the in-flight proposal"}
        base = llm({"verdict": "duplicate", "nearest_ids": ["pending:iter0001-002", "3"]})
        s = ScriptedLLM(lambda p, sc: confirm_fn(p, sc) if "same_mechanism_as" in sc.get("properties", {})
                        else base.fn(p, sc))
        pending = [{"node": "iter0001-002", "ticket": "t", "proposal": "a two-pass merge of radix buckets by density"}]
        r = check(self.t, FAMS, s, "radix buckets merged in two passes by density", pending=pending, confirm=True)
        self.assertIn("two-pass merge of radix buckets by density", seen["prompt"])
        self.assertEqual(r["verdict"], "duplicate")

    def test_a_duplicate_is_confirmed_before_it_is_final(self):
        confirm = {"same_mechanism_as": "3", "cited_was_measured": True, "proposal_names_located_fix": False,
                   "addresses_recorded_stopper": False, "verdict": "confirm_duplicate", "rationale": "same"}
        s = llm({"verdict": "duplicate", "nearest_ids": ["3"]}, confirm=confirm)
        r = check(self.t, FAMS, s, "radix bucket merge, cache line density, again", confirm=True)
        self.assertEqual(r["verdict"], "duplicate")
        self.assertTrue(any("same_mechanism_as" in json.dumps(p) or "CITED RECORDS" in p for p in s.prompts))

    def test_a_duplicate_the_confirmation_overturns_is_not_final(self):
        confirm = {"same_mechanism_as": "", "cited_was_measured": True, "proposal_names_located_fix": False,
                   "addresses_recorded_stopper": True, "verdict": "variant", "rationale": "differs in the merge"}
        r = check(self.t, FAMS, llm({"verdict": "duplicate", "nearest_ids": ["3"]}, confirm=confirm),
                  "radix bucket merge with a two-pass density split", confirm=True)
        self.assertEqual(r["verdict"], "variant")
        self.assertIn("confirmation", r["rule"])

    def test_the_confirmation_may_find_a_retry(self):
        confirm = {"same_mechanism_as": "", "cited_was_measured": False, "proposal_names_located_fix": True,
                   "addresses_recorded_stopper": True, "verdict": "retry", "rationale": "never measured"}
        r = check(self.t, FAMS, llm({"verdict": "duplicate", "nearest_ids": ["4"]}, confirm=confirm),
                  "cache oblivious bucket merge with the crash at merge step 3 fixed", confirm=True)
        self.assertEqual((r["verdict"], r["retry_of"]), ("retry", "4"))


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = tree(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_querys_fingerprint_finds_a_renamed_repeat(self):
        proposal = "a gizmo ledger keyed by filler scheme labels"
        plain = llm({"verdict": "variant", "what_differs": "d"}, fp="")
        check(self.t, FAMS, plain, proposal, k=2, query_fp=True)
        self.assertNotIn("#5", nearest_section(judge_prompt(plain)))
        fp = llm({"verdict": "variant", "what_differs": "d"}, fp="alpha widget table lookup")
        check(self.t, FAMS, fp, proposal, k=2, query_fp=True)
        self.assertIn("#5", nearest_section(judge_prompt(fp)))

    def test_attempts_the_proposal_cites_come_first(self):
        s = llm({"verdict": "variant", "what_differs": "d"})
        check(self.t, FAMS, s, "builds on #12 with a filler scheme twist", k=3)
        self.assertEqual(nearest_section(judge_prompt(s))[0], "#12")

    def test_only_citations_the_judge_was_shown_count(self):
        r = check(self.t, FAMS, llm({"verdict": "variant", "what_differs": "d", "nearest_ids": ["3", "999", "5"]}),
                  "radix bucket merge with cache line density, reweighted per pass", k=1)
        self.assertEqual(r["verdict"], "variant")
        self.assertEqual([b["id"] for b in r["nearest"]], ["3"])


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "novelty_canary"


class CanaryRetrievalTest(unittest.TestCase):
    """The planted test set's 45 probes that repeat or build on a recorded attempt: the attempt they target must be
    among those the judge is shown. No model calls: each probe's fingerprint is the one the classifier gave it."""

    def shown_rate(self, query_fp: bool) -> tuple[int, int]:
        import shutil
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(FIXTURE / "tree.jsonl", Path(d) / "tree.jsonl")
            t = Tree(Path(d) / "tree.jsonl")
            fams = json.loads((FIXTURE / "families.json").read_text())
            fps = json.loads((FIXTURE / "probe_fingerprints.json").read_text())
            probes = [json.loads(line) for line in (FIXTURE / "probes.jsonl").read_text().splitlines()]
            probes = [p for p in probes if p["targets"] and all(x in t for x in p["targets"]) and not p["pending"]]
            hit = 0
            for p in probes:
                fp = fps[p["pid"]]
                s = ScriptedLLM(lambda prompt, schema, fp=fp: {"items": [dict(fp, id="proposal")]}
                                if "items" in schema.get("properties", {}) else
                                {"verdict": "variant", "family": "F00", "nearest_ids": [], "what_differs": "d",
                                 "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": "r",
                                 "retry_of": ""})
                r = check(t, fams, s, p["text"], k=8, plateau=3, query_fp=query_fp)
                if r["rule"] == "identical proposal":  # matched by its text hash, before retrieval
                    ids = {b["id"] for b in r["nearest"]}
                else:
                    prompt = judge_prompt(s)
                    shown = prompt.split("NEAREST PRIOR ATTEMPTS\n", 1)[1].split("IN-FLIGHT PROPOSALS", 1)[0]
                    ids = {line.split()[0].lstrip("#") for line in shown.splitlines() if line.startswith("#")}
                hit += any(x in ids for x in p["targets"])
            return hit, len(probes)

    def test_every_targeted_attempt_is_shown_to_the_judge(self):
        hit, n = self.shown_rate(query_fp=True)
        self.assertEqual(n, 45)
        self.assertEqual(hit, n)

    def test_without_the_query_fingerprint_some_are_missed(self):
        hit, n = self.shown_rate(query_fp=False)
        self.assertLess(hit, n)  # the measured baseline was 42 of 45 (renamed repeats)


class ProductionPathTest(unittest.TestCase):
    def test_every_check_run_fingerprints_the_query_and_confirms_a_duplicate(self):
        from drsi.checks import run_check
        from drsi.store import Campaign
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("t", {"goal": "g"}, home=Path(d))
            tree(camp.root)  # the record lives at the campaign's tree.jsonl
            camp.families_path.write_text(json.dumps(FAMS))
            s = llm({"verdict": "duplicate", "nearest_ids": ["3"]})
            r = run_check(camp, "radix bucket merge, cache line density, once more", s)
            kinds = ["fp" if "ATTEMPT id=\"proposal\"" in p else "confirm" if "CITED RECORDS" in p else "judge"
                     for p in s.prompts]
            self.assertEqual(kinds, ["fp", "judge", "confirm"])
            self.assertEqual(r["verdict"], "duplicate")


class ConsumersTest(unittest.TestCase):
    def test_a_retry_is_a_passing_claim_and_off_target_is_not(self):
        from drsi.checks import pending_checks
        from drsi.store import Campaign
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("t", {"goal": "g"}, home=Path(d))
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            camp.checks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(camp.checks_path, "w") as fh:
                fh.write(json.dumps({"node": "iter0001-001", "verdict": "retry", "proposal": "p", "checked": now}) + "\n")
                fh.write(json.dumps({"node": "iter0001-002", "verdict": "off_target", "proposal": "q",
                                     "checked": now}) + "\n")
            self.assertEqual([p["node"] for p in pending_checks(camp)], ["iter0001-001"])

    def test_the_live_loop_asks_for_a_revision_on_off_target(self):
        from drsi.live import LiveRunner
        from drsi.question import ROOT
        from tests.test_live import fixed_checker, stub_worker
        from tests.test_round10 import WorkerModelsTest
        t = WorkerModelsTest()
        t.setUp()
        try:
            camp = t.campaign({})
            w = stub_worker(proposals=["an off-target idea", "an idea aimed at the stopper"])
            r = LiveRunner(camp, w, indexer=lambda ids: None, round_id="iter0001",
                           checker=fixed_checker("off_target", "variant"))
            out = r.run_batch([f"{ROOT}0"])
            node = camp.tree.get(out[0]["id"])
            self.assertEqual(node["fail_class"], "ok")
            second = [c for c in w.calls if "PHASE: PROPOSE" in c["system"]][1]["system"]
            self.assertIn("DOES NOT TARGET WHAT STOPPED", second)
            r2 = LiveRunner(camp, stub_worker(), indexer=lambda ids: None, round_id="iter0002",
                            checker=fixed_checker("off_target"))
            out2 = r2.run_batch([f"{ROOT}0"])
            self.assertEqual(camp.tree.get(out2[0]["id"])["fail_class"], "not_novel")
        finally:
            t.tearDown()


if __name__ == "__main__":
    unittest.main()
