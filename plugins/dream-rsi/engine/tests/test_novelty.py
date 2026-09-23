import json
import tempfile
import unittest
from pathlib import Path

from drsi.families import OTHER
from drsi.novelty import EXIT, check, record_check
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM

FAMS = {"families": [
    {"id": "F01", "name": "shellsort gap sequences", "description": "d", "boundary": "b"},
    {"id": "F02", "name": "lattice parameter moves", "description": "d", "boundary": "b"},
    {"id": OTHER, "name": "other", "description": "", "boundary": ""},
]}


def _tree(d):
    t = Tree(Path(d) / "tree.jsonl")
    rows = [
        ("1", "shellsort gap sequence with short tail gaps", "refuted", "speed", "F01"),
        ("2", "shellsort index decoupled from tail stability", "refuted", "stability", "F01"),
        ("3", "radix bucket width and cache line density sweep", "partial", "memory", "F02"),
    ]
    for i, mech, outcome, killed, fam in rows:
        t.add(make_node(id=i, parent=None if i == "1" else str(int(i) - 1), proposal=mech,
                        fingerprint={"mechanism": mech, "object": "o", "key_move": "k", "kind": "construction",
                                     "outcome": outcome, "killed_by": killed, "why": "w", "family_hint": "h",
                                     "family": fam}))
    return t


def judge(verdict, family="F01", addresses=True, what_differs="uses a different commitment", nearest=("1",),
          doubts=""):
    return lambda p, s: {"verdict": verdict, "family": family, "nearest_ids": list(nearest),
                         "what_differs": what_differs, "targets_gate": "speed",
                         "addresses_stopper": addresses, "doubts": doubts, "rationale": "r"}


class NoveltyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = _tree(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prompt_contains_nearest_prior_attempt(self):
        llm = ScriptedLLM(judge("novel"))
        check(self.tree, FAMS, llm, "a short tail shellsort gap trick")
        self.assertIn("short tail gaps", llm.prompts[0])
        self.assertIn("shellsort gap sequences", llm.prompts[0])  # family table present

    def test_novel_exit_zero_with_ticket(self):
        r = check(self.tree, FAMS, ScriptedLLM(judge("novel", family=OTHER)), "a totally new code-based idea")
        self.assertEqual(r["verdict"], "novel")
        self.assertEqual(r["exit_code"], EXIT["novel"])
        self.assertRegex(r["ticket"], r"^[0-9a-f]{12}$")

    def test_variant_that_can_flip_the_gate_stays_variant(self):
        r = check(self.tree, FAMS, ScriptedLLM(judge("variant")), "short tail shellsort tweak")
        self.assertEqual(r["verdict"], "variant")
        self.assertEqual(r["exit_code"], EXIT["variant"])

    def test_variant_that_does_not_target_the_stopper_is_a_duplicate(self):
        r = check(self.tree, FAMS, ScriptedLLM(judge("variant", addresses=False)), "short tail shellsort tweak")
        self.assertEqual(r["verdict"], "duplicate")
        self.assertEqual(r["exit_code"], EXIT["duplicate"])
        self.assertIn("does not target", r["rule"])

    def test_doubted_variant_that_targets_the_stopper_stays_variant(self):
        # a feasibility prediction is advice, not a veto: history decides duplicates
        r = check(self.tree, FAMS, ScriptedLLM(judge("variant", addresses=True, doubts="likely too large")),
                  "short tail shellsort tweak")
        self.assertEqual(r["verdict"], "variant")
        self.assertEqual(r["doubts"], "likely too large")

    def test_variant_without_stated_difference_is_a_duplicate(self):
        r = check(self.tree, FAMS, ScriptedLLM(judge("variant", what_differs="  ")), "short tail shellsort tweak")
        self.assertEqual(r["verdict"], "duplicate")

    def test_novel_inside_dead_family_stays_novel_with_a_warning(self):
        # F01 has two refuted attempts -> dead
        r = check(self.tree, FAMS, ScriptedLLM(judge("novel", family="F01", addresses=False)), "shellsort idea")
        self.assertEqual(r["verdict"], "novel")
        self.assertTrue(any("dead" in w for w in r["warnings"]))

    def test_unknown_nearest_ids_dropped_and_known_ones_expanded(self):
        r = check(self.tree, FAMS, ScriptedLLM(judge("variant", nearest=("1", "999"))), "short walk")
        self.assertEqual([n["id"] for n in r["nearest"]], ["1"])
        self.assertEqual(r["nearest"][0]["killed_by"], "speed")

    def test_empty_history_is_novel_without_llm_call(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Tree(Path(d) / "t.jsonl")
            llm = ScriptedLLM(judge("duplicate"))
            r = check(empty, {"families": []}, llm, "anything")
            self.assertEqual(r["verdict"], "novel")
            self.assertEqual(llm.prompts, [])

    def test_in_flight_proposals_are_shown_and_citable(self):
        pending = [{"ticket": "aaaaaaaaaaaa", "proposal": "odd-only bytearray sieve of eratosthenes"}]
        llm = ScriptedLLM(judge("duplicate", family=OTHER, nearest=("pending:aaaaaaaaaaaa", "1")))
        r = check(self.tree, FAMS, llm, "a sieve of eratosthenes on odd numbers", pending=pending)
        self.assertIn("IN-FLIGHT", llm.prompts[0])
        self.assertIn("odd-only bytearray sieve", llm.prompts[0])
        self.assertEqual([n["id"] for n in r["nearest"]], ["pending:aaaaaaaaaaaa", "1"])
        self.assertEqual(r["verdict"], "duplicate")

    def test_in_flight_proposals_count_as_history(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Tree(Path(d) / "t.jsonl")
            llm = ScriptedLLM(judge("duplicate", family=OTHER, nearest=("pending:bbbbbbbbbbbb",)))
            r = check(empty, {"families": []}, llm, "same sieve",
                      pending=[{"ticket": "bbbbbbbbbbbb", "proposal": "sieve"}])
            self.assertEqual(len(llm.prompts), 1)  # no longer "empty history"
            self.assertEqual(r["verdict"], "duplicate")

    def test_record_check_appends(self):
        path = Path(self.tmp.name) / "logs" / "checks.jsonl"
        r = check(self.tree, FAMS, ScriptedLLM(judge("novel", family=OTHER)), "new")
        record_check(path, r)
        record_check(path, r)
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["ticket"], r["ticket"])


if __name__ == "__main__":
    unittest.main()
