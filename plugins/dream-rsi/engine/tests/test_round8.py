"""Round 8: a variant that attacks a family's second stopper is not a repeat.

Found in use (2026-09-26): every attempt in the campaign's main families had failed one gate first, so that gate was
each family's most common stopper, and the check's rule (a variant that does not attack what stopped its family is a
repeat) read every idea aimed at the other open gates as a duplicate -- including one the judge itself called the first
of its kind, whose family it listed as also stopped by exactly the gates the idea attacked. The session built it anyway,
on its own authority: the override a correct verdict must make unnecessary.
"""
import tempfile
import unittest
from pathlib import Path

from drsi.families import OTHER, family_stats
from drsi.novelty import check
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM

FAMS = {"families": [
    {"id": "F01", "name": "hash-table speed-ups", "description": "d", "boundary": "b"},
    {"id": OTHER, "name": "other", "description": "", "boundary": ""},
]}


def _tree(d):
    t = Tree(Path(d) / "tree.jsonl")
    rows = [("1", "open addressing", "G-CORRECT"), ("2", "robin hood probing", "G-CORRECT"),
            ("3", "vector-only probe estimate", "G-CORRECT"), ("4", "probe chain priced", "G-LATENCY ceiling"),
            ("5", "lookups at four lanes", "G-LATENCY ceiling"), ("6", "lazy deletion", "G-MEMORY")]
    for i, mech, killed in rows:
        t.add(make_node(id=i, parent=None if i == "1" else str(int(i) - 1), proposal=mech,
                        fingerprint={"mechanism": mech, "object": "o", "key_move": "k", "kind": "construction",
                                     "outcome": "refuted", "killed_by": killed, "why": "w", "family_hint": "h",
                                     "family": "F01"}))
    return t


class SecondStopperTest(unittest.TestCase):
    def test_family_stats_name_the_second_stopper_too(self):
        with tempfile.TemporaryDirectory() as d:
            st = {s["id"]: s for s in family_stats(_tree(d), FAMS)}["F01"]
            self.assertEqual((st["killed_by_top"], st["killed_by_next"]), ("G-CORRECT", "G-LATENCY ceiling"))

    def test_the_judge_sees_both_stoppers_and_is_told_either_counts(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(lambda p, s: {"verdict": "variant", "family": "F01", "nearest_ids": ["5"],
                                            "what_differs": "splits the probes across scalar and vector units",
                                            "targets_gate": "G-LATENCY ceiling", "addresses_stopper": True,
                                            "doubts": "", "rationale": "r"})
            r = check(_tree(d), FAMS, llm, "a mixed scalar-and-vector probe table")
            prompt = llm.prompts[-1]
            self.assertIn("stopped by: G-CORRECT; also G-LATENCY ceiling", prompt)
            self.assertIn("the most common one or another", prompt)
            self.assertEqual(r["verdict"], "variant")

    def test_a_variant_aimed_at_no_stopper_is_still_a_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(lambda p, s: {"verdict": "variant", "family": "F01", "nearest_ids": ["2"],
                                            "what_differs": "renames the probes", "targets_gate": "G-CORRECT",
                                            "addresses_stopper": False, "doubts": "", "rationale": "r"})
            self.assertEqual(check(_tree(d), FAMS, llm, "the probe table, renamed")["verdict"], "duplicate")


if __name__ == "__main__":
    unittest.main()
