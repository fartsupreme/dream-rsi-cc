import json
import re
import tempfile
import unittest
from pathlib import Path

from drsi.families import (OTHER, assign_families, build_frontier, build_taxonomy, family_stats,
                           load_families)
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM


def _tree(tmp, specs):
    """specs: list of (id, parent, family_hint, outcome, killed_by)."""
    t = Tree(Path(tmp) / "tree.jsonl")
    for i, parent, hint, outcome, killed in specs:
        t.add(make_node(id=i, parent=parent, created=f"2026-09-{int(i):02d}T00:00:00Z",
                        fingerprint={"mechanism": f"m{i}", "object": "o", "key_move": "k", "kind": "construction",
                                     "outcome": outcome, "killed_by": killed, "why": "w", "family_hint": hint}))
    return t


TAXO = {"families": [
    {"id": "F01", "name": "shellsort gaps", "description": "d1", "boundary": "b1"},
    {"id": "F02", "name": "lattice params", "description": "d2", "boundary": "b2"},
]}


class TaxonomyTest(unittest.TestCase):
    def test_build_saves_families_with_other_bucket(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "iso", "refuted", "speed"), ("2", "1", "lat", "partial", ""),
                          ("3", "2", "iso", "refuted", ""), ("4", "3", "lat", "refuted", ""),
                          ("5", "4", "iso", "partial", "")])
            path = Path(d) / "families.json"
            llm = ScriptedLLM(lambda p, s: TAXO)
            fams = build_taxonomy(t, llm, goal="g", path=path)
            ids = [f["id"] for f in fams["families"]]
            self.assertEqual(ids, ["F01", "F02", OTHER])
            self.assertEqual(load_families(path)["families"][0]["name"], "shellsort gaps")
            self.assertIn("m1", llm.prompts[0])  # summaries carry the mechanism

    def test_duplicate_family_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "iso", "refuted", "")])
            bad = {"families": [TAXO["families"][0], TAXO["families"][0]]}
            with self.assertRaises(ValueError):
                build_taxonomy(t, ScriptedLLM(lambda p, s: bad), goal="g", path=Path(d) / "f.json")


class AssignTest(unittest.TestCase):
    def test_assigns_known_ids_and_maps_unknown_to_other(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "iso", "refuted", ""), ("2", "1", "lat", "partial", ""),
                          ("3", "2", "???", "refuted", "")])
            t.update("3", fingerprint={"error": "omitted"})

            def answer(p, s):
                ids = re.findall(r"^(\d+)\|", p, re.M)
                table = {"1": "F01", "2": "F99"}
                return {"items": [{"id": i, "family": table.get(i, "F02")} for i in ids]}
            fams = {"families": TAXO["families"] + [{"id": OTHER, "name": "other", "description": "", "boundary": ""}]}
            n = assign_families(t, fams, ScriptedLLM(answer), batch=10, workers=1)
            again = Tree(t.path)
            self.assertEqual(again.get("1")["fingerprint"]["family"], "F01")
            self.assertEqual(again.get("2")["fingerprint"]["family"], OTHER)   # F99 unknown
            self.assertNotIn("family", again.get("3")["fingerprint"])  # unclassified: waits, not F00
            self.assertEqual(n, 2)

    def test_only_unassigned(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "iso", "refuted", ""), ("2", "1", "lat", "partial", "")])
            fp = dict(t.get("1")["fingerprint"], family="F02")
            t.update("1", fingerprint=fp)
            seen = []

            def answer(p, s):
                ids = re.findall(r"^(\d+)\|", p, re.M)
                seen.extend(ids)
                return {"items": [{"id": i, "family": "F01"} for i in ids]}
            assign_families(t, {"families": TAXO["families"]}, ScriptedLLM(answer), only_unassigned=True)
            self.assertEqual(seen, ["2"])
            self.assertEqual(t.get("1")["fingerprint"]["family"], "F02")


class StatsTest(unittest.TestCase):
    def _fams(self):
        return {"families": TAXO["families"] + [{"id": OTHER, "name": "other", "description": "", "boundary": ""}]}

    def test_dead_family_all_refuted(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "", "refuted", "speed"), ("2", "1", "", "killed", "speed"),
                          ("3", "2", "", "refuted", "ten-year clause")])
            t.update_many({i: {"fingerprint": dict(t.get(i)["fingerprint"], family="F01")} for i in "123"})
            st = {s["id"]: s for s in family_stats(t, self._fams(), plateau=3)}
            self.assertEqual(st["F01"]["n"], 3)
            self.assertEqual(st["F01"]["status"], "dead")
            self.assertEqual(st["F01"]["killed_by_top"], "speed")
            self.assertEqual(st["F01"]["last"], "3")
            self.assertEqual(st["F02"]["status"], "untried")

    def test_plateau_when_recent_attempts_do_not_beat_earlier_best(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "", "partial", ""), ("2", "1", "", "refuted", ""),
                          ("3", "2", "", "refuted", ""), ("4", "3", "", "measured", "")])
            t.update_many({i: {"fingerprint": dict(t.get(i)["fingerprint"], family="F02")} for i in "1234"})
            st = {s["id"]: s for s in family_stats(t, self._fams(), plateau=3)}
            self.assertEqual(st["F02"]["status"], "plateau")
            self.assertEqual(st["F02"]["best"], "partial")

    def test_since_best_counts_attempts_after_first_best(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "", "refuted", ""), ("2", "1", "", "partial", ""),
                          ("3", "2", "", "refuted", ""), ("4", "3", "", "partial", ""), ("5", "4", "", "killed", "")])
            t.update_many({i: {"fingerprint": dict(t.get(i)["fingerprint"], family="F02")} for i in "12345"})
            st = {s["id"]: s for s in family_stats(t, self._fams(), plateau=3)}
            self.assertEqual(st["F02"]["best_at"], "2")
            self.assertEqual(st["F02"]["since_best"], 3)

    def test_open_when_recent_attempt_improves(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "", "refuted", ""), ("2", "1", "", "partial", "")])
            t.update_many({i: {"fingerprint": dict(t.get(i)["fingerprint"], family="F02")} for i in "12"})
            st = {s["id"]: s for s in family_stats(t, self._fams(), plateau=3)}
            self.assertEqual(st["F02"]["status"], "open")


class FrontierTest(unittest.TestCase):
    def test_frontier_saved_into_families_file(self):
        with tempfile.TemporaryDirectory() as d:
            t = _tree(d, [("1", None, "", "refuted", "speed")])
            t.update("1", fingerprint=dict(t.get("1")["fingerprint"], family="F01"))
            path = Path(d) / "families.json"
            path.write_text(json.dumps({"families": TAXO["families"]}))
            sugg = {"directions": [{"direction": "try Y", "rationale": "r", "avoids": ["F01"]}]}
            llm = ScriptedLLM(lambda p, s: sugg)
            build_frontier(t, load_families(path), llm, goal="g", path=path)
            self.assertEqual(load_families(path)["frontier"][0]["direction"], "try Y")
            self.assertIn("shellsort gaps", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
