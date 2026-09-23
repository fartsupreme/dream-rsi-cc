import tempfile
import unittest
from pathlib import Path

from drsi.families import OTHER
from drsi.mapview import render_map
from drsi.store import Tree, make_node


def _tree(d, n=12, fams=("F01", "F02")):
    t = Tree(Path(d) / "tree.jsonl")
    for i in range(1, n + 1):
        fam = fams[i % len(fams)]
        outcome = "refuted" if fam == "F01" else ("partial" if i == 11 else "refuted")
        t.add(make_node(id=str(i), parent=None if i == 1 else str(i - 1), proposal=f"proposal {i}",
                        created=f"2026-09-{i:02d}T00:00:00Z",
                        fingerprint={"mechanism": f"mechanism {i}", "outcome": outcome, "killed_by": "speed",
                                     "family": fam, "why": "w", "family_hint": "h"}))
    return t


FAMS = {"families": [
    {"id": "F01", "name": "dead idea", "description": "d", "boundary": ""},
    {"id": "F02", "name": "live idea", "description": "d", "boundary": ""},
    {"id": "F03", "name": "never tried", "description": "d", "boundary": ""},
    {"id": OTHER, "name": "other", "description": "", "boundary": ""},
], "frontier": [{"direction": "try the unexplored thing", "rationale": "r", "avoids": ["F01"]}]}


class MapTest(unittest.TestCase):
    def test_sections_and_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            text = render_map(_tree(d), FAMS, goal="break the wall", max_chars=16000)
            self.assertIn("break the wall", text)
            self.assertIn("12 attempts", text)
            self.assertIn("drsi check", text)
            self.assertIn("| F01 | dead idea |", text)
            self.assertIn("dead", text)
            self.assertIn("try the unexplored thing", text)
            self.assertIn("suggestion", text.lower())
            self.assertIn("#12", text)

    def test_status_order_open_before_dead_before_untried(self):
        with tempfile.TemporaryDirectory() as d:
            text = render_map(_tree(d), FAMS, goal="g", max_chars=16000)
            self.assertLess(text.index("| F02 |"), text.index("| F01 |"))
            self.assertLess(text.index("| F01 |"), text.index("| F03 |"))

    def test_recent_section_lists_last_ten(self):
        with tempfile.TemporaryDirectory() as d:
            text = render_map(_tree(d), FAMS, goal="g", max_chars=16000)
            recent = text.split("## Last 10 attempts")[1]
            self.assertIn("#3 ", recent)
            self.assertNotIn("#2 ", recent)

    def test_hard_cap_truncates_family_rows(self):
        many = {"families": [{"id": f"F{i:02d}", "name": f"family number {i} " + "x" * 60, "description": "d",
                              "boundary": ""} for i in range(1, 80)] + [FAMS["families"][-1]]}
        with tempfile.TemporaryDirectory() as d:
            text = render_map(_tree(d), many, goal="g", max_chars=3000)
            self.assertLessEqual(len(text), 3000)
            self.assertIn("more families", text)

    def test_without_families_still_renders(self):
        with tempfile.TemporaryDirectory() as d:
            text = render_map(_tree(d), None, goal="g", max_chars=16000)
            self.assertIn("12 attempts", text)
            self.assertIn("drsi families", text)


class GoalLengthTest(unittest.TestCase):
    def test_a_long_goal_keeps_its_tail_in_the_map(self):
        import tempfile
        from drsi.store import Tree
        with tempfile.TemporaryDirectory() as d:
            goal = "Four gates at once. " * 40 + "KNOWN WALL: prove the heuristic step or design around it."
            out = render_map(Tree(Path(d) / "t.jsonl"), None, goal=goal)
            self.assertIn("KNOWN WALL", out)


if __name__ == "__main__":
    unittest.main()
