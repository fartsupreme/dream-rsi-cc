import json
import tempfile
import unittest
from pathlib import Path

from drsi.importer import import_jsonl
from drsi.store import Tree

FIXTURE = Path(__file__).parent / "fixtures" / "attempts_sample.jsonl"


class AttemptLedgerPresetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.tmp.name) / "tree.jsonl")
        self.added = import_jsonl(self.tree, FIXTURE, preset="attempt-ledger")

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_rows_imported_blank_line_skipped(self):
        self.assertEqual(self.added, 9)
        self.assertEqual([n["id"] for n in self.tree.nodes()], [str(i) for i in range(1, 10)])

    def test_rows_without_id_take_their_row_number(self):
        self.assertEqual(self.tree.get("3")["text"]["verdict"], "CONFIRMED at 1.8x")

    def test_parent_from_supersedes(self):
        n = self.tree.get("4")
        self.assertEqual(n["parent"], "2")
        self.assertEqual(n["ext"]["link"], "supersedes")

    def test_parent_from_discharges_reference(self):
        n = self.tree.get("5")
        self.assertEqual(n["parent"], "3")
        self.assertEqual(n["ext"]["link"], "discharges")

    def test_parent_from_refutes_when_no_supersedes(self):
        n = self.tree.get("7")
        self.assertEqual(n["parent"], "4")
        self.assertEqual(n["ext"]["link"], "refutes")

    def test_unlinked_row_chains_to_previous(self):
        n = self.tree.get("8")
        self.assertEqual(n["parent"], "7")
        self.assertEqual(n["ext"]["link"], "sequential")

    def test_dangling_reference_falls_back_to_sequential(self):
        n = self.tree.get("9")
        self.assertEqual(n["parent"], "8")
        self.assertEqual(n["ext"]["link"], "sequential")

    def test_first_row_is_root(self):
        self.assertIsNone(self.tree.get("1")["parent"])

    def test_proposal_prefers_candidate_then_construction_name(self):
        self.assertEqual(self.tree.get("1")["proposal"], "SHELLSORT-GAPS")
        self.assertEqual(self.tree.get("3")["proposal"], "the three-way merge at width 768")
        self.assertEqual(self.tree.get("6")["text"]["construction"], "THE MERGE'S LAW BY GALLOPING")

    def test_long_fields_trimmed(self):
        self.assertLessEqual(len(self.tree.get("9")["text"]["verdict"]), 1501)

    def test_ext_keeps_provenance(self):
        ext = self.tree.get("6")["ext"]
        self.assertEqual(ext["mode"], "invent")
        self.assertEqual(ext["row"], 6)
        self.assertEqual(ext["source_file"], str(FIXTURE))
        self.assertEqual(self.tree.get("6")["source"], "import")

    def test_created_from_epoch_ts_and_iso(self):
        self.assertTrue(self.tree.get("1")["created"].startswith("2026-08-"))
        self.assertEqual(self.tree.get("6")["created"], "2026-09-01T10:00:00Z")
        self.assertEqual(self.tree.get("7")["created"], "2026-09-02T00:00:00Z")

    def test_reimport_is_incremental(self):
        again = import_jsonl(self.tree, FIXTURE, preset="attempt-ledger")
        self.assertEqual(again, 0)
        self.assertEqual(len(Tree(self.tree.path)), 9)

    def test_sync_appends_only_new_rows(self):
        grown = Path(self.tmp.name) / "grown.jsonl"
        text = FIXTURE.read_text()
        extra = {"id": 10, "date": "2026-09-05", "mode": "invent", "supersedes": [9],
                 "candidate": "NEW", "falsifiable": "f", "verdict": "v", "check_cmd": "c", "next": "n"}
        grown.write_text(text + json.dumps(extra) + "\n")
        self.assertEqual(import_jsonl(self.tree, grown, preset="attempt-ledger"), 1)
        self.assertEqual(self.tree.get("10")["parent"], "9")


class GenericPresetTest(unittest.TestCase):
    def test_field_map(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "log.jsonl"
            src.write_text("\n".join(json.dumps(r) for r in [
                {"key": "a", "idea": "first idea", "outcome": "failed"},
                {"key": "b", "from": "a", "idea": "second idea", "outcome": "better"},
            ]) + "\n")
            tree = Tree(Path(d) / "tree.jsonl")
            n = import_jsonl(tree, src, preset="generic",
                             field_map={"id": "key", "parent": "from", "proposal": "idea",
                                        "text": ["idea", "outcome"]})
            self.assertEqual(n, 2)
            self.assertEqual(tree.get("b")["parent"], "a")
            self.assertEqual(tree.get("b")["text"]["outcome"], "better")
            self.assertEqual(tree.get("a")["proposal"], "first idea")


if __name__ == "__main__":
    unittest.main()
