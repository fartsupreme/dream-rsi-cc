"""Round 7: a ledger row corrected after import, and fingerprints read under a goal that has since changed.

Found in use (2026-09-25): reviewers corrected an attempt's figures after it was synced, and `drsi sync`
kept the pre-review text, because an id already in the tree was skipped. The campaign's goal had changed
two days earlier, and the map still named a removed rule as what stopped five families, because every
fingerprint was read under the old goal and nothing marked it.
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from drsi import cli
from drsi.families import OTHER
from drsi.fingerprint import build_prompt, fingerprint_nodes, goal_sha
from drsi.importer import import_jsonl
from drsi.mapview import render_map
from drsi.novelty import text_hash
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM, fp_for, ids_in_block
from tests.test_cli import FIXTURE, fake_llm


def _write(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _fp(**kw):
    return {"mechanism": "m", "outcome": "refuted", "killed_by": "old rule", "family": "F01"} | kw


class EditedRowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.tree = Tree(self.d / "t.jsonl")
        self.rows = [{"id": 1, "candidate": "a", "verdict": "v"},
                     {"id": 2, "supersedes": [1], "candidate": "b", "verdict": "2^-335 a trial"}]
        self.ledger = _write(self.d / "ledger.jsonl", self.rows)
        import_jsonl(self.tree, self.ledger)
        self.tree.update_many({"1": {"fingerprint": _fp()}, "2": {"fingerprint": _fp()}})

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_corrected_verdict_replaces_the_stored_text_and_is_read_again(self):
        self.rows[1]["verdict"] = "2^-156 a trial"
        _write(self.ledger, self.rows)
        report = {}
        self.assertEqual(import_jsonl(self.tree, self.ledger, report=report), 0)
        self.assertEqual(report["refreshed"], 1)
        node = Tree(self.tree.path).get("2")
        self.assertEqual(node["text"]["verdict"], "2^-156 a trial")
        self.assertNotIn("mechanism", node["fingerprint"])  # read again by the next fingerprint pass
        self.assertEqual(node["fingerprint"]["family"], "F01")  # a correction keeps its family meanwhile
        self.assertEqual(node["parent"], "1")

    def test_a_corrected_proposal_in_the_same_ledger_is_an_edit_not_a_second_ledger(self):
        self.rows[1]["candidate"] = "b, corrected"
        _write(self.ledger, self.rows)
        report = {}
        import_jsonl(self.tree, self.ledger, report=report)
        self.assertEqual(report["refreshed"], 1)
        self.assertEqual(Tree(self.tree.path).get("2")["proposal"], "b, corrected")

    def test_an_unchanged_row_keeps_its_fingerprint(self):
        report = {}
        import_jsonl(self.tree, self.ledger, report=report)
        self.assertEqual(report["refreshed"], 0)
        self.assertEqual(Tree(self.tree.path).get("2")["fingerprint"]["family"], "F01")

    def test_a_second_ledger_with_a_colliding_id_is_still_refused(self):
        other = _write(self.d / "other.jsonl", [{"id": 2, "candidate": "something else", "verdict": "v"}])
        with self.assertRaises(ValueError):
            import_jsonl(self.tree, other)

    def test_a_moved_ledger_takes_its_new_path_and_its_edits_are_then_refreshed(self):
        moved = _write(self.d / "new-worktree.jsonl", self.rows)
        import_jsonl(self.tree, moved)
        self.assertEqual(Tree(self.tree.path).get("2")["ext"]["source_file"], str(moved.resolve()))
        self.rows[1]["candidate"] = "b, corrected at the new path"
        _write(moved, self.rows)
        report = {}
        import_jsonl(self.tree, moved, report=report)
        self.assertEqual(report["refreshed"], 1)
        self.assertEqual(Tree(self.tree.path).get("2")["proposal"], "b, corrected at the new path")

    def test_a_live_attempt_is_never_overwritten_by_an_import(self):
        self.tree.add(make_node(id="3", parent="2", source="live", proposal="live idea"))
        _write(self.ledger, self.rows + [{"id": 3, "candidate": "ledger idea", "verdict": "v"}])
        import_jsonl(self.tree, self.ledger)
        self.assertEqual(Tree(self.tree.path).get("3")["proposal"], "live idea")

    def test_the_generic_preset_refreshes_an_edited_row_too(self):
        tree = Tree(self.d / "g.jsonl")
        src = _write(self.d / "g-ledger.jsonl", [{"key": "x", "idea": "first", "note": "n1"}])
        fm = {"id": "key", "proposal": "idea", "text": ["note"]}
        import_jsonl(tree, src, preset="generic", field_map=fm)
        _write(src, [{"key": "x", "idea": "first", "note": "n2"}])
        report = {}
        import_jsonl(tree, src, preset="generic", field_map=fm, report=report)
        self.assertEqual(report["refreshed"], 1)
        self.assertEqual(Tree(tree.path).get("x")["text"]["note"], "n2")


class ReviewFindingsTest(unittest.TestCase):
    """The cross-vendor review of the first round-7 change (2026-09-25)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.tree = Tree(self.d / "t.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_reading_of_text_that_changed_during_the_call_is_not_installed(self):
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "old"},
                                    {"id": 2, "supersedes": [1], "candidate": "b", "verdict": "old"}])
        import_jsonl(self.tree, self.d / "l.jsonl")

        def answer(prompt, schema):  # another process corrects #2 while this reading is in flight
            Tree(self.tree.path).update_many({"2": {"text": {"candidate": "b", "verdict": "corrected"}}})
            return {"items": [fp_for(i) for i in ids_in_block(prompt)]}
        stats = fingerprint_nodes(self.tree, ScriptedLLM(answer), goal="g", batch=5, workers=1)
        again = Tree(self.tree.path)
        self.assertIn("mechanism", again.get("1")["fingerprint"])
        self.assertNotIn("mechanism", again.get("2")["fingerprint"] or {})
        self.assertEqual(stats["done"], 1)

    def test_a_correction_past_the_stored_cut_updates_the_full_text_hash(self):
        long = "x" * 1700
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": long, "verdict": "v"}])
        import_jsonl(self.tree, self.d / "l.jsonl")
        self.tree.update_many({"1": {"fingerprint": _fp()}})
        fixed = long[:1650] + "y" + long[1651:]
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": fixed, "verdict": "v"}])
        report = {}
        import_jsonl(self.tree, self.d / "l.jsonl", report=report)
        node = Tree(self.tree.path).get("1")
        self.assertEqual(report["refreshed"], 1)
        self.assertEqual(node["ext"]["proposal_sha"], text_hash(fixed))
        self.assertEqual(node["fingerprint"]["mechanism"], "m")  # what the classifier reads did not change

    def test_a_failed_reread_of_a_corrected_row_keeps_its_family(self):
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "old"}])
        import_jsonl(self.tree, self.d / "l.jsonl")
        self.tree.update_many({"1": {"fingerprint": _fp()}})
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "corrected"}])
        import_jsonl(self.tree, self.d / "l.jsonl")
        fingerprint_nodes(self.tree, ScriptedLLM(lambda p, s: {"items": []}), goal="g")
        self.assertEqual(Tree(self.tree.path).get("1")["fingerprint"]["family"], "F01")

    def test_a_metadata_only_correction_is_stored_and_a_removed_field_goes(self):
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "v", "result": "2^-335"}])
        import_jsonl(self.tree, self.d / "l.jsonl")
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "v", "result": "2^-156"}])
        report = {}
        import_jsonl(self.tree, self.d / "l.jsonl", report=report)
        self.assertEqual(report["refreshed"], 1)
        self.assertEqual(Tree(self.tree.path).get("1")["ext"]["result"], "2^-156")
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "v"}])
        import_jsonl(self.tree, self.d / "l.jsonl")
        self.assertNotIn("result", Tree(self.tree.path).get("1")["ext"])

    def test_rows_imported_before_row_hashes_are_not_all_read_again(self):
        _write(self.d / "l.jsonl", [{"id": 1, "candidate": "a", "verdict": "v"}])
        import_jsonl(self.tree, self.d / "l.jsonl")
        self.tree.modify({"1": lambda n: (n["ext"].pop("row_sha", None), n.update(fingerprint=_fp()))})
        report = {}
        import_jsonl(self.tree, self.d / "l.jsonl", report=report)
        node = Tree(self.tree.path).get("1")
        self.assertEqual(report["refreshed"], 0)
        self.assertEqual(node["fingerprint"]["mechanism"], "m")
        self.assertTrue(node["ext"]["row_sha"])


class SecondReviewTest(unittest.TestCase):
    """The cross-vendor re-review of the fixes (2026-09-25)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.tree = Tree(self.d / "t.jsonl")
        self.ledger = self.d / "l.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_family_survives_a_failed_reread_followed_by_a_good_one(self):
        _write(self.ledger, [{"id": 1, "candidate": "a", "verdict": "old"}])
        import_jsonl(self.tree, self.ledger)
        self.tree.update_many({"1": {"fingerprint": _fp()}})
        _write(self.ledger, [{"id": 1, "candidate": "a", "verdict": "corrected"}])
        import_jsonl(self.tree, self.ledger)
        fingerprint_nodes(self.tree, ScriptedLLM(lambda p, s: {"items": []}), goal="g")
        fingerprint_nodes(self.tree, ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]}),
                          goal="g")
        fp = Tree(self.tree.path).get("1")["fingerprint"]
        self.assertEqual((fp.get("family"), fp.get("mechanism")), ("F01", "mechanism of 1"))

    def _legacy(self, row):
        _write(self.ledger, [row])
        import_jsonl(self.tree, self.ledger)
        self.tree.modify({"1": lambda n: n["ext"].pop("row_sha", None)})

    def test_a_legacy_row_whose_metadata_changed_before_the_upgrade_is_corrected(self):
        self._legacy({"id": 1, "candidate": "a", "verdict": "v", "result": "2^-335"})
        _write(self.ledger, [{"id": 1, "candidate": "a", "verdict": "v", "result": "2^-156"}])
        report = {}
        import_jsonl(self.tree, self.ledger, report=report)
        self.assertEqual((report["refreshed"], Tree(self.tree.path).get("1")["ext"]["result"]), (1, "2^-156"))

    def test_a_legacy_row_whose_long_candidate_changed_past_the_cut_is_corrected(self):
        long = "x" * 1700
        self._legacy({"id": 1, "candidate": long, "verdict": "v"})
        fixed = long[:1650] + "y" + long[1651:]
        _write(self.ledger, [{"id": 1, "candidate": fixed, "verdict": "v"}])
        import_jsonl(self.tree, self.ledger)
        self.assertEqual(Tree(self.tree.path).get("1")["ext"]["proposal_sha"], text_hash(fixed))

    def test_a_correction_that_changes_a_rows_links_is_reported_and_keeps_its_edge(self):
        rows = [{"id": 1, "candidate": "a", "verdict": "v"}, {"id": 4, "candidate": "d", "verdict": "v"},
                {"id": 2, "supersedes": [1], "candidate": "b", "verdict": "v"}]
        _write(self.ledger, rows)
        import_jsonl(self.tree, self.ledger)
        rows[2]["supersedes"] = [4]
        _write(self.ledger, rows)
        report = {}
        import_jsonl(self.tree, self.ledger, report=report)
        node = Tree(self.tree.path).get("2")
        self.assertEqual(report["relinked"], ["2"])
        self.assertEqual((node["parent"], node["ext"]["supersedes"]), ("1", [4]))

    def test_a_refresh_computed_before_another_process_wrote_a_newer_one_is_not_applied(self):
        from drsi import importer
        _write(self.ledger, [{"id": 1, "candidate": "a", "verdict": "v1"}])
        import_jsonl(self.tree, self.ledger)
        _write(self.ledger, [{"id": 1, "candidate": "a", "verdict": "v2"}])
        backfill, refresh, report = {}, {}, {}
        importer._ledger_nodes(self.tree, self.ledger.resolve(), report, backfill, refresh, set())
        self.assertIn("1", refresh)
        _write(self.ledger, [{"id": 1, "candidate": "a", "verdict": "v3"}])
        import_jsonl(Tree(self.tree.path), self.ledger)  # the other process, finishing first
        importer._apply_refresh(Tree(self.tree.path), refresh)
        self.assertEqual(Tree(self.tree.path).get("1")["text"]["verdict"], "v3")


class GoalDriftTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.tmp.name) / "t.jsonl")
        for i in ("1", "2", "3"):
            self.tree.add(make_node(id=i, parent=None, source="import", proposal=f"idea {i}"))
        self.llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i, killed_by="the rule") for i in ids_in_block(p)]})

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_fingerprint_records_the_goal_it_was_read_under(self):
        fingerprint_nodes(self.tree, self.llm, goal="goal one")
        for n in Tree(self.tree.path).nodes():
            self.assertEqual(n["fingerprint"]["goal_sha"], goal_sha("goal one"))

    def test_stale_rereads_only_fingerprints_read_under_another_goal(self):
        fingerprint_nodes(self.tree, self.llm, goal="goal one")
        asked = len(self.llm.prompts)
        stats = fingerprint_nodes(self.tree, self.llm, goal="goal one", stale=True)
        self.assertEqual((stats["done"], len(self.llm.prompts)), (0, asked))
        stats = fingerprint_nodes(self.tree, self.llm, goal="goal two", stale=True)
        self.assertEqual(stats["done"], 3)
        for n in Tree(self.tree.path).nodes():
            self.assertEqual(n["fingerprint"]["goal_sha"], goal_sha("goal two"))

    def test_a_fingerprint_without_a_goal_stamp_is_stale(self):
        self.tree.update_many({"1": {"fingerprint": _fp()}})
        stats = fingerprint_nodes(self.tree, self.llm, goal="goal one", stale=True, ids=None)
        self.assertEqual(stats["done"], 3)

    def test_rereading_under_a_new_goal_keeps_the_family(self):
        self.tree.update_many({i: {"fingerprint": _fp(goal_sha=goal_sha("goal one"))} for i in ("1", "2", "3")})
        fingerprint_nodes(self.tree, self.llm, goal="goal two", stale=True)
        self.assertEqual({n["fingerprint"]["family"] for n in Tree(self.tree.path).nodes()}, {"F01"})

    def test_the_classifier_is_told_to_judge_the_stopper_against_the_current_goal(self):
        prompt = build_prompt(self.tree.nodes()[:1], "goal two")
        self.assertIn("no longer contains", prompt)

    def test_the_map_says_how_many_attempts_were_read_under_an_earlier_goal(self):
        self.tree.update_many({"1": {"fingerprint": _fp(goal_sha=goal_sha("goal one"))},
                               "2": {"fingerprint": _fp(goal_sha=goal_sha("goal two"))},
                               "3": {"fingerprint": _fp()}})
        fams = {"families": [{"id": "F01", "name": "f", "description": "d", "boundary": ""},
                             {"id": OTHER, "name": "other", "description": "", "boundary": ""}]}
        text = render_map(Tree(self.tree.path), fams, goal="goal two")
        self.assertIn("2 attempts were read under an earlier goal", text)
        self.assertIn("drsi fingerprint --stale", text)
        self.tree.update_many({i: {"fingerprint": _fp(goal_sha=goal_sha("goal two"))} for i in ("1", "3")})
        self.assertNotIn("earlier goal", render_map(Tree(self.tree.path), fams, goal="goal two"))


class GoalInMapTest(unittest.TestCase):
    def test_a_goal_that_grew_past_1500_characters_keeps_its_last_sentence(self):
        with tempfile.TemporaryDirectory() as d:
            goal = "Four gates at once, each counted without a hardware limit. " * 30 + "THE LAST SENTENCE."
            self.assertGreater(len(goal), 1700)
            self.assertIn("THE LAST SENTENCE.", render_map(Tree(Path(d) / "t.jsonl"), None, goal=goal))


class CLIRound7Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("DRSI_HOME")
        os.environ["DRSI_HOME"] = self.tmp.name
        self.llm = ScriptedLLM(fake_llm)
        cli.LLM_FACTORY = lambda cfg, role: self.llm

    def tearDown(self):
        if self._old is None:
            os.environ.pop("DRSI_HOME", None)
        else:
            os.environ["DRSI_HOME"] = self._old
        cli.LLM_FACTORY = None
        self.tmp.cleanup()

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_sync_rereads_a_corrected_row_and_reports_it(self):
        ledger = Path(self.tmp.name) / "ledger.jsonl"
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
        _write(ledger, rows)
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(ledger))
        self.run_cli("fingerprint", "-c", "demo")
        self.run_cli("families", "-c", "demo")
        rows[-1]["verdict"] = "corrected figure"
        _write(ledger, rows)
        code, out = self.run_cli("sync", "-c", "demo")
        self.assertEqual(code, 0)
        self.assertIn("1 refreshed", out)
        node = cli.resolve_campaign("demo").tree.nodes()[-1]
        self.assertEqual(node["text"]["verdict"], "corrected figure")
        self.assertEqual(node["fingerprint"]["family"], "F01")
        self.assertIn("mechanism", node["fingerprint"])

    def sources(self):
        return [src["path"] for src in cli.resolve_campaign("demo").config["sources"]]

    def test_importing_the_ledger_from_a_new_path_moves_the_campaign_there(self):
        rows = [{"id": 1, "candidate": "a", "verdict": "refuted"},
                {"id": 2, "supersedes": [1], "candidate": "b", "verdict": "refuted: 1.14x"}]
        old = _write(Path(self.tmp.name) / "old.jsonl", rows)
        rows[1]["verdict"] = "confirmed at 8x"
        new = _write(Path(self.tmp.name) / "new.jsonl", rows)
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(old))
        self.run_cli("import", "-c", "demo", str(new))
        self.assertEqual(self.sources(), [str(new.resolve())])
        self.assertEqual(self.run_cli("sync", "-c", "demo")[0], 0)
        self.assertEqual(cli.resolve_campaign("demo").tree.get("2")["text"]["verdict"], "confirmed at 8x")

    def test_after_a_move_an_edit_at_the_new_path_syncs(self):
        rows = [{"id": 1, "candidate": "a", "verdict": "v"}, {"id": 2, "supersedes": [1], "candidate": "b", "verdict": "v"}]
        old = _write(Path(self.tmp.name) / "old.jsonl", rows)
        new = _write(Path(self.tmp.name) / "new.jsonl", rows)
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(old))
        self.run_cli("import", "-c", "demo", str(new))
        rows[1]["candidate"] = "b, corrected"
        _write(new, rows)
        code, out = self.run_cli("sync", "-c", "demo")
        self.assertEqual(code, 0)
        self.assertIn("1 refreshed", out)
        self.assertEqual(cli.resolve_campaign("demo").tree.get("2")["proposal"], "b, corrected")

    def test_a_stale_copy_still_listed_as_a_source_neither_reverts_nor_breaks_sync(self):
        rows = [{"id": 1, "candidate": "a", "verdict": "v"}, {"id": 2, "supersedes": [1], "candidate": "b", "verdict": "v"}]
        old = _write(Path(self.tmp.name) / "old.jsonl", rows)
        corrected = [rows[0], dict(rows[1], candidate="b, corrected")]
        new = _write(Path(self.tmp.name) / "new.jsonl", corrected)
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(new))
        both = [{"path": str(old.resolve()), "preset": "attempt-ledger"},
                {"path": str(new.resolve()), "preset": "attempt-ledger"}]
        cli.resolve_campaign("demo").update_config(lambda raw: raw.__setitem__("sources", both))
        code, _ = self.run_cli("sync", "-c", "demo")
        self.assertEqual(code, 0)
        self.assertEqual(cli.resolve_campaign("demo").tree.get("2")["proposal"], "b, corrected")

    def test_a_second_file_sharing_one_attempt_does_not_drop_the_first_ledger(self):
        ledger = _write(Path(self.tmp.name) / "ledger.jsonl",
                        [{"id": i, "candidate": c, "verdict": "v"} for i, c in ((1, "a"), (2, "b"), (3, "c"))])
        notes = _write(Path(self.tmp.name) / "notes.jsonl",
                       [{"id": 1, "candidate": "a", "verdict": "v"}, {"id": 9, "candidate": "z", "verdict": "v"}])
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(ledger))
        self.run_cli("import", "-c", "demo", str(notes))
        self.assertEqual(sorted(self.sources()), sorted([str(ledger.resolve()), str(notes.resolve())]))

    def test_fingerprint_stale_rereads_after_the_goal_changes(self):
        ledger = _write(Path(self.tmp.name) / "ledger.jsonl", [{"id": 1, "candidate": "a", "verdict": "v"}])
        self.run_cli("init", "demo", "--goal", "goal one")
        self.run_cli("import", "-c", "demo", str(ledger))
        self.run_cli("fingerprint", "-c", "demo")
        self.run_cli("config", "-c", "demo", "--set", "goal=goal two")
        code, out = self.run_cli("fingerprint", "-c", "demo", "--stale")
        self.assertEqual(code, 0)
        self.assertIn("1 fingerprinted", out)
        node = cli.resolve_campaign("demo").tree.get("1")
        self.assertEqual(node["fingerprint"]["goal_sha"], goal_sha("goal two"))


if __name__ == "__main__":
    unittest.main()
