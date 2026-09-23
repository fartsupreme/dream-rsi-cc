"""Layer 1 robustness: history must survive crashes, bad rows, concurrency and odd input."""
import io
import json
import multiprocessing
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from drsi import cli
from drsi.bm25 import tokenize
from drsi.families import OTHER, assign_families, build_taxonomy, family_stats
from drsi.fingerprint import fingerprint_nodes
from drsi.importer import import_jsonl
from drsi.novelty import check
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM, fp_for, ids_in_block


def _adder(path, start):
    t = Tree(path)
    for i in range(start, start + 25):
        t.add(make_node(id=f"p{i}", parent=None))


class TreeCrashTest(unittest.TestCase):
    def test_add_after_a_torn_tail_keeps_every_committed_node(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            t = Tree(path)
            t.add(make_node(id="1", parent=None))
            with open(path, "a") as fh:
                fh.write('{"id": "torn", "par')
            Tree(path).add(make_node(id="2", parent="1"))
            Tree(path).add(make_node(id="3", parent="2"))
            self.assertEqual([n["id"] for n in Tree(path).nodes()], ["1", "2", "3"])

    def test_four_processes_adding_concurrently_lose_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            ctx = multiprocessing.get_context("spawn")
            procs = [ctx.Process(target=_adder, args=(path, k * 100)) for k in range(4)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
            self.assertEqual(len(Tree(path)), 100)

    def test_lone_surrogate_text_is_stored_and_reloaded(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="s", parent=None, proposal="bad \udc80 char"))
            self.assertIn("bad", Tree(t.path).get("s")["proposal"])


class RetryTest(unittest.TestCase):
    def test_failed_fingerprints_are_retried_next_pass(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, proposal="p", fingerprint={"error": "omitted"}))
            llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
            st = fingerprint_nodes(t, llm, goal="g", batch=5, workers=1)
            self.assertEqual(st["done"], 1)
            self.assertNotIn("error", Tree(t.path).get("1")["fingerprint"])

    def test_failed_assignment_leaves_nodes_unassigned_for_a_retry(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, fingerprint=fp_for("1")))

            def boom(p, s):
                raise RuntimeError("down")
            assign_families(t, {"families": [{"id": "F01", "name": "n", "description": "", "boundary": ""}]},
                            ScriptedLLM(boom))
            self.assertNotIn("family", Tree(t.path).get("1")["fingerprint"])


class ImporterTest(unittest.TestCase):
    def write(self, d, rows, name="l.jsonl"):
        p = Path(d) / name
        p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
        return p

    def test_malformed_row_is_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.write(d, [{"id": 1, "candidate": "a", "verdict": "v"}, "{not json",
                                 {"id": 3, "candidate": "c", "verdict": "v"}])
            t = Tree(Path(d) / "t.jsonl")
            report = {}
            self.assertEqual(import_jsonl(t, src, report=report), 2)
            self.assertEqual(report["skipped"], 1)

    def test_string_and_int_links_are_ids_not_characters(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": i, "candidate": f"c{i}", "verdict": "v"} for i in range(1, 13)]
            rows.append({"id": 13, "supersedes": "12", "candidate": "x", "verdict": "v"})
            rows.append({"id": 14, "refutes": 3, "candidate": "y", "verdict": "v"})
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, self.write(d, rows))
            self.assertEqual(t.get("13")["parent"], "12")
            self.assertEqual(t.get("14")["parent"], "3")

    def test_millisecond_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, self.write(d, [{"ts": 1786655220000, "candidate": "a", "verdict": "v"}]))
            self.assertTrue(t.get("1")["created"].startswith("2026-"))

    def test_generic_null_id_uses_row_position(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, self.write(d, [{"key": None, "idea": "a"}, {"key": None, "idea": "b"}]),
                         preset="generic", field_map={"id": "key", "proposal": "idea"})
            self.assertEqual([n["id"] for n in t.nodes()], ["1", "2"])

    def test_second_ledger_with_colliding_ids_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, self.write(d, [{"id": 1, "candidate": "a", "verdict": "v"}], "one.jsonl"))
            with self.assertRaises(ValueError):
                import_jsonl(t, self.write(d, [{"id": 1, "candidate": "other", "verdict": "v"}], "two.jsonl"))

    def test_same_ledger_at_a_new_path_syncs_normally(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            rows = [{"id": 1, "candidate": "a", "verdict": "v"}]
            import_jsonl(t, self.write(d, rows, "old-worktree.jsonl"))
            moved = self.write(d, rows + [{"id": 2, "supersedes": [1], "candidate": "b", "verdict": "v"}],
                               "new-worktree.jsonl")
            self.assertEqual(import_jsonl(t, moved), 1)

    def test_concurrent_import_of_the_same_rows_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.write(d, [{"id": 1, "candidate": "a", "verdict": "v"}])
            a, b = Tree(Path(d) / "t.jsonl"), Tree(Path(d) / "t.jsonl")
            import_jsonl(a, src)
            self.assertEqual(import_jsonl(b, src), 0)  # b's snapshot was stale; the rows are already there
            self.assertEqual(len(Tree(a.path)), 1)


class NoveltyEdgeTest(unittest.TestCase):
    def tree(self, d):
        t = Tree(Path(d) / "t.jsonl")
        for i in ("1", "2", "3"):
            t.add(make_node(id=i, parent=None, proposal=f"idea {i}",
                            fingerprint={"mechanism": f"sieve variant {i}", "outcome": "refuted", "family": OTHER}))
        return t

    def judge(self, **kw):
        base = {"verdict": "novel", "family": OTHER, "nearest_ids": ["#2"], "what_differs": "d",
                "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": "r"}
        return lambda p, s: base | kw

    def test_other_bucket_gets_no_family_warning(self):
        with tempfile.TemporaryDirectory() as d:
            fams = {"families": [{"id": OTHER, "name": "other", "description": "", "boundary": ""}]}
            r = check(self.tree(d), fams, ScriptedLLM(self.judge()), "a sieve variant")
            self.assertEqual(r["warnings"], [])

    def test_hash_prefixed_nearest_ids_are_kept(self):
        with tempfile.TemporaryDirectory() as d:
            r = check(self.tree(d), {"families": []}, ScriptedLLM(self.judge()), "a sieve variant")
            self.assertEqual([n["id"] for n in r["nearest"]], ["2"])


class TextAndStatsTest(unittest.TestCase):
    def test_tokenizer_keeps_non_ascii_words(self):
        toks = tokenize("Équation de Fourier, решётка, 格")
        self.assertIn("équation", toks)
        self.assertIn("решётка", toks)
        self.assertIn("格", toks)

    def test_plateau_zero_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, fingerprint={"outcome": "refuted", "family": "F01"}))
            fams = {"families": [{"id": "F01", "name": "n", "description": "", "boundary": ""}]}
            self.assertEqual(family_stats(t, fams, plateau=0)[0]["n"], 1)

    def test_taxonomy_needs_enough_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            with self.assertRaises(ValueError):
                build_taxonomy(t, ScriptedLLM(lambda p, s: {"families": []}), "g", Path(d) / "f.json")


class CliValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("DRSI_HOME")
        os.environ["DRSI_HOME"] = self.tmp.name
        with redirect_stdout(io.StringIO()):
            cli.main(["init", "v", "--goal", "g"])

    def tearDown(self):
        cli.LLM_FACTORY = None
        if self.old is None:
            os.environ.pop("DRSI_HOME", None)
        else:
            os.environ["DRSI_HOME"] = self.old
        self.tmp.cleanup()

    def run_cli(self, *argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                return cli.main(list(argv))
            except SystemExit as e:
                return e.code

    def test_config_set_without_equals_is_an_error(self):
        self.assertEqual(self.run_cli("config", "-c", "v", "--set", "search.W"), 2)

    def test_config_set_through_a_scalar_is_an_error(self):
        self.assertNotEqual(self.run_cli("config", "-c", "v", "--set", "goal.x=1"), 0)

    def test_fingerprint_batch_must_be_positive(self):
        self.assertEqual(self.run_cli("fingerprint", "-c", "v", "--batch", "0"), 2)

    def test_sync_fingerprints_leftovers_even_without_new_rows(self):
        from drsi.store import Campaign
        camp = Campaign.open("v")
        camp.tree.add(make_node(id="1", parent=None, proposal="p"))
        cli.LLM_FACTORY = lambda cfg, role: ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
        self.assertEqual(self.run_cli("sync", "-c", "v"), 0)
        self.assertIn("mechanism", Campaign.open("v").tree.get("1")["fingerprint"])


if __name__ == "__main__":
    unittest.main()
