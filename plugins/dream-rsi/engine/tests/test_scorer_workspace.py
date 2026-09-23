import subprocess
import tempfile
import unittest
from pathlib import Path

from drsi.scorer import run_scorer
from drsi.workspace import Workspaces, out_of_scope


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def make_repo(root: Path) -> Path:
    src = root / "src"
    src.mkdir()
    git(src, "init", "-q")
    git(src, "config", "user.email", "t@t")
    git(src, "config", "user.name", "t")
    (src / "value.txt").write_text("1\n")
    (src / "README").write_text("r\n")
    git(src, "add", "-A")
    git(src, "commit", "-qm", "base")
    return src


class ScorerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_last_json_line_and_normalises_direction(self):
        cmd = "echo noise; echo '{\"score\": 3.5, \"valid\": true, \"gates\": {\"G\": {\"pass\": true}}}'"
        r = run_scorer(cmd, self.ws, timeout=10, direction="max")
        self.assertEqual((r["score"], r["raw_score"], r["valid"], r["fail_class"]), (3.5, 3.5, True, "ok"))
        self.assertEqual(r["gates"]["G"]["pass"], True)
        r = run_scorer(cmd, self.ws, timeout=10, direction="min")
        self.assertEqual(r["score"], -3.5)

    def test_workspace_env_var_set(self):
        r = run_scorer('echo "{\\"score\\": 1, \\"valid\\": true, \\"summary\\": \\"$DRSI_WORKSPACE\\"}"',
                       self.ws, timeout=10, direction="max")
        self.assertEqual(r["summary"], str(self.ws))

    def test_invalid_result_keeps_fail_class(self):
        r = run_scorer("echo '{\"score\": null, \"valid\": false, \"fail_class\": \"eval_error\", \"error\": \"x\"}'",
                       self.ws, timeout=10, direction="max")
        self.assertFalse(r["valid"])
        self.assertIsNone(r["score"])
        self.assertEqual(r["fail_class"], "eval_error")

    def test_no_json_is_eval_error(self):
        r = run_scorer("echo hello; exit 3", self.ws, timeout=10, direction="max")
        self.assertFalse(r["valid"])
        self.assertEqual(r["fail_class"], "eval_error")

    def test_valid_without_number_is_eval_error(self):
        r = run_scorer("echo '{\"valid\": true}'", self.ws, timeout=10, direction="max")
        self.assertEqual(r["fail_class"], "eval_error")

    def test_timeout(self):
        r = run_scorer("sleep 5", self.ws, timeout=1, direction="max")
        self.assertEqual(r["fail_class"], "timeout")


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src = make_repo(root)
        self.ws = Workspaces(root / "camp", source_repo=self.src, base=None)
        self.ws.ensure_clone()

    def tearDown(self):
        self.tmp.cleanup()

    def test_clone_is_campaign_owned(self):
        self.assertTrue((self.ws.repo / ".git").exists())
        self.assertNotEqual(self.ws.repo.resolve(), self.src.resolve())
        self.assertTrue(self.ws.base_commit)

    def test_node_workspace_branches_from_parent_commit(self):
        a = self.ws.create("n1", parent_commit=None)
        (a / "value.txt").write_text("2\n")
        c1 = self.ws.snapshot("n1", self.ws.base_commit)
        b = self.ws.create("n2", parent_commit=c1)
        self.assertEqual((b / "value.txt").read_text(), "2\n")
        c = self.ws.create("n3", parent_commit=None)
        self.assertEqual((c / "value.txt").read_text(), "1\n")
        self.assertEqual(git(b, "rev-parse", "--abbrev-ref", "HEAD"), "drsi/n2")

    def test_commit_records_changed_files(self):
        a = self.ws.create("n1", parent_commit=None)
        (a / "value.txt").write_text("5\n")
        (a / "new.txt").write_text("x\n")
        c = self.ws.snapshot("n1", self.ws.base_commit)
        self.assertEqual(sorted(self.ws.changed_since_base(c)), ["new.txt", "value.txt"])

    def test_generated_artifacts_are_never_committed(self):
        a = self.ws.create("n1", parent_commit=None)
        (a / "value.txt").write_text("3\n")
        (a / "__pycache__").mkdir()
        (a / "__pycache__" / "value.cpython-314.pyc").write_bytes(b"x")
        (a / "stray.pyc").write_bytes(b"x")
        (a / "target").mkdir()
        (a / "target" / "debug.o").write_bytes(b"x")
        (a / ".DS_Store").write_bytes(b"x")
        c = self.ws.snapshot("n1", self.ws.base_commit)
        self.assertEqual(self.ws.changed_since_base(c), ["value.txt"])

    def test_commit_with_no_changes_still_succeeds(self):
        self.ws.create("n1", parent_commit=None)
        c = self.ws.snapshot("n1", self.ws.base_commit)
        self.assertTrue(c)
        self.assertEqual(self.ws.changed_since_base(c), [])

    def test_source_repo_untouched(self):
        a = self.ws.create("n1", parent_commit=None)
        (a / "value.txt").write_text("9\n")
        self.ws.snapshot("n1", self.ws.base_commit)
        self.assertEqual((self.src / "value.txt").read_text(), "1\n")
        self.assertEqual(git(self.src, "branch", "--list", "drsi/*"), "")


class ScopeTest(unittest.TestCase):
    def test_out_of_scope(self):
        self.assertEqual(out_of_scope(["src/a.rs", "README"], ["src/**"]), ["README"])
        self.assertEqual(out_of_scope(["anything/at/all"], ["**"]), [])
        self.assertEqual(out_of_scope(["value.txt"], ["value.txt"]), [])


if __name__ == "__main__":
    unittest.main()
