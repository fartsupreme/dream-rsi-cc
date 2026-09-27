"""Round 9: a campaign's base can move forward.

Found in use (2026-09-27): the brief every worker reads lived in the workspace repository, was corrected there, and
`workspace.base` was moved to the corrected commit. The next round refused to start ("Start a new campaign"), and the
live loop sat dead until someone looked. A base that moves forward is taken up: new branches start on it, and a
continuation starts on it with its parent's own edits laid on top, so it sees the new fixed files and stays in scope.
A base that does not descend from the pinned one, or a different source, still needs a new campaign.
"""
import json
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.live import LiveRunner
from drsi.question import ROOT
from drsi.store import Campaign
from tests.test_scorer_workspace import git, make_repo

SCORER = "python3 -c \"import json;print(json.dumps({'score': int(open('value.txt').read()), 'valid': True}))\""


def worker(seen: list):
    def run(workspace: Path, prompt: str, system: str) -> AgentResult:
        seen.append((workspace.joinpath("README").read_text(), workspace.joinpath("value.txt").read_text()))
        workspace.joinpath("value.txt").write_text(f"{int(workspace.joinpath('value.txt').read_text()) + 1}\n")
        return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                "notes": ""})
    return run


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


class BaseMoveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = make_repo(self.root)
        self.base0 = head(self.src)
        self.camp = Campaign.create("t", {
            "goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
            "workspace": {"repo": str(self.src), "base": self.base0, "mutable": ["value.txt"]},
            "search": {"W": 1, "K1": 2, "plateau": 3}, "live": {"require_check": False},
        }, home=self.root / "home")
        self.seen: list = []

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, rid: str) -> LiveRunner:
        return LiveRunner(self.camp, worker(self.seen), indexer=lambda ids: None, round_id=rid)

    def batch(self, runner: LiveRunner, cells: list) -> list[dict]:
        return [self.camp.tree.get(n["id"]) for n in runner.run_batch(cells)]

    def set_base(self, sha: str) -> None:
        self.camp.update_config(lambda raw: raw["workspace"].__setitem__("base", sha))

    def move_base(self, readme: str) -> str:
        self.src.joinpath("README").write_text(readme)
        git(self.src, "commit", "-qam", "the brief, corrected")
        sha = head(self.src)
        self.set_base(sha)
        return sha

    def test_a_base_that_moves_forward_is_taken_up_and_recorded(self):
        self.batch(self.runner("iter0001"), [f"{ROOT}0"])
        b1 = self.move_base("r2\n")
        r = self.runner("iter0002")
        self.assertEqual(r.ws.base_commit, b1)
        pin = json.loads((self.camp.root / "base_commit.json").read_text())
        self.assertEqual((pin["base"], pin["commit"]), (b1, b1))
        self.assertEqual([h["commit"] for h in pin["history"]], [self.base0])

    def test_a_new_branch_starts_on_the_new_base(self):
        self.batch(self.runner("iter0001"), [f"{ROOT}0"])
        self.move_base("r2\n")
        node = self.batch(self.runner("iter0002"), [f"{ROOT}0"])[0]
        self.assertEqual(self.seen[-1], ("r2\n", "1\n"))
        self.assertEqual((node["fail_class"], node["score"], node["artifacts"]["changed"]), ("ok", 2.0, ["value.txt"]))

    def test_a_continuation_starts_on_the_new_base_with_its_parents_edits(self):
        n1 = self.batch(self.runner("iter0001"), [f"{ROOT}0"])[0]
        self.assertEqual(n1["score"], 2.0)
        self.move_base("r2\n")
        n2 = self.batch(self.runner("iter0002"), [n1["id"]])[0]
        self.assertEqual(self.seen[-1], ("r2\n", "2\n"))  # the new brief, and the parent's own work
        self.assertEqual((n2["fail_class"], n2["score"]), ("ok", 3.0))
        self.assertEqual((n2["artifacts"]["changed"], n2["artifacts"]["out_of_scope"]), (["value.txt"], []))

    def test_a_continuation_of_a_continuation_needs_no_second_move(self):
        n1 = self.batch(self.runner("iter0001"), [f"{ROOT}0"])[0]
        self.move_base("r2\n")
        n2 = self.batch(self.runner("iter0002"), [n1["id"]])[0]
        n3 = self.batch(self.runner("iter0003"), [n2["id"]])[0]
        self.assertEqual(self.seen[-1], ("r2\n", "3\n"))
        self.assertEqual((n3["fail_class"], n3["score"], n3["artifacts"]["out_of_scope"]), ("ok", 4.0, []))

    def test_an_out_of_scope_parent_stays_out_of_scope_after_the_move(self):
        def sneaky(workspace: Path, prompt: str, system: str) -> AgentResult:
            workspace.joinpath("notes.txt").write_text("not the attempt's to write\n")
            return worker(self.seen)(workspace, prompt, system)
        n1 = self.batch(LiveRunner(self.camp, sneaky, indexer=lambda ids: None, round_id="iter0001"), [f"{ROOT}0"])[0]
        self.assertEqual(n1["fail_class"], "out_of_scope")
        self.move_base("r2\n")
        n2 = self.batch(self.runner("iter0002"), [n1["id"]])[0]
        self.assertEqual(n2["fail_class"], "out_of_scope")
        self.assertIn("notes.txt", n2["artifacts"]["out_of_scope"])

    def test_naming_the_pinned_commit_differently_is_not_a_move(self):
        self.batch(self.runner("iter0001"), [f"{ROOT}0"])
        self.set_base(self.base0[:10])
        r = self.runner("iter0002")
        self.assertEqual(r.ws.base_commit, self.base0)
        pin = json.loads((self.camp.root / "base_commit.json").read_text())
        self.assertEqual((pin["base"], pin.get("history", [])), (self.base0[:10], []))

    def test_a_base_that_does_not_descend_from_the_pinned_one_is_refused(self):
        self.batch(self.runner("iter0001"), [f"{ROOT}0"])
        git(self.src, "checkout", "-q", "--orphan", "elsewhere")
        git(self.src, "commit", "-qm", "unrelated history")
        self.set_base(head(self.src))
        with self.assertRaisesRegex(RuntimeError, "Start a new campaign"):
            self.runner("iter0002")

    def test_a_different_source_is_refused(self):
        self.batch(self.runner("iter0001"), [f"{ROOT}0"])
        (self.root / "other").mkdir()
        other = make_repo(self.root / "other")
        self.camp.update_config(lambda raw: raw["workspace"].update(repo=str(other), base=head(other)))
        with self.assertRaisesRegex(RuntimeError, "Start a new campaign"):
            self.runner("iter0002")


class BaseMoveShapesTest(unittest.TestCase):
    """Second review (grok-4.7): a parent whose files change shape against the new base must not stop a round."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = make_repo(self.root)
        self.src.joinpath("pkg").mkdir()
        self.src.joinpath("pkg", "mod.txt").write_text("m\n")
        git(self.src, "add", "-A")
        git(self.src, "commit", "-qm", "a package")
        self.camp = Campaign.create("t", {
            "goal": "maximise value.txt", "scorer": {"cmd": SCORER, "timeout_s": 30},
            "workspace": {"repo": str(self.src), "base": head(self.src), "mutable": ["value.txt", "pkg/*", "notes*"]},
            "search": {"W": 1, "K1": 2, "plateau": 3}, "live": {"require_check": False},
        }, home=self.root / "home")
        self.seen: list = []

    def tearDown(self):
        self.tmp.cleanup()

    def batch(self, rid: str, cells: list, action=None) -> list[dict]:
        def run(workspace: Path, prompt: str, system: str) -> AgentResult:
            self.seen.append(sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if ".git" not in p.parts))
            if action:
                action(workspace)
            workspace.joinpath("value.txt").write_text(f"{int(workspace.joinpath('value.txt').read_text()) + 1}\n")
            return AgentResult(ok=True, structured={"proposal": "p", "summary": "s", "self_reported_score": None,
                                                    "notes": ""})
        r = LiveRunner(self.camp, run, indexer=lambda ids: None, round_id=rid)
        return [self.camp.tree.get(n["id"]) for n in r.run_batch(cells)]

    def move_base(self, change) -> None:
        change(self.src)
        git(self.src, "add", "-A")
        git(self.src, "commit", "-qm", "the base moves")
        self.camp.update_config(lambda raw: raw["workspace"].__setitem__("base", head(self.src)))

    def test_a_parent_that_made_a_file_of_a_directory_continues_after_the_move(self):
        def shape(ws):
            ws.joinpath("pkg", "mod.txt").unlink()
            ws.joinpath("pkg").rmdir()
            ws.joinpath("pkg").write_text("now a file\n")
        n1 = self.batch("iter0001", [f"{ROOT}0"], shape)[0]
        self.move_base(lambda src: src.joinpath("README").write_text("r2\n"))
        n2, n3 = self.batch("iter0002", [n1["id"], f"{ROOT}1"])
        self.assertIn("pkg", self.seen[1] + self.seen[2])
        self.assertTrue(n2["artifacts"]["changed"] and n3["fail_class"] == "ok")
        self.assertEqual(n2["fail_class"], n1["fail_class"])  # judged by scope as before, not crashed

    def test_a_base_that_made_a_file_of_the_parents_directory_is_judged_not_crashed(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"], lambda ws: ws.joinpath("pkg", "mod.txt").write_text("edited\n"))[0]
        self.assertEqual(n1["fail_class"], "ok")

        def to_file(src):
            src.joinpath("pkg", "mod.txt").unlink()
            src.joinpath("pkg").rmdir()
            src.joinpath("pkg").write_text("fixed file\n")
        self.move_base(to_file)
        n2, n3 = self.batch("iter0002", [n1["id"], f"{ROOT}1"])
        self.assertEqual(n2["fail_class"], "out_of_scope")  # its edit no longer fits the base's fixed file
        self.assertEqual(n3["fail_class"], "ok")

    def test_a_carriage_return_in_a_parents_file_name_is_kept(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"], lambda ws: ws.joinpath("notes\r.txt").write_text("n\n"))[0]
        self.move_base(lambda src: src.joinpath("README").write_text("r2\n"))
        self.batch("iter0002", [n1["id"]])
        self.assertIn("notes\r.txt", self.seen[-1])
        self.assertNotIn("notes\n.txt", self.seen[-1])

    def test_a_cell_that_cannot_be_set_up_is_recorded_and_the_batch_goes_on(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"])[0]
        r = LiveRunner(self.camp, lambda w, p, s: AgentResult(ok=True, structured={
            "proposal": "p", "summary": "s", "self_reported_score": None, "notes": ""}),
            indexer=lambda ids: None, round_id="iter0002")
        real = r.ws.start_for
        r.ws.start_for = lambda c: (_ for _ in ()).throw(RuntimeError("boom")) if c else real(c)
        out = r.run_batch([n1["id"], f"{ROOT}1"])
        by = {self.camp.tree.get(o["id"])["parent"]: self.camp.tree.get(o["id"]) for o in out}
        self.assertEqual(by[n1["id"]]["fail_class"], "orchestrator_error")
        self.assertIn("boom", by[n1["id"]]["text"]["orchestrator_error"])
        self.assertEqual(by[None]["fail_class"], "ok")

    def test_results_come_back_in_batch_order_when_a_later_cell_fails(self):
        n1 = self.batch("iter0001", [f"{ROOT}0"])[0]
        n2 = self.batch("iter0001b", [f"{ROOT}1"])[0]
        r = LiveRunner(self.camp, lambda w, p, s: AgentResult(ok=True, structured={
            "proposal": "p", "summary": "s", "self_reported_score": None, "notes": ""}),
            indexer=lambda ids: None, round_id="iter0002")
        real = r.ws.start_for
        r.ws.start_for = lambda c: (_ for _ in ()).throw(RuntimeError("boom")) if c == n2["artifacts"]["commit"] else real(c)
        out = r.run_batch([n1["id"], n2["id"]])  # probe_batch pairs these with the cells by position
        parents = [self.camp.tree.get(o["id"])["parent"] for o in out]
        self.assertEqual(parents, [n1["id"], n2["id"]])
        self.assertEqual([o["fail_class"] for o in out], ["ok", "orchestrator_error"])

    def test_a_file_the_parent_never_changed_follows_the_new_base(self):
        self.src.joinpath("pkg", "keep.txt").write_text("k\n")
        git(self.src, "add", "-A")
        git(self.src, "commit", "-qm", "another file")
        self.camp.update_config(lambda raw: raw["workspace"].__setitem__("base", head(self.src)))
        n1 = self.batch("iter0001", [f"{ROOT}0"], lambda ws: ws.joinpath("pkg", "mod.txt").write_text("edited\n"))[0]
        self.move_base(lambda src: src.joinpath("pkg", "keep.txt").unlink())  # the base drops a file
        self.batch("iter0002", [n1["id"]])
        self.assertIn("pkg/mod.txt", self.seen[-1])       # the parent's own edit is carried
        self.assertNotIn("pkg/keep.txt", self.seen[-1])   # a file it never changed follows the base


if __name__ == "__main__":
    unittest.main()
