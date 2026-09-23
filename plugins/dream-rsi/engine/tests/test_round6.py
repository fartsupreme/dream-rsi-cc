"""Sixth-review findings (grok-4.7 on the round-5 code), each reproduced here first."""
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from drsi.agent import AgentResult
from drsi.families import rebuild_families
from drsi.guard import check_policy_source
from drsi.live import LiveRunner, live_round, load_policy
from drsi.novelty import check
from drsi.scorer import run_scorer
from drsi.store import Tree, make_node
from tests.helpers import ScriptedLLM
from tests.test_integrity_live import LiveIntegrityBase, bump
from tests.test_live import fixed_checker
from tests.test_replay import HEADER
from tests.test_scorer_workspace import git


class ProposePhaseTest(LiveIntegrityBase):
    def setUp(self):
        super().setUp()
        raw = json.loads(self.camp.config_path.read_text())
        raw["live"]["require_check"] = True
        raw["search"]["K1"] = 1
        self.camp.save_config(raw)

    def run_with(self, propose_action):
        def w(ws, prompt, system):
            if "PHASE: PROPOSE" in system:
                propose_action(ws)
                return AgentResult(ok=True, structured={"proposal": "x", "summary": "", "self_reported_score": None,
                                                        "notes": ""})
            bump(ws)
            return AgentResult(ok=True, structured={})
        r = LiveRunner(self.camp, w, indexer=lambda ids: None, round_id="iter0001", checker=fixed_checker("novel"))
        live_round(self.camp, self.policy, r)
        return self.camp.tree.get(r.ids[0])

    def test_skip_worktree_flag_does_not_carry_a_pre_verdict_edit(self):
        def act(ws):
            git(ws, "update-index", "--skip-worktree", "value.txt")
            (ws / "value.txt").write_text("100\n")
        self.assertEqual(self.run_with(act)["score"], 2.0)

    def test_nested_git_dir_does_not_survive_into_implement(self):
        def act(ws):
            (ws / "payload").mkdir()
            subprocess.run(["git", "init", "-q", str(ws / "payload")], check=True)
            (ws / "payload" / "notes").write_text("secret\n")
        n = self.run_with(act)
        self.assertEqual(n["fail_class"], "ok")
        self.assertNotIn("payload", " ".join(n["artifacts"]["changed"]))


class GitInjectionTest(LiveIntegrityBase):
    def _hook(self):
        marker = Path(self.tmp.name) / "PWNED"
        hook = Path(self.tmp.name) / "hook.sh"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
        hook.chmod(0o755)
        return marker, hook

    def _evil_repo(self, gitdir: Path, hook: Path):
        subprocess.run(["git", "init", "-q", "--bare", str(gitdir)], check=True)
        for k, v in (("core.bare", "false"), ("core.fsmonitor", str(hook)), ("diff.external", str(hook))):
            subprocess.run(["git", "--git-dir", str(gitdir), "config", k, v], check=True)

    def test_worker_rewritten_dot_git_cannot_run_code_in_the_orchestrator(self):
        marker, hook = self._hook()

        def act(ws):
            bump(ws)
            (ws / ".git").unlink()
            self._evil_repo(ws / ".git", hook)
        nodes = self.round(act)
        self.assertFalse(marker.exists())
        self.assertEqual(nodes[0]["score"], 2.0)  # the attempt is the files it left, not its git metadata

    def test_nested_git_dir_is_out_of_scope_and_never_read(self):
        raw = json.loads(self.camp.config_path.read_text())
        raw["workspace"]["mutable"] = ["value.txt", "sub/*"]
        self.camp.save_config(raw)
        marker, hook = self._hook()

        def act(ws):
            bump(ws)
            (ws / "sub").mkdir()
            (ws / "sub" / "x").write_text("1\n")
            self._evil_repo(ws / "sub" / ".git", hook)
        n = self.round(act)[0]
        self.assertFalse(marker.exists())
        self.assertEqual(n["fail_class"], "out_of_scope")
        self.assertTrue(any(".git" in p for p in n["artifacts"]["out_of_scope"]))


class GitConfigIsolationTest(GitInjectionTest):
    def test_case_variant_nested_git_dir_is_refused(self):
        raw = json.loads(self.camp.config_path.read_text())
        raw["workspace"]["mutable"] = ["value.txt", "sub/*"]
        self.camp.save_config(raw)
        marker, hook = self._hook()

        def act(ws):
            bump(ws)
            (ws / "sub").mkdir()
            self._evil_repo(ws / "sub" / ".GIT", hook)
        n = self.round(act)[0]
        self.assertFalse(marker.exists())
        self.assertEqual(n["fail_class"], "out_of_scope")

    def test_worker_attributes_cannot_invoke_a_global_filter_driver(self):
        import os
        raw = json.loads(self.camp.config_path.read_text())
        raw["workspace"]["mutable"] = ["value.txt", ".gitattributes"]
        self.camp.save_config(raw)
        marker, hook = self._hook()
        home = Path(self.tmp.name) / "home-with-filter"
        home.mkdir()
        (home / ".gitconfig").write_text(f"[filter \"evil\"]\n\tclean = {hook}\n\tsmudge = {hook}\n")
        old = os.environ.get("HOME")
        os.environ["HOME"] = str(home)
        try:
            def act(ws):
                bump(ws)
                (ws / ".gitattributes").write_text("* filter=evil\n")
            self.round(act)
        finally:
            os.environ["HOME"] = old
        self.assertFalse(marker.exists())


class AdminDirTest(LiveIntegrityBase):
    def test_worker_settings_do_not_grant_the_git_admin_dir(self):
        from drsi.cli import worker_agent
        ws = self.camp.root / "work" / "iter0001-001"
        agent = worker_agent(self.camp, ws, "sys")
        self.assertNotIn(".git", json.dumps(agent.settings["sandbox"]["filesystem"]["allowWrite"]))

    def test_fifo_planted_in_an_admin_dir_does_not_wedge_other_attempts(self):
        import os
        import threading
        from drsi.workspace import Workspaces
        ws = Workspaces(self.camp.root, self.src)
        ws.ensure_clone()
        a = ws.create("a1", None)
        head = ws.repo / ".git" / "worktrees" / a.name / "HEAD"
        head.unlink()
        os.mkfifo(head)  # a parallel attempt's worker, while this one is still running
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("p", ws.create_detached("_score-b", ws.base_commit)),
                             daemon=True)
        t.start()
        t.join(60)
        self.assertFalse(t.is_alive())
        self.assertTrue((box["p"] / "value.txt").exists())


class LeftoverWorkerTest(unittest.TestCase):
    def tearDown(self):
        subprocess.run(["pkill", "-f", "sleep 7779.75"])

    def test_detached_child_working_in_the_workspace_is_killed(self):
        from drsi.agent import run_group
        with tempfile.TemporaryDirectory() as d:
            run_group(["python3", "-c", "import subprocess; subprocess.Popen(['sleep', '7779.75'], "
                                        "start_new_session=True)"], cwd=d, timeout=30, sweep=[d])
            time.sleep(0.3)
            self.assertNotEqual(subprocess.run(["pgrep", "-f", "sleep 7779.75"]).returncode, 0)


class AgentSweepTest(unittest.TestCase):
    def test_agent_runs_sweep_their_working_directory(self):
        from drsi.agent import ClaudeAgent
        from tests.test_agent_worlds import FakeRunner
        r = FakeRunner(json.dumps({"subtype": "success", "result": "ok"}))
        ClaudeAgent(model="opus", tools="Read", runner=r).run("/tmp/ws", "p")
        self.assertEqual(r.calls[0]["sweep"], ["/tmp/ws"])


class FamilyRevKeptTest(unittest.TestCase):
    def test_refingerprinting_keeps_the_rebuild_stamp(self):
        from drsi.fingerprint import _writer
        node = {"id": "1", "source": "live", "fingerprint": {"family": "F02", "family_rev": "r1", "outcome": "pass"},
                "artifacts": {"outcome": "pass"}}
        _writer({"mechanism": "m"})(node)
        self.assertEqual(node["fingerprint"].get("family_rev"), "r1")


class ProposalFileTest(ProposePhaseTest):
    def _pfile(self, ws):
        return self.camp.root / "work" / "_proposals" / ws.name / "proposal.txt"

    def test_fifo_proposal_file_does_not_hang_the_orchestrator(self):
        import os
        import threading
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("n", self.run_with(lambda ws: os.mkfifo(self._pfile(ws)))),
                             daemon=True)
        t.start()
        t.join(60)
        self.assertFalse(t.is_alive())
        self.assertEqual(box["n"]["proposal"], "x")

    def test_symlinked_proposal_file_is_not_followed(self):
        secret = Path(self.tmp.name) / "secret"
        secret.write_text("TOPSECRET\n")
        n = self.run_with(lambda ws: self._pfile(ws).symlink_to(secret))
        self.assertNotIn("TOPSECRET", n["proposal"])


class TimeoutSwallowTest(unittest.TestCase):
    def test_catch_all_handlers_are_rejected(self):
        for handler in ("except BaseException:", "except:", "except KeyboardInterrupt:", "except (ValueError, BaseException):"):
            src = HEADER + ("    def solve(self, question, budget=None):\n        try:\n            pass\n"
                            f"        {handler}\n            pass\n")
            self.assertTrue(check_policy_source(src), handler)

    def test_except_exception_is_still_allowed(self):
        src = HEADER + ("    def solve(self, question, budget=None):\n        try:\n            pass\n"
                        "        except Exception:\n            pass\n")
        self.assertEqual(check_policy_source(src), [])


class TimeoutFinallyTest(LiveIntegrityBase):
    def test_loop_inside_finally_is_still_stopped(self):
        raw = json.loads(self.camp.config_path.read_text())
        raw["live"]["think_timeout_s"] = 1
        self.camp.save_config(raw)
        p = Path(self.tmp.name) / "p.py"
        p.write_text(HEADER + ("    def solve(self, question, budget=None):\n        question.reset()\n"
                               "        question.probe_batch(question.legal_roots()[:1])\n"
                               "        try:\n            while True:\n                pass\n"
                               "        finally:\n            while True:\n                pass\n"))
        from tests.test_integrity_live import worker
        r = LiveRunner(self.camp, worker(bump), indexer=lambda ids: None, round_id="iter0001")
        t0 = time.time()
        live_round(self.camp, load_policy(p), r)
        self.assertLess(time.time() - t0, 30)


class GuardBuildClassTest(unittest.TestCase):
    def test_build_class_and_dunder_names_rejected(self):
        src = HEADER + ("    def solve(self, question, budget=None):\n        def body():\n"
                        "            __getattribute__ = 1\n        return __build_class__(body, 'E')\n")
        problems = check_policy_source(src)
        self.assertTrue(any("__build_class__" in p for p in problems))
        self.assertTrue(any("__getattribute__" in p for p in problems))


class ReflectionAttrTest(unittest.TestCase):
    def test_code_and_generator_introspection_rejected(self):
        for expr in ("g.gi_code", "g.gi_code.co_consts", "g.cr_code", "g.ag_code", "g.gi_yieldfrom", "g.f_builtins"):
            src = HEADER + ("    def solve(self, question, budget=None):\n        g = (x for x in [1])\n"
                            f"        return {expr}\n")
            self.assertTrue(check_policy_source(src), expr)


class GitlinkScopeTest(unittest.TestCase):
    def test_a_committed_gitlink_is_reported(self):
        from drsi.workspace import Workspaces
        from tests.test_scorer_workspace import make_repo
        with tempfile.TemporaryDirectory() as d:
            ws = Workspaces(Path(d) / "camp", make_repo(Path(d)))
            ws.ensure_clone()
            git(ws.repo, "update-index", "--add", "--cacheinfo", f"160000,{ws.base_commit},sub")
            tree = git(ws.repo, "write-tree")
            commit = git(ws.repo, "commit-tree", tree, "-p", ws.base_commit, "-m", "x")
            self.assertEqual(ws.gitlinks_since_base(commit), ["sub"])


class RebuildRollbackTest(unittest.TestCase):
    def test_crash_between_relabel_and_swap_is_recovered(self):
        from drsi import families as fam_mod
        from drsi.families import assign_new, load_families
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            for i in range(1, 7):
                t.add(make_node(id=str(i), parent=None, fingerprint={"mechanism": f"m{i}", "family": "F01"}))
            path = Path(d) / "f.json"
            path.write_text(json.dumps({"families": [{"id": "F01", "name": "old", "description": "", "boundary": ""}]}))
            llm = ScriptedLLM(lambda p, s: {"families": [{"id": "F01", "name": "new", "description": "", "boundary": ""},
                                                         {"id": "F02", "name": "new2", "description": "", "boundary": ""}]}
                              if "families" in s.get("properties", {}) else
                              {"items": [{"id": str(i), "family": "F02"} for i in range(1, 7)]})
            orig = fam_mod.os.replace

            def crash_on_swap(src, dst):
                if str(src).endswith(".new"):
                    raise OSError("killed")
                return orig(src, dst)
            fam_mod.os.replace = crash_on_swap
            try:
                with self.assertRaises(OSError):
                    rebuild_families(t, llm, "g", path)
            finally:
                fam_mod.os.replace = orig
            # the tree already carries the new taxonomy's ids; the next families operation finishes the swap
            assign_new(t, path, llm)
            names = [f["name"] for f in load_families(path)["families"]]
            self.assertIn("new2", names)
            self.assertNotIn("old", names)
            self.assertFalse(Path(str(path) + ".new").exists())

    def test_crash_before_relabel_discards_the_staged_taxonomy(self):
        from drsi.families import assign_new, load_families
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None, fingerprint={"mechanism": "m", "family": "F01"}))
            path = Path(d) / "f.json"
            path.write_text(json.dumps({"families": [{"id": "F01", "name": "old", "description": "", "boundary": ""}]}))
            Path(str(path) + ".new").write_text(json.dumps({"rev": "abc", "families": [
                {"id": "F01", "name": "new", "description": "", "boundary": ""}]}))
            assign_new(t, path, ScriptedLLM(lambda p, s: {"items": []}))
            self.assertEqual([f["name"] for f in load_families(path)["families"]], ["old"])
            self.assertFalse(Path(str(path) + ".new").exists())


class TextHashTest(unittest.TestCase):
    def test_shared_long_preamble_is_not_a_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            pre = "Background and constraints of the campaign, restated at length. " * 12
            t.add(make_node(id="1", parent=None, proposal=(pre + "Idea one.")[:600] + "…",
                            fingerprint={"mechanism": "m"}))
            llm = ScriptedLLM(lambda p, s: {"verdict": "novel", "family": "F00", "nearest_ids": [], "what_differs": "d",
                                            "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": ""})
            r = check(t, {"families": []}, llm, pre + "A completely different idea two.")
            self.assertEqual(r["verdict"], "novel")
            self.assertEqual(len(llm.prompts), 1)

    def test_verbatim_resubmission_matches_by_hash(self):
        from drsi.novelty import text_hash
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            full = "long proposal text " * 700
            t.add(make_node(id="1", parent=None, proposal=full[:8000], ext={"proposal_sha": text_hash(full)},
                            fingerprint={"mechanism": "m"}))
            r = check(t, {"families": []}, ScriptedLLM(lambda p, s: {"verdict": "novel"}), full)
            self.assertEqual(r["verdict"], "duplicate")


PROBE = """
import socket, subprocess
try:
    open('/dev/tty', 'w')
    print('TTY_OPEN')
except OSError as e:
    print('TTY_ERRNO', e.errno)
with open('/dev/null', 'w') as fh:
    fh.write('x')
print('NULL_OK')
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('1.1.1.1', 80))
    print('NET_OPEN')
except OSError as e:
    print('NET_ERR', e.errno)
subprocess.Popen(['sleep', '7779.25'], start_new_session=True)
"""


class ImportHashTest(unittest.TestCase):
    def test_import_stores_and_backfills_the_full_text_hash(self):
        from drsi.importer import import_jsonl
        with tempfile.TemporaryDirectory() as d:
            long = "a long ledger proposal about one construction " * 40
            src = Path(d) / "l.jsonl"
            src.write_text(json.dumps({"id": 1, "candidate": long, "verdict": "v"}) + "\n")
            t = Tree(Path(d) / "t.jsonl")
            import_jsonl(t, src)
            t.modify({"1": lambda n: n["ext"].pop("proposal_sha")})  # as imported by an older version
            import_jsonl(t, src)
            r = check(Tree(t.path), {"families": []}, ScriptedLLM(lambda p, s: {"verdict": "novel"}), long)
            self.assertEqual(r["verdict"], "duplicate")


class ScorerRound6Test(unittest.TestCase):
    def tearDown(self):
        subprocess.run(["pkill", "-f", "sleep 7779.25"])

    @unittest.skipUnless(Path("/usr/bin/sandbox-exec").exists(), "macOS only")
    def test_dev_writes_and_network_are_denied_and_leftovers_killed(self):
        with tempfile.TemporaryDirectory() as ws:
            Path(ws, "probe.py").write_text(PROBE)
            r = run_scorer("python3 probe.py; echo '{\"score\": 1, \"valid\": true}'", ws, timeout=30, sandbox=True)
            self.assertTrue(r["valid"], r)
            out = r["stdout_tail"]
            self.assertIn("TTY_ERRNO 1", out)  # EPERM from the sandbox, not ENXIO from a missing terminal
            self.assertIn("NULL_OK", out)
            self.assertNotIn("NET_OPEN", out)
            time.sleep(0.3)
            self.assertNotEqual(subprocess.run(["pgrep", "-f", "sleep 7779.25"]).returncode, 0)

    @unittest.skipUnless(Path("/usr/bin/sandbox-exec").exists(), "macOS only")
    def test_network_can_be_allowed_explicitly(self):
        from drsi.scorer import _sandbox_argv
        self.assertIn("(deny network*)", _sandbox_argv("true", Path("/tmp"), [])[2])
        self.assertNotIn("(deny network*)", _sandbox_argv("true", Path("/tmp"), [], network=True)[2])


class ModifyParentTest(unittest.TestCase):
    def test_modify_cannot_change_parent(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="1", parent=None))
            with self.assertRaises(TypeError):
                t.modify({"1": lambda n: n.__setitem__("parent", "missing")})


if __name__ == "__main__":
    unittest.main()
