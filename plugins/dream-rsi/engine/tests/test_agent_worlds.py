import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from drsi.agent import ClaudeAgent
from drsi.store import Tree, make_node
from drsi.worlds import freeze_world, load_worlds, outcome_score, world_from_tree


class FakeRunner:
    def __init__(self, stdout, rc=0):
        self.stdout, self.rc, self.calls = stdout, rc, []

    def __call__(self, args, input=None, capture_output=None, text=None, timeout=None, cwd=None, env=None,
                 sweep=None):
        self.calls.append({"args": args, "input": input, "cwd": cwd, "sweep": sweep})
        return subprocess.CompletedProcess(args, self.rc, stdout=self.stdout, stderr="")


class AgentTest(unittest.TestCase):
    def test_args_and_result(self):
        out = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done",
                          "session_id": "s9", "structured_output": {"a": 1}})
        r = FakeRunner(out)
        agent = ClaudeAgent(model="opus", tools="Read,Edit,Write", allowed_tools=["Bash(drsi check *)"],
                            append_system_prompt="brief", json_schema={"type": "object"}, runner=r)
        res = agent.run("/tmp/ws", "do the thing", add_dirs=["/tmp/extra"])
        args = r.calls[0]["args"]
        self.assertEqual(args[args.index("--tools") + 1], "Read,Edit,Write")
        self.assertEqual(args[args.index("--permission-mode") + 1], "acceptEdits")
        self.assertEqual(args[args.index("--allowedTools") + 1], "Bash(drsi check *)")
        self.assertEqual(args[args.index("--append-system-prompt") + 1], "brief")
        self.assertEqual(args[args.index("--add-dir") + 1], "/tmp/extra")
        self.assertEqual(args[args.index("--setting-sources") + 1], "project,local")
        self.assertEqual(r.calls[0]["cwd"], "/tmp/ws")
        self.assertEqual(r.calls[0]["input"], "do the thing")
        self.assertTrue(res.ok)
        self.assertEqual(res.structured, {"a": 1})
        self.assertEqual(res.session_id, "s9")

    def test_failure_is_reported_not_raised(self):
        res = ClaudeAgent(model="opus", tools="Read", runner=FakeRunner("garbage", rc=1)).run("/tmp", "p")
        self.assertFalse(res.ok)
        self.assertTrue(res.error)


class WorldsTest(unittest.TestCase):
    def test_outcome_score_ranks(self):
        self.assertGreater(outcome_score({"fingerprint": {"outcome": "partial"}}),
                           outcome_score({"fingerprint": {"outcome": "refuted"}}))
        self.assertIsNone(outcome_score({"fingerprint": None}))

    def test_world_from_tree_uses_score_then_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(Path(d) / "t.jsonl")
            t.add(make_node(id="a", parent=None, score=0.7, valid=True))
            t.add(make_node(id="b", parent="a", fingerprint={"outcome": "partial", "family": "F01"}))
            t.add(make_node(id="c", parent="b", valid=False, fail_class="timeout"))
            w = world_from_tree(t, "w", baseline=0.1)
            nodes = {n["id"]: n for n in w["nodes"]}
            self.assertEqual(nodes["a"]["score"], 0.7)
            self.assertAlmostEqual(nodes["b"]["score"], outcome_score(t.get("b")))
            self.assertEqual(nodes["b"]["family"], "F01")
            self.assertFalse(nodes["c"]["valid"])
            self.assertEqual(w["baseline"], 0.1)

    def test_freeze_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            pool = Path(d) / "trace_pool"
            freeze_world(pool, {"id": "iter0001", "baseline": 0.0, "nodes": [{"id": "x", "parent": None, "score": 1}]})
            freeze_world(pool, {"id": "iter0002", "baseline": 0.0, "nodes": []})
            ws = load_worlds(pool)
            self.assertEqual([w["id"] for w in ws], ["iter0001"])  # empty worlds skipped
            with self.assertRaises(FileExistsError):
                freeze_world(pool, {"id": "iter0001", "baseline": 0.0, "nodes": []})  # frozen means immutable


if __name__ == "__main__":
    unittest.main()
