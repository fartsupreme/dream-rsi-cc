import json
import subprocess
import unittest

from drsi.llm import ClaudeCLI, LLMError

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, args, input=None, capture_output=None, text=None, timeout=None, cwd=None, env=None):
        self.calls.append({"args": args, "input": input, "cwd": cwd})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        rc, stdout = out
        return subprocess.CompletedProcess(args, rc, stdout=stdout, stderr="")


def ok(obj):
    return (0, json.dumps({"type": "result", "subtype": "success", "is_error": False,
                           "structured_output": obj, "session_id": "s1"}))


class ClaudeCLITest(unittest.TestCase):
    def test_args_are_lean_and_isolated(self):
        r = FakeRunner([ok({"x": 1})])
        ClaudeCLI(model="opus", system_prompt="be terse", runner=r).json("hello", SCHEMA)
        args = r.calls[0]["args"]
        self.assertEqual(args[0], "claude")
        self.assertIn("-p", args)
        self.assertEqual(args[args.index("--model") + 1], "opus")
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertEqual(args[args.index("--system-prompt") + 1], "be terse")
        self.assertEqual(args[args.index("--setting-sources") + 1], "project,local")
        self.assertIn("--strict-mcp-config", args)
        self.assertIn("--no-session-persistence", args)
        self.assertEqual(json.loads(args[args.index("--json-schema") + 1]), SCHEMA)

    def test_prompt_goes_through_stdin_not_argv(self):
        r = FakeRunner([ok({"x": 1})])
        ClaudeCLI(model="opus", runner=r).json("secret prompt body", SCHEMA)
        self.assertEqual(r.calls[0]["input"], "secret prompt body")
        self.assertNotIn("secret prompt body", r.calls[0]["args"])

    def test_returns_structured_output(self):
        r = FakeRunner([ok({"x": 7})])
        self.assertEqual(ClaudeCLI(model="opus", runner=r).json("p", SCHEMA), {"x": 7})

    def test_retries_once_on_error_then_succeeds(self):
        bad = (0, json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True}))
        r = FakeRunner([bad, ok({"x": 2})])
        self.assertEqual(ClaudeCLI(model="opus", runner=r).json("p", SCHEMA), {"x": 2})
        self.assertEqual(len(r.calls), 2)

    def test_raises_after_second_failure(self):
        r = FakeRunner([(1, "boom"), (0, "not json")])
        with self.assertRaises(LLMError):
            ClaudeCLI(model="opus", runner=r).json("p", SCHEMA)

    def test_missing_structured_output_is_a_failure(self):
        noso = (0, json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "hi"}))
        r = FakeRunner([noso, noso])
        with self.assertRaises(LLMError):
            ClaudeCLI(model="opus", runner=r).json("p", SCHEMA)

    def test_timeout_is_retried_then_raised(self):
        r = FakeRunner([subprocess.TimeoutExpired("claude", 1), subprocess.TimeoutExpired("claude", 1)])
        with self.assertRaises(LLMError):
            ClaudeCLI(model="opus", runner=r, timeout=1).json("p", SCHEMA)


if __name__ == "__main__":
    unittest.main()
