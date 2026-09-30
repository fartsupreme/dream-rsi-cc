"""Round 33: every worker and policy-developer call keeps its transcript.

Workers ran `claude -p --output-format json --no-session-persistence`: a call that finished left only its final JSON,
and one killed at live.timeout_s left nothing, so why a call ran long (still working, rate-limited, API errors) could
not be told afterwards. A call given a transcript path now runs with `--output-format stream-json --verbose`, and
Claude Code writes every event of the session (the prompt it was given, each message and tool call and result,
rate-limit events, the final result) straight into that file as it happens, so a killed call keeps everything up to
the kill. The final result event carries the same fields the JSON output did. Worker calls write
logs/workers/<attempt>/<time>.jsonl in the campaign (each propose and implement call its own file), developer calls
logs/developer/<time>.jsonl, and an attempt's record names the transcript of its last call.
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import cli
from drsi.agent import AgentResult, ClaudeAgent, run_group
from drsi.store import Campaign

RESULT = {"type": "result", "subtype": "success", "is_error": False, "result": "done", "session_id": "s9",
          "structured_output": {"a": 1}}
RATE = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "rateLimitType": "five_hour",
                                                        "utilization": 0.93}}


def stream(*events) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


class StreamRunner:
    """Writes the given stdout to the transcript file as the real runner would, and records the call."""

    def __init__(self, stdout, rc=0, raise_timeout=False):
        self.stdout, self.rc, self.raise_timeout, self.calls = stdout, rc, raise_timeout, []

    def __call__(self, args, input=None, capture_output=None, text=None, timeout=None, cwd=None, env=None,
                 sweep=None, stdout_path=None, stderr_path=None):
        self.calls.append({"args": args, "stdout_path": stdout_path, "stderr_path": stderr_path})
        if stdout_path is not None:
            with open(stdout_path, "a") as fh:
                fh.write(self.stdout)
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, self.rc, stdout=self.stdout, stderr="")


class AgentTranscriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t" / "call.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_call_with_a_transcript_streams_every_event_and_reads_the_result(self):
        r = StreamRunner(stream({"type": "system", "subtype": "init"}, {"type": "assistant", "message": {}}, RATE,
                                RESULT))
        res = ClaudeAgent(model="opus", tools="Read", json_schema={"type": "object"}, runner=r).run(
            "/tmp/ws", "p", transcript=self.path)
        args = r.calls[0]["args"]
        self.assertEqual(args[args.index("--output-format") + 1], "stream-json")
        self.assertIn("--verbose", args)  # Claude Code needs it for stream-json in print mode
        self.assertEqual(Path(r.calls[0]["stdout_path"]), self.path)
        self.assertTrue(self.path.parent.is_dir())
        self.assertTrue(res.ok, res)
        self.assertEqual((res.structured, res.session_id, res.result_text), ({"a": 1}, "s9", "done"))
        self.assertEqual(res.transcript, str(self.path))

    def test_without_a_transcript_the_call_is_unchanged(self):
        r = StreamRunner(json.dumps(RESULT))
        res = ClaudeAgent(model="opus", tools="Read", runner=r).run("/tmp/ws", "p")
        args = r.calls[0]["args"]
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertNotIn("--verbose", args)
        self.assertIsNone(r.calls[0]["stdout_path"])
        self.assertTrue(res.ok)
        self.assertIsNone(res.transcript)

    def test_a_failed_result_is_an_error_that_names_the_transcript(self):
        bad = dict(RESULT, subtype="error_max_turns", is_error=True, result="ran out")
        res = ClaudeAgent(model="opus", tools="Read", runner=StreamRunner(stream(bad), rc=1)).run(
            "/tmp/ws", "p", transcript=self.path)
        self.assertFalse(res.ok)
        self.assertIn("error_max_turns", res.error)
        self.assertIn(str(self.path), res.error)

    def test_a_stream_without_a_result_is_an_error_not_a_crash(self):
        res = ClaudeAgent(model="opus", tools="Read", runner=StreamRunner(stream({"type": "assistant"}) + "junk\n",
                                                                          rc=1)).run("/tmp/ws", "p",
                                                                                      transcript=self.path)
        self.assertFalse(res.ok)
        self.assertIn("no result", res.error)
        self.assertEqual(res.transcript, str(self.path))

    def test_a_timed_out_call_keeps_its_transcript_and_says_what_it_last_saw(self):
        r = StreamRunner(stream({"type": "system", "subtype": "init"}, RATE, {"type": "assistant", "message": {}}),
                         raise_timeout=True)
        res = ClaudeAgent(model="opus", tools="Read", runner=r, timeout=7).run("/tmp/ws", "p",
                                                                                transcript=self.path)
        self.assertFalse(res.ok)
        self.assertIn("timed out after 7s", res.error)
        self.assertIn(str(self.path), res.error)
        self.assertIn("3 events", res.error)
        self.assertIn("allowed_warning", res.error)  # the last usage status the call saw
        self.assertEqual(res.transcript, str(self.path))


class RunGroupStdoutTest(unittest.TestCase):
    def test_a_killed_process_keeps_what_it_wrote(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.jsonl"
            code = "import sys,time\nprint('{\"type\": \"system\"}', flush=True)\ntime.sleep(60)\n"
            t0 = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                run_group([sys.executable, "-c", code], timeout=3, stdout_path=out)
            self.assertLess(time.monotonic() - t0, 30)
            self.assertEqual(out.read_text(), '{"type": "system"}\n')

    def test_a_finished_process_returns_its_stdout_and_keeps_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.jsonl"
            proc = run_group([sys.executable, "-c", "print('hello')"], timeout=30, stdout_path=out)
            self.assertEqual(proc.stdout, "")  # round 34: the file holds it; nothing is read back into memory
            self.assertEqual(out.read_text(), "hello\n")


class CampaignTranscriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camp = Campaign.create("t", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_worker_call_writes_its_own_transcript_under_its_attempt(self):
        seen = []

        def fake_run(agent, cwd, prompt, add_dirs=(), transcript=None):
            seen.append(Path(transcript))
            return AgentResult(ok=True, transcript=str(transcript))
        ws = self.camp.root / "work" / "iter0007-003"
        with mock.patch.object(ClaudeAgent, "run", fake_run):
            worker = cli.make_worker(self.camp)
            worker(ws, "propose", "system")
            worker(ws, "implement", "system")
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])
        for p in seen:
            self.assertEqual(p.parent, self.camp.root / "logs" / "workers" / "iter0007-003")
            self.assertEqual(p.suffix, ".jsonl")
        self.assertLess(seen[0].name, seen[1].name)  # named by time, so they sort in call order

    def test_each_developer_call_writes_its_own_transcript(self):
        seen = []

        def fake_run(agent, cwd, prompt, add_dirs=(), transcript=None):
            seen.append(Path(transcript))
            return AgentResult(ok=True)
        with mock.patch.object(ClaudeAgent, "run", fake_run):
            dev = cli.make_developer(self.camp)
            dev(Path(self.tmp.name), "revise")
        self.assertEqual(seen[0].parent, self.camp.root / "logs" / "developer")

    def test_a_workers_transcript_lives_outside_what_it_may_write(self):
        ws = self.camp.root / "work" / "iter0007-003"
        settings = json.loads((a := cli.worker_agent(self.camp, ws, "s").build_args())[a.index("--settings") + 1])
        for w in settings["sandbox"]["filesystem"]["allowWrite"]:
            self.assertFalse(str(self.camp.root / "logs").startswith(w))


class RecordTest(unittest.TestCase):
    def test_an_attempt_records_the_transcript_of_its_last_call(self):
        from drsi.live import live_round
        from tests.test_live import LiveTest, stub_worker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        try:
            inner = stub_worker()

            def worker(workspace, prompt, system):
                res = inner(workspace, prompt, system)
                res.transcript = f"/logs/workers/{Path(workspace).name}/{len(inner.calls)}.jsonl"
                return res
            r = base.runner(worker)
            live_round(base.camp, base.policy, r)
            nodes = [base.camp.tree.get(i) for i in r.ids]
            self.assertTrue(nodes)
            for n in nodes:
                self.assertRegex(n["worker"]["transcript"], rf"^/logs/workers/{n['id']}/[0-9]+\.jsonl$")
        finally:
            base.tearDown()

    def test_a_worker_without_a_transcript_records_none(self):
        from drsi.live import live_round
        from tests.test_live import LiveTest, stub_worker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        try:
            r = base.runner(stub_worker())
            live_round(base.camp, base.policy, r)
            self.assertTrue(all(base.camp.tree.get(i)["worker"]["transcript"] is None for i in r.ids))
        finally:
            base.tearDown()


if __name__ == "__main__":
    unittest.main()
