"""Round 36: the review of rounds 34 and 35 (Opus), each finding reproduced here first.

- Round 35 took a call as finished only if its stream held exactly one result event, as the last line. Real sessions
  do not always end so: a worker that leaves a background task running gets its success result and then system
  events (background_tasks_changed, task_updated, task_notification); one whose background task ends in time gets a
  second turn and a second success result. Both were failed as agent errors, losing finished attempts (and, in the
  propose phase, their proposals). The session's result is now its final result event, as Claude Code's own JSON
  output reports it; the call fails if any earlier result in the session was an error (so a success appended after
  an error still fails), and after the final result only whole system events may follow.
- A call that timed out lost its stderr (the runner discarded it on a kill): stderr now goes to <name>.stderr.txt as
  it is written, like stdout to the transcript, and a timeout's error carries its tail. An empty stderr file is
  removed.
- Transcripts were decoded in the locale's encoding and the header written in it: under ISO 8859-1 a result read
  back garbled, under US-ASCII writing failed. Both are UTF-8 now.
- An attempt that failed in the orchestrator after its worker's call (the novelty judge raising, say) recorded no
  worker, so no transcript; an attempt whose proposals never passed recorded no session and no time.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from drsi.agent import AgentResult, ClaudeAgent, _stream_result
from tests.test_round33 import RESULT, stream
from tests.test_round34 import FileOnlyRunner

ENGINE = Path(__file__).resolve().parents[1]
ERROR = dict(RESULT, subtype="error_during_execution", is_error=True, result="failed")
ASSISTANT, USER = {"type": "assistant", "message": {}}, {"type": "user", "message": {}}


def system(sub):
    return {"type": "system", "subtype": sub}


class SessionShapeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def result_of(self, text):
        self.n += 1
        path = Path(self.tmp.name) / f"{self.n}.jsonl"
        path.write_text(text)
        return _stream_result(path)

    def test_a_background_task_left_running_ends_in_system_events_after_the_result(self):
        env, why = self.result_of(stream(system("init"), ASSISTANT, USER, RESULT, system("background_tasks_changed"),
                                         system("task_updated"), system("task_notification")))
        self.assertEqual(env, RESULT, why)

    def test_a_background_task_that_ends_in_time_gives_a_second_turn_and_result(self):
        second = dict(RESULT, result="second turn")
        env, why = self.result_of(stream(system("init"), ASSISTANT, RESULT, system("task_notification"), ASSISTANT,
                                         USER, second))
        self.assertEqual(env, second, why)  # the final result, as Claude Code's JSON output reports

    def test_an_error_earlier_in_the_session_fails_it(self):
        env, why = self.result_of(stream(system("init"), ERROR, RESULT))
        self.assertIsNone(env)
        self.assertIn("error", why)

    def test_a_turn_begun_after_the_last_result_fails_it(self):
        env, why = self.result_of(stream(RESULT, system("task_notification"), ASSISTANT))
        self.assertIsNone(env)

    def test_a_torn_line_after_the_last_result_fails_it(self):
        env, why = self.result_of(stream(RESULT) + '{"type": "system", "subt')
        self.assertIsNone(env)

    def test_a_calls_own_error_result_is_returned_for_the_caller_to_report(self):
        env, why = self.result_of(stream(system("init"), ERROR))
        self.assertEqual(env, ERROR)  # ClaudeAgent.run reports it as a failure with its subtype


class TimeoutStderrTest(unittest.TestCase):
    def test_a_timed_out_call_keeps_its_stderr(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "claude"
            fake.write_text("#!/bin/sh\necho 'API Error: 529 overloaded' >&2\nsleep 30\n")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            t = Path(d) / "w" / "call.jsonl"
            res = ClaudeAgent(model="opus", tools="Read", binary=str(fake), timeout=2).run(d, "p", transcript=t)
            self.assertFalse(res.ok)
            self.assertIn("timed out", res.error)
            self.assertIn("529 overloaded", res.error)
            self.assertIn("529 overloaded", t.with_name("call.stderr.txt").read_text())

    def test_an_empty_stderr_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "claude"
            fake.write_text("#!/bin/sh\necho '" + json.dumps(RESULT) + "'\n")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            t = Path(d) / "w" / "call.jsonl"
            res = ClaudeAgent(model="opus", tools="Read", binary=str(fake), timeout=20).run(d, "p", transcript=t)
            self.assertTrue(res.ok, res)
            self.assertFalse(t.with_name("call.stderr.txt").exists())


class LocaleTest(unittest.TestCase):
    def run_in_locale(self, loc, code):
        env = dict(os.environ, LC_ALL=loc, LANG=loc, PYTHONUTF8="0", PYTHONPATH=str(ENGINE))
        env.pop("PYTHONIOENCODING", None)
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                              encoding="utf-8", timeout=60)

    def test_a_transcript_reads_and_writes_as_utf8_in_any_locale(self):
        with tempfile.TemporaryDirectory() as d:
            text = "café ✓"
            code = f'''
import json, subprocess
from pathlib import Path
from drsi.agent import ClaudeAgent
d = Path({d!r})
def runner(args, stdout_path=None, stderr_path=None, **kw):
    with open(stdout_path, "ab") as fh:
        fh.write((json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                              "result": {text!r}}}, ensure_ascii=False) + "\\n").encode("utf-8"))
    with open(stderr_path, "ab") as fh:
        fh.write({text!r}.encode("utf-8"))
    return subprocess.CompletedProcess(args, 0, stdout="", stderr={text!r})
res = ClaudeAgent(model="opus", tools="Read", runner=runner).run(d, "prompt {text}", transcript=d / "t.jsonl")
print(json.dumps([res.ok, res.result_text]))
'''
            for loc in ("en_US.ISO8859-1", "en_US.US-ASCII"):
                out = self.run_in_locale(loc, code)
                self.assertEqual(out.returncode, 0, out.stderr[-600:])
                self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1]), [True, text], loc)
                head = json.loads((Path(d) / "t.jsonl").read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(head["prompt"], f"prompt {text}")
                (Path(d) / "t.jsonl").unlink()
                for f in Path(d).glob("t.stderr.txt"):
                    f.unlink()


class RecordTest(unittest.TestCase):
    def setUp(self):
        from tests.test_live import LiveTest
        self.base = LiveTest("test_baseline_measured_on_untouched_base")
        self.base.setUp()
        self.addCleanup(self.base.tearDown)
        self.base.camp.update_config(lambda c: c["live"].update(require_check=True, max_proposals=1) or c)

    def worker(self, workspace, prompt, system):
        return AgentResult(ok=True, structured={"proposal": "an idea", "summary": "", "self_reported_score": None,
                                                "notes": ""}, session_id="s-prop", secs=12.5,
                           transcript=f"/t/{Path(workspace).name}.jsonl")

    def run_round(self, checker):
        from drsi.live import live_round
        r = self.base.runner(self.worker, checker=checker)
        live_round(self.base.camp, self.base.policy, r)
        return [self.base.camp.tree.get(i) for i in r.ids]

    def test_an_orchestrator_failure_after_the_call_records_its_transcript(self):
        def checker(proposal, node):
            raise RuntimeError("the judge is down")
        nodes = self.run_round(checker)
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual(n["fail_class"], "orchestrator_error")
            self.assertEqual(n["worker"]["transcript"], f"/t/{n['id']}.jsonl")

    def test_proposals_that_never_passed_record_the_calls_session_and_time(self):
        from tests.test_live import fixed_checker
        nodes = self.run_round(fixed_checker("duplicate"))
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual(n["fail_class"], "not_novel")
            self.assertEqual((n["worker"]["session"], n["worker"]["secs"]), ("s-prop", 12.5))


if __name__ == "__main__":
    unittest.main()
