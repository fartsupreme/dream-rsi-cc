"""Round 34: the review of round 33 (Opus), each finding reproduced here first.

- Every transcript was read back into memory whole and parsed line by line to find the last result: a 100 MB
  transcript peaked at 542 MB. The runner no longer reads the file back, and the result and the error summary are
  found by streaming the file a line at a time.
- A call that failed before it streamed anything (an invalid key, a bad flag) lost its stderr: the error said only
  "no result in the stream". The error now carries stderr's tail, and stderr is kept beside the transcript.
- The transcript did not hold what the call was told: the prompt goes in on stdin and the brief as an appended
  system prompt, and no stream event repeats them. The file's first line is now the call itself: model, prompt,
  appended system prompt.
- An attempt whose worker wrote no proposal, or whose proposals never passed the check, recorded no transcript,
  though its calls had one.
- Every timeout error said "rate limit ...", since each call sees a rate-limit event; `drsi prune --error-match
  "rate limit"` would have matched them all. It now says "usage".
- The dream evaluated the incumbent twice whenever a world was left out as not comparable (the history world, a
  world without cells).
- A transcript path that already exists fails loudly instead of being overwritten.
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import dream
from drsi.agent import AgentResult, ClaudeAgent
from drsi.dream import SEED_POLICY, run_dream
from drsi.store import DEFAULT_CONFIG
from tests.helpers import own_worlds
from tests.test_dream import Dev
from tests.test_round33 import RATE, RESULT, StreamRunner, stream


class FileOnlyRunner(StreamRunner):
    """Writes the stream to the file and returns no stdout, as the real runner does with a transcript."""

    def __init__(self, stdout, rc=0, stderr="", **kw):
        super().__init__(stdout, rc, **kw)
        self.stderr = stderr

    def __call__(self, args, stdout_path=None, stderr_path=None, **kw):
        if stderr_path is not None and self.stderr:  # as the real runner does: stderr goes to its file
            with open(stderr_path, "a") as fh:
                fh.write(self.stderr)
        proc = super().__call__(args, stdout_path=stdout_path, stderr_path=stderr_path, **kw)
        return subprocess.CompletedProcess(args, proc.returncode, stdout="", stderr=self.stderr)


class TranscriptFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "w" / "call.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, runner, **kw):
        return ClaudeAgent(model="opus", tools="Read", append_system_prompt="the brief", runner=runner, **kw)

    def test_the_result_is_read_from_the_file_not_from_memory(self):
        res = self.agent(FileOnlyRunner(stream({"type": "assistant"}) * 1000 + stream(RESULT))).run(
            "/tmp/ws", "p", transcript=self.path)
        self.assertTrue(res.ok, res)
        self.assertEqual(res.structured, {"a": 1})

    def test_a_torn_line_after_the_result_is_a_failure(self):
        # round 35: Claude Code ends a session with exactly one result as its last line; anything after it (a torn
        # line here) is not what Claude Code wrote, so the call fails rather than falling back to an earlier result
        res = self.agent(FileOnlyRunner(stream(RESULT) + '{"type": "result", "subtype": "succ')).run(
            "/tmp/ws", "p", transcript=self.path)
        self.assertFalse(res.ok, res)

    def test_the_first_line_is_the_call_itself(self):
        self.agent(FileOnlyRunner(stream(RESULT))).run("/tmp/ws", "do the thing", transcript=self.path)
        head = json.loads(self.path.read_text().splitlines()[0])
        self.assertEqual(head["type"], "drsi_call")
        self.assertEqual((head["model"], head["prompt"], head["system"]), ("opus", "do the thing", "the brief"))
        self.assertEqual(json.loads(self.path.read_text().splitlines()[1]), RESULT)

    def test_a_call_that_streamed_nothing_keeps_its_stderr(self):
        res = self.agent(FileOnlyRunner("", rc=1, stderr="Error: Invalid API key\n")).run(
            "/tmp/ws", "p", transcript=self.path)
        self.assertFalse(res.ok)
        self.assertIn("Invalid API key", res.error)
        side = self.path.with_name(self.path.stem + ".stderr.txt")
        self.assertEqual(side.read_text(), "Error: Invalid API key\n")

    def test_no_stderr_file_when_there_was_none(self):
        self.agent(FileOnlyRunner(stream(RESULT))).run("/tmp/ws", "p", transcript=self.path)
        self.assertFalse(self.path.with_name(self.path.stem + ".stderr.txt").exists())

    def test_a_timeout_error_says_usage_not_rate_limit(self):
        res = self.agent(FileOnlyRunner(stream(RATE), raise_timeout=True), timeout=5).run(
            "/tmp/ws", "p", transcript=self.path)
        self.assertIn("allowed_warning", res.error)
        self.assertNotIn("rate limit", res.error.lower())
        self.assertIn("1 events", res.error)  # the call's own first line is not a stream event

    def test_an_existing_transcript_is_never_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("earlier call\n")
        with self.assertRaises(FileExistsError):
            self.agent(FileOnlyRunner(stream(RESULT))).run("/tmp/ws", "p", transcript=self.path)
        self.assertEqual(self.path.read_text(), "earlier call\n")


class RecordTest(unittest.TestCase):
    def run_round(self, worker, verdicts=None):
        from drsi.live import live_round
        from tests.test_live import LiveTest, fixed_checker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        self.addCleanup(base.tearDown)
        if verdicts:
            base.camp.update_config(lambda c: c["live"].update(require_check=True, max_proposals=1) or c)
        r = base.runner(worker, checker=fixed_checker(*verdicts) if verdicts else None)
        live_round(base.camp, base.policy, r)
        return [base.camp.tree.get(i) for i in r.ids]

    def test_a_worker_that_wrote_no_proposal_records_its_transcript(self):
        def worker(workspace, prompt, system):
            return AgentResult(ok=True, structured={}, transcript=f"/t/{Path(workspace).name}.jsonl")
        nodes = self.run_round(worker, verdicts=["novel"])
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual(n["worker"]["transcript"], f"/t/{n['id']}.jsonl")

    def test_an_attempt_whose_proposals_never_passed_records_its_transcript(self):
        def worker(workspace, prompt, system):
            return AgentResult(ok=True, structured={"proposal": "the same old thing", "summary": "",
                                                    "self_reported_score": None, "notes": ""},
                               transcript=f"/t/{Path(workspace).name}.jsonl")
        nodes = self.run_round(worker, verdicts=["duplicate"])
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual(n["fail_class"], "not_novel")
            self.assertEqual(n["worker"]["transcript"], f"/t/{n['id']}.jsonl")


class DreamOnceTest(unittest.TestCase):
    def test_a_world_left_out_does_not_evaluate_the_incumbent_twice(self):
        seed = SEED_POLICY.read_text()
        worlds = own_worlds(seed, 4, 24, n=4) + [dict(own_worlds(seed, 4, 24, n=1)[0], id="history", live=False)]
        calls = []
        real = dream.evaluate_policy

        def counted(path, ws, **kw):
            calls.append((Path(path).name, len(ws)))
            return real(path, ws, **kw)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(dream, "evaluate_policy", counted):
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(seed)
            cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"], M=0)}
            run_dream(pdir, worlds, Dev(), cfg, Path(d) / "logs")
        self.assertEqual([c for c in calls if c[0] == "method.py"], [("method.py", 4)])


if __name__ == "__main__":
    unittest.main()
