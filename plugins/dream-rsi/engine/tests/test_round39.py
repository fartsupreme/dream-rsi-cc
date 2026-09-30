"""Round 39: a worker's heavy computation runs on the campaign's compute host, and its local processes have a memory cap.

The machine that runs the workers ran out of memory three times; the third time, one worker's own experiment reached
11.5 GB. Workers were told to test from their checkout, so everything they ran ran on that machine. Now:
- live.offload names a command the orchestrator runs, outside the worker's sandbox, to run a worker's experiment
  elsewhere (for one campaign, on a compute host in a sandbox with its own cores and a memory limit). A worker asks for
  a run by writing a request file into its proposal directory with drsi/offload.py; the orchestrator validates it
  (the command as an argument list, a directory inside the checkout, memory and time within the configured maximum),
  runs it, streams its output back into a file the helper prints, and records its exit status. The worker's sandbox
  is unchanged: it never gets network or a way out, and the orchestrator runs nothing but the configured command.
  The runs end with the worker's call.
- live.worker_mem_gb caps the memory a worker's processes on this machine hold together: the tracker that already
  follows each worker's process tree kills the largest of them (never the worker itself) while their total is over
  the cap, and the attempt records each kill. A cap on each process alone would let a pool of processes under it
  (every recent attempt imports multiprocessing) fill the machine. With the offload in place the cap enforces a rule
  that has a proper alternative.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import cli, offload
from drsi.agent import AgentResult, ClaudeAgent, run_group
from drsi.store import Campaign

MB = 1024 ** 2
CLIENT = Path(offload.__file__)


def spawn_many(sizes_mb: list[int], sleep: float = 20) -> list[str]:
    """A process that starts one child per size, holding that many MB each, and prints their exit statuses in order."""
    kids = [f"a = bytearray({mb} * 1024 * 1024); import time; time.sleep({sleep})" for mb in sizes_mb]
    return [sys.executable, "-c", "import subprocess, sys\n"
            f"ps = [subprocess.Popen([sys.executable, '-c', c]) for c in {kids!r}]\n"
            "print(' '.join(str(p.wait()) for p in ps))"]


def spawn(child_mb: int, sleep: float = 20) -> list[str]:
    """A process that starts a child holding child_mb MB, and prints the child's exit status."""
    child = f"a = bytearray({child_mb} * 1024 * 1024); import time; time.sleep({sleep})"
    return [sys.executable, "-c",
            f"import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', {child!r}]); print(p.wait())"]


class MemoryCapTest(unittest.TestCase):
    def test_a_process_over_the_cap_is_killed_and_reported(self):
        t0 = time.monotonic()
        proc = run_group(spawn(400), timeout=60, mem_cap_bytes=150 * MB)
        self.assertEqual(proc.stdout.strip(), "-9")
        self.assertLess(time.monotonic() - t0, 15)
        self.assertEqual(len(proc.mem_kills), 1)
        self.assertGreater(proc.mem_kills[0]["gb"], 0.14)

    def test_processes_together_over_the_cap_lose_the_largest_until_under_it(self):
        # each child is under the 200 MB cap; together they hold 250 MB: the 120 MB one goes, the others stay
        proc = run_group(spawn_many([60, 120, 70], sleep=4), timeout=60, mem_cap_bytes=200 * MB)
        self.assertEqual(proc.stdout.split(), ["0", "-9", "0"])
        self.assertEqual(len(proc.mem_kills), 1)
        self.assertGreater(proc.mem_kills[0]["gb"], 0.11)
        self.assertGreater(proc.mem_kills[0]["total_gb"], 0.2)

    def test_a_process_under_the_cap_is_left_alone(self):
        proc = run_group(spawn(40, sleep=1), timeout=60, mem_cap_bytes=150 * MB)
        self.assertEqual(proc.stdout.strip(), "0")
        self.assertEqual(proc.mem_kills, [])

    def test_the_worker_itself_is_never_killed_by_the_cap(self):
        code = "a = bytearray(400 * 1024 * 1024); import time; time.sleep(1.5); print('done')"
        proc = run_group([sys.executable, "-c", code], timeout=60, mem_cap_bytes=150 * MB)
        self.assertEqual(proc.stdout.strip(), "done")
        self.assertEqual(proc.mem_kills, [])

    def test_without_a_cap_nothing_is_watched(self):
        proc = run_group([sys.executable, "-c", "print(1)"], timeout=60)
        self.assertEqual(proc.mem_kills, [])

    def test_the_agent_passes_the_cap_and_reports_the_kills(self):
        seen = {}

        def runner(args, **kw):
            seen.update(kw)
            cp = subprocess.CompletedProcess(args, 0, stdout=json.dumps(
                {"type": "result", "subtype": "success", "is_error": False, "result": "ok"}), stderr="")
            cp.mem_kills = [{"pid": 7, "gb": 5.1, "command": "python3"}]
            return cp
        res = ClaudeAgent(model="opus", tools="Read", runner=runner, mem_cap_gb=4).run("/tmp", "p")
        self.assertEqual(seen["mem_cap_bytes"], 4 * 1024 ** 3)
        self.assertTrue(res.ok)
        self.assertEqual(res.mem_kills, [{"pid": 7, "gb": 5.1, "command": "python3"}])
        seen.clear()
        ClaudeAgent(model="opus", tools="Read", runner=runner).run("/tmp", "p")
        self.assertNotIn("mem_cap_bytes", seen)

    def test_the_worker_takes_the_campaigns_cap(self):
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("w", {"workspace": {"repo": "/x", "mutable": ["a"]}, "live": {"worker_mem_gb": 3}},
                                   home=Path(d))
            self.assertEqual(cli.worker_agent(camp, camp.root / "work" / "iter0001-001", "s").mem_cap_gb, 3)
            other = Campaign.create("v", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(d))
            self.assertIsNone(cli.worker_agent(other, other.root / "work" / "iter0001-001", "s").mem_cap_gb)

    def test_the_worker_is_told_the_cap_is_on_its_processes_together(self):
        from drsi.live import LiveRunner
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("w", {"workspace": {"repo": "/x", "mutable": ["a"]}, "live": {"worker_mem_gb": 3}},
                                   home=Path(d))
            r = LiveRunner.__new__(LiveRunner)
            r.camp, r.objective = camp, None
            text = "\n".join(r._context(None, 0, camp.root / "work" / "iter0001-001"))
        self.assertIn("together go over 3 GB", text)
        self.assertIn("largest first", text)

    def test_an_attempt_records_the_kills(self):
        from drsi.live import live_round
        from tests.test_live import LiveTest, stub_worker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        self.addCleanup(base.tearDown)
        inner = stub_worker()

        def worker(workspace, prompt, system):
            res = inner(workspace, prompt, system)
            res.mem_kills = [{"pid": 9, "gb": 11.5, "command": "python3"}]
            return res
        r = base.runner(worker)
        live_round(base.camp, base.policy, r)
        nodes = [base.camp.tree.get(i) for i in r.ids]
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual(n["worker"]["mem_kills"], [{"pid": 9, "gb": 11.5, "command": "python3"}])


FAKE = """#!/bin/sh
# stands in for the compute host: records how it was called, prints, exits with the status the command names
printf '%s\\n' "$@" > "$(dirname "$0")/called.txt"
echo "remote says hello"
echo "remote error line" >&2
shift 5
case "$1" in sleep) sleep "$2" ;; esac
exit "${EXIT_WITH:-3}"
"""


class _OffloadBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fake = root / "bin" / "fake_remote.sh"
        self.fake.parent.mkdir()
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)
        self.camp = Campaign.create("o", {"workspace": {"repo": "/x", "mutable": ["a"]}, "live": {"offload": {
            "cmd": str(self.fake), "mem_gb": 8, "max_mem_gb": 16, "secs": 600, "max_secs": 1800}}},
            home=root / "home")
        self.ws = self.camp.root / "work" / "iter0001-001"
        (self.ws / "sub").mkdir(parents=True)
        self.env = dict(os.environ, **offload.worker_env(self.camp, self.ws))

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, *args, cwd=None, timeout=30):
        return subprocess.run([sys.executable, str(CLIENT), *args], cwd=cwd or self.ws / "sub", env=self.env,
                              capture_output=True, text=True, timeout=timeout)


class OffloadTest(_OffloadBase):
    def test_a_request_runs_the_configured_command_and_streams_back(self):
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            out = self.client("--mem", "12", "--", "python3", "probe.py", "--size", "64")
        self.assertEqual(out.returncode, 3)
        self.assertIn("remote says hello", out.stdout)
        self.assertIn("remote error line", out.stdout)
        called = (self.fake.parent / "called.txt").read_text().splitlines()
        self.assertEqual(called, [str(self.ws), "12", "600", "sub", "--", "python3", "probe.py", "--size", "64"])

    def test_a_command_that_cannot_start_ends_the_request_instead_of_leaving_it_waiting(self):
        self.fake.unlink()  # the configured command is missing
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            out = self.client("--", "true", timeout=20)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("could not run", out.stdout)

    def test_limits_over_the_maximum_are_refused(self):
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            for args in (["--mem", "17"], ["--secs", "1801"], ["--mem", "0"]):
                out = self.client(*args, "--", "true")
                self.assertEqual(out.returncode, 2, args)
                self.assertIn("refused", out.stdout + out.stderr)
        self.assertFalse((self.fake.parent / "called.txt").exists())

    def test_a_directory_outside_the_checkout_is_refused(self):
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            out = self.client("--", "true", cwd=self.camp.root / "work")
        self.assertEqual(out.returncode, 2)
        self.assertFalse((self.fake.parent / "called.txt").exists())

    def test_a_hand_written_request_is_validated_as_well(self):
        d = offload.request_dir(self.camp, "iter0001-001")
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            d.mkdir(parents=True, exist_ok=True)
            for rid, req in (("a", {"argv": "rm -rf /", "cwd": "."}), ("b", {"argv": ["x"], "cwd": "../../.."}),
                             ("c", {"argv": [], "cwd": "."}), ("d", {"argv": ["x"], "cwd": "/etc"})):
                (d / f"{rid}.req.json").write_text(json.dumps(req))
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not all((d / f"{r}.done").exists() for r in "abcd"):
                time.sleep(0.2)
        for r in "abcd":
            self.assertEqual(json.loads((d / f"{r}.done").read_text())["exit"], 2, r)
        self.assertFalse((self.fake.parent / "called.txt").exists())

    def test_a_run_ends_with_the_workers_call(self):
        d = offload.request_dir(self.camp, "iter0001-001")
        proc = None
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            proc = subprocess.Popen([sys.executable, str(CLIENT), "--", "sleep", "60"], cwd=self.ws, env=self.env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not (self.fake.parent / "called.txt").exists():
                time.sleep(0.2)
            t0 = time.monotonic()
        self.assertLess(time.monotonic() - t0, 10)  # stopping the server killed the run
        out, _ = proc.communicate(timeout=20)
        self.assertEqual(proc.returncode, 143)
        done = [json.loads(p.read_text()) for p in d.glob("*.done")]
        self.assertEqual([x["exit"] for x in done], [143])

    def test_without_an_offload_the_helper_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if not k.startswith("DRSI_OFFLOAD")}
            out = subprocess.run([sys.executable, str(CLIENT), "--", "true"], cwd=d, env=env, capture_output=True,
                                 text=True, timeout=30)
        self.assertEqual(out.returncode, 2)
        self.assertIn("not configured", out.stderr)

    def test_the_worker_is_told_how_and_given_the_environment(self):
        from drsi.live import LiveRunner
        r = LiveRunner.__new__(LiveRunner)
        r.camp = self.camp
        r.objective = None
        text = "\n".join(r._context(None, 0, self.ws))
        self.assertIn(str(CLIENT), text)
        self.assertIn("16 GB", text)
        agent = cli.worker_agent(self.camp, self.ws, "s")
        self.assertEqual(agent.env["DRSI_OFFLOAD_DIR"], str(offload.request_dir(self.camp, "iter0001-001")))
        self.assertEqual(agent.env["DRSI_OFFLOAD_WORKSPACE"], str(self.ws))

    def test_a_campaign_without_it_says_nothing_about_it(self):
        from drsi.live import LiveRunner
        camp = Campaign.create("n", {"workspace": {"repo": "/x", "mutable": ["a"]}}, home=Path(self.tmp.name) / "h2")
        r = LiveRunner.__new__(LiveRunner)
        r.camp = camp
        r.objective = None
        self.assertNotIn("offload", "\n".join(r._context(None, 0, camp.root / "work" / "iter0001-001")))
        self.assertNotIn("DRSI_OFFLOAD_DIR", cli.worker_agent(camp, camp.root / "work" / "iter0001-001", "s").env or {})

    def test_the_live_call_serves_requests_while_the_worker_runs(self):
        from drsi.live import live_round
        from tests.test_live import LiveTest, stub_worker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        self.addCleanup(base.tearDown)
        base.camp.update_config(lambda c: c["live"].update(offload={"cmd": str(self.fake), "mem_gb": 8,
                                                                     "max_mem_gb": 16, "secs": 600,
                                                                     "max_secs": 1800}) or c)
        inner = stub_worker()
        outputs = []

        def worker(workspace, prompt, system):
            env = dict(os.environ, **offload.worker_env(base.camp, workspace))
            outputs.append(subprocess.run([sys.executable, str(CLIENT), "--", "true"], cwd=workspace, env=env,
                                          capture_output=True, text=True, timeout=60))
            return inner(workspace, prompt, system)
        r = base.runner(worker)
        live_round(base.camp, base.policy, r)
        self.assertTrue(outputs)
        self.assertTrue(all(o.returncode == 3 and "remote says hello" in o.stdout for o in outputs))


class HardeningTest(_OffloadBase):
    """The orchestrator serves requests outside the worker's sandbox, in a directory the worker can write: nothing a
    worker plants there may make it write anywhere else, read another file, or stall."""

    def setUp(self):
        super().setUp()
        self.d = offload.request_dir(self.camp, "iter0001-001")
        self.outside = Path(self.tmp.name) / "outside.txt"
        self.outside.write_text("untouched\n")

    def wait_done(self, *rids, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not all((self.d / f"{r}.done").exists() for r in rids):
            time.sleep(0.2)
        return {r: json.loads((self.d / f"{r}.done").read_text()) for r in rids}

    def test_a_planted_output_link_is_not_followed(self):
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            (self.d / "r1.out").symlink_to(self.outside)
            (self.d / "r1.req.json").write_text(json.dumps({"argv": ["x"], "cwd": "."}))
            done = self.wait_done("r1")
        self.assertEqual(self.outside.read_text(), "untouched\n")
        self.assertNotEqual(done["r1"]["exit"], 0)
        self.assertFalse((self.fake.parent / "called.txt").exists())

    def test_a_linked_request_is_not_read(self):
        runnable = json.dumps({"argv": ["x"], "cwd": "."})  # a request that would run, were it read through a link
        by_symlink, by_hard_link = Path(self.tmp.name) / "one.json", Path(self.tmp.name) / "two.json"
        by_symlink.write_text(runnable)
        by_hard_link.write_text(runnable)
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            (self.d / "s1.req.json").symlink_to(by_symlink)  # its target has one link: only not following stops it
            os.link(by_hard_link, self.d / "s2.req.json")
            done = self.wait_done("s1", "s2")
        for r in ("s1", "s2"):
            self.assertEqual(done[r]["exit"], 2, r)
        self.assertFalse((self.fake.parent / "called.txt").exists())

    def test_a_refusal_repeats_no_value_from_the_request(self):
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            (self.d / "s3.req.json").write_text(json.dumps({"argv": ["x"], "cwd": ".", "mem_gb": "SECRET-VALUE"}))
            done = self.wait_done("s3")
        self.assertEqual(done["s3"]["exit"], 2)
        self.assertNotIn("SECRET-VALUE", json.dumps(done["s3"]) + (self.d / "s3.out").read_text())

    def test_a_fifo_does_not_stall_the_server(self):
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            os.mkfifo(self.d / "0.req.json")
            try:
                out = self.client("--", "true", timeout=20)
            finally:  # a server stuck opening the pipe is let go, so the test fails rather than hangs
                try:
                    os.close(os.open(self.d / "0.req.json", os.O_WRONLY | os.O_NONBLOCK))
                except OSError:
                    pass
        self.assertEqual(out.returncode, 3)

    def test_a_directory_link_left_in_its_place_is_not_followed(self):
        elsewhere = Path(self.tmp.name) / "elsewhere"
        elsewhere.mkdir()
        self.d.parent.mkdir(parents=True, exist_ok=True)
        self.d.symlink_to(elsewhere)
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            out = self.client("--", "true")
        self.assertEqual(out.returncode, 3)
        self.assertEqual(list(elsewhere.iterdir()), [])
        self.assertFalse(self.d.is_symlink())

    def test_a_directory_swapped_out_mid_call_is_not_written_through(self):
        elsewhere = Path(self.tmp.name) / "elsewhere"
        elsewhere.mkdir()
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            self.d.rename(self.d.parent / "moved")
            self.d.symlink_to(elsewhere)
            (elsewhere / "w1.req.json").write_text(json.dumps({"argv": ["x"], "cwd": "."}))
            time.sleep(1.5)
        self.assertEqual(sorted(p.name for p in elsewhere.iterdir()), ["w1.req.json"])
        self.assertFalse((self.fake.parent / "called.txt").exists())

    def test_a_request_an_earlier_call_left_is_not_run(self):
        self.d.mkdir(parents=True)
        (self.d / "old.req.json").write_text(json.dumps({"argv": ["x"], "cwd": "."}))
        with offload.serve(self.camp, self.ws, "iter0001-001"):
            time.sleep(1.5)
        self.assertFalse((self.fake.parent / "called.txt").exists())


if __name__ == "__main__":
    unittest.main()
