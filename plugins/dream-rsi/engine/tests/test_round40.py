"""Round 40: the review of round 39 (Opus), each finding reproduced here first.

- A worker's shell stops a foreground command after 2 minutes (10 at most), so a long run's helper was killed while
  the run went on, with no way to cancel it or read its output again, and the worker's next request queued behind it.
  The helper now holds a lock on <id>.lock for as long as it lives and names the output file; a run whose helper is
  gone is stopped, a request no helper holds is not run, and the brief says how to run long commands.
- One worker could load the orchestrator that serves every worker: a directory flooded with entries was listed and
  sorted every 0.3 s, and emptying a very deep tree held a handle per level. A flooded directory is listed once and
  no longer served, and the previous call's directory is moved aside, never walked.
- The configured command's own children were not tracked, so one that left its process group outlived the call; a
  request picked up as the call ended could still start. Both are closed.
- A directory argument starting with "-" reached the configured command before its "--".
- Memory-cap kills from a worker's proposal calls were lost from the record; the final poll at the end of a call
  could record processes that were being torn down anyway.
- macOS's resident size leaves out compressed and swapped pages, so the cap read low exactly under memory pressure:
  on macOS it now reads each process's physical footprint.
- (Grok) The configured command ran with the checkout as its directory, so a relative cmd, or one inside a checkout,
  was a program the worker chose: cmd must be an absolute path, and one inside the workers' checkouts is refused.
- With neither setting, a record carried an empty mem_kills and the process table was read with sizes: both only
  with a cap now. Bad settings stop the run at its start; the log strips control characters from a worker's command.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import agent, offload
from drsi.agent import Descendants
from drsi.store import Campaign

CLIENT = Path(offload.__file__)
MB = 1024 ** 2

FAKE = """#!/bin/sh
# stands in for the compute host
printf '%s\\n' "$@" > "$(dirname "$0")/called.txt"
echo "remote says hello"
shift 5
case "$1" in
  sleep) sleep "$2" ;;
  detach) python3 -c 'import os, sys, time; os.setsid(); open(sys.argv[1], "w").write(str(os.getpid())); time.sleep(60)' \
            "$(dirname "$0")/grandchild.pid" & sleep 60 ;;
esac
exit 3
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.bin = root / "bin"
        self.bin.mkdir()
        self.fake = self.bin / "fake_remote.sh"
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)
        self.camp = Campaign.create("o", {"workspace": {"repo": "/x", "mutable": ["a"]}, "live": {"offload": {
            "cmd": str(self.fake), "mem_gb": 8, "max_mem_gb": 16, "secs": 600, "max_secs": 1800}}},
            home=root / "home")
        self.ws = self.camp.root / "work" / "iter0001-001"
        self.ws.mkdir(parents=True)
        self.d = offload.request_dir(self.camp, "iter0001-001")
        self.env = dict(os.environ, **offload.worker_env(self.camp, self.ws))
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def serve(self):
        return offload.serve(self.camp, self.ws, "iter0001-001", log=self.logs.append)

    def client(self, *args, cwd=None, timeout=30):
        return subprocess.run([sys.executable, str(CLIENT), *args], cwd=cwd or self.ws, env=self.env,
                              capture_output=True, text=True, timeout=timeout)

    def wait_for(self, cond, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not cond():
            time.sleep(0.1)
        return cond()


class HelperTest(Base):
    def test_a_run_whose_helper_is_gone_is_stopped(self):
        with self.serve():
            helper = subprocess.Popen([sys.executable, str(CLIENT), "--", "sleep", "60"], cwd=self.ws, env=self.env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertTrue(self.wait_for((self.bin / "called.txt").exists))
            helper.kill()  # as the worker's shell does at its command time limit
            helper.wait()
            done = self.wait_for(lambda: any(self.d.glob("*.done")), timeout=10)
            self.assertTrue(done, "the run went on after its helper was gone")
            status = json.loads(next(self.d.glob("*.done")).read_text())
        self.assertEqual(status["exit"], 143)
        self.assertIn("helper", status["error"])

    def test_the_helper_names_its_output_file(self):
        with self.serve():
            out = self.client("--", "true")
        self.assertEqual(out.returncode, 3)
        self.assertRegex(out.stderr, r"output is also in .*/offload/[0-9a-f]+\.out")

    def test_a_request_no_helper_holds_is_not_run(self):
        with self.serve():
            (self.d / "h1.req.json").write_text(json.dumps({"argv": ["x"], "cwd": "."}))
            self.assertTrue(self.wait_for(lambda: (self.d / "h1.done").exists()))
        self.assertEqual(json.loads((self.d / "h1.done").read_text())["exit"], 2)
        self.assertFalse((self.bin / "called.txt").exists())

    def test_the_brief_says_how_to_run_long_commands(self):
        text = "\n".join(offload.brief(self.camp))
        self.assertIn("2 minutes", text)
        self.assertIn("background", text)


class LoadTest(Base):
    def test_a_flooded_directory_is_listed_once_and_no_longer_served(self):
        calls = []
        real = os.listdir

        def counting(p=None):
            calls.append(p)
            return real(p)
        with mock.patch.object(offload, "MAX_ENTRIES", 50):
            with self.serve():
                for i in range(60):
                    (self.d / f"junk{i}").write_text("")
                with mock.patch.object(offload.os, "listdir", side_effect=counting):
                    self.assertTrue(self.wait_for(lambda: any("flooded" in m for m in self.logs), timeout=5))
                    n = len(calls)
                    time.sleep(1.0)
                    self.assertEqual(len(calls), n)  # not listed again
                with self.assertRaises(subprocess.TimeoutExpired):
                    self.client("--", "true", timeout=3)
        self.assertFalse((self.bin / "called.txt").exists())

    def test_the_previous_directory_is_moved_aside_not_walked(self):
        deep = self.d
        for _ in range(300):  # deeper than a process's default limit on open files
            deep = deep / "d"
        deep.mkdir(parents=True)
        t0 = time.monotonic()
        with self.serve():
            self.assertEqual(os.listdir(self.d), [])
        self.assertLess(time.monotonic() - t0, 5)
        aside = [p for p in self.d.parent.iterdir() if p.name.startswith(".offload-")]
        self.assertEqual(len(aside), 1)
        self.assertTrue((aside[0] / "d").is_dir())


class RunTest(Base):
    def test_the_commands_detached_children_end_with_the_call(self):
        with self.serve():
            helper = subprocess.Popen([sys.executable, str(CLIENT), "--", "detach"], cwd=self.ws, env=self.env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pidfile = self.bin / "grandchild.pid"
            self.assertTrue(self.wait_for(lambda: pidfile.exists() and pidfile.read_text().strip()))
            time.sleep(1.0)  # the tracker polls every 0.5 s
            pid = int(pidfile.read_text())
        helper.wait(timeout=20)
        self.assertTrue(self.wait_for(lambda: not alive(pid), timeout=5), "a child in its own session outlived the call")

    def test_a_request_is_not_started_once_the_call_has_ended(self):
        s = offload.serve(self.camp, self.ws, "iter0001-001")
        server = s.server
        lock = os.open(self.d / "late.lock", os.O_CREAT | os.O_WRONLY, 0o644)
        fcntl.flock(lock, fcntl.LOCK_EX)
        (self.d / "late.req.json").write_text(json.dumps({"argv": ["x"], "cwd": "."}))
        server.stop_event.set()
        server._handle("late", "late.req.json")
        os.close(lock)
        os.close(server.dfd)
        self.assertEqual(json.loads((self.d / "late.done").read_text())["exit"], 143)
        self.assertFalse((self.bin / "called.txt").exists())

    def test_a_directory_starting_with_a_dash_is_refused(self):
        (self.ws / "-x").mkdir()
        with self.serve():
            out = self.client("--", "true", cwd=self.ws / "-x")
        self.assertEqual(out.returncode, 2)
        self.assertFalse((self.bin / "called.txt").exists())

    def test_a_command_inside_the_workers_checkouts_is_refused(self):
        planted = self.ws / "run.sh"
        planted.write_text("#!/bin/sh\necho PLANTED > \"$(dirname \"$0\")/planted.txt\"\n")
        planted.chmod(0o755)
        self.camp.update_config(lambda c: c["live"]["offload"].update(cmd=str(planted)) or c)
        with self.serve():
            out = self.client("--", "true")
        self.assertNotEqual(out.returncode, 0)
        self.assertFalse((self.ws / "planted.txt").exists())

    def test_the_log_carries_no_control_characters(self):
        with self.serve():
            self.client("--", "echo", "\x1b[31mred\x07")
        line = next(m for m in self.logs if "echo" in m)
        self.assertNotIn("\x1b", line)
        self.assertNotIn("\x07", line)


class SettingsTest(unittest.TestCase):
    def test_bad_settings_are_refused(self):
        exe = sys.executable  # round 41: cmd must be an executable file, so each case fails for its own reason
        for live in ({"offload": {"cmd": exe, "mem_gb": 20, "max_mem_gb": 16}},
                     {"offload": {"cmd": exe, "secs": 0}},
                     {"offload": {"cmd": exe, "max_secs": "long"}},
                     {"offload": {"cmd": 7}},
                     {"offload": {"cmd": "scripts/run.sh"}},
                     {"worker_mem_gb": "three"},
                     {"worker_mem_gb": -1},
                     {"worker_mem_gb": True}):
            with self.assertRaises(ValueError, msg=live):
                offload.check({"live": live})
        offload.check({"live": {"offload": {"cmd": exe, "mem_gb": 8, "max_mem_gb": 24}, "worker_mem_gb": 3}})
        offload.check({"live": {"worker_mem_gb": 2.5}})
        offload.check({"live": {}})

    def test_a_run_with_bad_settings_stops_at_its_start(self):
        from drsi.live import LiveRunner
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("b", {"workspace": {"repo": "/x", "mutable": ["a"]},
                                         "live": {"worker_mem_gb": "three"}}, home=Path(d))
            with self.assertRaises(ValueError):
                LiveRunner(camp, lambda *a, **k: None, None, "iter0001", workspaces=mock.Mock())


class RecordTest(unittest.TestCase):
    def live(self, worker, cap=None):
        from drsi.live import live_round
        from tests.test_live import LiveTest, fixed_checker
        base = LiveTest("test_baseline_measured_on_untouched_base")
        base.setUp()
        self.addCleanup(base.tearDown)
        base._require_check()  # each attempt proposes, is checked, then builds: two calls
        if cap:
            base.camp.update_config(lambda c: c["live"].update(worker_mem_gb=cap) or c)
        r = base.runner(worker, checker=fixed_checker("novel"))
        live_round(base.camp, base.policy, r)
        return [base.camp.tree.get(i) for i in r.ids]

    def test_kills_in_every_call_of_an_attempt_are_recorded(self):
        from tests.test_live import stub_worker
        inner = stub_worker()

        def worker(workspace, prompt, system):
            res = inner(workspace, prompt, system)
            phase = "propose" if "PHASE: PROPOSE" in system else "implement"
            res.mem_kills = [{"pid": 1, "gb": 5.0, "total_gb": 5.0, "command": phase}]
            return res
        nodes = self.live(worker, cap=3)
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual([k["command"] for k in n["worker"]["mem_kills"]], ["propose", "implement"])

    def test_without_kills_the_record_is_as_before(self):
        from tests.test_live import stub_worker
        for n in self.live(stub_worker()):
            self.assertNotIn("mem_kills", n["worker"])


class TrackerTest(unittest.TestCase):
    def test_without_a_cap_the_table_is_read_as_before(self):
        seen = []
        real = subprocess.run

        def run(args, **kw):
            if args and args[0] == "ps":
                seen.append(list(args))
            return real(args, **kw)
        root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(root.kill)
        with mock.patch.object(agent.subprocess, "run", side_effect=run):
            t = Descendants(root.pid, interval=100)
            t.poll()
            t.kill([])
        self.assertTrue(seen)
        self.assertTrue(all(a == ["ps", "-A", "-o", "pid=", "-o", "ppid="] for a in seen), seen)

    def test_the_cap_reads_the_footprint_where_there_is_one(self):
        code = "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(30)"
        root = subprocess.Popen([sys.executable, "-c", code])
        self.addCleanup(root.kill)
        time.sleep(1.0)
        with mock.patch.object(agent, "_footprint", return_value=10 * 1024 ** 3):  # small resident, large footprint
            t = Descendants(root.pid, interval=100, mem_cap_bytes=1024 ** 3)
            t.poll()
            t.kill([])
        self.assertEqual(len(t.mem_kills), 1)
        self.assertEqual(t.mem_kills[0]["gb"], 10.0)

    @unittest.skipUnless(sys.platform == "darwin", "the footprint is read on macOS")
    def test_the_footprint_counts_what_a_process_holds(self):
        code = "a = bytearray(300 * 1024 * 1024); import time; time.sleep(30)"
        p = subprocess.Popen([sys.executable, "-c", code])
        self.addCleanup(p.kill)
        time.sleep(1.5)
        self.assertGreater(agent._footprint(p.pid), 280 * MB)

    def test_the_teardown_poll_kills_but_records_nothing(self):
        code = ("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
                "'a = bytearray(200 * 1024 * 1024); import time; time.sleep(30)']); time.sleep(30)")
        root = subprocess.Popen([sys.executable, "-c", code])
        self.addCleanup(root.kill)
        time.sleep(1.5)
        t = Descendants(root.pid, interval=100, mem_cap_bytes=50 * MB)  # the thread never polls on its own
        t.kill([])
        self.assertEqual(t.mem_kills, [])
        child = [p for p in t.pids if p != root.pid]
        self.assertTrue(child)
        time.sleep(0.3)
        self.assertFalse(any(alive(p) and p != root.pid for p in child))


if __name__ == "__main__":
    unittest.main()
