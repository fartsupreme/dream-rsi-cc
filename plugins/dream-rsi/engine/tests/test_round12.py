"""Round 12: nothing a run started outlives it.

Found in use (2026-09-28): stopping `drsi run` with Ctrl-C twice left its workers running with no orchestrator above
them, and commands a worker's shell had started (each in a session of its own) survived for days. The run's own cleanup
cannot cover a run that dies abruptly, so a run now records every process group it starts in a registry, and a guardian
in its own session kills what is recorded, and anything still working in the run's workspaces, as soon as the run is
gone without having closed the registry. `drsi stop` does the same on demand.
"""
import fcntl
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from drsi import cli, guardian
from drsi.guardian import Registry, guard, reap

ENGINE = str(Path(__file__).resolve().parents[1])
IDLE = [sys.executable, "-c", "import time; time.sleep(120)"]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()[:1] != "Z"


def gone_within(pid: int, secs: float) -> bool:
    end = time.time() + secs
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return not alive(pid)


class GuardianTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        (self.work / "iter0001-001").mkdir(parents=True)
        (self.root / "logs").mkdir()
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        self.tmp.cleanup()

    def start(self, cwd, session=True):
        p = subprocess.Popen(IDLE, cwd=cwd, start_new_session=session, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        time.sleep(0.3)
        return p

    def open_dead(self, reg):
        """A registry whose run was a real process, now gone (its run lock is free)."""
        run = subprocess.Popen(IDLE, start_new_session=True)
        reg.open(orchestrator=run.pid)
        run.kill()
        run.wait()

    def hold_lock(self):
        """This process as the live run: it holds the run lock."""
        fh = open(self.root / "logs" / "run.lock", "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fh

    def test_the_registry_records_a_group_and_forgets_it(self):
        reg = Registry(self.root / "logs" / "run-children.json", self.work)
        reg.open()
        worker = self.start(self.work / "iter0001-001")
        reg.add(worker.pid)
        self.assertIn(str(worker.pid), json.loads(reg.path.read_text())["groups"])
        reg.discard(worker.pid)
        self.assertNotIn(str(worker.pid), json.loads(reg.path.read_text())["groups"])
        reg.close()
        self.assertFalse(reg.path.exists())

    def test_reaping_a_dead_run_kills_its_groups_and_what_works_in_its_workspaces(self):
        reg = Registry(self.root / "logs" / "run-children.json", self.work)
        self.open_dead(reg)
        worker = self.start(self.work / "iter0001-001")
        reg.add(worker.pid)
        shell_child = self.start(self.work / "iter0001-001")   # its own session, never recorded
        elsewhere = self.start(self.root)                      # not the run's: left alone
        rep = reap(reg.path)
        self.assertTrue(gone_within(worker.pid, 5))
        self.assertTrue(gone_within(shell_child.pid, 5))
        self.assertTrue(alive(elsewhere.pid))
        self.assertFalse(reg.path.exists())
        self.assertEqual((rep["groups"], rep["in_work"] >= 1), (1, True))

    def test_the_guardian_stands_down_when_the_run_closes_its_registry(self):
        lock = self.hold_lock()
        reg = Registry(self.root / "logs" / "run-children.json", self.work)
        reg.open()  # this process is the live run
        worker = self.start(self.work / "iter0001-001")
        reg.add(worker.pid)
        threading.Timer(0.5, reg.close).start()
        self.assertEqual(guard(reg.path, interval=0.1), "closed")
        self.assertTrue(alive(worker.pid))
        lock.close()

    def test_the_guardian_reaps_when_the_run_is_gone(self):
        reg = Registry(self.root / "logs" / "run-children.json", self.work)
        self.open_dead(reg)
        worker = self.start(self.work / "iter0001-001")
        reg.add(worker.pid)
        self.assertEqual(guard(reg.path, interval=0.1), "reaped")
        self.assertTrue(gone_within(worker.pid, 5))

    def test_a_run_that_dies_abruptly_leaves_nothing_running(self):
        """The whole path: a run records a worker, starts its guardian and exits without any cleanup."""
        registry = self.root / "logs" / "run-children.json"
        pid_file = self.root / "worker.pid"
        code = (
            "import fcntl, os, subprocess, sys, time\n"
            f"sys.path.insert(0, {ENGINE!r})\n"
            "from drsi.guardian import Registry, spawn_guardian\n"
            f"lock = open({str(self.root / 'logs' / 'run.lock')!r}, 'w'); fcntl.flock(lock, fcntl.LOCK_EX)\n"
            f"reg = Registry({str(registry)!r}, {str(self.work)!r})\n"
            "reg.open()\n"
            "spawn_guardian(reg.path, interval=0.2)\n"
            f"w = subprocess.Popen({IDLE!r}, cwd={str(self.work / 'iter0001-001')!r}, start_new_session=True)\n"
            "reg.add(w.pid)\n"
            f"open({str(pid_file)!r}, 'w').write(str(w.pid))\n"
            "time.sleep(0.5)\n"
            "os._exit(0)\n")
        subprocess.run([sys.executable, "-c", code], timeout=30)
        worker = int(pid_file.read_text())
        self.assertTrue(gone_within(worker, 10), "the guardian did not stop the dead run's worker")
        end = time.time() + 10  # the registry goes once the workspaces are swept
        while registry.exists() and time.time() < end:
            time.sleep(0.1)
        self.assertFalse(registry.exists())


class StopCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        from tests.test_scorer_workspace import make_repo
        from drsi.store import Campaign
        self.camp = Campaign.create("t", {
            "goal": "g", "scorer": {"cmd": "true", "timeout_s": 30},
            "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]}}, home=root / "home")
        (self.camp.root / "work" / "iter0001-001").mkdir(parents=True)
        (self.camp.root / "logs").mkdir(exist_ok=True)
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        self.tmp.cleanup()

    def test_drsi_stop_ends_the_run_and_everything_it_started(self):
        lock = str(self.camp.root / "logs" / "run.lock")
        run = subprocess.Popen([sys.executable, "-c", f"import fcntl, time; fh = open({lock!r}, 'w'); "
                                "fcntl.flock(fh, fcntl.LOCK_EX); time.sleep(120)"], start_new_session=True)
        worker = subprocess.Popen(IDLE, cwd=self.camp.root / "work" / "iter0001-001", start_new_session=True)
        self.procs += [run, worker]
        reg = Registry(self.camp.root / "logs" / guardian.REGISTRY, self.camp.root / "work")
        end = time.time() + 10
        while not guardian.run_lock_held(reg.path) and time.time() < end:
            time.sleep(0.05)
        reg.open(orchestrator=run.pid)
        reg.add(worker.pid)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--grace", "2"]), 0)
        self.assertTrue(gone_within(run.pid, 5))
        self.assertTrue(gone_within(worker.pid, 5))
        self.assertIn("stopped", out.getvalue())

    def test_drsi_stop_with_nothing_recorded_says_so(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root)]), 0)
        self.assertIn("no run", out.getvalue())

    def test_drsi_run_keeps_a_registry_and_a_guardian_only_while_it_runs(self):
        from drsi import cli as cli_mod
        seen = {}

        def fake_cycles(camp, n, **kw):
            path = camp.root / "logs" / guardian.REGISTRY
            seen["registry"] = json.loads(path.read_text())["orchestrator"] == os.getpid()
            out = subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout
            seen["guardian"] = [int(l.split()[0]) for l in out.splitlines() if "drsi.guardian" in l and str(path) in l]
            return {"rounds": []}
        real, ready = cli_mod.run_cycles, cli_mod._live_ready
        cli_mod.run_cycles, cli_mod._live_ready = fake_cycles, (lambda camp: None)
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["run", "-c", str(self.camp.root), "--rounds", "1"]), 0)
        finally:
            cli_mod.run_cycles, cli_mod._live_ready = real, ready
            for sig in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_DFL)
        self.assertTrue(seen["registry"])
        self.assertEqual(len(seen["guardian"]), 1)
        self.assertFalse((self.camp.root / "logs" / guardian.REGISTRY).exists())
        self.assertTrue(gone_within(seen["guardian"][0], 10), "the guardian did not exit with the run")


if __name__ == "__main__":
    unittest.main()
