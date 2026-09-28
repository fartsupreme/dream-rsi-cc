"""Round 13: review findings on the guardian (round 12), each reproduced here first.

1. A worker's shell command in a session of its own, working outside the swept workspaces (a temp dir, the proposal
   directory), outlived the run: the guardian knew only the recorded groups and the workspaces.
2. The workspace sweep killed any process of the user's in those directories, a terminal or an editor included.
3. A `ps` that failed or timed out read as a dead run, and a group recorded without a start time matched any process
   that later held its pid.
4. The guardian of a dead run could reap the next run's registry.
5. On Ctrl-C the run closed its registry before the commands its workers had detached were swept, and threads still
   running could start new children after the cleanup.
6. `lsof` had no deadline, so one stuck call hung the sweep.
7. `drsi stop` crashed, skipping the cleanup, when the run exited between its last check and SIGKILL.

The run's own lock (logs/run.lock, held for the life of `drsi run`, released by the kernel however the run ends) is now
what says a run is gone; `ps` only identifies processes, and an unreadable table is "unknown", never "dead".
"""
import fcntl
import io
import json
import os
import pty
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from drsi import agent, cli, guardian
from drsi.guardian import Registry, guard, reap
from tests.test_round12 import ENGINE, IDLE, alive, gone_within


def hold_lock(logs: Path):
    """This process as the live run: it holds the run's lock."""
    fh = open(logs / "run.lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(os.path.realpath(self.tmp.name))
        self.work = self.root / "work"
        (self.work / "iter0001-001").mkdir(parents=True)
        (self.work / "_proposals" / "iter0001-001").mkdir(parents=True)
        self.outside = self.root / "elsewhere"
        self.outside.mkdir()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.path = self.logs / guardian.REGISTRY
        self.pids = []
        self.procs = []

    def tearDown(self):
        agent.allow_children()
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        self.tmp.cleanup()

    def start(self, cwd):
        p = subprocess.Popen(IDLE, cwd=cwd, start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        time.sleep(0.3)
        return p

    def dead_run(self, reg: Registry):
        """A registry whose run was a real process, now gone (its lock is free)."""
        run = subprocess.Popen(IDLE, start_new_session=True)
        reg.open(orchestrator=run.pid)
        run.kill()
        run.wait()
        return reg


class GuardianReviewTest(Base):
    def test_a_detached_command_working_outside_the_workspaces_dies_with_the_run(self):
        pid_file = self.root / "grandchild.pid"
        worker = (
            "import subprocess, time\n"
            f"p = subprocess.Popen({IDLE!r}, cwd={str(self.outside)!r}, start_new_session=True)\n"
            f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
            "time.sleep(120)\n")
        code = (
            "import fcntl, os, subprocess, sys, time\n"
            f"sys.path.insert(0, {ENGINE!r})\n"
            "from drsi.guardian import Registry, spawn_guardian\n"
            f"lock = open({str(self.logs / 'run.lock')!r}, 'w'); fcntl.flock(lock, fcntl.LOCK_EX)\n"
            f"reg = Registry({str(self.path)!r}, {str(self.work)!r})\n"
            "reg.open()\n"
            "spawn_guardian(reg.path, interval=0.2)\n"
            f"w = subprocess.Popen([sys.executable, '-c', {worker!r}], cwd={str(self.work / 'iter0001-001')!r},"
            " start_new_session=True)\n"
            "reg.add(w.pid)\n"
            f"while not os.path.exists({str(pid_file)!r}): time.sleep(0.05)\n"
            "time.sleep(1.5)\n"  # the guardian has seen the command in the run's tree
            "os._exit(0)\n")
        subprocess.run([sys.executable, "-c", code], timeout=60)
        grandchild = int(pid_file.read_text())
        self.pids.append(grandchild)
        self.assertTrue(gone_within(grandchild, 15), "a detached command outside the workspaces outlived the run")

    def test_a_process_left_in_the_proposal_directory_is_swept(self):
        reg = self.dead_run(Registry(self.path, self.work))
        leftover = self.start(self.work / "_proposals" / "iter0001-001")
        reap(reg.path)
        self.assertTrue(gone_within(leftover.pid, 5))

    def test_a_process_with_a_terminal_in_a_workspace_is_left_alone(self):
        reg = self.dead_run(Registry(self.path, self.work))
        pid, fd = pty.fork()
        if pid == 0:  # the user's shell, opened in a workspace
            os.chdir(self.work / "iter0001-001")
            os.execvp(IDLE[0], IDLE)
        self.pids.append(pid)
        time.sleep(0.5)
        headless = self.start(self.work / "iter0001-001")
        reap(reg.path)
        self.assertTrue(gone_within(headless.pid, 5))
        self.assertTrue(alive(pid), "the sweep killed a process with a terminal")
        os.close(fd)

    def test_an_unreadable_process_table_is_not_taken_for_a_dead_run(self):
        lock = hold_lock(self.logs)
        reg = Registry(self.path, self.work)
        reg.open()
        worker = self.start(self.work / "iter0001-001")
        reg.add(worker.pid)
        threading.Timer(1.0, reg.close).start()
        with mock.patch.object(guardian, "start_time", return_value=None), \
                mock.patch.object(guardian, "process_table", return_value=None, create=True):
            self.assertEqual(guard(reg.path, interval=0.1), "closed")
        self.assertTrue(alive(worker.pid))
        lock.close()

    def test_a_run_whose_start_time_cannot_be_read_is_not_started(self):
        reg = Registry(self.path, self.work)
        with mock.patch.object(guardian, "start_time", return_value=None):
            with self.assertRaises(RuntimeError):
                reg.open()
        self.assertFalse(self.path.exists())

    def test_a_group_recorded_without_a_start_time_is_not_killed(self):
        reg = self.dead_run(Registry(self.path, self.work))
        bystander = self.start(self.outside)  # holds a pid the registry names, with no start time to match
        data = json.loads(self.path.read_text())
        data["groups"][str(bystander.pid)] = None
        self.path.write_text(json.dumps(data))
        reap(reg.path)
        time.sleep(0.5)
        self.assertTrue(alive(bystander.pid))

    def test_a_dead_runs_guardian_leaves_the_next_run_alone(self):
        dead = self.dead_run(Registry(self.path, self.work))
        first = guardian.identity(json.loads(self.path.read_text()))
        lock = hold_lock(self.logs)  # the next run: it holds the lock and has its own registry
        nxt = Registry(self.path, self.work)
        nxt.open()
        worker = self.start(self.work / "iter0001-001")
        nxt.add(worker.pid)
        rep = reap(dead.path, first)
        self.assertTrue(rep.get("superseded"))
        self.assertEqual(guard(self.path, interval=0.1, ident=first), "superseded")
        self.assertTrue(alive(worker.pid))
        self.assertTrue(self.path.exists())
        nxt.close()
        lock.close()

    def test_a_stuck_lsof_does_not_hang_the_sweep(self):
        if Path("/proc").is_dir():
            self.skipTest("the /proc path does not run lsof")
        real = subprocess.run

        def fake(args, **kw):
            if "lsof" in str(args[0]):
                if kw.get("timeout") is None:
                    time.sleep(15)
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
                raise subprocess.TimeoutExpired(args, kw["timeout"])
            return real(args, **kw)
        t0 = time.time()
        with mock.patch.object(agent.subprocess, "run", side_effect=fake):
            found = agent._working_in([self.work])
        self.assertLess(time.time() - t0, 10)
        self.assertIsNone(found)  # a failed look, not an empty one (round 14)


class RunAndStopReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(os.path.realpath(self.tmp.name))
        from tests.test_scorer_workspace import make_repo
        from drsi.store import Campaign
        self.camp = Campaign.create("t", {
            "goal": "g", "scorer": {"cmd": "true", "timeout_s": 30},
            "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]}}, home=root / "home")
        (self.camp.root / "work" / "iter0001-001").mkdir(parents=True)
        (self.camp.root / "logs").mkdir(exist_ok=True)
        self.outside = root / "elsewhere"
        self.outside.mkdir()
        self.procs = []

    def tearDown(self):
        agent.allow_children()
        try:  # an interrupted in-process run leaves its guardian waiting on this (still live) test process
            data = json.loads((self.camp.root / "logs" / guardian.REGISTRY).read_text())
            if data.get("guardian"):
                os.kill(data["guardian"], signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, json.JSONDecodeError):
            pass
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_DFL)
        self.tmp.cleanup()

    def interrupted_run(self, body):
        from drsi import cli as cli_mod
        real, ready = cli_mod.run_cycles, cli_mod._live_ready

        def fake_cycles(camp, n, **kw):
            body()
            raise KeyboardInterrupt
        cli_mod.run_cycles, cli_mod._live_ready = fake_cycles, (lambda camp: None)
        try:
            with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                cli.main(["run", "-c", str(self.camp.root), "--rounds", "1"])
        finally:
            cli_mod.run_cycles, cli_mod._live_ready = real, ready

    def test_an_interrupted_run_sweeps_what_its_workers_detached_before_it_closes(self):
        seen = {}

        def body():  # a worker's command in a session of its own, working outside the workspaces, recorded nowhere
            p = subprocess.Popen(IDLE, cwd=self.outside, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.procs.append(p)
            seen["pid"] = p.pid
            time.sleep(0.3)
        self.interrupted_run(body)
        self.assertTrue(gone_within(seen["pid"], 5))
        # the registry stays for the guardian, which reaps once the run has exited (round 14)
        self.assertTrue((self.camp.root / "logs" / guardian.REGISTRY).exists())

    def test_after_an_interrupt_no_new_child_is_let_run(self):
        self.interrupted_run(lambda: None)
        t0 = time.time()
        with self.assertRaises(RuntimeError):  # none is started at all (round 14)
            agent.run_group(IDLE, timeout=60)
        self.assertLess(time.time() - t0, 10)

    def test_drsi_stop_survives_the_run_exiting_under_it(self):
        logs = self.camp.root / "logs"
        run = subprocess.Popen([sys.executable, "-c",
                                "import fcntl, time\n"
                                f"fh = open({str(logs / 'run.lock')!r}, 'w'); fcntl.flock(fh, fcntl.LOCK_EX)\n"
                                "time.sleep(120)\n"], start_new_session=True)
        self.procs.append(run)
        time.sleep(0.5)
        worker = subprocess.Popen(IDLE, cwd=self.camp.root / "work" / "iter0001-001", start_new_session=True)
        self.procs.append(worker)
        reg = Registry(logs / guardian.REGISTRY, self.camp.root / "work")
        reg.open(orchestrator=run.pid)
        reg.add(worker.pid)
        real_kill = os.kill

        def kill(pid, sig):  # the run ignores SIGTERM, then exits just before SIGKILL lands
            if pid == run.pid:
                if sig == signal.SIGKILL:
                    real_kill(pid, signal.SIGKILL)
                    run.wait()
                    raise ProcessLookupError
                return None
            return real_kill(pid, sig)
        out = io.StringIO()
        with mock.patch.object(cli.os, "kill", side_effect=kill), redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--grace", "0.5"]), 0)
        self.assertTrue(gone_within(worker.pid, 5))
        self.assertIn("stopped", out.getvalue())

    def test_drsi_stop_does_not_claim_a_run_it_could_not_stop(self):
        run = subprocess.Popen(IDLE, start_new_session=True)  # alive, but not holding the run lock
        worker = subprocess.Popen(IDLE, cwd=self.camp.root / "work" / "iter0001-001", start_new_session=True)
        self.procs += [run, worker]
        time.sleep(0.3)
        reg = Registry(self.camp.root / "logs" / guardian.REGISTRY, self.camp.root / "work")
        reg.open(orchestrator=run.pid)
        reg.add(worker.pid)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["stop", "-c", str(self.camp.root), "--grace", "0.5"]), 1)
        self.assertIn("nothing was stopped", out.getvalue())
        self.assertTrue(alive(run.pid) and alive(worker.pid))


if __name__ == "__main__":
    unittest.main()
