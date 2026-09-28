"""Round 14: the second review of the guardian (round 13's fixes), each finding reproduced here first.

1. `drsi run` overwrote a dead run's registry when reaping it failed (an unreadable process table), so the dead run's
   processes were forgotten.
2. An unreadable process table, or a failed `lsof`, read as "nothing left", and the registry was then closed or removed.
3. The run's tree as the guardian saw it lived only in the guardian's memory: `drsi stop` (or the next run) reaped
   without it whenever it could not confirm the guardian, and removed the registry under it.
4. On Ctrl-C the run closed its registry while a thread could still be starting a process that detached a grandchild:
   now an interrupted run leaves its registry to the guardian, which reaps once the run has exited and nothing can
   start, and no process starts at all once the run is stopping.
5. On Linux the workspace sweep read every user's processes, with no deadline on /proc.
"""
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from drsi import agent, cli, guardian
from drsi.guardian import Registry, reap
from tests.test_round12 import ENGINE, IDLE, alive, gone_within
from tests.test_round13 import Base, hold_lock


def wait_for(cond, secs=15.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


class ReapFailuresTest(Base):
    def test_a_failed_workspace_sweep_leaves_the_registry_for_a_later_reap(self):
        reg = self.dead_run(Registry(self.path, self.work))
        with mock.patch.object(agent, "_working_in", return_value=None):
            rep = reap(reg.path)
        self.assertTrue(rep.get("unknown"))
        self.assertTrue(self.path.exists())

    def test_the_guardian_keeps_the_runs_tree_on_disk(self):
        pid_file = self.root / "grandchild.pid"
        worker = (
            "import subprocess, time\n"
            f"p = subprocess.Popen({IDLE!r}, cwd={str(self.outside)!r}, start_new_session=True)\n"
            f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
            "time.sleep(120)\n")
        run = subprocess.Popen([sys.executable, "-c",
                                "import fcntl, subprocess, sys, time\n"
                                f"sys.path.insert(0, {ENGINE!r})\n"
                                "from drsi.guardian import Registry, spawn_guardian\n"
                                f"lock = open({str(self.logs / 'run.lock')!r}, 'w'); fcntl.flock(lock, fcntl.LOCK_EX)\n"
                                f"reg = Registry({str(self.path)!r}, {str(self.work)!r}); reg.open()\n"
                                "g = spawn_guardian(reg.path, interval=0.2); reg.set_guardian(g.pid)\n"
                                f"w = subprocess.Popen([sys.executable, '-c', {worker!r}], start_new_session=True,"
                                f" cwd={str(self.work / 'iter0001-001')!r})\n"
                                "reg.add(w.pid)\n"
                                "time.sleep(120)\n"], start_new_session=True)
        self.procs.append(run)
        self.assertTrue(wait_for(pid_file.exists))
        grandchild = int(pid_file.read_text())
        self.pids.append(grandchild)
        seen_file = guardian.seen_path(self.path, guardian.identity(json.loads(self.path.read_text())))
        self.assertTrue(wait_for(lambda: seen_file.exists() and str(grandchild) in json.loads(seen_file.read_text())["seen"]),
                        "the guardian did not write the run's tree to disk")
        data = json.loads(self.path.read_text())
        os.kill(data["guardian"], signal.SIGKILL)  # the guardian is gone; then the run is killed
        run.kill()
        run.wait()
        with guardian.holding_run_lock(self.path, timeout=10):
            reap(self.path)
        self.assertTrue(gone_within(grandchild, 5), "a reap without the guardian missed the tree it had recorded")
        self.assertFalse(seen_file.exists())


class RunStartTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(os.path.realpath(self.tmp.name))
        from tests.test_scorer_workspace import make_repo
        from drsi.store import Campaign
        self.camp = Campaign.create("t", {
            "goal": "g", "scorer": {"cmd": "true", "timeout_s": 30},
            "workspace": {"repo": str(make_repo(root)), "mutable": ["value.txt"]}}, home=root / "home")
        (self.camp.root / "work" / "iter0001-001").mkdir(parents=True)
        self.logs = self.camp.root / "logs"
        self.logs.mkdir(exist_ok=True)
        self.path = self.logs / guardian.REGISTRY
        self.outside = root / "elsewhere"
        self.outside.mkdir()
        self.procs = []
        from drsi import cli as cli_mod
        self.cli_mod = cli_mod
        self.real = cli_mod.run_cycles, cli_mod._live_ready

    def tearDown(self):
        self.cli_mod.run_cycles, self.cli_mod._live_ready = self.real
        agent.allow_children()
        try:
            data = json.loads(self.path.read_text())
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

    def fake(self, body):
        def fake_cycles(camp, n, **kw):
            return body()
        self.cli_mod.run_cycles, self.cli_mod._live_ready = fake_cycles, (lambda camp: None)

    def test_a_run_does_not_overwrite_a_registry_it_could_not_reap(self):
        dead = subprocess.Popen(IDLE, start_new_session=True)
        Registry(self.path, self.camp.root / "work").open(orchestrator=dead.pid)
        dead.kill()
        dead.wait()
        before = self.path.read_text()
        self.fake(lambda: {"rounds": []})
        with mock.patch.object(guardian, "process_table", return_value=None), redirect_stdout(io.StringIO()):
            self.assertNotEqual(cli.main(["run", "-c", str(self.camp.root), "--rounds", "1"]), 0)
        self.assertEqual(self.path.read_text(), before)

    def test_an_interrupted_run_leaves_its_registry_to_the_guardian(self):
        def body():
            raise KeyboardInterrupt
        self.fake(body)
        with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            cli.main(["run", "-c", str(self.camp.root), "--rounds", "1"])
        data = json.loads(self.path.read_text())
        self.assertEqual(data["orchestrator"], os.getpid())
        self.assertTrue(alive(data["guardian"]))

    def test_a_run_that_cannot_sweep_leaves_its_registry_to_the_guardian(self):
        real_table = guardian.process_table
        calls = {}

        def table():  # readable while the run starts, unreadable when it sweeps at the end
            return None if calls.get("sweeping") else real_table()

        def body():
            calls["sweeping"] = True
            return {"rounds": []}
        self.fake(body)
        with mock.patch.object(guardian, "process_table", side_effect=table), redirect_stdout(io.StringIO()):
            cli.main(["run", "-c", str(self.camp.root), "--rounds", "1"])
        self.assertTrue(self.path.exists(), "a run that could not sweep closed its registry")

    def test_no_process_starts_once_the_run_is_stopping(self):
        agent.stop_children()
        t0 = time.time()
        with self.assertRaises(RuntimeError):
            agent.run_group(IDLE, timeout=60)
        self.assertLess(time.time() - t0, 5)


class ProcSweepTest(unittest.TestCase):
    def test_the_proc_scan_reads_only_this_users_processes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work = root / "work"
            work.mkdir()
            entry = root / "proc" / "4242"
            entry.mkdir(parents=True)
            (entry / "cwd").symlink_to(work)
            self.assertEqual(agent._proc_cwds(root / "proc", os.getuid()), {4242: str(work)})
            self.assertEqual(agent._proc_cwds(root / "proc", os.getuid() + 1), {})


if __name__ == "__main__":
    unittest.main()
