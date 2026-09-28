"""Round 16: the fourth review of the guardian (round 15's fixes), each finding reproduced here first.

1. The in-process cleanup (Ctrl-C in a batch, the end of `drsi run`) killed the workers' groups before anything walked
   their detached children, whose parentage died with them.
2. A freeze whose process table became unreadable killed what it had stopped, so the next look lost their children.
3. SIGSTOP is delivered asynchronously: a look taken before it lands proved nothing, and a fork in flight was missed.
4. An `lsof` that failed after printing part of the table, or a `ps` that left out a process still there, counted as a
   complete look.
"""
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import agent, guardian
from tests.test_round12 import IDLE, alive, gone_within
from tests.test_round13 import Base


class FourthReviewTest(Base):
    def detaching_worker(self, register=True):
        pid_file = self.root / f"detached-{time.time_ns()}.pid"
        worker = subprocess.Popen([sys.executable, "-c",
                                   "import subprocess, time\n"
                                   f"p = subprocess.Popen({IDLE!r}, cwd={str(self.outside)!r}, start_new_session=True)\n"
                                   f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
                                   "time.sleep(120)\n"], cwd=self.work / "iter0001-001", start_new_session=True)
        self.procs.append(worker)
        end = time.time() + 10
        while not pid_file.exists() and time.time() < end:
            time.sleep(0.05)
        detached = int(pid_file.read_text())
        self.pids.append(detached)
        if register:
            agent.register_child(worker)
        return worker, detached

    def test_the_in_process_cleanup_walks_a_workers_detached_children_before_it_kills(self):
        worker, detached = self.detaching_worker()
        try:
            agent.kill_all_children()
        finally:
            agent.unregister_child(worker)
        self.assertTrue(gone_within(worker.pid, 5))
        self.assertTrue(gone_within(detached, 5), "a worker's detached child outlived the in-process cleanup")

    def test_a_freeze_that_loses_the_table_kills_nothing(self):
        worker, detached = self.detaching_worker(register=False)
        with mock.patch.object(guardian, "process_table", return_value=None):
            self.assertIsNone(guardian.freeze_and_kill({worker.pid}))
        self.assertTrue(alive(worker.pid), "a failed freeze killed its roots")
        os.kill(worker.pid, signal.SIGCONT)

    def test_a_look_counts_only_once_the_stopped_processes_show_stopped(self):
        root = self.start(self.outside)
        child = self.start(self.outside)  # stands for a child forked while the stop was in flight
        real = guardian.process_table()
        r, c = real[root.pid], real[child.pid]
        looks = iter([
            {root.pid: (r[0], r[1], r[2], "S")},                                   # not stopped yet, no child seen
            {root.pid: (r[0], r[1], r[2], "T"), child.pid: (root.pid, c[1], c[2], "S")},  # stopped, and a child
            {root.pid: (r[0], r[1], r[2], "T"), child.pid: (root.pid, c[1], c[2], "T")},
        ] + [{root.pid: (r[0], r[1], r[2], "T"), child.pid: (root.pid, c[1], c[2], "T")}] * 40)
        with mock.patch.object(guardian, "process_table", side_effect=lambda: next(looks)):
            guardian.freeze_and_kill({root.pid})
        self.assertTrue(gone_within(root.pid, 5))
        self.assertTrue(gone_within(child.pid, 5), "a look taken before the stop landed missed a forked child")

    def test_a_partial_lsof_is_a_failed_look(self):
        if Path("/proc").is_dir():
            self.skipTest("the /proc path does not run lsof")
        real = subprocess.run

        def fake(args, **kw):
            if "lsof" in str(args[0]):
                return subprocess.CompletedProcess(args, 1, stdout=f"p{os.getpid()}\nn/\n", stderr="lsof: skipped one")
            return real(args, **kw)
        with mock.patch.object(agent.subprocess, "run", side_effect=fake):
            self.assertIsNone(agent._working_in([self.work]))

    def test_a_ps_that_leaves_out_a_live_process_is_a_failed_look(self):
        a, b = self.start(self.work / "iter0001-001"), self.start(self.work / "iter0001-001")
        real = subprocess.run

        def fake(args, **kw):
            if args[0] == "ps" and "tty=" in args:
                return subprocess.CompletedProcess(args, 1, stdout=f"{a.pid} ??\n", stderr="")
            return real(args, **kw)
        with mock.patch.object(agent.subprocess, "run", side_effect=fake):
            self.assertIsNone(agent._headless({a.pid, b.pid}))


if __name__ == "__main__":
    unittest.main()
