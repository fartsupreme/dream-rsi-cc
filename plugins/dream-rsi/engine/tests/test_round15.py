"""Round 15: the third review of the guardian (round 14's fixes), each finding reproduced here first.

1. Every run's tree went to one file, so a dead run's guardian, finishing a slow look, could overwrite the next run's.
2. The reap walked only from processes the guardian had already seen, and after killing the recorded groups: a command a
   worker detached between two looks (a session of its own, working outside the workspaces) survived.
3. The workspace sweep could report success without a last empty look, and an `lsof` or `ps` that failed with no output
   read as nothing there.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import agent, guardian
from drsi.guardian import Registry, reap
from tests.test_round12 import IDLE, alive, gone_within
from tests.test_round13 import Base


class ThirdReviewTest(Base):
    def test_each_runs_tree_has_a_file_of_its_own(self):
        old = self.dead_run(Registry(self.path, self.work))
        old_ident = guardian.identity(json.loads(self.path.read_text()))
        run = subprocess.Popen(IDLE, start_new_session=True)  # the next run, with its own registry
        self.procs.append(run)
        time.sleep(0.3)
        nxt = Registry(self.path, self.work)
        nxt.open(orchestrator=run.pid)
        ident = guardian.identity(json.loads(self.path.read_text()))
        detached = self.start(self.outside)
        guardian._write_seen(self.path, ident, {detached.pid: guardian.start_time(detached.pid)})
        guardian._write_seen(self.path, old_ident, {})  # the old run's guardian, finishing late
        run.kill()
        run.wait()
        reap(self.path)
        self.assertTrue(gone_within(detached.pid, 5), "the next run's tree was lost to the old run's write")

    def test_a_command_a_worker_detached_before_the_guardian_looked_dies_with_the_run(self):
        reg = self.dead_run(Registry(self.path, self.work))
        pid_file = self.root / "detached.pid"
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
        reg.add(worker.pid)  # recorded, but the guardian never saw its child
        reap(reg.path)
        self.assertTrue(gone_within(worker.pid, 5))
        self.assertTrue(gone_within(detached, 5), "a recorded worker's detached child outlived the reap")

    def test_a_sweep_that_never_finds_the_workspaces_quiet_is_not_a_success(self):
        ended = subprocess.Popen([sys.executable, "-c", "pass"])
        ended.wait()
        with mock.patch.object(agent, "_working_in", return_value={ended.pid}):
            self.assertIsNone(guardian.sweep_workspaces(self.work))

    def test_a_failed_lsof_with_no_output_is_a_failed_look(self):
        if Path("/proc").is_dir():
            self.skipTest("the /proc path does not run lsof")
        real = subprocess.run

        def fake(args, **kw):
            if "lsof" in str(args[0]):
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="lsof: something broke")
            return real(args, **kw)
        with mock.patch.object(agent.subprocess, "run", side_effect=fake):
            self.assertIsNone(agent._working_in([self.work]))


if __name__ == "__main__":
    unittest.main()
