"""Round 41: the last check before 0.4.2 (Opus), each finding reproduced here first.

- Round 40 moved each call's request directory aside and never removed it, so every call's run output stayed for
  the campaign's life, and a run's output file had no size limit. The previous directories are now removed through
  a handle, a bounded number of levels and entries deep (what is past the bound stays aside), and a run whose output
  passes MAX_OUT is stopped.
- The lock check took any locked file: a hard link to a file another process held locked passed. A lock counts only
  from a plain file with one link.
- When the configured command exited on its own, its process group was not killed, so what it left running in the
  background survived; the group is now killed whether or not the command is still running.
- check() refused a config setting only a maximum below a built-in default (the default is now the smaller of the
  two), and `offload: {}`, `offload: false` and `worker_mem_gb: 0`, which are off; it accepted a cmd that is not an
  executable file, and one inside the workers' checkouts. A refusal is `drsi run`'s (and `drsi baseline`'s) message
  before anything starts, not a traceback.
- A flooded directory was listed whole once; the listing now stops past MAX_ENTRIES.
- The footprint reader raised when its library could not load (the poll, and with it the cap, stopped) and could be
  read half set up by a concurrent first poll; the teardown's directory sweep ran lsof with no directory to look in.
"""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from drsi import agent, cli, offload
from drsi.store import Campaign
from tests.test_round40 import Base as Round40Base, alive

CLIENT = Path(offload.__file__)

FAKE = """#!/bin/sh
printf '%s\\n' "$@" > "$(dirname "$0")/called.txt"
shift 5
case "$1" in
  spew) while :; do echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"; done ;;
  leave) sleep 300 & echo $! > "$(dirname "$0")/left.pid"; exit 3 ;;
esac
exit 3
"""


class Base(Round40Base):
    def setUp(self):
        super().setUp()
        self.fake.write_text(FAKE)


class AsideTest(Base):
    def test_the_previous_directory_is_removed(self):
        for _ in range(3):
            with self.serve():
                self.assertEqual(self.client("--", "true").returncode, 3)
        self.assertEqual([p.name for p in self.d.parent.iterdir() if p.name.startswith(".offload-")], [])

    def test_a_tree_past_the_bound_stays_aside_and_the_call_still_starts(self):
        deep = self.d
        for _ in range(offload.MAX_DEPTH + 5):
            deep = deep / "d"
        deep.mkdir(parents=True)
        with self.serve():
            self.assertEqual(os.listdir(self.d), [])
        self.assertEqual(len([p for p in self.d.parent.iterdir() if p.name.startswith(".offload-")]), 1)

    def test_a_run_whose_output_passes_the_limit_is_stopped(self):
        with mock.patch.object(offload, "MAX_OUT", 100_000):
            with self.serve():
                out = self.client("--", "spew", timeout=30)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("output", out.stdout.splitlines()[-1])
        size = next(self.d.glob("*.out")).stat().st_size
        self.assertLess(size, 5_000_000)


class LockTest(Base):
    def test_a_hard_linked_lock_does_not_count(self):
        held = Path(self.tmp.name) / "held"
        held.write_text("")
        fd = os.open(held, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with self.serve():
                os.link(held, self.d / "k1.lock")
                (self.d / "k1.req.json").write_text(json.dumps({"argv": ["x"], "cwd": "."}))
                self.assertTrue(self.wait_for(lambda: (self.d / "k1.done").exists()))
        finally:
            os.close(fd)
        self.assertEqual(json.loads((self.d / "k1.done").read_text())["exit"], 2)
        self.assertFalse((self.bin / "called.txt").exists())


class GroupTest(Base):
    def test_what_the_command_left_in_its_group_dies_when_it_exits(self):
        with self.serve():
            out = self.client("--", "leave")
            self.assertEqual(out.returncode, 3)
            pid = int((self.bin / "left.pid").read_text())
            self.assertTrue(self.wait_for(lambda: not alive(pid), timeout=5), "a background process outlived its run")


class SettingsTest(unittest.TestCase):
    def test_a_maximum_alone_lowers_the_default(self):
        off = {"cmd": sys.executable, "max_secs": 600, "max_mem_gb": 4}
        offload.check({"live": {"offload": off}})
        camp = mock.Mock(config={"live": {"offload": off}})
        text = "\n".join(offload.brief(camp))
        self.assertIn("defaults 4 GB and 600 s", text)
        ok, _ = offload._validate({"argv": ["x"], "cwd": "."}, off)
        self.assertEqual((ok["mem_gb"], ok["secs"]), (4, 600))

    def test_off_is_off(self):
        for live in ({"offload": {}}, {"offload": False}, {"offload": None}, {"worker_mem_gb": 0},
                     {"worker_mem_gb": None}):
            offload.check({"live": live})

    def test_a_cmd_that_cannot_run_is_refused_at_the_start(self):
        with tempfile.TemporaryDirectory() as d:
            plain = Path(d) / "plain.sh"
            plain.write_text("echo hi\n")  # not executable
            for cmd in ("/nonexistent/run.sh", f"{sys.executable} -c pass", str(plain), d):
                with self.assertRaises(ValueError, msg=cmd):
                    offload.check({"live": {"offload": {"cmd": cmd}}})

    def test_a_cmd_inside_the_workers_checkouts_is_refused_at_the_start(self):
        with tempfile.TemporaryDirectory() as d:
            planted = Path(d) / "work" / "iter0001-001" / "run.sh"
            planted.parent.mkdir(parents=True)
            planted.write_text("#!/bin/sh\n")
            planted.chmod(0o755)
            with self.assertRaises(ValueError):
                offload.check({"live": {"offload": {"cmd": str(planted)}}}, root=Path(d))
            offload.check({"live": {"offload": {"cmd": str(planted)}}})  # without a campaign root: not checkable

    def test_drsi_run_reports_bad_settings_before_it_starts_anything(self):
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("s", {"workspace": {"repo": "/x", "mutable": ["a"]}, "scorer": {"cmd": "true"},
                                         "live": {"worker_mem_gb": "three"}}, home=Path(d))
            self.assertIn("worker_mem_gb", cli._live_ready(camp))


class FloodTest(Base):
    def test_the_listing_stops_past_the_limit(self):
        seen = []
        real = os.scandir

        def counting(p=None):
            it = real(p)

            class Counted:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    it.close()

                def __iter__(self):
                    for e in it:
                        seen.append(e.name)
                        yield e

                def close(self):
                    it.close()
            return Counted()
        with mock.patch.object(offload, "MAX_ENTRIES", 50):
            with self.serve():
                for i in range(500):
                    (self.d / f"junk{i}").write_text("")
                real_listdir = os.listdir

                def listing(p=None):  # a whole listing counts every name it read
                    names = real_listdir(p)
                    seen.extend(names)
                    return names
                with mock.patch.object(offload.os, "scandir", side_effect=counting), \
                        mock.patch.object(offload.os, "listdir", side_effect=listing):
                    self.assertTrue(self.wait_for(lambda: any("flooded" in m for m in self.logs), timeout=5))
        self.assertLessEqual(len(seen), 51 + 5)


class FootprintTest(unittest.TestCase):
    def test_a_library_that_does_not_load_leaves_the_resident_size(self):
        saved = dict(vars(agent._RusageInfoV0))
        try:
            agent._RusageInfoV0.struct = None
            agent._RusageInfoV0.failed = False
            with mock.patch("ctypes.CDLL", side_effect=OSError("no libproc")):
                self.assertIsNone(agent._footprint(os.getpid()))
        finally:
            for k, v in saved.items():
                if not k.startswith("__"):
                    setattr(agent._RusageInfoV0, k, v)

    def test_the_teardown_sweep_with_no_directory_reads_nothing(self):
        with mock.patch.object(agent.subprocess, "run", side_effect=AssertionError("no process listing expected")), \
                mock.patch.object(agent, "_proc_cwds", side_effect=AssertionError("no /proc scan expected")):
            self.assertEqual(agent._working_in([]), set())


if __name__ == "__main__":
    unittest.main()
