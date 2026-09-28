"""A run's guardian: what a run started does not outlive it.

`drsi run` holds the campaign's run lock (logs/run.lock) for its whole life; the kernel releases it however the run
ends, so a free lock is what says the run is gone. The run also records every process group it starts in a registry
file beside the lock, with its own pid and start time, and starts a guardian in a session of its own.

Every couple of seconds the guardian notes every process in the run's tree (pid and start time) and keeps that list on
disk beside the registry, since a worker's shell commands run in sessions of their own and can leave the workspaces.
When the lock comes free with the registry still in place (Ctrl-C, a crash, SIGKILL), it takes the lock, kills the
recorded groups and every process in that list that is still the same process, then every headless process of this
user still working in the run's workspaces (a terminal or an editor the user opened there is left alone), removes the
registry and exits. A run that ends normally sweeps its own tree and closes the registry first, and the guardian exits
with it; an interrupted run sweeps and leaves the registry, so the guardian finishes once the run has exited and
nothing can start. A registry that names another run means a newer run owns the campaign, and the guardian exits
without touching it.

`ps` only identifies processes: an unreadable process table or a failed look into the workspaces is "unknown", never
"dead" or "nothing left", and the registry then stays for the next attempt. A process recorded with no start time is
never killed by pid. `drsi stop` asks the run to end through its own cleanup, then makes sure.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REGISTRY = "run-children.json"
SEEN = "run-children.seen"  # + ".<run>.json": each run's tree as its guardian saw it, for whoever reaps
LOCK = "run.lock"  # held by `drsi run` for its whole life (cli._run_lock)
ENGINE_DIR = Path(__file__).resolve().parents[1]
PS_TIMEOUT = 30


def _ps(*args) -> subprocess.CompletedProcess | None:
    try:  # the C locale, so a start time reads the same from every shell
        return subprocess.run(["ps", *args], capture_output=True, text=True, timeout=PS_TIMEOUT,
                              env=dict(os.environ, LC_ALL="C"))
    except (OSError, subprocess.SubprocessError):
        return None


def start_time(pid: int) -> str | None:
    """The process's start time as ps reports it, which with the pid identifies a process despite pid reuse;
    "" when there is no such process, None when it could not be read."""
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except ProcessLookupError:
        return ""
    except PermissionError:
        pass
    except (TypeError, ValueError):
        return None
    out = _ps("-o", "lstart=", "-p", str(pid))
    if out is None:
        return None
    return out.stdout.strip() or ("" if out.returncode else None)


def alive(pid: int, started: str | None) -> bool | None:
    """True while the process is the one recorded, False once it is gone, None when that cannot be told."""
    if not started:
        return None
    now = start_time(pid)
    if now is None:
        return None
    return now == started


def process_table() -> dict[int, tuple[int, str, str, str]] | None:
    """{pid: (ppid, tty, start time, state)} for every live process (a zombie has ended), or None when the table cannot
    be read."""
    out = _ps("-A", "-o", "pid=", "-o", "ppid=", "-o", "stat=", "-o", "tty=", "-o", "lstart=")
    if out is None or out.returncode:
        return None
    table = {}
    for line in out.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[0].isdigit() and parts[1].isdigit() and not parts[2].startswith("Z"):
            table[int(parts[0])] = (int(parts[1]), parts[3], parts[4].strip(), parts[2])
    return table or None


def tree(table: dict, root: int, skip=()) -> dict[int, str]:
    """{pid: start time} of every process descended from `root` (not `root` itself), leaving out `skip`."""
    children: dict[int, list[int]] = {}
    for pid, row in table.items():
        children.setdefault(row[0], []).append(pid)
    found, todo = {}, list(children.get(root, []))
    while todo:
        pid = todo.pop()
        if pid in found or pid in skip or pid == root:
            continue
        found[pid] = table[pid][2]
        todo.extend(children.get(pid, []))
    return found


def identity(data: dict) -> tuple:
    return data.get("orchestrator"), data.get("orchestrator_start")


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _lock_path(path: Path) -> Path:
    return Path(path).parent / LOCK


def run_lock_held(path) -> bool:
    """Whether some process holds the run lock beside this registry (a live `drsi run`, or a reap in progress)."""
    with open(_lock_path(path), "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False


@contextmanager
def holding_run_lock(path, timeout: float | None = None):
    """Take the run lock beside this registry: yields True once held, False if `timeout` passed first."""
    with open(_lock_path(path), "a") as fh:
        end = None if timeout is None else time.time() + timeout
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if end is not None and time.time() >= end:
                    yield False
                    return
                time.sleep(0.2)
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class Registry:
    """The run's record of the process groups it started, kept on disk for the guardian and `drsi stop`."""

    def __init__(self, path, work_dir):
        self.path, self.work = Path(path), Path(work_dir)
        self._lock = threading.Lock()
        self._data: dict = {}
        self._closed = True

    def open(self, orchestrator: int | None = None) -> None:
        pid = os.getpid() if orchestrator is None else int(orchestrator)
        started = None
        for _ in range(3):
            started = start_time(pid)
            if started:
                break
            time.sleep(0.5)
        if not started:  # without it no one can tell this run from a later holder of its pid
            raise RuntimeError(f"cannot read the start time of process {pid}; not starting a run without it")
        with self._lock:
            self._closed = False
            self._data = {"orchestrator": pid, "orchestrator_start": started, "work": str(self.work),
                          "groups": {}}
            _write(self.path, self._data)

    def set_guardian(self, pid: int) -> None:
        started = None
        for _ in range(3):
            started = start_time(pid)
            if started:
                break
            time.sleep(0.5)
        with self._lock:
            if self._closed:
                return
            self._data.update(guardian=pid, guardian_start=started or None)
            _write(self.path, self._data)

    def add(self, pid: int) -> None:
        started = start_time(pid) or None  # unknown: never killed by pid, the tree and the sweep still cover it
        with self._lock:
            if self._closed:
                return
            self._data["groups"][str(pid)] = started
            _write(self.path, self._data)

    def discard(self, pid: int) -> None:
        with self._lock:
            if self._closed:
                return  # a thread finishing after the run closed must not bring the registry back
            self._data["groups"].pop(str(pid), None)
            _write(self.path, self._data)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            files = [self.path] + ([seen_path(self.path, identity(self._data))] if self._data else [])
            for f in files:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass


def _run_dirs(work: Path) -> list[Path]:
    """The run's own workspaces: attempt worktrees, scoring checkouts and proposal directories (a separate
    `drsi rescore` keeps its own)."""
    dirs = [d for d in work.glob("*") if d.is_dir() and d.name.startswith(("iter", "_score-"))]
    return dirs + ([work / "_proposals"] if (work / "_proposals").is_dir() else [])


def _kill(pid: int, sig=signal.SIGKILL, group: bool = False) -> bool:
    try:
        (os.killpg if group else os.kill)(int(pid), sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def sweep_workspaces(work: Path, spare=()) -> int | None:
    """Kill every headless process of this user still working in the run's workspaces; returns how many once a look
    finds them quiet, or None when they could not be looked into or never went quiet (a failed look is not a quiet
    one)."""
    from . import agent
    if not work.is_dir():
        return 0
    killed = 0
    for _ in range(6):  # a child can appear while its parent dies: sweep until a look finds nothing
        found = agent._working_in(_run_dirs(work))
        if found is None:
            return None
        found = found - set(spare) - {os.getpid()}
        if not found:
            return killed
        killed += sum(_kill(pid) for pid in found)
        time.sleep(0.2)
    return None


def freeze_and_kill(roots, spare=()) -> int | None:
    """Stop every process in `roots` and everything descended from them (a stopped process cannot start another),
    looking again until a look finds every one of them stopped and nothing new, then kill them all. A stop is
    delivered asynchronously, so a look counts only once ps shows each process stopped: a fork in flight has then
    landed, and its child is in the look. `roots` must already be known to be the right processes. Returns how many
    were killed, or None when the table could not be read or never settled; then nothing is killed, and what was
    stopped stays stopped, with its children, for the next look."""
    spare = set(spare) | {os.getpid()}
    stopped: set[int] = set()
    frontier = set(roots) - spare
    for _ in range(40):
        for pid in frontier:
            _kill(pid, signal.SIGSTOP)
        stopped |= frontier
        table = process_table()
        if table is None:
            return None
        running = {pid for pid in stopped if pid in table and table[pid][3][:1] not in ("T", "t")}
        grown = set()
        for pid in stopped:
            grown |= set(tree(table, pid, skip=spare))
        frontier = grown - stopped
        if not frontier and not running:
            return sum(_kill(pid) for pid in stopped)
        time.sleep(0.05)
    return None


def sweep_tree(root: int, work: Path, spare=()) -> int | None:
    """What a run does as it ends: kill every process in its tree but `spare` (its guardian), then every headless
    leftover in its workspaces. Returns how many were killed, or None when either could not be looked at."""
    table = process_table()
    if table is None:
        return None
    children = {pid for pid, row in table.items() if row[0] == root} - set(spare)
    killed = freeze_and_kill(children, spare=set(spare) | {root})
    if killed is None:
        return None
    in_work = sweep_workspaces(Path(work), spare=set(spare) | {root})
    return None if in_work is None else killed + in_work


def seen_path(path: Path, ident: tuple) -> Path:
    """Where a run's tree is kept: a file of its own, so a late write from an earlier run's guardian cannot replace
    it."""
    key = hashlib.sha256(f"{ident[0]} {ident[1]}".encode()).hexdigest()[:16]
    return Path(path).with_name(f"{SEEN}.{key}.json")


def _write_seen(path: Path, ident: tuple, seen: dict) -> None:
    _write(seen_path(path, ident), {"orchestrator": ident[0], "orchestrator_start": ident[1],
                                    "seen": {str(pid): st for pid, st in seen.items()}})


def _seen_on_disk(path: Path, ident: tuple) -> dict[int, str]:
    try:
        data = json.loads(seen_path(path, ident).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if tuple(identity(data)) != tuple(ident):
        return {}
    return {int(pid): st for pid, st in (data.get("seen") or {}).items()}


def reap(path, ident: tuple | None = None, seen: dict | None = None) -> dict:
    """Kill what a gone run left: its recorded groups, every process in `seen` (pid: start time, the run's tree as
    its guardian saw it) that is still the same process, and the headless processes still working in its
    workspaces; then remove the registry. The caller holds the run lock. With `ident`, a registry that names another
    run is left alone. A run that is visibly alive is never reaped."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {"groups": 0, "processes": 0, "in_work": 0}
    except json.JSONDecodeError:
        return {"groups": 0, "processes": 0, "in_work": 0, "unreadable": True}
    if ident is not None and tuple(identity(data)) != tuple(ident):
        return {"groups": 0, "processes": 0, "in_work": 0, "superseded": True}
    run, run_start = identity(data)
    seen = {**_seen_on_disk(path, (run, run_start)), **(seen or {})}
    table = None
    for _ in range(3):
        table = process_table()
        if table is not None:
            break
        time.sleep(1)
    if table is None:
        return {"groups": 0, "processes": 0, "in_work": 0, "unknown": True}  # retried by the next caller
    if run in table and table[run][2] == run_start:
        return {"groups": 0, "processes": 0, "in_work": 0, "alive": True}
    spare = {data.get("guardian")} - {None}

    def same(pid, started):
        return bool(started) and pid in table and table[pid][2] == started
    leaders = {int(pid) for pid, st in (data.get("groups") or {}).items() if same(int(pid), st)} - spare
    known = {int(pid) for pid, st in (seen or {}).items() if same(int(pid), st)} - spare
    # the recorded groups' leaders and every process seen in the run's tree are the roots: each is stopped with
    # everything descended from it before anything is killed, so no parentage is lost to a kill
    processes = freeze_and_kill(leaders | known, spare=spare)
    for pid in leaders:  # and anything left in their groups
        _kill(pid, group=True)
    groups = len(leaders)
    if processes is None:
        return {"groups": groups, "processes": 0, "in_work": 0, "unknown": True}
    in_work = sweep_workspaces(Path(data.get("work") or ""), spare=spare)
    if in_work is None:  # the workspaces could not be looked into: the registry stays for the next attempt
        return {"groups": groups, "processes": processes, "in_work": 0, "unknown": True}
    for f in (path, seen_path(path, (run, run_start))):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    return {"groups": groups, "processes": processes, "in_work": in_work}


def guard(path, interval: float = 2.0, ident: tuple | None = None) -> str:
    """Watch a run: "closed" when it closes its registry, "superseded" when the registry names another run,
    "reaped" once the run is gone and what it left is killed."""
    path = Path(path)
    seen: dict[int, str] = {}
    written: dict[int, str] = {}
    while True:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            if ident is not None:
                try:
                    seen_path(path, ident).unlink()
                except FileNotFoundError:
                    pass
            return "closed"
        except json.JSONDecodeError:  # caught between a write and its rename: read again
            time.sleep(interval)
            continue
        if ident is None:
            ident = identity(data)
        elif tuple(identity(data)) != tuple(ident):
            try:
                seen_path(path, ident).unlink()
            except FileNotFoundError:
                pass
            return "superseded"
        table = process_table()
        if table is not None:
            seen = {pid: st for pid, st in seen.items() if pid in table and table[pid][2] == st}
            if ident[0] in table and table[ident[0]][2] == ident[1]:
                seen.update(tree(table, ident[0], skip={os.getpid()}))
            if seen != written:
                _write_seen(path, ident, seen)
                written = dict(seen)
        if not run_lock_held(path):
            with holding_run_lock(path, timeout=interval) as held:
                if held:
                    if not path.exists():
                        return "closed"
                    rep = reap(path, ident, seen)
                    if rep.get("superseded"):
                        return "superseded"
                    if not (rep.get("alive") or rep.get("unknown") or rep.get("unreadable")):
                        return "reaped"
        time.sleep(interval)


def spawn_guardian(path, interval: float = 2.0) -> subprocess.Popen:
    """The guardian, in a session of its own so that a signal meant for the run does not reach it."""
    env = dict(os.environ, PYTHONPATH=str(ENGINE_DIR) + (os.pathsep + os.environ["PYTHONPATH"]
                                                         if os.environ.get("PYTHONPATH") else ""))
    return subprocess.Popen([sys.executable, "-m", "drsi.guardian", str(path), str(interval)], env=env,
                            start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    guard(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
