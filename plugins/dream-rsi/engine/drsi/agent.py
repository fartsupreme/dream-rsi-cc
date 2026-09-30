"""Headless Claude Code agent runs (tools enabled), for workers and the policy developer.

Runs without user-level settings, so the operator's own hooks and plugins do not
load inside worker sessions; project/local settings of the working directory do.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_CHILDREN: set = set()
_CHILDREN_LOCK = threading.Lock()


def _killpg(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


_REGISTRY = None  # a guardian.Registry while `drsi run` runs: every child group is recorded on disk too
_STOPPING = threading.Event()  # set while a run shuts down: a child started now is killed at once
PS_TIMEOUT = 30
LSOF_TIMEOUT = 60


def attach_registry(registry) -> None:
    global _REGISTRY
    _REGISTRY = registry


def stop_children() -> None:
    """A run is ending: kill every child it started, and every child a thread still running starts from now on."""
    _STOPPING.set()
    kill_all_children()


def allow_children() -> None:
    _STOPPING.clear()


def refuse_if_stopping() -> None:
    """Called before a child is started: none is, once the run is stopping."""
    if _STOPPING.is_set():
        raise RuntimeError("the run is stopping; no new process is started")


def register_child(proc) -> None:
    with _CHILDREN_LOCK:
        _CHILDREN.add(proc)
    if _STOPPING.is_set():  # started in the instant the run began stopping: killed before anything else
        _killpg(proc)
    if _REGISTRY is not None:
        _REGISTRY.add(proc.pid)


def unregister_child(proc) -> None:
    with _CHILDREN_LOCK:
        _CHILDREN.discard(proc)
    if _REGISTRY is not None:
        _REGISTRY.discard(proc.pid)


def kill_all_children() -> None:
    """Kill every worker and scorer process group this process started (they run in their own
    sessions, so a terminal Ctrl-C does not reach them on its own). Each is frozen first with everything
    descended from it, so a command a worker detached into a session of its own is found through its parent before
    the parent dies. If the freeze cannot finish (an unreadable process table) during a run, the processes stay
    stopped for the run's guardian, which reaps with their parentage intact; outside a run the groups are killed."""
    from .guardian import freeze_and_kill
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN)
    if not procs:
        return
    if freeze_and_kill({p.pid for p in procs}) is None and _REGISTRY is not None:
        return
    for proc in procs:
        _killpg(proc)


class Descendants:
    """Tracks every process descended from a child (worker or scorer) while it runs, by polling the process
    table, so children that left its process group (start_new_session, setsid) are still killed when it
    ends. A process that detaches and loses its parent between two polls is not seen; kill() also takes
    directories, and kills this user's processes still working inside them."""

    def __init__(self, root_pid: int, interval: float = 0.25):
        self.pids = {root_pid}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()

    @staticmethod
    def _table() -> dict[int, int]:
        out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True,
                             timeout=PS_TIMEOUT).stdout
        table = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                table[int(parts[0])] = int(parts[1])
        return table

    def poll(self) -> None:
        table = self._table()
        grew = True
        while grew:
            new = {pid for pid, ppid in table.items() if ppid in self.pids and pid not in self.pids}
            self.pids |= new
            grew = bool(new)
        self.pids = {pid for pid in self.pids if pid in table}  # gone: forget, so a reused pid is never hit

    def _loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - tracking is best effort; the group kill still happens
                pass

    def kill(self, dirs: list[Path]) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self.poll()
        except Exception:  # noqa: BLE001
            pass
        for pid in self.pids | (_working_in(dirs) or set()):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _proc_cwds(root: Path, uid: int, timeout: float = LSOF_TIMEOUT) -> dict[int, str] | None:
    """{pid: working directory} of `uid`'s processes from a /proc tree, or None when it cannot be read in time (a
    readlink on a dead mount blocks, so the scan runs in a thread of its own)."""
    out: dict[int, str] = {}
    done = threading.Event()

    def scan():
        try:
            for entry in root.iterdir():
                if entry.name.isdigit():
                    try:
                        if entry.stat().st_uid == uid:
                            out[int(entry.name)] = os.readlink(entry / "cwd")
                    except OSError:
                        pass
            done.set()
        except OSError:
            pass
    t = threading.Thread(target=scan, daemon=True)
    t.start()
    t.join(timeout)
    return dict(out) if done.is_set() else None


def _working_in(dirs: list[Path]) -> set[int] | None:
    """This user's headless processes whose working directory is inside one of `dirs` (the scoring checkout and its
    temp dir): leftovers that detached too fast to be tracked but never left where they were started. A process with
    a controlling terminal is the user's own (a shell or an editor opened there) and is never included. None when the
    process table cannot be read in time: a failed look is not an empty one."""
    roots = [str(Path(d).resolve()) for d in dirs]

    def inside(p: str) -> bool:
        return any(p == r or p.startswith(r + "/") for r in roots)
    found, me = set(), os.getpid()
    if Path("/proc").is_dir():
        cwds = _proc_cwds(Path("/proc"), os.getuid())
        if cwds is None:
            return None
        found = {pid for pid, cwd in cwds.items() if inside(cwd)}
    elif Path("/usr/sbin/lsof").exists():
        try:
            proc = subprocess.run(["/usr/sbin/lsof", "-a", "-d", "cwd", "-u", str(os.getuid()), "-Fpn"],
                                  capture_output=True, text=True, timeout=LSOF_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return None
        out = proc.stdout
        if proc.returncode != 0:  # it exits 0 on a full look: a failure may have skipped the very process sought
            return None
        pid = None
        for line in out.splitlines():
            if line.startswith("p") and line[1:].isdigit():
                pid = int(line[1:])
            elif line.startswith("n") and pid is not None and inside(line[1:]):
                found.add(pid)
    found.discard(me)
    return _headless(found)


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _headless(pids: set[int]) -> set[int] | None:
    """The processes among `pids` with no controlling terminal (ps shows "??" on macOS, "?" on Linux); None when ps
    cannot be read."""
    if not pids:
        return set()
    try:
        proc = subprocess.run(["ps", "-o", "pid=", "-o", "tty=", "-p", ",".join(str(p) for p in sorted(pids))],
                              capture_output=True, text=True, timeout=PS_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    keep, listed = set(), set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            listed.add(int(parts[0]))
            if parts[1] in ("??", "?", "-"):
                keep.add(int(parts[0]))
    if any(_exists(p) for p in pids - listed):  # ps exits 1 for a process that ended, and leaves out a live one only
        return None                             # when it failed
    return keep & pids


def run_group(args, input=None, capture_output=True, text=True, timeout=None, cwd=None, env=None,
              sweep: list | None = None, stdout_path=None):
    """subprocess.run with the child in its own process group, killed on exit or timeout, so tool
    subprocesses cannot outlive the run or keep writing after it; output goes through files so a
    lingering grandchild cannot hold a pipe open. With `sweep`, descendants that left the group and
    processes still working inside those directories are killed too. With `stdout_path`, stdout goes to that file,
    which stays: whatever the child wrote before a kill or a timeout is kept."""
    refuse_if_stopping()
    out_file = open(stdout_path, "w+b") if stdout_path is not None else tempfile.TemporaryFile()
    with tempfile.TemporaryFile() as fin, out_file as fout, tempfile.TemporaryFile() as ferr:
        fin.write((input or "").encode())
        fin.seek(0)
        proc = subprocess.Popen(args, stdin=fin, stdout=fout, stderr=ferr, cwd=cwd, env=env,
                                start_new_session=True)
        register_child(proc)
        tracked = Descendants(proc.pid, interval=0.5) if sweep else None
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _killpg(proc)
            proc.wait()
            raise
        except BaseException:
            _killpg(proc)
            raise
        finally:
            _killpg(proc)
            if tracked is not None:
                tracked.kill([Path(d) for d in sweep])
            unregister_child(proc)
        fout.seek(0)
        ferr.seek(0)
        return subprocess.CompletedProcess(args, rc, stdout=fout.read().decode("utf-8", "replace"),
                                           stderr=ferr.read().decode("utf-8", "replace"))


@dataclass
class AgentResult:
    ok: bool
    result_text: str = ""
    structured: dict | None = None
    session_id: str | None = None
    secs: float = 0.0
    error: str = ""
    transcript: str | None = None  # the call's stream-json transcript, when it was given one


class ClaudeAgent:
    def __init__(self, model: str, tools: str, permission_mode: str = "acceptEdits",
                 allowed_tools: list[str] | None = None, append_system_prompt: str | None = None,
                 json_schema: dict | None = None, binary: str = "claude", runner=run_group,
                 timeout: int = 6 * 3600, extra_args: tuple = (), env: dict | None = None,
                 settings: dict | None = None, setting_sources: str = "project,local"):
        self.model, self.tools, self.permission_mode = model, tools, permission_mode
        self.allowed_tools = allowed_tools or []
        self.append_system_prompt = append_system_prompt
        self.json_schema = json_schema
        self.binary, self.runner, self.timeout = binary, runner, timeout
        self.extra_args = tuple(extra_args)
        self.env = env
        self.settings = settings
        self.setting_sources = setting_sources

    def build_args(self, add_dirs=(), stream: bool = False) -> list[str]:
        # stream: every event of the session goes to stdout as it happens (print mode needs --verbose for it)
        out = ["stream-json", "--verbose"] if stream else ["json"]
        args = [self.binary, "-p", "--model", self.model, "--output-format", *out,
                "--no-session-persistence", "--setting-sources", self.setting_sources, "--strict-mcp-config",
                "--tools", self.tools, "--permission-mode", self.permission_mode]
        if self.settings:
            args += ["--settings", json.dumps(self.settings)]
        if self.allowed_tools:
            args += ["--allowedTools", ",".join(self.allowed_tools)]
        if self.append_system_prompt:
            args += ["--append-system-prompt", self.append_system_prompt]
        if self.json_schema:
            args += ["--json-schema", json.dumps(self.json_schema)]
        for d in add_dirs:
            args += ["--add-dir", str(d)]
        return args + list(self.extra_args)

    def run(self, cwd, prompt: str, add_dirs=(), transcript=None) -> AgentResult:
        """With `transcript` (a file path), the call streams every event of its session into that file as it
        happens, so even a call killed at the timeout leaves its whole record; the result is read from the stream's
        final result event, which carries the fields the plain JSON output does."""
        t0 = time.time()
        kw = {}
        if transcript is not None:
            Path(transcript).parent.mkdir(parents=True, exist_ok=True)
            kw["stdout_path"] = str(transcript)
        tpath = str(transcript) if transcript is not None else None
        try:
            proc = self.runner(self.build_args(add_dirs, stream=transcript is not None), input=prompt,
                               capture_output=True, text=True, timeout=self.timeout, cwd=str(cwd), sweep=[str(cwd)],
                               env=dict(os.environ, **{k: str(v) for k, v in self.env.items()}) if self.env else None,
                               **kw)
        except subprocess.TimeoutExpired:
            where = f"; {_stream_summary(tpath)}" if tpath else ""
            return AgentResult(ok=False, secs=time.time() - t0, transcript=tpath,
                               error=f"agent timed out after {self.timeout}s{where}")
        secs = time.time() - t0
        if tpath:
            env = _stream_result(proc.stdout)
            if env is None:
                return AgentResult(ok=False, secs=secs, transcript=tpath,
                                   error=f"exit {proc.returncode}; no result in the stream; {_stream_summary(tpath)}")
        else:
            try:
                env = json.loads(proc.stdout)
            except json.JSONDecodeError:
                return AgentResult(ok=False, secs=secs, error=f"exit {proc.returncode}; non-JSON output: "
                                                              f"{(proc.stdout or proc.stderr)[-400:]}")
        ok = proc.returncode == 0 and not env.get("is_error") and env.get("subtype") == "success"
        err = "" if ok else f"subtype={env.get('subtype')} {str(env.get('result'))[:300]}"
        if err and tpath:
            err += f"; transcript {tpath}"
        return AgentResult(ok=ok, result_text=str(env.get("result") or ""), structured=env.get("structured_output"),
                           session_id=env.get("session_id"), secs=secs, error=err, transcript=tpath)


def _stream_events(text: str) -> list[dict]:
    out = []
    for line in (text or "").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def _stream_result(text: str) -> dict | None:
    """The stream's final result event (the one Claude Code writes when the session ends)."""
    return next((e for e in reversed(_stream_events(text)) if e.get("type") == "result"), None)


def _stream_summary(path) -> str:
    """One line on a transcript: where it is, how many events it holds, the last event and the last rate-limit
    status the call saw, so an error says where the call stood without opening the file."""
    try:
        events = _stream_events(Path(path).read_text(errors="replace"))
    except OSError:
        return f"transcript {path} (unreadable)"
    parts = [f"transcript {path}: {len(events)} events"]
    if events:
        last = events[-1]
        parts.append(f"last {last.get('type')}" + (f"/{last['subtype']}" if last.get("subtype") else ""))
    rate = next((e.get("rate_limit_info") or {} for e in reversed(events) if e.get("type") == "rate_limit_event"), None)
    if rate is not None:
        parts.append(f"rate limit {rate.get('status')} ({rate.get('rateLimitType')}, "
                     f"utilization {rate.get('utilization')})")
    return ", ".join(parts)
