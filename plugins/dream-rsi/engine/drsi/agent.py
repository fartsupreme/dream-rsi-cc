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
# Claude Code's auto memory loads a project's MEMORY.md into every session and lets the file tools write notes beside
# it, outside the Bash sandbox: a channel between calls the orchestrator neither sees nor checks. Every call runs
# with it off, and with CLAUDE.md files off as well (beside loading no settings source, which already keeps a
# CLAUDE.local.md a worker leaves in its checkout from a continuation of it).
NO_MEMORY = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}


def call_env(extra: dict | None = None) -> dict:
    """The environment of every Claude call: this process's, the call's own settings, and auto memory off; never the
    switch that makes an added directory's CLAUDE.md load (a worker's proposal directory is one)."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (extra or {}).items()})
    env.pop("CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD", None)
    env.update(NO_MEMORY)
    return env


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


class _RusageInfoV0:  # filled in on first use (macOS only)
    struct = None
    call = None
    failed = False


def _footprint(pid: int) -> int | None:
    """A process's physical footprint in bytes on macOS (what Activity Monitor calls its memory: the memory it owns,
    resident, compressed or swapped out; not clean pages of files it maps), or None where it cannot be read. The
    resident size leaves out compressed and swapped pages, so it reads low exactly when memory is short."""
    import sys
    if sys.platform != "darwin" or _RusageInfoV0.failed:
        return None
    try:
        import ctypes
        if _RusageInfoV0.struct is None:
            import ctypes.util

            class Info(ctypes.Structure):  # rusage_info_v0, <sys/resource.h>
                _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
                    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
                    "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]
            fn = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib", use_errno=True).proc_pid_rusage
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            fn.restype = ctypes.c_int
            _RusageInfoV0.call = fn
            _RusageInfoV0.struct = Info  # set last: a concurrent first reader sees all of it or none
    except (ImportError, OSError, AttributeError):  # no library: the resident size stands in, the cap still holds
        _RusageInfoV0.failed = True
        return None
    info = _RusageInfoV0.struct()
    if _RusageInfoV0.call(pid, 0, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V0
        return None
    return int(info.phys_footprint)


class Descendants:
    """Tracks every process descended from a child (worker or scorer) while it runs, by polling the process
    table, so children that left its process group (start_new_session, setsid) are still killed when it
    ends. A process that detaches and loses its parent between two polls is not seen; kill() also takes
    directories, and kills this user's processes still working inside them."""

    def __init__(self, root_pid: int, interval: float = 0.25, mem_cap_bytes: int | None = None):
        self.root = root_pid
        self.pids = {root_pid}
        self.mem_cap = mem_cap_bytes  # the descendants' (never the root's) total; the largest go while over it
        self.mem_kills: list[dict] = []
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

    @staticmethod
    def _rows() -> dict[int, tuple[int, int, str]]:
        """{pid: (ppid, resident KB, command name)} for every process (read only when there is a cap)."""
        out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "rss=", "-o", "comm="],
                             capture_output=True, text=True, timeout=PS_TIMEOUT).stdout
        rows = {}
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) >= 3 and all(x.isdigit() for x in parts[:3]):
                rows[int(parts[0])] = (int(parts[1]), int(parts[2]), parts[3] if len(parts) == 4 else "")
        return rows

    def poll(self, enforce: bool = True) -> None:
        """Follow the tree; with a cap (and `enforce`: not in the poll that ends the call, where everything is
        killed anyway), kill the largest descendants while together they hold more than the cap."""
        rows = self._rows() if self.mem_cap else None
        table = {pid: r[0] for pid, r in rows.items()} if rows is not None else self._table()
        grew = True
        while grew:
            new = {pid for pid, ppid in table.items() if ppid in self.pids and pid not in self.pids}
            self.pids |= new
            grew = bool(new)
        self.pids = {pid for pid in self.pids if pid in table}  # gone: forget, so a reused pid is never hit
        if self.mem_cap and enforce:
            # the cap is on the processes together: a pool of processes each under it can fill the machine as well
            held = {}
            for pid in self.pids - {self.root}:
                size = _footprint(pid)
                held[pid] = size if size is not None else rows[pid][1] * 1024
            total = sum(held.values())
            for pid in sorted(held, key=held.get, reverse=True):
                if total <= self.mem_cap:
                    break
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    continue
                self.mem_kills.append({"pid": pid, "gb": round(held[pid] / 1024 ** 3, 2),
                                       "total_gb": round(total / 1024 ** 3, 2),
                                       "command": os.path.basename(rows[pid][2])[:60]})
                total -= held[pid]

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
            self.poll(enforce=False)  # everything is killed below: nothing here is a kill for the cap
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
    if not dirs:
        return set()  # nothing to look in: no process listing needed
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
              sweep: list | None = None, stdout_path=None, stderr_path=None, mem_cap_bytes: int | None = None):
    """subprocess.run with the child in its own process group, killed on exit or timeout, so tool
    subprocesses cannot outlive the run or keep writing after it; output goes through files so a
    lingering grandchild cannot hold a pipe open. With `sweep`, descendants that left the group and
    processes still working inside those directories are killed too. With `stdout_path`, stdout is appended to that
    file, which stays (whatever the child wrote before a kill or a timeout is kept) and is not read back: the
    returned stdout is empty, so a large transcript never has to fit in memory. With `stderr_path`, stderr is
    appended to that file the same way (and still returned), so it outlives a kill too. With `mem_cap_bytes`, while
    the child's descendants (never the child itself) together hold more resident memory than that, the largest of them
    is killed; the returned process (or the TimeoutExpired raised) carries the kills in `mem_kills`."""
    refuse_if_stopping()
    out_file = open(stdout_path, "ab") if stdout_path is not None else tempfile.TemporaryFile()
    try:
        err_file = open(stderr_path, "a+b") if stderr_path is not None else tempfile.TemporaryFile()
    except BaseException:
        out_file.close()
        raise
    with tempfile.TemporaryFile() as fin, out_file as fout, err_file as ferr:
        fin.write((input or "").encode())
        fin.seek(0)
        proc = subprocess.Popen(args, stdin=fin, stdout=fout, stderr=ferr, cwd=cwd, env=env,
                                start_new_session=True)
        register_child(proc)
        tracked = Descendants(proc.pid, interval=0.5, mem_cap_bytes=mem_cap_bytes) if (sweep or mem_cap_bytes) \
            else None
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _killpg(proc)
            proc.wait()
            e.mem_kills = tracked.mem_kills if tracked is not None else []
            raise
        except BaseException:
            _killpg(proc)
            raise
        finally:
            _killpg(proc)
            if tracked is not None:
                tracked.kill([Path(d) for d in (sweep or [])])
            unregister_child(proc)
        fout.seek(0)
        ferr.seek(0)
        out = "" if stdout_path is not None else fout.read().decode("utf-8", "replace")
        done = subprocess.CompletedProcess(args, rc, stdout=out, stderr=ferr.read().decode("utf-8", "replace"))
        done.mem_kills = tracked.mem_kills if tracked is not None else []
        return done


@dataclass
class AgentResult:
    ok: bool
    result_text: str = ""
    structured: dict | None = None
    session_id: str | None = None
    secs: float = 0.0
    error: str = ""
    transcript: str | None = None  # the call's stream-json transcript, when it was given one
    mem_kills: list | None = None  # processes of the call killed over the memory cap: {pid, gb, total_gb, command}
    limited: bool = False  # refused for the model's usage limit before the model ran (see _refused_up_front)


class ClaudeAgent:
    def __init__(self, model: str, tools: str, permission_mode: str = "acceptEdits",
                 allowed_tools: list[str] | None = None, append_system_prompt: str | None = None,
                 json_schema: dict | None = None, binary: str = "claude", runner=run_group,
                 timeout: int = 6 * 3600, extra_args: tuple = (), env: dict | None = None,
                 settings: dict | None = None, setting_sources: str = "", mem_cap_gb: float | None = None):
        self.model, self.tools, self.permission_mode = model, tools, permission_mode
        self.allowed_tools = allowed_tools or []
        self.append_system_prompt = append_system_prompt
        self.json_schema = json_schema
        self.binary, self.runner, self.timeout = binary, runner, timeout
        self.extra_args = tuple(extra_args)
        self.env = env
        self.settings = settings
        self.setting_sources = setting_sources
        self.mem_cap_gb = mem_cap_gb  # the call's processes together (never the call itself); the largest go

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
        """With `transcript` (a new file's path), the file's first line is the call itself (model, prompt, appended
        system prompt: no stream event repeats them), and every event of the session is streamed after it as it
        happens, so even a call killed at the timeout leaves its whole record. The result is read from the stream's
        final result event, which carries the fields the plain JSON output does; stderr, if any, is kept beside the
        file. An existing file is never overwritten (FileExistsError)."""
        t0 = time.time()
        kw = {}
        tpath = str(transcript) if transcript is not None else None
        epath = str(Path(tpath).with_name(Path(tpath).stem + ".stderr.txt")) if tpath else None
        if tpath:
            Path(tpath).parent.mkdir(parents=True, exist_ok=True)
            with open(tpath, "x", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "drsi_call", "model": self.model, "prompt": prompt,
                                     "system": self.append_system_prompt, "cwd": str(cwd)}) + "\n")
            kw = {"stdout_path": tpath, "stderr_path": epath}
        if self.mem_cap_gb:
            kw["mem_cap_bytes"] = int(self.mem_cap_gb * 1024 ** 3)
        try:
            proc = self.runner(self.build_args(add_dirs, stream=transcript is not None), input=prompt,
                               capture_output=True, text=True, timeout=self.timeout, cwd=str(cwd), sweep=[str(cwd)],
                               env=call_env(self.env),
                               **kw)
        except subprocess.TimeoutExpired as e:
            where = f"; {_stream_summary(tpath)}{_stderr_tail(epath)}" if tpath else ""
            return AgentResult(ok=False, secs=time.time() - t0, transcript=tpath,
                               error=f"agent timed out after {self.timeout}s{where}",
                               mem_kills=list(getattr(e, "mem_kills", []) or []))
        finally:
            if epath and os.path.exists(epath) and os.path.getsize(epath) == 0:
                os.unlink(epath)  # nothing was written to stderr
        secs = time.time() - t0
        kills = list(getattr(proc, "mem_kills", []) or [])
        if tpath:
            env, why = _stream_result(tpath)
            if env is None:
                return AgentResult(ok=False, secs=secs, transcript=tpath, mem_kills=kills,
                                   error=f"exit {proc.returncode}; {why}; {_stream_summary(tpath)}{_stderr_tail(epath)}")
        else:
            try:
                env = json.loads(proc.stdout)
            except json.JSONDecodeError:
                return AgentResult(ok=False, secs=secs, mem_kills=kills, error=f"exit {proc.returncode}; non-JSON output: "
                                                              f"{(proc.stdout or proc.stderr)[-400:]}")
        ok = proc.returncode == 0 and not env.get("is_error") and env.get("subtype") == "success"
        err = "" if ok else f"subtype={env.get('subtype')} {str(env.get('result'))[:300]}"
        if err and tpath:
            err += f"; transcript {tpath}"
        return AgentResult(ok=ok, result_text=str(env.get("result") or ""), structured=env.get("structured_output"),
                           session_id=env.get("session_id"), secs=secs, error=err, transcript=tpath, mem_kills=kills,
                           limited=not ok and _refused_up_front(env, tpath))


def _refused_up_front(env: dict, tpath) -> bool:
    """A call refused for its model's usage limit before the model ran: its result shows no turn of work (one turn, no
    API time, no model usage) and the stream's last usage event says rejected, whatever the error's status (a real
    refusal's is 429); without a stream, a 429 says so. A limit crossed after work, or a short-term rate limit that
    Claude Code retried, is not this: its result shows the work, or its usage status is not rejected."""
    ran = (env.get("num_turns") or 0) > 1 or (env.get("duration_api_ms") or 0) > 0 or bool(env.get("modelUsage"))
    if ran:
        return False
    if tpath:
        return _last_usage_status(tpath) == "rejected"
    return env.get("api_error_status") == 429


def _last_usage_status(path) -> str | None:
    status = None
    for line in _lines(path):
        if "rate_limit_event" in line:
            ev = _json(line)
            if ev is not None and ev.get("type") == "rate_limit_event":
                status = (ev.get("rate_limit_info") or {}).get("status")
    return status


def _lines(path):
    """The file's lines one at a time (a transcript can be larger than memory should hold)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        yield from fh


def _json(line: str) -> dict | None:
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return None
    return e if isinstance(e, dict) else None


def _failed(result: dict) -> bool:
    return bool(result.get("is_error")) or result.get("subtype") != "success"


def _stream_result(path) -> tuple[dict | None, str]:
    """The session's result. A session can hold more than one: when a background task ends, Claude Code starts a turn
    of its own, whose result carries an origin ({"kind": "task-notification"}); one still running when the session
    ends leaves system events after the last result. The worker's own last result is the call's, and an automatic
    turn's replaces it only if it succeeded with a report of its own: an automatic turn that failed or reported
    nothing does not undo a finished attempt. The call fails if an earlier result of the worker's own turns was an
    error, or if anything but whole system events follows the last result; a failure gives (None, why)."""
    own, auto, last, earlier_error, after = None, None, None, False, None
    for line in _lines(path):
        if not line.strip():
            continue
        ev = _json(line)
        if ev is not None and ev.get("type") == "result":
            if ev.get("origin"):
                auto = ev
            else:
                if own is not None and _failed(own):
                    earlier_error = True
                own, auto = ev, None
            last, after = ev, None
        elif last is not None and (ev is None or ev.get("type") != "system"):
            after = "a torn line" if ev is None else f"a {ev.get('type')} event"
    if last is None:
        return None, "no result in the stream"
    if earlier_error:
        return None, "an earlier result of the session was an error"
    if after:
        return None, f"the stream goes on after its final result ({after})"
    if own is None:
        return last, ""
    if not _failed(own) and auto is not None and not _failed(auto) and auto.get("structured_output") is not None:
        return auto, ""
    return own, ""


def _stderr_tail(epath) -> str:
    try:
        text = Path(epath).read_text(encoding="utf-8", errors="replace").strip() if epath else ""
    except OSError:
        return ""
    return f"; stderr: {text[-300:]}" if text else ""


def _stream_summary(path) -> str:
    """One line on a transcript: where it is, how many stream events it holds, its last event and the last usage
    status the call saw, so an error says where the call stood without opening the file."""
    n, e, rate = 0, None, None
    try:
        for line in _lines(path):
            if not line.strip():
                continue
            ev = _json(line)
            if ev is not None and ev.get("type") == "drsi_call":
                continue
            n, e = n + 1, ev
            if ev is not None and ev.get("type") == "rate_limit_event":
                rate = ev
    except OSError:
        return f"transcript {path} (unreadable)"
    parts = [f"transcript {path}: {n} events"]
    if e is not None:
        parts.append(f"last {e.get('type')}" + (f"/{e['subtype']}" if e.get("subtype") else ""))
    info = (rate.get("rate_limit_info") or {}) if rate else None
    if info is not None:
        parts.append(f"last usage status {info.get('status')} ({info.get('rateLimitType')} at "
                     f"{info.get('utilization')})")
    return ", ".join(parts)
