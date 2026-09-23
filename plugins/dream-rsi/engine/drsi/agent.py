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


def register_child(proc) -> None:
    with _CHILDREN_LOCK:
        _CHILDREN.add(proc)


def unregister_child(proc) -> None:
    with _CHILDREN_LOCK:
        _CHILDREN.discard(proc)


def kill_all_children() -> None:
    """Kill every worker and scorer process group this process started (they run in their own
    sessions, so a terminal Ctrl-C does not reach them on its own)."""
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN)
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
        out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True).stdout
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
        for pid in self.pids | _working_in(dirs):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _working_in(dirs: list[Path]) -> set[int]:
    """This user's processes whose working directory is inside one of `dirs` (the scoring checkout and its
    temp dir): leftovers that detached too fast to be tracked but never left where they were started."""
    roots = [str(Path(d).resolve()) for d in dirs]

    def inside(p: str) -> bool:
        return any(p == r or p.startswith(r + "/") for r in roots)
    found, me = set(), os.getpid()
    if Path("/proc").is_dir():
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    if inside(os.readlink(entry / "cwd")):
                        found.add(int(entry.name))
                except OSError:
                    pass
    elif Path("/usr/sbin/lsof").exists():
        out = subprocess.run(["/usr/sbin/lsof", "-a", "-d", "cwd", "-u", str(os.getuid()), "-Fpn"],
                             capture_output=True, text=True).stdout
        pid = None
        for line in out.splitlines():
            if line.startswith("p") and line[1:].isdigit():
                pid = int(line[1:])
            elif line.startswith("n") and pid is not None and inside(line[1:]):
                found.add(pid)
    found.discard(me)
    return found


def run_group(args, input=None, capture_output=True, text=True, timeout=None, cwd=None, env=None,
              sweep: list | None = None):
    """subprocess.run with the child in its own process group, killed on exit or timeout, so tool
    subprocesses cannot outlive the run or keep writing after it; output goes through files so a
    lingering grandchild cannot hold a pipe open. With `sweep`, descendants that left the group and
    processes still working inside those directories are killed too."""
    with tempfile.TemporaryFile() as fin, tempfile.TemporaryFile() as fout, tempfile.TemporaryFile() as ferr:
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

    def build_args(self, add_dirs=()) -> list[str]:
        args = [self.binary, "-p", "--model", self.model, "--output-format", "json",
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

    def run(self, cwd, prompt: str, add_dirs=()) -> AgentResult:
        t0 = time.time()
        try:
            proc = self.runner(self.build_args(add_dirs), input=prompt, capture_output=True, text=True,
                               timeout=self.timeout, cwd=str(cwd), sweep=[str(cwd)],
                               env=dict(os.environ, **{k: str(v) for k, v in self.env.items()}) if self.env else None)
        except subprocess.TimeoutExpired:
            return AgentResult(ok=False, secs=time.time() - t0, error=f"agent timed out after {self.timeout}s")
        secs = time.time() - t0
        try:
            env = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return AgentResult(ok=False, secs=secs,
                               error=f"exit {proc.returncode}; non-JSON output: {(proc.stdout or proc.stderr)[-400:]}")
        ok = proc.returncode == 0 and not env.get("is_error") and env.get("subtype") == "success"
        return AgentResult(ok=ok, result_text=str(env.get("result") or ""), structured=env.get("structured_output"),
                           session_id=env.get("session_id"), secs=secs,
                           error="" if ok else f"subtype={env.get('subtype')} {str(env.get('result'))[:300]}")
