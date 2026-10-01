"""Heavy computation off the machine that runs the workers (live.offload).

A worker's shell is sandboxed on the machine that runs the loop: it can write only its workspace and proposal
directory and has no network. With live.offload set, a worker asks for an experiment to run elsewhere by running this
file as a script from its checkout:

    python3 offload.py [--mem GB] [--secs S] -- COMMAND [ARGS...]

The script (the helper) writes a request (the command as an argument list, the directory relative to the checkout,
the limits) into the worker's proposal directory, holds a lock on it for as long as it lives, prints the output as it
arrives, and exits with the command's status. The orchestrator, which runs outside the sandbox, serves the requests
while the worker's call runs (serve()): it checks each one and runs the configured command (live.offload.cmd) as

    CMD CHECKOUT MEM_GB SECONDS DIR -- COMMAND [ARGS...]

with no shell, appends its output to the file the helper prints, and records its exit status. What CMD does with it
is the campaign's (ship the checkout to a compute host and run the command there in a sandbox with its own limits,
say); CHECKOUT and DIR are the worker's to write, so CMD treats them as such. The worker never gets a way out of its
sandbox, and the orchestrator runs nothing but CMD. A run is stopped when its helper ends or the worker's call does:
CMD's process group and every process it started are killed, and a CMD that starts work on another machine stops that
work when it is killed.

Config (live.offload): cmd (required, an executable path), mem_gb and secs (defaults per request), max_mem_gb and
max_secs (the most a request may ask for), note (one line for the worker: what the host has).

The helper runs under whatever python3 the worker has and uses the standard library only.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import sys
import time
import uuid

POLL_S = 0.3
QUEUE_S = 3900  # time allowed on top of a request's own limit (shipping it, waiting for a free slot there)
STILL_RUNNING = 75  # --wait's status when its time is up and the run goes on
WAIT_S = 540  # --wait's default: inside the worker shell's 10-minute limit


def _running(d: str) -> list[str]:
    """The requests in the directory that have not ended, oldest first."""
    try:
        names = os.listdir(d)
    except OSError:
        return []
    ids = [n[:-len(".req.json")] for n in names if n.endswith(".req.json")]
    live = [i for i in ids if not os.path.exists(os.path.join(d, i + ".done"))]
    return sorted(live, key=lambda i: os.path.getmtime(os.path.join(d, i + ".req.json")))


def _follow(out_path: str, done_path: str, deadline: float | None = None) -> int | None:
    """Print a request's output from the start as it arrives; its exit status when it ends, None at the deadline."""
    pos = 0
    out = sys.stdout.buffer

    def drain() -> None:
        nonlocal pos
        try:
            with open(out_path, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
        except FileNotFoundError:
            return
        if chunk:
            pos += len(chunk)
            out.write(chunk)
            out.flush()
    while True:
        done = os.path.exists(done_path)
        drain()
        if done:
            try:
                with open(done_path) as fh:
                    status = json.load(fh)
            except (ValueError, OSError):
                return 1
            if status.get("error") and not status.get("told"):
                print(f"offload: {status['error']}", file=sys.stderr)
            return int(status.get("exit", 1))
        if deadline is not None and time.monotonic() > deadline:
            return None
        time.sleep(POLL_S)


def _client(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="offload", description="run a command on the campaign's compute host")
    ap.add_argument("--mem", type=int, default=None, help="memory limit, GB")
    ap.add_argument("--secs", type=int, default=None, help="time limit, seconds")
    ap.add_argument("--wait", metavar="ID", nargs="?", const="", default=None,
                    help="attach to a running request (with no ID, the only one): print its output, exit with its "
                         "status when it ends")
    ap.add_argument("--for", dest="for_s", type=int, default=WAIT_S, metavar="S",
                    help=f"with --wait, return after S seconds if the run goes on (status {STILL_RUNNING})")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    d, ws = os.environ.get("DRSI_OFFLOAD_DIR"), os.environ.get("DRSI_OFFLOAD_WORKSPACE")
    if not d or not ws:
        print("offload: not configured for this campaign (no DRSI_OFFLOAD_DIR)", file=sys.stderr)
        return 2
    if a.wait is not None:
        if cmd:
            print("offload: give either a command to run or --wait, not both", file=sys.stderr)
            return 2
        running = _running(d)
        rid = a.wait
        if not rid:
            if len(running) != 1:
                print("offload: " + ("no request is running" if not running else
                                     f"{len(running)} requests are running ({', '.join(running)}): wait for one with "
                                     "--wait ID"), file=sys.stderr)
                return 2
            rid = running[0]
        base = os.path.join(d, rid)
        if not _RID.fullmatch(rid) or not (os.path.exists(base + ".req.json") or os.path.exists(base + ".done")):
            print(f"offload: no request {rid} (a request ID is the 12 characters the helper prints first, not the "
                  f"shell's task ID); running now: {', '.join(running) or 'none'}", file=sys.stderr)
            return 2
        code = _follow(base + ".out", base + ".done", time.monotonic() + max(1, a.for_s))
        if code is None:
            print(f"offload: the run is still going; wait again with --wait {rid}", file=sys.stderr)
            return STILL_RUNNING
        return code
    if not cmd:
        print("offload: give a command after --", file=sys.stderr)
        return 2
    rid = uuid.uuid4().hex[:12]
    req = {"argv": cmd, "cwd": os.path.relpath(os.path.realpath(os.getcwd()), os.path.realpath(ws)),
           "mem_gb": a.mem, "secs": a.secs}
    os.makedirs(d, exist_ok=True)
    # held until this process ends (the descriptor is never closed): a run whose helper is gone is stopped
    lock = os.open(os.path.join(d, rid + ".lock"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    fcntl.flock(lock, fcntl.LOCK_EX)
    tmp = os.path.join(d, rid + ".req.tmp")
    with open(tmp, "w") as fh:
        json.dump(req, fh)
    os.replace(tmp, os.path.join(d, rid + ".req.json"))
    out_path, done_path = os.path.join(d, rid + ".out"), os.path.join(d, rid + ".done")
    print(f"offload: request {rid}; its output is also in {out_path}; wait for it from another command with "
          f"--wait {rid}", file=sys.stderr, flush=True)
    return _follow(out_path, done_path)


# -- the orchestrator's half -------------------------------------------------------------------------------------------

def config(camp) -> dict | None:
    off = (camp.config.get("live") or {}).get("offload")
    return off if isinstance(off, dict) and off.get("cmd") else None


def _whole(v, what: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(f"{what} must be a whole number of at least 1")
    return v


def check(cfg: dict, root=None) -> None:
    """Refuse bad live.offload and live.worker_mem_gb settings; `drsi run` and `drsi baseline` check them before they
    start anything. `root` is the campaign's directory, when known. Unset, empty, false and 0 are off."""
    live = cfg.get("live") or {}
    cap = live.get("worker_mem_gb")
    if cap not in (None, 0) and (isinstance(cap, bool) or not isinstance(cap, (int, float)) or not cap > 0):
        raise ValueError("live.worker_mem_gb must be a positive number of GB (or 0 for no cap)")
    off = live.get("offload")
    if not off:
        return
    if not isinstance(off, dict) or not isinstance(off.get("cmd"), str) or not off["cmd"]:
        raise ValueError("live.offload needs cmd, the path of the command that runs a request")
    cmd = off["cmd"]
    if not os.path.isabs(cmd):  # it runs with a checkout as its directory: a relative path is the worker's
        raise ValueError("live.offload.cmd must be an absolute path")
    if not (os.path.isfile(cmd) and os.access(cmd, os.X_OK)):
        raise ValueError(f"live.offload.cmd is not an executable file: {cmd} (it takes no arguments of its own)")
    if root is not None and os.path.realpath(cmd).startswith(os.path.realpath(os.path.join(root, "work")) + os.sep):
        raise ValueError("live.offload.cmd is inside the workers' checkouts")
    for k in ("mem_gb", "secs", "max_mem_gb", "max_secs"):
        if k in off:
            _whole(off[k], f"live.offload.{k}")
    for d, m in (("mem_gb", "max_mem_gb"), ("secs", "max_secs")):
        if d in off and m in off and off[d] > off[m]:
            raise ValueError(f"live.offload.{d} is over live.offload.{m}")


def _limits(off: dict) -> dict:
    """{key: (default, maximum)} for mem_gb and secs: a maximum set alone lowers the built-in default to it."""
    out = {}
    for key, builtin in (("mem_gb", 8), ("secs", 1800)):
        top = off.get(f"max_{key}")
        default = off.get(key, builtin if top is None else min(builtin, top))
        out[key] = (int(default), int(top if top is not None else default))
    return out


def request_dir(camp, nid: str):
    return camp.root / "work" / "_proposals" / nid / "offload"


def worker_env(camp, workspace) -> dict:
    """The environment a worker's shell needs for the helper (nothing when the campaign has no offload)."""
    from pathlib import Path
    if config(camp) is None:
        return {}
    return {"DRSI_OFFLOAD_DIR": str(request_dir(camp, Path(workspace).name)), "DRSI_OFFLOAD_WORKSPACE": str(workspace)}


def brief(camp) -> list[str]:
    """What a worker is told (nothing when the campaign has no offload)."""
    off = config(camp)
    if off is None:
        return []
    lim = _limits(off)
    (mem, max_mem), (secs, max_secs) = lim["mem_gb"], lim["secs"]
    lines = ["HEAVY COMPUTATION",
             "Run anything more than a quick check on the campaign's compute host, not on this machine: from your "
             f"checkout, `{sys.executable} {os.path.abspath(__file__)} [--mem GB] [--secs S] -- <command> [args]` "
             f"(defaults {mem} GB and {secs} s, at most {max_mem} GB and {max_secs} s). It ships your checkout there, runs the command in the same "
             "directory, streams its output back and exits with its status; files it writes there do not come back. "
             "Your session has no later turn: when you stop, it ends, and every run still going is stopped, so wait "
             "for a run you need before you finish. A command that runs past your shell's time limit (2 minutes "
             "unless you ask for more, 10 at most) is moved to the background or stopped: start a longer run in the "
             "background, then wait for it in the foreground, giving the Bash tool its 10-minute timeout "
             f"(600000 ms): `{sys.executable} {os.path.abspath(__file__)} --wait` with no ID attaches to your one "
             "running request (with several, give `--wait ID`, the 12-character request ID the helper prints first, "
             "not the shell's task ID), prints the run's output and exits with its status when it ends, or after "
             f"{WAIT_S // 60} minutes prints `offload: the run is still going` (then wait again). A run also stops "
             "when the helper that started it ends."]
    if off.get("note"):
        lines.append(str(off["note"]))
    return lines + [""]


MAX_REQUEST = 1 << 20  # bytes
MAX_ENTRIES = 4096  # a request directory holding more is flooded: no longer served
MAX_OUT = 64 * 1024 ** 2  # bytes of a run's output; a run that writes more is stopped
MAX_DEPTH = 32  # levels an earlier call's directory is removed to; anything deeper stays aside
MAX_REMOVALS = 100_000  # entries removed per call
_RID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK


def _printable(text: str) -> str:
    return "".join(c if c.isprintable() else "?" for c in text)


def _validate(req, off: dict) -> tuple[dict | None, str]:
    """The request as it will run, or (None, why). A refusal never repeats a value from the request."""
    if not isinstance(req, dict):
        return None, "the request must be a JSON object in a plain file"
    argv = req.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv) or len(argv) > 256 \
            or sum(len(x) for x in argv) > 65536:
        return None, "the command must be a non-empty list of strings"
    cwd = req.get("cwd", ".")
    norm = os.path.normpath(cwd) if isinstance(cwd, str) else ""
    if not isinstance(cwd, str) or os.path.isabs(cwd) or norm == ".." or norm.startswith("../") or "\0" in cwd:
        return None, "the directory must be inside the checkout"
    if norm.startswith("-"):  # it reaches the configured command before its "--"
        return None, "the directory must not start with -"
    limits = {}
    for key, (default, top) in _limits(off).items():
        v = req.get(key)
        v = default if v is None else v
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= top:
            return None, f"{key} must be a whole number from 1 to {top}"
        limits[key] = v
    return {"argv": argv, "cwd": norm, **limits}, ""


def _remove_bounded(parent_fd: int, name: str, budget: list, depth: int = 0) -> bool:
    """Remove parent/name without following a link, through handles, at most MAX_DEPTH levels deep and budget[0]
    entries in all; False (leaving the rest) when a bound is reached."""
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    if budget[0] <= 0:
        return False
    budget[0] -= 1
    if not stat.S_ISDIR(st.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return True
    if depth >= MAX_DEPTH:
        return False
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    try:
        done = True
        while True:  # an entry at a time: a directory of any width is never listed whole
            with os.scandir(fd) as it:
                entry = next((e.name for e in it), None)
            if entry is None:
                break
            if not _remove_bounded(fd, entry, budget, depth + 1):
                done = False
                break
    finally:
        os.close(fd)
    if done:
        os.rmdir(name, dir_fd=parent_fd)
    return done


def _open_fresh_dir(base, parts) -> int:
    """A handle on a new, empty base/part/.../last, each part opened without following a link: a link or anything but
    a directory in a part's place is removed (never what it points to) and the directory made anew. Whatever stood at
    the last part (an earlier call's requests, a link, a tree of any depth) is moved aside by a rename, never walked,
    so requests an earlier call left are not run, then removed within bounds (a deeper or larger tree stays aside).
    Every later read and write goes through the handle, so a directory swapped out while the call runs changes
    nothing the orchestrator touches."""
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        *path, last = parts
        for part in path:
            try:
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(part, 0o755, dir_fd=fd)
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError:  # a link (ELOOP) or a file (ENOTDIR) in its place
                os.unlink(part, dir_fd=fd)
                os.mkdir(part, 0o755, dir_fd=fd)
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nfd
        for _ in range(3):
            try:  # a rename moves a link itself, not what it points to
                os.rename(last, f".{last}-{uuid.uuid4().hex[:12]}", src_dir_fd=fd, dst_dir_fd=fd)
            except FileNotFoundError:
                pass
            try:
                os.mkdir(last, 0o755, dir_fd=fd)
                break
            except FileExistsError:  # something took its place in between: move that aside too
                continue
        else:
            raise OSError(f"could not make a fresh {last} directory")
        budget = [MAX_REMOVALS]  # earlier calls' directories: removed within the bounds, the rest stays aside
        with os.scandir(fd) as it:
            aside = [e.name for e in it if e.name.startswith(f".{last}-")][:64]
        for name in aside:
            _remove_bounded(fd, name, budget)
        nfd = os.open(last, _DIR_FLAGS, dir_fd=fd)
        os.close(fd)
        return nfd
    except BaseException:
        os.close(fd)
        raise


class _Server:
    """Serves one worker call's requests, one at a time. The directory is the worker's to write, and this runs outside
    its sandbox: every file is reached through the directory's handle, never by following a link; a request is read
    only from a plain file with one link; the files written here are created anew (the output file) or put in place
    by a rename (the status file), so nothing planted in the directory is written through. A request runs only while
    the helper that made it holds its lock, and a directory flooded with entries is no longer served."""

    def __init__(self, dfd: int, workspace, off: dict, log):
        import threading
        self.dfd, self.workspace, self.off, self.log = dfd, workspace, off, log or (lambda m: None)
        self.seen: set[str] = set()
        self.flooded = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _exists(self, name: str) -> bool:
        try:
            os.stat(name, dir_fd=self.dfd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.flooded:
                try:
                    names = []
                    with os.scandir(self.dfd) as it:  # stops past the limit: a flood is never listed whole
                        for e in it:
                            names.append(e.name)
                            if len(names) > MAX_ENTRIES:
                                break
                    if len(names) > MAX_ENTRIES:
                        self.flooded = True
                        self.log(f"offload: the request directory holds more than {MAX_ENTRIES} entries: flooded, "
                                 "no longer served in this call")
                        names = []
                    for name in sorted(n for n in names if n.endswith(".req.json")):
                        rid = name[:-len(".req.json")]
                        if rid in self.seen or not _RID.fullmatch(rid) or self._exists(f"{rid}.done"):
                            continue
                        self.seen.add(rid)
                        try:
                            self._handle(rid, name)
                        except Exception as e:  # noqa: BLE001 - end the request rather than leave it waiting
                            self.log(f"offload {rid}: {type(e).__name__}: {_printable(str(e))}")
                            if not self._exists(f"{rid}.done"):
                                self._finish(rid, 1, f"could not run the request: {type(e).__name__}: {e}")
                        if self.stop_event.is_set():
                            break
                except Exception as e:  # noqa: BLE001 - a bad request never stops the serving
                    self.log(f"offload: {type(e).__name__}: {_printable(str(e))}")
            self.stop_event.wait(POLL_S)

    def _read(self, name: str):
        """The request in a plain file with one link and at most MAX_REQUEST bytes, opened without following a link or
        waiting on a pipe; None for anything else."""
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.dfd)
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > MAX_REQUEST:
                return None
            chunks, n = [], 0
            while n <= MAX_REQUEST:
                b = os.read(fd, 65536)
                if not b:
                    break
                chunks.append(b)
                n += len(b)
            return json.loads(b"".join(chunks)) if n <= MAX_REQUEST else None
        except (OSError, ValueError):
            return None
        finally:
            os.close(fd)

    def _helper_alive(self, rid: str) -> bool:
        """Whether the helper that made the request still holds its lock (a lock is let go when its holder ends)."""
        try:
            fd = os.open(f"{rid}.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.dfd)
        except OSError:
            return False
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:  # a link to someone else's locked file is not it
                return False
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except OSError:
            return False
        finally:
            os.close(fd)

    def _finish(self, rid: str, code: int, message: str = "", out_fd: int | None = None) -> None:
        told = False
        if message and out_fd is not None:
            os.write(out_fd, f"offload: {message}\n".encode())
            told = True
        tmp = f".{rid}.{uuid.uuid4().hex}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=self.dfd)
        try:
            os.write(fd, json.dumps({"exit": code, "error": message, "told": told}).encode())
        finally:
            os.close(fd)
        os.rename(tmp, f"{rid}.done", src_dir_fd=self.dfd, dst_dir_fd=self.dfd)  # replaces whatever was there

    def _handle(self, rid: str, name: str) -> None:
        import subprocess
        from .agent import Descendants, _killpg, refuse_if_stopping, register_child, unregister_child
        try:
            out_fd = os.open(f"{rid}.out", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_APPEND, 0o644,
                             dir_fd=self.dfd)
        except OSError:
            self._finish(rid, 2, "refused: its output file is already there")
            return
        try:
            ok, why = _validate(self._read(name), self.off)
            if ok is None:
                self._finish(rid, 2, f"refused: {why}", out_fd)
                return
            if not self._helper_alive(rid):
                self._finish(rid, 2, "refused: no helper holds the request's lock (make requests with the helper)",
                             out_fd)
                return
            if self.stop_event.is_set():
                self._finish(rid, 143, "the worker's call ended before the run started", out_fd)
                return
            cmd = os.path.realpath(str(self.off["cmd"]))
            checkouts = os.path.realpath(os.path.dirname(os.path.abspath(str(self.workspace))))
            if not os.path.isabs(str(self.off["cmd"])) or cmd.startswith(checkouts + os.sep):
                self._finish(rid, 2, "refused: the configured command is not an absolute path outside the workers' "
                                     "checkouts", out_fd)
                return
            argv = [cmd, str(self.workspace), str(ok["mem_gb"]), str(ok["secs"]), ok["cwd"], "--", *ok["argv"]]
            try:
                refuse_if_stopping()
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out_fd, stderr=subprocess.STDOUT,
                                        cwd=str(self.workspace), start_new_session=True)
            except Exception as e:  # noqa: BLE001 - a missing command, say, or a run that is stopping
                self.log(f"offload {rid}: {type(e).__name__}: {_printable(str(e))}")
                self._finish(rid, 1, f"could not run the request: {type(e).__name__}: {e}", out_fd)
                return
            register_child(proc)
            tracked = Descendants(proc.pid, interval=0.5)  # its children that leave its group end with it too
            self.log(f"offload {rid}: {_printable(' '.join(ok['argv']))[:120]} ({ok['mem_gb']} GB, {ok['secs']} s)")
            deadline = time.monotonic() + ok["secs"] + QUEUE_S
            code, message = None, ""
            try:
                while code is None:
                    try:
                        code = proc.wait(timeout=POLL_S)
                    except subprocess.TimeoutExpired:
                        if self.stop_event.is_set():
                            code, message = 143, "the worker's call ended; the run was stopped"
                        elif not self._helper_alive(rid):
                            code, message = 143, "the helper that asked for the run is gone; the run was stopped"
                        elif os.fstat(out_fd).st_size > MAX_OUT:
                            code, message = 1, f"the run's output passed {MAX_OUT // 1024 ** 2} MB; the run was stopped"
                        elif time.monotonic() > deadline:
                            code, message = 124, "the run took longer than its limit and the queue allowance"
            finally:
                _killpg(proc)  # also when the command has exited: what it left in its group goes with it
                tracked.kill([])
                proc.wait()
                unregister_child(proc)
            self._finish(rid, code, message, out_fd)
        finally:
            os.close(out_fd)


class serve:
    """Serves a worker's requests while its call runs; a no-op for a campaign without live.offload."""

    def __init__(self, camp, workspace, nid: str, log=None):
        from pathlib import Path
        self.off = config(camp)
        self.server = None
        if self.off is not None:
            base = camp.root / "work" / "_proposals"
            base.mkdir(parents=True, exist_ok=True)
            self.server = _Server(_open_fresh_dir(base, (nid, "offload")), Path(workspace), self.off, log)

    def __enter__(self):
        if self.server is not None:
            self.server.thread.start()
        return self.server

    def __exit__(self, *exc):
        if self.server is not None:
            self.server.stop_event.set()
            self.server.thread.join()
            os.close(self.server.dfd)
        return False


if __name__ == "__main__":
    sys.exit(_client(sys.argv[1:]))
