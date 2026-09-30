"""Heavy computation off the machine that runs the workers (live.offload).

A worker's shell is sandboxed on the machine that runs the loop: it can write only its workspace and proposal
directory and has no network. With live.offload set, a worker asks for an experiment to run elsewhere by running this
file as a script from its checkout:

    python3 offload.py [--mem GB] [--secs S] -- COMMAND [ARGS...]

The script writes a request (the command as an argument list, the directory relative to the checkout, the limits)
into the worker's proposal directory, prints the output as it arrives, and exits with the command's status. The
orchestrator, which runs outside the sandbox, serves the requests while the worker's call runs (serve()): it checks
each one, runs the configured command (live.offload.cmd) as

    CMD CHECKOUT MEM_GB SECONDS DIR -- COMMAND [ARGS...]

with no shell, appends its output to the file the script prints, and records its exit status. What CMD does with it
is the campaign's (ship the checkout to a compute host and run the command there in a sandbox with its own limits,
say). The worker never gets a way out of its sandbox, and the orchestrator runs nothing but CMD. A worker's runs end
with its call.

Config (live.offload): cmd (required, an executable path), mem_gb and secs (defaults per request), max_mem_gb and
max_secs (the most a request may ask for), note (one line for the worker: what the host has).

The client half runs under whatever python3 the worker has and uses the standard library only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import time
import uuid

POLL_S = 0.3
QUEUE_S = 3900  # time allowed on top of a request's own limit (shipping it, waiting for a free slot there)


def _client(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="offload", description="run a command on the campaign's compute host")
    ap.add_argument("--mem", type=int, default=None, help="memory limit, GB")
    ap.add_argument("--secs", type=int, default=None, help="time limit, seconds")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    d, ws = os.environ.get("DRSI_OFFLOAD_DIR"), os.environ.get("DRSI_OFFLOAD_WORKSPACE")
    if not d or not ws:
        print("offload: not configured for this campaign (no DRSI_OFFLOAD_DIR)", file=sys.stderr)
        return 2
    if not cmd:
        print("offload: give a command after --", file=sys.stderr)
        return 2
    rid = uuid.uuid4().hex[:12]
    req = {"argv": cmd, "cwd": os.path.relpath(os.path.realpath(os.getcwd()), os.path.realpath(ws)),
           "mem_gb": a.mem, "secs": a.secs}
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, rid + ".req.tmp")
    with open(tmp, "w") as fh:
        json.dump(req, fh)
    os.replace(tmp, os.path.join(d, rid + ".req.json"))
    out_path, done_path = os.path.join(d, rid + ".out"), os.path.join(d, rid + ".done")
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
        time.sleep(POLL_S)


# -- the orchestrator's half -------------------------------------------------------------------------------------------

def config(camp) -> dict | None:
    off = (camp.config.get("live") or {}).get("offload")
    return off if isinstance(off, dict) and off.get("cmd") else None


def request_dir(camp, nid: str):
    return camp.root / "work" / "_proposals" / nid / "offload"


def worker_env(camp, workspace) -> dict:
    """The environment a worker's shell needs for the client (nothing when the campaign has no offload)."""
    from pathlib import Path
    if config(camp) is None:
        return {}
    return {"DRSI_OFFLOAD_DIR": str(request_dir(camp, Path(workspace).name)), "DRSI_OFFLOAD_WORKSPACE": str(workspace)}


def brief(camp) -> list[str]:
    """What a worker is told (nothing when the campaign has no offload)."""
    off = config(camp)
    if off is None:
        return []
    mem, secs = int(off.get("mem_gb", 8)), int(off.get("secs", 1800))
    lines = ["HEAVY COMPUTATION",
             "Run anything more than a quick check on the campaign's compute host, not on this machine: from your "
             f"checkout, `{sys.executable} {os.path.abspath(__file__)} [--mem GB] [--secs S] -- <command> [args]` "
             f"(defaults {mem} GB and {secs} s, at most {int(off.get('max_mem_gb', mem))} GB and "
             f"{int(off.get('max_secs', secs))} s). It ships your checkout there, runs the command in the same "
             "directory, streams its output back and exits with its status; files it writes there do not come back."]
    if off.get("note"):
        lines.append(str(off["note"]))
    return lines + [""]


MAX_REQUEST = 1 << 20  # bytes
_RID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK


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
    limits = {}
    for key, default_key, max_key in (("mem_gb", "mem_gb", "max_mem_gb"), ("secs", "secs", "max_secs")):
        default = int(off.get(default_key, 8 if key == "mem_gb" else 1800))
        top = int(off.get(max_key, default))
        v = req.get(key)
        v = default if v is None else v
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= top:
            return None, f"{key} must be a whole number from 1 to {top}"
        limits[key] = v
    return {"argv": argv, "cwd": norm, **limits}, ""


def _open_fresh_dir(base, parts) -> int:
    """A handle on base/part/part/..., each part opened without following a link: a link or anything but a directory
    in a part's place is removed (never what it points to) and the directory made anew. The last part is emptied, so
    requests an earlier call left are not run. Every later read and write goes through the handle, so a directory
    swapped out while the call runs changes nothing the orchestrator touches."""
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts:
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
        for name in os.listdir(fd):
            if stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                shutil.rmtree(name, dir_fd=fd)
            else:
                os.unlink(name, dir_fd=fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


class _Server:
    """Serves one worker call's requests, one at a time. The directory is the worker's to write, and this runs outside
    its sandbox: every file is reached through the directory's handle, never by following a link; a request is read
    only from a plain file with one link; the files written here are created anew (the output file) or put in place
    by a rename (the status file), so nothing planted in the directory is written through."""

    def __init__(self, dfd: int, workspace, off: dict, log):
        import threading
        self.dfd, self.workspace, self.off, self.log = dfd, workspace, off, log or (lambda m: None)
        self.seen: set[str] = set()
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
            try:
                for name in sorted(n for n in os.listdir(self.dfd) if n.endswith(".req.json")):
                    rid = name[:-len(".req.json")]
                    if rid in self.seen or not _RID.fullmatch(rid) or self._exists(f"{rid}.done"):
                        continue
                    self.seen.add(rid)
                    try:
                        self._handle(rid, name)
                    except Exception as e:  # noqa: BLE001 - the command could not start, say: end the request
                        self.log(f"offload {rid}: {type(e).__name__}: {e}")
                        if not self._exists(f"{rid}.done"):
                            self._finish(rid, 1, f"could not run the request: {type(e).__name__}: {e}")
                    if self.stop_event.is_set():
                        break
            except Exception as e:  # noqa: BLE001 - a bad request never stops the serving
                self.log(f"offload: {type(e).__name__}: {e}")
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
        from .agent import _killpg, refuse_if_stopping, register_child, unregister_child
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
            argv = [str(self.off["cmd"]), str(self.workspace), str(ok["mem_gb"]), str(ok["secs"]), ok["cwd"], "--",
                    *ok["argv"]]
            try:
                refuse_if_stopping()
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out_fd, stderr=subprocess.STDOUT,
                                        cwd=str(self.workspace), start_new_session=True)
            except Exception as e:  # noqa: BLE001 - a missing command, say, or a run that is stopping
                self.log(f"offload {rid}: {type(e).__name__}: {e}")
                self._finish(rid, 1, f"could not run the request: {type(e).__name__}: {e}", out_fd)
                return
            register_child(proc)
            self.log(f"offload {rid}: {' '.join(ok['argv'])[:120]} ({ok['mem_gb']} GB, {ok['secs']} s)")
            deadline = time.monotonic() + ok["secs"] + QUEUE_S
            code, message = None, ""
            try:
                while code is None:
                    try:
                        code = proc.wait(timeout=POLL_S)
                    except subprocess.TimeoutExpired:
                        if self.stop_event.is_set():
                            code, message = 143, "the worker's call ended; the run was stopped"
                        elif time.monotonic() > deadline:
                            code, message = 124, "the run took longer than its limit and the queue allowance"
            finally:
                if proc.poll() is None:
                    _killpg(proc)
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
