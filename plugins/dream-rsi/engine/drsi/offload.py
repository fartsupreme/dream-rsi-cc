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
import sys
import time

POLL_S = 0.3
QUEUE_S = 3900  # time allowed on top of a request's own limit (shipping it, waiting for a free slot there)


def _client(argv: list[str]) -> int:
    import argparse
    import uuid
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
                    return int(json.load(fh).get("exit", 1))
            except (ValueError, OSError):
                return 1
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


def _validate(req, off: dict) -> tuple[dict | None, str]:
    if not isinstance(req, dict):
        return None, "the request is not an object"
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
            return None, f"{key} {v!r} is outside 1..{top}"
        limits[key] = v
    return {"argv": argv, "cwd": norm, **limits}, ""


class _Server:
    def __init__(self, d, workspace, off: dict, log):
        import threading
        self.d, self.workspace, self.off, self.log = d, workspace, off, log or (lambda m: None)
        self.seen: set[str] = set()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                for p in sorted(self.d.glob("*.req.json")):
                    rid = p.name[:-len(".req.json")]
                    if rid in self.seen or (self.d / f"{rid}.done").exists():
                        continue
                    self.seen.add(rid)
                    self._handle(rid, p)
                    if self.stop_event.is_set():
                        break
            except Exception as e:  # noqa: BLE001 - a bad request never stops the serving
                self.log(f"offload: {type(e).__name__}: {e}")
            self.stop_event.wait(POLL_S)

    def _finish(self, rid: str, code: int, message: str = "") -> None:
        if message:
            with open(self.d / f"{rid}.out", "ab") as fh:
                fh.write(f"offload: {message}\n".encode())
        tmp = self.d / f"{rid}.done.tmp"
        tmp.write_text(json.dumps({"exit": code, "error": message}))
        os.replace(tmp, self.d / f"{rid}.done")

    def _handle(self, rid: str, path) -> None:
        import subprocess
        from .agent import _killpg, refuse_if_stopping, register_child, unregister_child
        try:
            req = json.loads(path.read_text())
        except (OSError, ValueError):
            req = None
        ok, why = _validate(req, self.off)
        if ok is None:
            self._finish(rid, 2, f"refused: {why}")
            return
        argv = [str(self.off["cmd"]), str(self.workspace), str(ok["mem_gb"]), str(ok["secs"]), ok["cwd"], "--",
                *ok["argv"]]
        refuse_if_stopping()
        with open(self.d / f"{rid}.out", "ab") as out:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    cwd=str(self.workspace), start_new_session=True)
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
        self._finish(rid, code, message)


class serve:
    """Serves a worker's requests while its call runs; a no-op for a campaign without live.offload."""

    def __init__(self, camp, workspace, nid: str, log=None):
        from pathlib import Path
        self.off = config(camp)
        self.server = None
        if self.off is not None:
            d = request_dir(camp, nid)
            d.mkdir(parents=True, exist_ok=True)
            self.server = _Server(d, Path(workspace), self.off, log)

    def __enter__(self):
        if self.server is not None:
            self.server.thread.start()
        return self.server

    def __exit__(self, *exc):
        if self.server is not None:
            self.server.stop_event.set()
            self.server.thread.join()
        return False


if __name__ == "__main__":
    sys.exit(_client(sys.argv[1:]))
