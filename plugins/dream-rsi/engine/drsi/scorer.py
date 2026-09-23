"""Run a campaign's scorer on a workspace and normalise its verdict.

Contract: the scorer runs with cwd = a clean checkout of the attempt and DRSI_WORKSPACE
set. It must exit 0 and print one JSON object as its LAST non-empty stdout line:
  {"score": number|null, "valid": bool, "fail_class": "ok"|..., "gates": {...}, "summary": str, "error": str}
A non-zero exit, a last line that is not that object, or a non-finite or boolean score is an
eval_error. Scorers should run the candidate code in a child process and print the final line
themselves, so candidate output can never be the last line. The process group is killed when
the scorer finishes or times out, so background children cannot linger or hold it open.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from .agent import Descendants, register_child, unregister_child


def _fail(fail_class: str, error: str, stdout: str = "") -> dict:
    return {"score": None, "raw_score": None, "valid": False, "fail_class": fail_class, "gates": {},
            "summary": "", "error": error, "stdout_tail": stdout[-2000:]}


def _killpg(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


SANDBOX_EXEC = "/usr/bin/sandbox-exec"


DEV_WRITABLE = ('(literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper") '
                '(literal "/dev/stdout") (literal "/dev/stderr") (regex #"^/dev/fd/[0-9]+$")')


def _sandbox_argv(cmd: str, workspace: Path, allow_write: list[str], network: bool = False) -> list[str]:
    """macOS seatbelt profile: the scorer (and the candidate code it runs) may read anything but write only
    to its checkout, its own private temp dir (in allow_write, exported as TMPDIR), the paths the campaign
    lists in scorer.allow_write and the null and descriptor devices (not terminals). Shared temp
    directories are not writable, and there is no network unless scorer.network is on."""
    def q(p):
        return '"' + str(Path(p).expanduser().resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'
    allowed = [workspace] + list(allow_write)
    profile = ("(version 1)(allow default)(deny file-write*)"
               "(allow file-write* " + " ".join(f"(subpath {q(p)})" for p in allowed) + " " + DEV_WRITABLE + ")"
               + "(deny lsopen)" + ("" if network else "(deny network*)"))
    return [SANDBOX_EXEC, "-p", profile, "bash", "-c", cmd]


def run_scorer(cmd: str, workspace, timeout: int, direction: str = "max", env: dict | None = None,
               sandbox="auto", allow_write: list[str] | None = None, network: bool = False) -> dict:
    """sandbox: True, False or "auto" (use macOS sandbox-exec when present)."""
    workspace = Path(workspace)
    full_env = dict(os.environ, DRSI_WORKSPACE=str(workspace), **{k: str(v) for k, v in (env or {}).items()})
    use_sandbox = (sandbox is True) or (sandbox == "auto" and Path(SANDBOX_EXEC).exists())
    private_tmp = tempfile.mkdtemp(prefix="drsi-score-")
    full_env.update(TMPDIR=private_tmp, TMP=private_tmp, TEMP=private_tmp)
    argv = (_sandbox_argv(cmd, workspace.resolve(), list(allow_write or []) + [private_tmp], network)
            if use_sandbox else ["bash", "-c", cmd])
    try:
        return _run(argv, workspace, full_env, timeout, direction, sweep=[workspace, Path(private_tmp)])
    finally:
        import shutil
        shutil.rmtree(private_tmp, ignore_errors=True)


def _run(argv, workspace, full_env, timeout, direction, sweep: list[Path]) -> dict:
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, cwd=workspace, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=err, env=full_env, start_new_session=True)
        register_child(proc)
        tracked = Descendants(proc.pid)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _killpg(proc)
            proc.wait()
            return _fail("timeout", f"scorer timed out after {timeout}s")
        finally:
            _killpg(proc)
            tracked.kill(sweep)
            unregister_child(proc)
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode("utf-8", "replace")
        stderr = err.read().decode("utf-8", "replace")
    if rc != 0:
        return _fail("eval_error", f"scorer exited {rc}: {stderr[-400:]}", stdout)
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    try:
        parsed = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return _fail("eval_error", "the scorer's last stdout line is not a JSON object", stdout)
    valid = parsed.get("valid", parsed.get("score") is not None)
    if not isinstance(valid, bool):
        return _fail("eval_error", "scorer 'valid' must be a boolean", stdout)
    raw = parsed.get("score")
    if valid and (isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw)):
        return _fail("eval_error", "scorer said valid but gave no finite numeric score", stdout)
    score = None if not valid else (float(raw) if direction == "max" else -float(raw))
    return {"score": score, "raw_score": raw if valid else None, "valid": valid,
            "fail_class": parsed.get("fail_class") or ("ok" if valid else "eval_error"),
            "gates": parsed.get("gates") or {}, "summary": str(parsed.get("summary") or ""),
            "error": str(parsed.get("error") or ""), "stdout_tail": stdout[-2000:]}
