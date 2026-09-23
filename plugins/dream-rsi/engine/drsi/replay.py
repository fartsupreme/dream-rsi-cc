"""Dreaming: score a policy by replaying it over frozen discovery trees.

No agent or scorer runs here; every outcome is read from the record, so one
evaluation costs seconds. Each policy runs twice in isolated subprocesses under
different hash seeds; any difference marks the policy unsound.

Ranking (B.2's Pareto AUC): every prefix of a run is an operating point
(stopping earlier is always available), so each run contributes its anytime
curve of (work so far, attainment so far). For each swept beta those curves are
averaged over the worlds with signal; the frontier over the per-beta mean curves
is integrated over work in [0, 1]. Reaching good attempts sooner scores higher
even when every run explores the whole world, and one beta applies to all worlds.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .guard import check_policy_source
from .question import IllegalBatch, ReplayQuestion
from .reward import attainment, eq1_value, mean_curve_auc, parallel_penalty

ENGINE_DIR = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "replay_runner.py"


def _run_once(job: dict, workdir: Path, seed: int, timeout: int) -> dict:
    out = workdir / f"result-{seed}.json"
    job_path = workdir / f"job-{seed}.json"
    job_path.write_text(json.dumps(job | {"out": str(out)}))
    env = {"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"}
    try:
        proc = subprocess.run([sys.executable, "-s", "-P", str(RUNNER), str(ENGINE_DIR), str(job_path)],
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    if not out.exists():
        return {"ok": False, "error": f"runner wrote no result (exit {proc.returncode}): {proc.stderr[-400:]}"}
    try:
        return json.loads(out.read_text())
    except json.JSONDecodeError:
        return {"ok": False, "error": "runner result is not JSON"}


def reachable(world: dict) -> list[dict]:
    """Nodes replay can ever reveal: each root and its chain of first recorded children (a leaf reveals
    its first child, after which its parent is no longer a leaf). Later siblings are unreachable, so they
    must not set the target a policy is measured against or the work it is normalised by."""
    kids: dict = {}
    for n in world["nodes"]:
        kids.setdefault(n.get("parent"), []).append(n)
    out = []
    for root in kids.get(None, []):
        cur = root
        while cur is not None:
            out.append(cur)
            nxt = kids.get(cur["id"], [])
            cur = nxt[0] if nxt else None
    return out


def world_best(world: dict) -> float | None:
    vals = [n["score"] for n in reachable(world)
            if n.get("valid", n.get("score") is not None) and n.get("score") is not None]
    return max(vals) if vals else None


def _aggregate(raw: dict, worlds: list[dict], W: int, lam: float, beta1: float, beta2: float) -> dict:
    """Ranking: per swept beta, average the anytime curves over the worlds that carry signal (at least one
    valid reachable score); integrate the frontier over those per-beta mean curves; subtract lambda times
    the parallel penalty of the runs on those worlds. Worlds without signal cannot favour any policy."""
    swept_keys = set(raw.get("swept", raw["runs"].keys()))
    signal = [i for i, w in enumerate(worlds) if world_best(w) is not None]
    per_beta, curves = {}, {}
    for beta, rows in raw["runs"].items():
        atts, works, eq1s, sizes, pts = [], [], [], [], []
        for wi in signal:
            world, row = worlds[wi], rows[wi]
            size = max(1, len(reachable(world)))
            best_w = world_best(world)
            base = world.get("baseline", 0.0)
            a = attainment(row["best"], base, best_w)
            atts.append(a)
            works.append(row["probes"] / size)
            eq1s.append(eq1_value(a, row["probes"], row["rounds"], beta1, beta2))
            sizes += row["batch_sizes"]
            pts.append([(p / size, attainment(b, base, best_w)) for p, b in row.get("curve", [])])
        n = max(1, len(signal))
        per_beta[beta] = {"attainment": sum(atts) / n, "work": sum(works) / n, "eq1": sum(eq1s) / n,
                          "batch_sizes": sizes, "per_world_attainment": atts}
        if beta in swept_keys:
            curves[beta] = pts
    default = str(float(raw["default_beta"]))
    swept = {b: r for b, r in per_beta.items() if b in swept_keys}
    auc = mean_curve_auc(curves) if signal else 0.0
    pens = [parallel_penalty(r["batch_sizes"], W) for r in swept.values()]
    pen = sum(pens) / len(pens) if pens else 1.0

    def public(r):
        return {k: v for k, v in r.items() if k != "batch_sizes"} | {
            "mean_batch": sum(r["batch_sizes"]) / len(r["batch_sizes"]) if r["batch_sizes"] else 0.0}
    return {"ok": True, "reward": auc - lam * pen, "auc": auc, "parallel_penalty": pen,
            "signal_worlds": len(signal), "default_beta": raw["default_beta"],
            "eq1_default_beta": per_beta[default]["eq1"], "default": public(per_beta[default]),
            "per_beta": {b: public(r) for b, r in swept.items()}}


def evaluate_policy(policy_path, worlds: list[dict], W: int, betas, budget, lam: float, beta1: float,
                    beta2: float, timeout: int = 120) -> dict:
    policy_path = Path(policy_path).resolve()
    problems = check_policy_source(policy_path.read_text())
    if problems:
        return {"ok": False, "stage": "guard", "problems": problems, "error": "; ".join(problems)}
    with tempfile.TemporaryDirectory() as d:
        job = {"policy": str(policy_path), "worlds": worlds, "W": W,
               "betas": [float(b) for b in betas], "budget": budget}
        first = _run_once(job, Path(d), 1, timeout)
        if not first.get("ok"):
            return {"ok": False, "stage": "run", "error": first.get("error", "unknown failure")}
        second = _run_once(job, Path(d), 2, timeout)
        if not second.get("ok"):
            return {"ok": False, "stage": "run", "error": second.get("error", "unknown failure")}
    if first != second:
        return {"ok": False, "stage": "determinism",
                "error": "the policy produced different traces on two identical replays"}
    try:
        measured = _replay_traces(first, worlds, W, budget)
    except IllegalBatch as e:
        return {"ok": False, "stage": "run", "error": f"trace does not replay: {e}"}
    measured["swept"] = [str(float(b)) for b in betas]
    rep = _aggregate(measured, worlds, W, lam, beta1, beta2)
    rep["traces_replayed"] = True
    return rep


def _replay_traces(raw: dict, worlds: list[dict], W: int, budget) -> dict:
    """Recompute every metric in this (trusted) process by replaying the batches each run requested."""
    runs = {}
    for beta, rows in raw["runs"].items():
        out = []
        for world, row in zip(worlds, rows):
            q = ReplayQuestion(world, W, max_probes=budget)
            for batch in row["trace"]:
                if not isinstance(batch, list) or any(type(c) is not str for c in batch):
                    raise IllegalBatch("malformed trace")
                q.probe_batch(batch)
            out.append({"probes": q._probes, "rounds": q._rounds, "best": q.best_score(),
                        "batch_sizes": list(q._batch_sizes), "curve": [list(p) for p in q._curve]})
        runs[beta] = out
    return {"ok": True, "default_beta": raw["default_beta"], "runs": runs}
