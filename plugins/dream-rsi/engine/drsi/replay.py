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
from .question import IllegalBatch, RecordEnd, ReplayQuestion
from .reward import attainment, eq1_value, live_penalty, mean_curve_auc

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


SCORES = ("default", "sweep")
CURVES = ("canonical", "reveal", "batch", "clock")


def _ranking(score: str, penalty: str, curve: str) -> None:
    if score not in SCORES:
        raise ValueError(f"score {score!r} is not one of {SCORES}")
    if penalty != "live":
        raise ValueError(f"penalty {penalty!r} is not supported; the parallel penalty is live fill (\"live\")")
    if curve not in CURVES:
        raise ValueError(f"curve {curve!r} is not one of {CURVES}")


def _aggregate(raw: dict, worlds: list[dict], W: int, lam: float, beta1: float, beta2: float,
               score: str = "default", penalty: str = "live", curve: str = "canonical") -> dict:
    """Ranking: average the anytime curves over the worlds that carry signal (at least one valid reachable
    score), integrate over work, subtract lambda times the parallel penalty. Worlds without signal cannot favour
    any policy.

    score="default" ranks the policy at the beta it runs live (its class default; live_runner instantiates
    OptimalPolicy() with no beta), so behaviour at betas that never run live cannot earn reward; the swept betas
    stay in the report. score="sweep" is the old frontier over the per-beta mean curves.
    penalty="live" is the only penalty: live fill, cells requested per batch out of W (every probe is an attempt),
    with a run's unspent batches empty (reward.live_penalty). The earlier "support" and "realized" penalties paid
    for gains that do not exist live and were removed (round 25).
    curve="canonical" (default) keeps one anytime point per revealed cell but credits the cells of a batch in a
    fixed order (by node id): live they run in parallel, so the order a policy lists them in must not earn
    reward. curve="reveal" is the old policy-ordered point per cell; "batch" is one point per batch (this
    favours serial policies: a wide batch is credited only when all of it is spent); "clock" puts decision
    rounds / K1 on the work axis."""
    _ranking(score, penalty, curve)
    swept_keys = set(raw.get("swept", raw["runs"].keys()))
    default = str(float(raw["default_beta"]))
    scored_keys = {default} if score == "default" else swept_keys
    signal = [i for i, w in enumerate(worlds) if world_best(w) is not None]
    per_beta, curves, pens = {}, {}, {}
    for beta, rows in raw["runs"].items():
        atts, works, eq1s, sizes, pts, requested, unspent = [], [], [], [], [], [], 0
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
            requested += list(row.get("requested", []))
            unspent += int(row.get("unspent", 0))
            if curve == "clock":  # x = live decision rounds used / K1: the wall clock of a live round
                kmax = max(1, int(raw.get("kmax") or 1))
                pts.append([(t / kmax, attainment(b, base, best_w)) for t, b in row.get("curve_clock", [])])
            else:
                key = {"batch": "curve_batch", "canonical": "curve_canonical"}.get(curve, "curve")
                raw_pts = row.get(key, row.get("curve", []))
                pts.append([(p / size, attainment(b, base, best_w)) for p, b in raw_pts])
        n = max(1, len(signal))
        per_beta[beta] = {"attainment": sum(atts) / n, "work": sum(works) / n, "eq1": sum(eq1s) / n,
                          "batch_sizes": sizes, "per_world_attainment": atts}
        if beta in scored_keys:
            curves[beta] = pts
            pens[beta] = live_penalty(requested, W, unspent)
    swept = {b: r for b, r in per_beta.items() if b in swept_keys}
    auc = mean_curve_auc(curves) if signal else 0.0
    pen = sum(pens.values()) / len(pens) if pens else 1.0

    def public(r):
        return {k: v for k, v in r.items() if k != "batch_sizes"} | {
            "mean_batch": sum(r["batch_sizes"]) / len(r["batch_sizes"]) if r["batch_sizes"] else 0.0}
    return {"ok": True, "reward": auc - lam * pen, "auc": auc, "parallel_penalty": pen,
            "signal_worlds": len(signal), "default_beta": raw["default_beta"], "scored_betas": sorted(scored_keys),
            "eq1_default_beta": per_beta[default]["eq1"], "default": public(per_beta[default]),
            "per_beta": {b: public(r) for b, r in swept.items()}}


def resampled_reward(measured: dict, worlds: list[dict], idx: list[int], W: int, lam: float, beta1: float,
                     beta2: float, score: str = "default", penalty: str = "live", curve: str = "canonical") -> float:
    """The reward on a multiset of worlds (indices may repeat), for a paired bootstrap across worlds."""
    sub = {"default_beta": measured["default_beta"], "swept": measured.get("swept"), "kmax": measured.get("kmax"),
           "runs": {b: [rows[i] for i in idx] for b, rows in measured["runs"].items()}}
    return _aggregate(sub, [worlds[i] for i in idx], W, lam, beta1, beta2, score, penalty, curve)["reward"]


def evaluate_policy(policy_path, worlds: list[dict], W: int, betas, budget, lam: float, beta1: float,
                    beta2: float, timeout: int = 120, score: str = "default", penalty: str = "live",
                    curve: str = "canonical") -> dict:
    """budget: probes per world, K1 x W as a live round has; required, since without one the recorded roots cap a
    run, which no live round has."""
    if budget is None:
        raise ValueError("evaluate_policy needs a budget (K1 x W): replay ranks a policy as a live round runs it")
    _ranking(score, penalty, curve)
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
    rep = _aggregate(measured, worlds, W, lam, beta1, beta2, score, penalty, curve)
    rep["traces_replayed"] = True
    rep["measured"] = measured  # per-world metrics, for the deploy rule's paired bootstrap
    rep["traces"] = first  # the raw action traces, for the live-behaviour gate
    return rep


def _replay_traces(raw: dict, worlds: list[dict], W: int, budget) -> dict:
    """Recompute every metric in this (trusted) process by replaying the batches each run requested."""
    runs = {}
    for beta, rows in raw["runs"].items():
        out = []
        for world, row in zip(worlds, rows):
            q = ReplayQuestion(world, W, max_probes=budget)
            curve_batch, curve_clock, curve_canonical, t, best = [], [], [], 0, None
            for bi, batch in enumerate(row["trace"]):
                if not isinstance(batch, list) or any(type(c) is not str for c in batch):
                    raise IllegalBatch("malformed trace")
                seen, done = len(q._order), q._probes
                try:
                    q.probe_batch(batch)
                except RecordEnd:  # the run's last batch reached past the record
                    if bi != len(row["trace"]) - 1:
                        raise IllegalBatch("the trace goes on past the end of the record")
                t += 1  # every batch is a live batch: each probe is an attempt
                curve_clock.append([t, q.best_score()])
                # a batch's attempts run in parallel live: credit them in a fixed order the policy cannot choose
                for k, nid in enumerate(sorted(q._order[seen:])):
                    o = q._obs[nid]
                    if o.valid and o.score is not None and (best is None or o.score > best):
                        best = o.score
                    curve_canonical.append([done + k + 1, best])
                curve_batch.append([q._probes, q.best_score()])
            # a run that stops before its budget, or reaches the end of its record, leaves batches a live round would
            # have run (root slots never run out); each counts as an empty batch
            unspent = -(-(budget - q._probes) // max(1, W)) if budget is not None and q._probes < budget else 0
            out.append({"probes": q._probes, "rounds": q._rounds, "best": q.best_score(), "unspent": unspent,
                        "off_record": q._off,  # probes the record could not answer: where replay is not what happened
                        "requested": list(q._requested), "batch_sizes": list(q._batch_sizes),
                        "curve": [list(p) for p in q._curve], "curve_batch": curve_batch, "curve_clock": curve_clock,
                        "curve_canonical": curve_canonical})
        runs[beta] = out
    return {"ok": True, "default_beta": raw["default_beta"], "runs": runs,
            "kmax": max(1, (budget or W) // max(1, W))}
