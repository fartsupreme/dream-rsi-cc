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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import _killpg, refuse_if_stopping, register_child, unregister_child
from .guard import check_policy_source
from .question import POLICY_HASH_SEED, IllegalBatch, RecordEnd, ReplayQuestion
from .reward import attainment, eq1_value, live_penalty, mean_curve_auc

ENGINE_DIR = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "replay_runner.py"


def _run_once(job: dict, workdir: Path, seed: int, timeout: float, deadline: float | None = None) -> dict:
    """The policy's trace on every (beta, world) of the job, each run in a process of its own (a live round runs one
    policy per process, so a run must not see what an earlier run left in its process). The default beta is added to
    the job's betas; at it the policy is built with no argument, as live. Each run has the timeout (the work of a job
    grows with its worlds, and one deadline for all of them would fail a policy only for the size of the pool); a
    deadline (time.monotonic()), when given, bounds them all as well. The first failure ends the job, and the
    processes run in sessions of their own, registered with the run's children, so an interrupt or `drsi stop` kills
    every one."""
    base = {"policy": job["policy"], "W": job["W"], "budget": job["budget"]}
    running = _Runs()
    results: dict = {}
    pool = None
    try:
        info = _spawn(base | {"world": None}, workdir, f"{seed}-describe", seed, timeout, deadline, running)
        if not info.get("ok"):
            return info
        default = info["default_beta"]
        betas = list(job["betas"]) + ([default] if default not in job["betas"] else [])
        tasks = [(b, i) for b in betas for i in range(len(job["worlds"]))]
        pool = ThreadPoolExecutor(max_workers=max(1, min(4, len(tasks))))
        futures = {pool.submit(_spawn, base | {"world": job["worlds"][i], "beta": None if b == default else b},
                               workdir, f"{seed}-{b}-{i}", seed, timeout, deadline, running): (b, i)
                   for b, i in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            if not r.get("ok"):
                return r
            results[futures[fut]] = r
    finally:  # a failure or an interrupt: nothing queued starts, and what runs is killed
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        running.halt()
    runs: dict = {}
    for b, i in tasks:
        runs.setdefault(str(float(b)), []).append({"trace": results[(b, i)]["trace"]})
    return {"ok": True, "default_beta": default, "runs": runs}


class _Runs:
    """The processes of one evaluation. Once it halts (a failure or an interrupt) none starts, even in a pool thread
    that was about to start one, and each one started is killed."""

    def __init__(self):
        self._lock = threading.Lock()
        self._procs: set = set()
        self._halted = False

    def start(self, argv: list[str], **kw):
        with self._lock:
            if self._halted:
                return None
            refuse_if_stopping()
            proc = subprocess.Popen(argv, start_new_session=True, **kw)
            self._procs.add(proc)
        register_child(proc)
        return proc

    def done(self, proc) -> None:
        with self._lock:
            self._procs.discard(proc)
        unregister_child(proc)

    def halt(self) -> None:
        with self._lock:
            self._halted = True
            procs = list(self._procs)
        for proc in procs:
            _killpg(proc)


def _spawn(job: dict, workdir: Path, tag: str, seed: int, timeout: float, deadline: float | None,
           running: _Runs) -> dict:
    limit = timeout if deadline is None else min(timeout, deadline - time.monotonic())
    if limit <= 0:
        return {"ok": False, "error": "timeout: the evaluation's time limit ran out"}
    out = workdir / f"result-{tag}.json"
    job_path = workdir / f"job-{tag}.json"
    job_path.write_text(json.dumps(job | {"out": str(out)}))
    env = {"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"}
    proc = running.start([sys.executable, "-s", "-P", str(RUNNER), str(ENGINE_DIR), str(job_path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    if proc is None:
        return {"ok": False, "error": "the evaluation has stopped"}
    try:
        try:
            _, err = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            _killpg(proc)
            proc.communicate()
            return {"ok": False, "error": f"timeout: a replay run took over {timeout:g} s" if limit == timeout
                    else "timeout: the evaluation's time limit ran out"}
    finally:
        if proc.poll() is None:  # an interrupt, here or in the caller's thread: the run dies with the evaluation
            _killpg(proc)
        running.done(proc)
    if not out.exists():
        return {"ok": False, "error": f"runner wrote no result (exit {proc.returncode}): {(err or '')[-400:]}"}
    try:
        return json.loads(out.read_text())
    except json.JSONDecodeError:
        return {"ok": False, "error": "runner result is not JSON"}


def reachable(world: dict) -> list[dict]:
    """Nodes replay can ever reveal, by the rules it reveals by: each root through its slot, then the chain of
    children each opened from its parent (a leaf reveals one child, after which it is no longer a leaf). Anything
    else must not set the target a policy is measured against or the work it is normalised by."""
    q = ReplayQuestion(world, 1)
    out = []
    for j in sorted(q._slot):
        node = q._rec[q._slot[j]]
        while node is not None:
            out.append(node)
            node = q._recorded(str(node["id"]))
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
    curve="canonical" (default) keeps one anytime point per revealed cell but credits the cells of a batch worst
    first, best last: live they run in parallel, so the order a policy lists them in must not earn reward (nor the
    ids, which live numbers in that order). curve="reveal" is the old policy-ordered point per cell; "batch" is one point per batch (this
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
                    curve: str = "canonical", total_timeout: float | None = None) -> dict:
    """budget: probes per world, K1 x W as a live round has; required, since without one the recorded roots cap a
    run, which no live round has. timeout bounds each replay run; total_timeout, when given, the whole evaluation."""
    if budget is None:
        raise ValueError("evaluate_policy needs a budget (K1 x W): replay ranks a policy as a live round runs it")
    _ranking(score, penalty, curve)
    policy_path = Path(policy_path).resolve()
    problems = check_policy_source(policy_path.read_text())
    if problems:
        return {"ok": False, "stage": "guard", "problems": problems, "error": "; ".join(problems)}
    deadline = None if total_timeout is None else time.monotonic() + total_timeout
    with tempfile.TemporaryDirectory() as d:
        job = {"policy": str(policy_path), "worlds": worlds, "W": W,
               "betas": [float(b) for b in betas], "budget": budget}
        first = _run_once(job, Path(d), POLICY_HASH_SEED, timeout, deadline)  # the seed a live round runs at
        if not first.get("ok"):
            return {"ok": False, "stage": "run", "error": first.get("error", "unknown failure")}
        second = _run_once(job, Path(d), POLICY_HASH_SEED + 1, timeout, deadline)  # seed-keyed behaviour shows here
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


def _credit_key(o) -> float:
    return o.score if o.valid and o.score is not None else float("-inf")


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
                seen, done, off = len(q._order), q._probes, q._off
                try:
                    q.probe_batch(batch)
                except RecordEnd:  # the run's last batch reached past the record
                    if bi != len(row["trace"]) - 1:
                        raise IllegalBatch("the trace goes on past the end of the record")
                t += 1  # every batch is a live batch: each probe is an attempt
                curve_clock.append([t, q.best_score()])
                # a batch's attempts run in parallel live: credit them worst first, best last, by score, which
                # neither the policy's listing order nor the ids (numbered in that order live) decide. In the batch
                # that ends the record, the recorded cells come after the probes the record could not answer: live
                # those are attempts too, and each moves a recorded cell at most one place later in the order
                lead = q._off - off
                for k, nid in enumerate(sorted(q._order[seen:], key=lambda i: _credit_key(q._obs[i]))):
                    o = q._obs[nid]
                    if o.valid and o.score is not None and (best is None or o.score > best):
                        best = o.score
                    curve_canonical.append([done + lead + k + 1, best])
                curve_batch.append([q._probes, q.best_score()])
            # A run that stops before its budget leaves batches a live round would have run (root slots never run out),
            # each counted as an empty batch; live the same policy stops the same way. A run that reached the end of
            # its record would go on live in batches replay cannot see: each probe it had left counts as an empty
            # batch, which no live continuation can fill less (so the penalty never undercharges it).
            left = budget - q._probes if budget is not None and q._probes < budget else 0
            unspent = left if q._ended else -(-left // max(1, W))
            out.append({"probes": q._probes, "rounds": q._rounds, "best": q.best_score(), "unspent": unspent,
                        "ended": q._ended,
                        "off_record": q._off,  # probes the record could not answer: where replay is not what happened
                        "requested": list(q._requested), "batch_sizes": list(q._batch_sizes),
                        "curve": [list(p) for p in q._curve], "curve_batch": curve_batch, "curve_clock": curve_clock,
                        "curve_canonical": curve_canonical})
        runs[beta] = out
    return {"ok": True, "default_beta": raw["default_beta"], "runs": runs,
            "kmax": max(1, (budget or W) // max(1, W))}
