"""Removing recorded attempts that did no work (`drsi prune`).

A worker or the orchestrator can fail before any work is done (every call refused, a workspace that could not be
made). Such an attempt is still recorded, so the record, the map and the replay pool carry it as if it were a failed
idea. `drsi prune` removes these attempts, selected by id or by the error text they carry, but only ones that did no
work: a live attempt that failed as a worker or orchestration error, with no score and no changed files. It removes
them from the tree (anything that continued from one moves to its nearest kept ancestor), from the frozen round worlds
(a world left with nothing is removed), their branches, worktrees and proposal directories; it withdraws their novelty
claims, rewrites the map, and appends what it removed to logs/prune.jsonl. While a run is live, attempts of a round not
yet frozen are left alone. The tree is changed last, so a prune that fails midway can be run again to finish.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import guardian
from .novelty import record_check
from .store import Campaign, _atomic_write, utcnow
from .worlds import worlds_lock

NO_WORK = {"agent_error", "orchestrator_error"}


def _changed(node: dict) -> bool:
    c = (node.get("artifacts") or {}).get("changed")
    if isinstance(c, str):
        return c.strip() not in ("", "[]")
    return bool(c)


def did_no_work(node: dict) -> bool:
    return (node.get("source") == "live" and node.get("fail_class") in NO_WORK and node.get("score") is None
            and not node.get("valid") and not _changed(node))


def _error_text(node: dict) -> str:
    t = node.get("text") or {}
    return f"{t.get('worker_error') or ''}\n{t.get('orchestrator_error') or ''}"


def prune(camp: Campaign, ids: set[str] | None = None, error_match: str | None = None, dry_run: bool = False,
          reason: str = "", log=print) -> dict:
    if not ids and not error_match:
        raise ValueError("prune needs ids or an error text to match")
    tree = camp.tree
    chosen = [n for n in tree.nodes() if n.get("source") == "live"
              and ((ids and n["id"] in ids) or (error_match and error_match in _error_text(n)))]
    refused = [n["id"] for n in chosen if not did_no_work(n)]
    pool = camp.root / "trace_pool"
    frozen = {p.parent.name for p in pool.glob("*/world.json")}
    running = guardian.run_lock_held(camp.root / "logs" / guardian.REGISTRY)
    cur = camp.root / "logs" / "current_round"  # written by the run as each round starts
    current = cur.read_text().strip() if running and cur.exists() else None

    def live_round(rid):  # the round the run is on; without its note, any round not frozen yet
        return rid == current if current else rid not in frozen
    in_progress = [n["id"] for n in chosen if did_no_work(n) and running
                   and live_round((n.get("ext") or {}).get("round"))]
    gone = {n["id"] for n in chosen if did_no_work(n)} - set(in_progress)
    victims = [n["id"] for n in tree.nodes() if n["id"] in gone]
    rep = {"pruned": victims, "refused": refused, "in_progress": in_progress,
           "reparented": [n["id"] for n in tree.nodes() if n["parent"] in gone and n["id"] not in gone],
           "worlds_removed": [], "worlds_rewritten": []}
    if dry_run and victims:  # what the world step below would do, without doing it
        for wp in sorted(pool.glob("*/world.json")):
            try:
                ids = {n["id"] for n in json.loads(wp.read_text())["nodes"]}
            except FileNotFoundError:
                continue
            if ids & gone:
                rep["worlds_removed" if ids <= gone else "worlds_rewritten"].append(wp.parent.name)
    if dry_run or not victims:
        if not dry_run:  # a rerun after a failure past the tree step still refreshes the map
            from .live import write_map
            write_map(camp)
        return rep
    # The tree goes last: it is what selects the victims, so every step before it can simply be repeated by a rerun
    # if one fails, and nothing is left pointing at attempts the tree no longer has.
    with worlds_lock(pool):
        for wp in sorted(pool.glob("*/world.json")):
            try:
                w = json.loads(wp.read_text())
            except FileNotFoundError:
                continue
            if not {n["id"] for n in w["nodes"]} & gone:
                continue
            keep = [n for n in w["nodes"] if n["id"] not in gone]
            if not keep:
                shutil.rmtree(wp.parent)
                rep["worlds_removed"].append(wp.parent.name)
                continue
            parent_of = {n["id"]: n.get("parent") for n in w["nodes"]}
            for n in keep:
                q = n.get("parent")
                while q is not None and q in gone:
                    q = parent_of.get(q)
                n["parent"] = q
            w["nodes"] = keep
            _atomic_write(wp, json.dumps(w, ensure_ascii=True))
            rep["worlds_rewritten"].append(wp.parent.name)

    if camp.checks_path.exists():  # a claim of an attempt that is gone would read as in flight
        import fcntl
        with open(camp.checks_path.parent / "check.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            latest = {}
            for line in camp.checks_path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("node") in gone:
                    latest[row["node"]] = row.get("verdict")
            for nid in sorted(n for n, v in latest.items() if v != "withdrawn"):
                record_check(camp.checks_path, {"node": nid, "verdict": "withdrawn", "checked": utcnow(),
                                                "rationale": "the attempt was pruned"})

    work = camp.root / "work"
    if (camp.root / "repo" / ".git").exists():
        from .workspace import Workspaces, _rmtree
        cfg = camp.config
        ws = Workspaces(camp.root, cfg["workspace"]["repo"], cfg["workspace"].get("base"),
                        ignore=cfg["workspace"].get("ignore"))
        for nid in victims:
            if (work / nid).exists():
                _rmtree(work / nid)
        ws._in_repo("worktree", "prune", check=False)
        have = set(ws._in_repo("branch", "--list", "drsi/*", "--format=%(refname:short)").split())  # fails loudly
        doomed = [f"drsi/{nid}" for nid in victims if f"drsi/{nid}" in have]
        for k in range(0, len(doomed), 200):
            ws._in_repo("branch", "-D", "-q", *doomed[k:k + 200])  # a failure stops the prune before the tree
    for nid in victims:
        d = work / "_proposals" / nid
        if d.is_dir():
            shutil.rmtree(d)

    logp = camp.root / "logs" / "prune.jsonl"  # before the tree: the ids stay reserved even if what follows fails
    logp.parent.mkdir(parents=True, exist_ok=True)
    record_check(logp, {"at": utcnow(), "ids": victims, "reason": reason,  # ends a torn last line first
                        "criteria": {"ids": sorted(ids) if ids else None, "error_match": error_match},
                        "reparented": rep["reparented"], "worlds_removed": rep["worlds_removed"],
                        "worlds_rewritten": rep["worlds_rewritten"]})
    rep["reparented"] = tree.prune(gone)
    from .live import write_map
    write_map(camp)
    log(f"pruned {len(victims)} attempts that did no work")
    return rep
