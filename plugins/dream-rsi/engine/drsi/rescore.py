"""Re-score recorded live attempts with the campaign's current scorer.

A scorer is corrected over a campaign's life, and attempts scored by an earlier version keep readings that no longer mean
what the map and the frozen round worlds take them to mean. Rescoring runs the current scorer on each attempt's own
commit, on a fresh detached checkout, exactly as the live loop scores it; replaces the reading and keeps the old one on
the node (artifacts.rescored); judges every live attempt's outcome again against its parent's (possibly new) score; and
writes the new readings into the frozen round worlds the dream learns from.

Attempts that never reached the scorer (not novel, out of scope, a worker or an orchestration failure) are left alone:
scoring them now would skip the checks that stopped them. The exception is a finished build the run's stop left
unscored (worker.stopped_by_run with changed files): its checks had passed, and only the stop kept it from the scorer.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .live import PROCEDURAL, _file_lock, live_outcome, write_map
from .scorer import run_scorer
from .store import Campaign, _atomic_write, utcnow
from .workspace import Workspaces
from .worlds import worlds_lock


def scorable(node: dict) -> bool:
    art = node.get("artifacts") or {}
    if node.get("source") != "live" or not art.get("commit"):
        return False
    if (node.get("worker") or {}).get("stopped_by_run"):  # a finished build the run's stop left unscored (round 53)
        return bool(art.get("changed"))
    return node.get("fail_class") is not None and node.get("fail_class") not in PROCEDURAL


def rescore(camp: Campaign, ids: set[str] | None = None, parallel: int = 1, log=lambda msg: None) -> dict:
    """Rescore the live attempts in `ids` (all live attempts when None). Returns {"rescored": {id: (old, new)},
    "skipped": [ids that never reached the scorer]}."""
    cfg = camp.config
    ws = Workspaces(camp.root, cfg["workspace"]["repo"], cfg["workspace"].get("base"),
                    ignore=cfg["workspace"].get("ignore"))
    ws.ensure_clone()
    chosen = [n for n in camp.tree.nodes() if n.get("source") == "live" and (ids is None or n["id"] in ids)]
    skipped = [n["id"] for n in chosen if not scorable(n)]
    todo = [n for n in chosen if scorable(n)]
    serial = cfg["scorer"].get("serial", True)
    git_lock = threading.Lock()

    def one(node: dict) -> tuple[str, dict]:
        with _file_lock(camp.root / "logs" / "score.lock", serial):  # never beside a serial loop's scoring
            with git_lock:
                path = ws.create_detached(f"_rescore-{node['id']}", node["artifacts"]["commit"])
            try:
                sc = run_scorer(cfg["scorer"]["cmd"], path, cfg["scorer"]["timeout_s"], cfg.get("direction", "max"),
                                env=cfg["workspace"].get("env"), sandbox=cfg["scorer"].get("sandbox", "auto"),
                                allow_write=cfg["scorer"].get("allow_write") or [],
                                network=bool(cfg["scorer"].get("network", False)))
            finally:
                with git_lock:
                    ws.remove(path)
        log(f"{node['id']}: {node.get('score')} -> {sc['score']}" + ("" if sc["valid"] else f" (invalid: {sc['fail_class']})"))
        return node["id"], sc

    (camp.root / "logs").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=1 if serial else max(1, parallel)) as pool:
        results = list(pool.map(one, todo))

    stamp = utcnow()

    def replace(sc):
        def fn(node):
            art = node.setdefault("artifacts", {})
            art.setdefault("rescored", []).append({
                "at": stamp, "score": node.get("score"), "valid": node.get("valid"), "fail_class": node.get("fail_class"),
                "gates": node.get("gates"), "raw_score": art.get("raw_score"), "scorer_summary": art.get("scorer_summary")})
            node.update(score=sc["score"], valid=sc["valid"], gates=sc["gates"], fail_class=sc["fail_class"])
            art["raw_score"] = sc.get("raw_score")
            art["scorer_summary"] = str(sc.get("summary") or sc.get("error") or "")[:4000]
        return fn

    old = {n["id"]: n.get("score") for n in todo}
    if results:
        camp.tree.modify({nid: replace(sc) for nid, sc in results})
    _judge_outcomes(camp)
    _update_worlds(camp, {nid for nid, _ in results})
    write_map(camp)
    return {"rescored": {nid: (old[nid], sc["score"]) for nid, sc in results}, "skipped": skipped}


def _judge_outcomes(camp: Campaign) -> None:
    """Every live attempt's outcome again, against its parent's score now (the campaign baseline for a new branch),
    as the loop judged it when the attempt was made."""
    tree, baseline = camp.tree, camp.config.get("baseline")
    fns = {}
    for node in tree.nodes():
        if node.get("source") != "live":
            continue
        parent = tree.get(node["parent"]) if node.get("parent") else None
        ref = parent["score"] if parent and parent.get("valid") else baseline
        outcome, killed_by = live_outcome({"valid": node.get("valid"), "score": node.get("score"),
                                           "fail_class": node.get("fail_class")}, ref,
                                          float(camp.config["live"].get("pass_margin", 0.0)))
        art = node.get("artifacts") or {}
        if (art.get("outcome"), art.get("killed_by")) != (outcome, killed_by):
            def fn(n, outcome=outcome, killed_by=killed_by):
                n.setdefault("artifacts", {}).update(outcome=outcome, killed_by=killed_by)
                fp = n.get("fingerprint") or {}
                fp.update(outcome=outcome, killed_by=killed_by)  # a live attempt's outcome is the scorer's
                n["fingerprint"] = fp
            fns[node["id"]] = fn
    if fns:
        tree.modify(fns)


def _update_worlds(camp: Campaign, ids: set[str]) -> None:
    if not ids:
        return
    tree = camp.tree
    pool = camp.root / "trace_pool"
    with worlds_lock(pool):
        for path in sorted(pool.glob("*/world.json")):
            world = json.loads(path.read_text())
            changed = False
            for n in world.get("nodes", []):
                if n["id"] in ids and n["id"] in tree:
                    node = tree.get(n["id"])
                    valid = bool(node.get("valid"))
                    n.update(score=node.get("score") if valid else None, valid=valid,
                             fail_class=node.get("fail_class"))
                    changed = True
            if changed:
                _atomic_write(path, json.dumps(world, ensure_ascii=True))
