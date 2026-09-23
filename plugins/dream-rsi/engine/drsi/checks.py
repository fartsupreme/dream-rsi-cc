"""Novelty checks as a campaign operation: claim, judge, record.

Used by `drsi check` (in-session) and by the live orchestrator, which checks every
worker's proposal itself before the worker may build it.
"""
from __future__ import annotations

import calendar
import fcntl
import json
import time
from pathlib import Path

from .families import load_families
from .novelty import check as novelty_check
from .novelty import record_check
from .store import Campaign, utcnow


def pending_checks(camp: Campaign, hours: float = 12.0, tree=None) -> list[dict]:
    """In-flight claims: passing checks made for a live attempt (with a node id) in the last `hours` whose
    attempt is not recorded yet. In-session checks (no node) are not claims: the session records its own
    attempts in its ledger, and `drsi sync` brings them in as history."""
    if not camp.checks_path.exists():
        return []
    recorded = {n["id"] for n in (tree or camp.tree).nodes()}
    now, latest = time.time(), {}
    for line in camp.checks_path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
            when = calendar.timegm(time.strptime(row.get("checked", ""), "%Y-%m-%dT%H:%M:%SZ"))
        except (json.JSONDecodeError, ValueError):
            continue
        node = row.get("node")
        if node and now - when <= hours * 3600:
            latest[node] = row  # the node's latest row decides: its claim, then its verdict
    return [{"ticket": r.get("ticket", ""), "node": node, "proposal": r.get("proposal", "")}
            for node, r in latest.items()
            if node not in recorded and r.get("verdict") in ("claim", "novel", "variant")]


def run_check(camp: Campaign, proposal: str, llm, node: str | None = None) -> dict:
    """Judge `proposal` against the campaign's history and record the verdict.

    With a node id the check is a two-phase claim: under a lock, read the claims already made and record
    ours; then judge without the lock. Parallel attempts are judged concurrently, but each sees every claim
    made before it, so two of them cannot both be told the same idea is novel."""
    cfg = camp.config
    fams = load_families(camp.families_path) if camp.families_path.exists() else {"families": []}
    tree = camp.tree  # one snapshot for the whole check
    proposal = proposal.strip()
    camp.checks_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = camp.checks_path.parent / "check.lock"
    pending = []
    if node:
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            same_round = node.rsplit("-", 1)[0]  # claims from an interrupted earlier round are not in flight
            pending = [p for p in pending_checks(camp, tree=tree)
                       if p["node"] != node and p["node"].rsplit("-", 1)[0] == same_round]
            record_check(camp.checks_path, {"node": node, "verdict": "claim", "proposal": proposal,
                                            "checked": utcnow()})
    result = novelty_check(tree, fams, llm, proposal, goal=cfg.get("goal", ""),
                           plateau=cfg["search"]["plateau"], pending=pending)
    if node:
        result["node"] = node
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record_check(camp.checks_path, result)
    return result
