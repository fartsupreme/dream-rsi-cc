"""Import an existing attempt log (JSONL) into a campaign tree.

Imports are incremental: rows whose id is already in the tree are skipped, so
re-running against a ledger that keeps growing appends only the new attempts.
A malformed row is skipped and counted, never fatal. Ids from a second ledger that
collide with a first ledger's are refused rather than merged into its history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .novelty import text_hash
from .presets import attempt_ledger as ledger
from .store import Tree, make_node

TRIM = 1500


def _trim(s: str, cap: int = TRIM) -> str:
    return s if len(s) <= cap else s[:cap] + "…"


def _created(row: dict) -> str | None:
    ts = row.get("ts")
    if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
        secs = float(ts)
        if secs > 1e11:  # milliseconds
            secs /= 1000.0
        try:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(secs)))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(ts, str) and "T" in ts:
        return ts
    date = row.get("date")
    if isinstance(date, str) and len(date) == 10:
        return date + "T00:00:00Z"
    return None


def _rows(path: Path, report: dict):
    """(position, row) for every non-blank line; malformed lines keep their position but are skipped."""
    with open(path, errors="surrogateescape") as fh:
        position = 0
        for line in fh:
            if not line.strip():
                continue
            position += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                report["skipped"] = report.get("skipped", 0) + 1
                continue
            if not isinstance(row, dict):
                report["skipped"] = report.get("skipped", 0) + 1
                continue
            yield position, row


def _check_source(tree: Tree, rid: str, proposal: str, full: str, backfill: dict) -> None:
    """Same id and same attempt text is the same ledger (possibly moved to a new path, e.g. a new worktree);
    the same id with different text is a second ledger colliding with the first. Attempts imported before
    proposals carried a hash of their full text get one here."""
    existing = tree.get(rid)
    if existing.get("source") == "import" and existing.get("proposal") != proposal:
        other = (existing.get("ext") or {}).get("source_file")
        raise ValueError(f"attempt id {rid} already names a different attempt (imported from {other}); a second "
                         "ledger with overlapping ids needs its own campaign")
    if existing.get("source") == "import" and not (existing.get("ext") or {}).get("proposal_sha"):
        backfill[rid] = {"ext": {"proposal_sha": text_hash(full)}}


def _ledger_nodes(tree: Tree, path: Path, report: dict, backfill: dict) -> list[dict]:
    known = {n["id"] for n in tree.nodes()}
    previous = None
    new = []
    in_file: set[str] = set()
    for position, row in _rows(path, report):
        rid = ledger.row_id(row, position)
        if rid in in_file:  # the ledger repeats an id: the first occurrence is the attempt
            report["duplicates"] = report.get("duplicates", 0) + 1
            continue
        in_file.add(rid)
        if rid in known:
            if rid in tree:
                full = ledger.proposal(row)
                _check_source(tree, rid, _trim(full, 600), full, backfill)
            else:
                report["duplicates"] = report.get("duplicates", 0) + 1
            previous = rid
            continue
        parent, link = ledger.resolve_parent(row, rid, known, previous)
        full = ledger.proposal(row)
        extra = ledger.ext(row) | {"row": position, "link": link, "source_file": str(path),
                                    "proposal_sha": text_hash(full)}
        new.append(make_node(
            id=rid, parent=parent, source="import", created=_created(row),
            proposal=_trim(full, 600),
            text={k: _trim(v) for k, v in ledger.texts(row).items()},
            ext=extra))
        known.add(rid)
        previous = rid
    return new


def _generic_nodes(tree: Tree, path: Path, field_map: dict, report: dict, backfill: dict) -> list[dict]:
    fid, fparent = field_map.get("id", "id"), field_map.get("parent", "parent")
    fprop, ftext = field_map.get("proposal", "proposal"), field_map.get("text", [])
    known = {n["id"] for n in tree.nodes()}
    rows = list(_rows(path, report))
    in_file = {str(position) if r.get(fid) is None else str(r.get(fid)) for position, r in rows}
    new = []
    first: set[str] = set()
    for position, row in rows:
        raw_id = row.get(fid)
        rid = str(position) if raw_id is None else str(raw_id)
        if rid in first:
            report["duplicates"] = report.get("duplicates", 0) + 1
            continue
        first.add(rid)
        if rid in known:
            if rid in tree:
                full = str(row.get(fprop, ""))
                _check_source(tree, rid, _trim(full, 600), full, backfill)
            else:
                report["duplicates"] = report.get("duplicates", 0) + 1
            continue
        parent = row.get(fparent)
        parent = str(parent) if parent is not None and (str(parent) in known or str(parent) in in_file) \
            and str(parent) != rid else None
        new.append(make_node(
            id=rid, parent=parent, source="import", created=_created(row),
            proposal=_trim(str(row.get(fprop, "")), 600),
            text={k: _trim(str(row[k])) for k in ftext if k in row},
            ext={"row": position, "source_file": str(path), "proposal_sha": text_hash(str(row.get(fprop, "")))}))
        known.add(rid)
    return _parents_first(new, {n["id"] for n in tree.nodes()})


def _parents_first(nodes: list[dict], present: set[str]) -> list[dict]:
    """Order nodes so every parent precedes its children (linear time). A parent that never arrives,
    or a cycle, turns the child into a root rather than failing the import."""
    by_id = {n["id"]: n for n in nodes}
    kids: dict[str, list[dict]] = {}
    ready = []
    for n in nodes:
        p = n["parent"]
        if p is None or p in present:
            ready.append(n)
        elif p in by_id:
            kids.setdefault(p, []).append(n)
        else:
            n["parent"] = None
            ready.append(n)
    from collections import deque
    ready = deque(ready)  # file order: unrelated rows keep the order the ledger gave them
    out, placed = [], set()
    while True:
        while ready:
            n = ready.popleft()
            if n["id"] in placed:
                continue
            placed.add(n["id"])
            out.append(n)
            ready.extend(kids.pop(n["id"], []))
        stuck = [n for n in nodes if n["id"] not in placed]
        if not stuck:
            return out
        # walk parent links from a stuck node until one repeats: that node is on the cycle
        seen, cur = set(), stuck[0]
        while cur["id"] not in seen and cur["parent"] in by_id and cur["parent"] not in placed:
            seen.add(cur["id"])
            cur = by_id[cur["parent"]]
        cur["parent"] = None
        ready.append(cur)


def import_jsonl(tree: Tree, path, preset: str = "attempt-ledger", field_map: dict | None = None,
                 report: dict | None = None) -> int:
    """Returns the number of attempts added. `report` (optional) receives skipped/duplicates counts."""
    path = Path(path).resolve()
    report = report if report is not None else {}
    report.setdefault("skipped", 0)
    report.setdefault("duplicates", 0)
    backfill: dict = {}
    if preset == "attempt-ledger":
        nodes = _ledger_nodes(tree, path, report, backfill)
    elif preset == "generic":
        nodes = _generic_nodes(tree, path, field_map or {}, report, backfill)
    else:
        raise ValueError(f"unknown preset {preset!r}")
    if backfill:
        tree.merge_many(backfill)
    return tree.add_many(nodes, skip_existing=True) if nodes else 0
