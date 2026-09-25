"""Import an existing attempt log (JSONL) into a campaign tree.

Imports are incremental: rows whose id is already in the tree are not added again, so
re-running against a ledger that keeps growing appends only the new attempts. A row the
ledger has corrected since it was imported replaces the stored text and is read again. A
malformed row is skipped and counted, never fatal. Ids from a second ledger that collide
with a first ledger's are refused rather than merged into its history.
"""
from __future__ import annotations

import hashlib
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


def _row_sha(full: str, texts: dict, ext: dict) -> str:
    """Identity of everything an attempt's node is built from, untrimmed: a correction anywhere in it, even
    past the stored cut or only in a metadata field, changes it."""
    blob = json.dumps({"proposal": full, "text": texts, "ext": ext}, sort_keys=True, ensure_ascii=False,
                      default=str)
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


def _check_source(tree: Tree, rid: str, path: Path, full: str, texts: dict, ext: dict, position: int,
                  shadow: set, report: dict, backfill: dict, refresh: dict) -> None:
    """An id the tree already has, met again in a ledger.

    Each imported attempt belongs to one ledger file (`ext.source_file`). A changed row in that file is a
    correction: the stored text is replaced, and if what the classifier reads changed, the reading is
    dropped (its family kept) so the next pass reads it again. The same attempt at a path the campaign does
    not list alongside (`shadow`) is the ledger moved, e.g. to a new worktree: the attempt moves with it and
    the old path is reported in `moved_from`. A copy at another listed path is left alone, so two listed
    copies never overwrite each other. A different attempt under the same id is a second ledger colliding
    with the first, and is refused. A correction that changes a row's links (`supersedes`, `refutes`) is
    reported in `relinked`; its place in the tree stays where it was imported, because edges are fixed.

    What is written is decided here, before the tree's lock is taken again, so each write carries the node's
    state as scanned (`seen`) and is skipped if another process changed the node in between."""
    existing = tree.get(rid)
    if existing.get("source") != "import":
        return
    here = str(path)
    old_ext = existing.get("ext") or {}
    owner = old_ext.get("source_file")
    if owner != here and owner in shadow:
        report["shadowed"] = report.get("shadowed", 0) + 1
        return
    proposal, text = _trim(full, 600), {k: _trim(v) for k, v in texts.items()}
    if existing.get("proposal") != proposal and owner != here:
        raise ValueError(f"attempt id {rid} already names a different attempt (imported from {owner}); a "
                         "second ledger with overlapping ids needs its own campaign")
    sha = _row_sha(full, texts, ext)
    stored_same = existing.get("proposal") == proposal and existing.get("text") == text
    if old_ext.get("row_sha"):
        changed = old_ext["row_sha"] != sha
    else:  # imported before row hashes: judged on what it stores, so an upgrade re-reads only real changes
        changed = (not stored_same or any(old_ext.get(k) != ext.get(k) for k in ledger.EXT_KEYS)
                   or old_ext.get("proposal_sha") not in (None, text_hash(full)))
    if owner and owner != here:
        report.setdefault("moved_from", set()).add(owner)
    if changed and any(old_ext.get(k) != ext.get(k) for k in ("supersedes", "refutes")):
        report.setdefault("relinked", []).append(rid)
    stamp = {"source_file": here, "proposal_sha": text_hash(full), "row_sha": sha}
    if changed:
        refresh[rid] = {"proposal": proposal, "text": text, "reread": not stored_same,
                        "ext": ext | {"row": position} | stamp, "seen": _state(existing)}
    elif owner != here or any(old_ext.get(k) != v for k, v in stamp.items()):
        backfill[rid] = {"ext": stamp, "seen": _state(existing)}


def _state(node: dict) -> str:
    ext = node.get("ext") or {}
    return json.dumps([node.get("proposal"), node.get("text"), ext.get("row_sha"), ext.get("source_file")],
                      sort_keys=True, ensure_ascii=False, default=str)


def _apply_backfill(tree: Tree, backfill: dict) -> None:
    def stamp(fields):
        def apply(node):
            if _state(node) == fields["seen"]:
                node["ext"] = (node.get("ext") or {}) | fields["ext"]
        return apply
    tree.modify({rid: stamp(fields) for rid, fields in backfill.items()})


def _apply_refresh(tree: Tree, refresh: dict) -> None:
    def replace(fields):
        def apply(node):
            if _state(node) != fields["seen"]:  # another process got there first
                return
            node["proposal"], node["text"] = fields["proposal"], fields["text"]
            kept = {k: v for k, v in (node.get("ext") or {}).items() if k not in ledger.EXT_KEYS}
            node["ext"] = kept | fields["ext"]  # a field the corrected row dropped is dropped here too
            if fields["reread"]:
                fp = node.get("fingerprint") if isinstance(node.get("fingerprint"), dict) else {}
                node["fingerprint"] = {k: fp[k] for k in ("family", "family_rev") if k in fp} or None
        return apply
    tree.modify({rid: replace(fields) for rid, fields in refresh.items()})


def _ledger_nodes(tree: Tree, path: Path, report: dict, backfill: dict, refresh: dict,
                  shadow: set) -> list[dict]:
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
                _check_source(tree, rid, path, ledger.proposal(row), ledger.texts(row), ledger.ext(row),
                              position, shadow, report, backfill, refresh)
            else:
                report["duplicates"] = report.get("duplicates", 0) + 1
            previous = rid
            continue
        parent, link = ledger.resolve_parent(row, rid, known, previous)
        full = ledger.proposal(row)
        extra = ledger.ext(row) | {"row": position, "link": link, "source_file": str(path),
                                    "proposal_sha": text_hash(full),
                                    "row_sha": _row_sha(full, ledger.texts(row), ledger.ext(row))}
        new.append(make_node(
            id=rid, parent=parent, source="import", created=_created(row),
            proposal=_trim(full, 600),
            text={k: _trim(v) for k, v in ledger.texts(row).items()},
            ext=extra))
        known.add(rid)
        previous = rid
    return new


def _generic_nodes(tree: Tree, path: Path, field_map: dict, report: dict, backfill: dict,
                   refresh: dict, shadow: set) -> list[dict]:
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
                _check_source(tree, rid, path, str(row.get(fprop, "")), {k: str(row[k]) for k in ftext if k in row},
                              {}, position, shadow, report, backfill, refresh)
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
            ext={"row": position, "source_file": str(path), "proposal_sha": text_hash(str(row.get(fprop, ""))),
                 "row_sha": _row_sha(str(row.get(fprop, "")), {k: str(row[k]) for k in ftext if k in row}, {})}))
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
                 report: dict | None = None, shadow=()) -> int:
    """Returns the number of attempts added. `report` (optional) receives skipped, duplicates, refreshed
    (corrected rows whose stored text was replaced) and shadowed counts, and `moved_from`: the paths whose
    attempts this ledger took over. `shadow` lists the campaign's other sources; attempts they own are
    left to them."""
    shadow = {str(Path(p).resolve()) for p in shadow}
    path = Path(path).resolve()
    report = report if report is not None else {}
    report.setdefault("skipped", 0)
    report.setdefault("duplicates", 0)
    backfill: dict = {}
    refresh: dict = {}
    if preset == "attempt-ledger":
        nodes = _ledger_nodes(tree, path, report, backfill, refresh, shadow)
    elif preset == "generic":
        nodes = _generic_nodes(tree, path, field_map or {}, report, backfill, refresh, shadow)
    else:
        raise ValueError(f"unknown preset {preset!r}")
    if backfill:
        _apply_backfill(tree, backfill)
    if refresh:
        _apply_refresh(tree, refresh)
    report["refreshed"] = report.get("refreshed", 0) + len(refresh)
    return tree.add_many(nodes, skip_existing=True) if nodes else 0
