"""Preset for an attempt ledger: one JSON object per attempt, carrying `candidate`, `construction`,
`falsifiable`, `verdict`, `next` and `check_cmd`, with `supersedes`, `refutes` and `discharges` links.

The ledger has two row shapes. Early rows carry no `id`; their id is their
1-based position among non-blank rows (the ledger's own `#N` references use
that numbering). Later rows carry an explicit integer `id`.

Parent resolution, first match wins:
  supersedes[0] -> a `#N` in `discharges` -> refutes[0] -> the previous row.
A reference that points at an unknown or later id falls through.
"""
from __future__ import annotations

import json
import re

TEXT_FIELDS = ("candidate", "construction", "falsifiable", "verdict", "next", "check_cmd", "discharges")
_REF = re.compile(r"#(\d+)")
_NAME_IN_STR = (re.compile(r"'name':\s*'((?:[^'\\]|\\.)*)'"), re.compile(r'"name":\s*"((?:[^"\\]|\\.)*)"'))


def construction_name(value) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        return None if name is None else str(name)
    if isinstance(value, str):
        for pat in _NAME_IN_STR:
            m = pat.search(value)
            if m:
                return m.group(1)
        return value
    return None


def row_id(row: dict, position: int) -> str:
    return str(row["id"]) if row.get("id") is not None else str(position)


def _as_list(refs) -> list:
    if refs is None:
        return []
    if isinstance(refs, (str, int)):
        return [refs]
    return list(refs)


def _first_known(refs, known: set[str], own: int) -> str | None:
    for ref in _as_list(refs):
        try:
            ref_i = int(ref)
        except (TypeError, ValueError):
            continue
        if ref_i < own and str(ref_i) in known:
            return str(ref_i)
    return None


def resolve_parent(row: dict, own_id: str, known: set[str], previous: str | None) -> tuple[str | None, str]:
    own = int(own_id) if own_id.isdigit() else 10**12
    p = _first_known(row.get("supersedes"), known, own)
    if p:
        return p, "supersedes"
    disc = row.get("discharges")
    if isinstance(disc, str):
        p = _first_known(_REF.findall(disc), known, own)
        if p:
            return p, "discharges"
    p = _first_known(row.get("refutes"), known, own)
    if p:
        return p, "refutes"
    if previous is None:
        return None, "root"
    return previous, "sequential"


def texts(row: dict) -> dict:
    out = {}
    for key in TEXT_FIELDS:
        val = row.get(key)
        if key == "construction":
            val = construction_name(val)
        if val is None:
            continue
        out[key] = val if isinstance(val, str) else str(val)
    return out


def proposal(row: dict) -> str:
    for value in (row.get("candidate"), construction_name(row.get("construction")), row.get("falsifiable")):
        if value not in (None, "", [], {}):
            return (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).strip()
    return ""


def ext(row: dict) -> dict:
    keep = ("mode", "session", "supersedes", "refutes", "date", "ts", "result", "check_sha256")
    return {k: row[k] for k in keep if k in row}
