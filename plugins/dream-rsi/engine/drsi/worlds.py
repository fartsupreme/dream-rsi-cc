"""Replay worlds: frozen discovery trees in the form the simulator reads.

A world is {"id", "baseline", "nodes": [{"id", "parent", "score", "valid", "fail_class", "family"}]}.
Once written to the trace pool a world is never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

from .families import RANK
from .store import Tree, _atomic_write


def outcome_score(node: dict) -> float | None:
    """Proxy score for attempts that have no measured score: the classifier's outcome rank in [0, 1]."""
    outcome = (node.get("fingerprint") or {}).get("outcome")
    if outcome not in RANK:
        return None
    return RANK[outcome] / max(RANK.values())


def world_from_tree(tree: Tree, world_id: str, baseline: float = 0.0, ids: set[str] | None = None) -> dict:
    nodes = []
    for n in tree.nodes():
        if ids is not None and n["id"] not in ids:
            continue
        score = n.get("score")
        if score is None and n.get("valid") is not False:
            score = outcome_score(n)
        valid = bool(n.get("valid", True)) if n.get("valid") is not None else score is not None
        parent = n["parent"] if (ids is None or n["parent"] in ids) else None
        nodes.append({"id": n["id"], "parent": parent, "score": score if valid else None, "valid": valid,
                      "fail_class": n.get("fail_class"), "family": (n.get("fingerprint") or {}).get("family")})
    return {"id": world_id, "baseline": baseline, "nodes": nodes}


def freeze_world(pool, world: dict) -> Path:
    path = Path(pool) / world["id"] / "world.json"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(world, ensure_ascii=True))
    return path


def load_worlds(pool) -> list[dict]:
    out = []
    for p in sorted(Path(pool).glob("*/world.json")):
        w = json.loads(p.read_text())
        if w.get("nodes"):
            out.append(w)
    return out
