"""Replay worlds: frozen discovery trees in the form the simulator reads.

A world is {"id", "baseline", "nodes": [{"id", "parent", "score", "valid", "fail_class", "family"}]}.
A world is written once, when its round finishes. Only two commands change one afterwards: `drsi rescore`
(new readings of the same attempts) and `drsi prune` (attempts that did no work removed), each under the worlds lock.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from .families import RANK
from .store import Tree, _atomic_write, _locked


@contextmanager
def worlds_lock(pool):
    """Held by every rewrite of an existing world (rescore, prune), so neither overwrites the other's change."""
    Path(pool).mkdir(parents=True, exist_ok=True)
    with _locked(Path(pool) / "worlds"):
        yield


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
                      "fail_class": n.get("fail_class"), "family": (n.get("fingerprint") or {}).get("family"),
                      "model": (n.get("worker") or {}).get("model"), "cell": (n.get("ext") or {}).get("cell")})
    return {"id": world_id, "baseline": baseline, "nodes": nodes}


def with_cells(worlds: list[dict], tree: Tree) -> list[dict]:
    """Worlds frozen before worlds kept each attempt's cell (the root slot or parent it was opened from) get it from
    the tree, so replay opens each recorded root in the slot it was opened in live. Returns copies; the files stay."""
    out = []
    for w in worlds:
        nodes = []
        for n in w["nodes"]:
            if n.get("cell") is None and n["id"] in tree:
                n = dict(n, cell=(tree.get(n["id"]).get("ext") or {}).get("cell"))
            nodes.append(n)
        out.append(dict(w, nodes=nodes))
    return out


def informative(worlds: list[dict]) -> int:
    """Worlds that can separate two policies: a valid score and at least one continuation (a world of roots alone
    plays out the same for every policy)."""
    return sum(1 for w in worlds if any(n.get("valid") and n.get("score") is not None for n in w["nodes"])
               and any(n.get("parent") for n in w["nodes"]))


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
        try:
            w = json.loads(p.read_text())
        except FileNotFoundError:  # removed since the listing (a prune emptied it): it is no longer a world
            continue
        if w.get("nodes"):
            out.append(w)
    return out
