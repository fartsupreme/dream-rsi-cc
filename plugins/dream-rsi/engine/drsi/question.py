"""The paper's exploration interface (Dream-RSI, Listing 2), for replay and live use.

A policy sees only the revealed prefix of a discovery tree. Its actions are the
leaves of that prefix plus "root slots" (open a new branch). Probing a batch
reveals one child per probed cell. In replay the child comes from the frozen
record; in live mode a worker produces it.

Replay rule (paper: Child(v) is v's recorded child, or ∅): a leaf reveals its
first recorded child. A leaf with no recorded child reveals nothing and is
marked exhausted. Siblings recorded under an interior node are unreachable,
because only leaves are actions (A(T) = {r} ∪ leaves).
"""
from __future__ import annotations

from dataclasses import dataclass, field


class IllegalBatch(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    id: str
    parent_id: str | None
    branch: int
    attempt: int
    seq: int
    score: float | None
    valid: bool
    fail_class: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class CellMeta:
    branch: int
    attempt: int
    parent_id: str | None
    seq: int | None
    tags: tuple = field(default_factory=tuple)


ROOT = "root:"


class QuestionBase:
    """Metrics (probes, rounds, batch_sizes, curve) are private and exposed as copies: the
    evaluator reads them, so a policy must not be able to rewrite them."""

    def __init__(self, max_parallelism: int, baseline_score: float = 0.0, max_probes: int | None = None):
        self._max_parallelism = int(max_parallelism)
        self._baseline_score = baseline_score
        self._max_probes = max_probes  # enforced here, whatever the policy's solve() does
        self._clear()

    @property
    def max_parallelism(self) -> int:
        return self._max_parallelism

    @property
    def baseline_score(self) -> float:
        return self._baseline_score

    # -- state ---------------------------------------------------------------------------
    def _clear(self) -> None:
        self._obs: dict[str, Observation] = {}
        self._order: list[str] = []
        self._revealed_children: dict[str, int] = {}
        self._exhausted: set[str] = set()
        self._opened_slots: set[int] = set()
        self._probes = 0
        self._rounds = 0
        self._probing = False
        self._batch_sizes: list[int] = []
        self._curve: list[tuple[int, float | None]] = []  # (probes, best score) after every reveal

    def reset(self) -> None:
        """Start of a run. Only legal before the first probe: resetting after exploring would let
        a policy look at everything, forget the cost, and replay only the best path."""
        if self._rounds:
            raise IllegalBatch("reset() after probing is not allowed")
        self._clear()

    @property
    def probes(self) -> int:
        return self._probes

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def batch_sizes(self) -> list[int]:
        return list(self._batch_sizes)

    @property
    def curve(self) -> list[tuple[int, float | None]]:
        return list(self._curve)

    def observed(self) -> dict[str, Observation]:
        return dict(self._obs)

    def best_score(self) -> float | None:
        vals = [o.score for o in self._obs.values() if o.valid and o.score is not None]
        return max(vals) if vals else None

    def opened_branches(self) -> list[int]:
        return sorted(self._opened_slots)

    # -- actions -------------------------------------------------------------------------
    def _root_capacity(self) -> int | None:
        """How many root slots exist in total (None = unbounded)."""
        return None

    def legal_roots(self) -> list[str]:
        cap = self._root_capacity()
        out, j = [], 0
        while len(out) < self.max_parallelism and (cap is None or j < cap):
            if j not in self._opened_slots:
                out.append(f"{ROOT}{j}")
            j += 1
            if cap is None and j > len(self._opened_slots) + self.max_parallelism:
                break
        return out

    def _leaves(self) -> list[str]:
        return [i for i in self._order if not self._revealed_children.get(i) and i not in self._exhausted]

    def legal_actions(self) -> list[str]:
        return self.legal_roots() + self._leaves()

    def meta(self, cell_id: str) -> CellMeta:
        if cell_id.startswith(ROOT):
            return CellMeta(branch=int(cell_id[len(ROOT):]), attempt=0, parent_id=None, seq=None)
        o = self._obs[cell_id]
        return CellMeta(branch=o.branch, attempt=o.attempt, parent_id=o.parent_id, seq=o.seq,
                        tags=((f"family:{o.family}",) if o.family else ()))

    def probe_batch(self, cells, on_reveal=None) -> list[Observation | None]:
        if self._probing:
            raise IllegalBatch("probe_batch cannot be called from inside on_reveal")
        cells = list(cells)
        if not cells:
            raise IllegalBatch("empty batch")
        if any(type(c) is not str for c in cells):
            raise IllegalBatch("cells must be plain strings")
        if self._max_probes is not None and self._probes >= self._max_probes:
            raise IllegalBatch(f"budget of {self._max_probes} probes is spent")
        if len(set(cells)) != len(cells):
            raise IllegalBatch(f"duplicate cells in batch: {cells}")
        if len(cells) > self.max_parallelism:
            raise IllegalBatch(f"batch of {len(cells)} exceeds max_parallelism {self.max_parallelism}")
        legal = set(self.legal_actions())
        bad = [c for c in cells if c not in legal]
        if bad:
            raise IllegalBatch(f"illegal cells: {bad}")
        if self._max_probes is not None:
            cells = cells[:self._max_probes - self._probes]  # the budget is exact, not per batch
        self._probing = True
        try:
            children = self._expand(cells)
            self._rounds += 1
            self._batch_sizes.append(0)  # parallel work is what was revealed, counted as it is revealed
            out: list[Observation | None] = []
            for cell, node in zip(cells, children):
                if cell.startswith(ROOT):
                    self._opened_slots.add(int(cell[len(ROOT):]))
                if node is None:
                    if not cell.startswith(ROOT):
                        self._exhausted.add(cell)
                    out.append(None)
                    continue
                obs = self._reveal(cell, node)
                self._batch_sizes[-1] += 1
                out.append(obs)
                if on_reveal:
                    on_reveal(obs)
            return out
        finally:
            self._probing = False

    def _reveal(self, cell: str, node: dict) -> Observation:
        if cell.startswith(ROOT):
            parent, branch, attempt = None, int(cell[len(ROOT):]), 0
        else:
            p = self._obs[cell]
            parent, branch, attempt = cell, p.branch, p.attempt + 1
            self._revealed_children[cell] = self._revealed_children.get(cell, 0) + 1
        obs = Observation(id=str(node["id"]), parent_id=parent, branch=branch, attempt=attempt,
                          seq=len(self._order), score=node.get("score"),
                          valid=bool(node.get("valid", node.get("score") is not None)),
                          fail_class=node.get("fail_class"), family=node.get("family"))
        self._obs[obs.id] = obs
        self._order.append(obs.id)
        self._probes += 1
        self._curve.append((self._probes, self.best_score()))
        return obs

    def _expand(self, cells: list[str]) -> list[dict | None]:
        raise NotImplementedError


class ReplayQuestion(QuestionBase):
    """A frozen discovery tree used as a zero-cost simulator."""

    def __init__(self, world: dict, max_parallelism: int, max_probes: int | None = None):
        self._world = world
        nodes = world["nodes"]
        self._rec = {str(n["id"]): n for n in nodes}
        self._kids: dict[str | None, list[str]] = {}
        for n in nodes:
            p = n.get("parent")
            self._kids.setdefault(None if p is None else str(p), []).append(str(n["id"]))
        self._roots = self._kids.get(None, [])
        super().__init__(max_parallelism, world.get("baseline", 0.0), max_probes)

    def _root_capacity(self) -> int:
        return len(self._roots)

    def _expand(self, cells):
        out = []
        for c in cells:
            if c.startswith(ROOT):
                j = int(c[len(ROOT):])
                out.append(self._rec[self._roots[j]] if j < len(self._roots) else None)
            else:
                kids = [k for k in self._kids.get(c, []) if k not in self._obs]
                out.append(self._rec[kids[0]] if kids else None)
        return out
