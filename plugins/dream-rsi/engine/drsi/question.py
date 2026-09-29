"""The paper's exploration interface (Dream-RSI, Listing 2), for replay and live use.

A policy sees only the revealed prefix of a discovery tree. Its actions are the
leaves of that prefix plus "root slots" (open a new branch). Probing a batch
reveals one child per probed cell. In replay the child comes from the frozen
record; in live mode a worker produces it.

Replay rule (paper: Child(v) is v's recorded child, or ∅): a leaf reveals its
first recorded child. Siblings recorded under an interior node are unreachable,
because only leaves are actions (A(T) = {r} ∪ leaves). Under a budget, a probe
past the record (a leaf with no recorded child, a root slot beyond the recorded
roots) ends the run: replay cannot know what that work would have found, so the
batch's recorded cells are revealed, the probes past the record cost budget and
reveal nothing, and RecordEnd stops the policy. Nothing it could learn there
counts, so a replayed run never scores above what the same policy does live on
the same attempts. Without a budget the paper's ∅ stands: nothing is revealed,
the leaf is exhausted and the recorded roots are the limit, so a run that
explores until nothing is legal still ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class IllegalBatch(ValueError):
    pass


# The hash seed every policy process runs at: a live round, the behaviour gate and the first replay run alike. A policy
# can read the seed through set iteration order, so a live round at another seed could act on what no check sees;
# the second replay run uses another seed only to catch exactly that (its traces must match the first run's).
POLICY_HASH_SEED = 1


class RecordEnd(BaseException):
    """A replayed run reached the end of its record. Not an Exception, so a policy (which may catch only Exception
    or narrower) cannot catch it and go on; the runner that started the policy does."""


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
        self._ended = False  # a replayed run past its record: no probe counts after this
        self._batch_sizes: list[int] = []
        self._requested: list[int] = []  # cells asked for per batch (after the budget cut)
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
        if self._ended:
            raise RecordEnd("the replayed record has ended")
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
            self._requested.append(len(cells))
            out: list[Observation | None] = []
            for cell, node in zip(cells, children):
                if cell.startswith(ROOT):
                    self._opened_slots.add(int(cell[len(ROOT):]))
                if node is None:  # past the record: nothing revealed (budgeted replay ends after this batch)
                    if not cell.startswith(ROOT):
                        self._exhausted.add(cell)
                    self._probes += 1
                    self._curve.append((self._probes, self.best_score()))
                    out.append(None)
                    continue
                obs = self._reveal(cell, node)
                self._batch_sizes[-1] += 1
                out.append(obs)
                if on_reveal:
                    on_reveal(obs)
        finally:
            self._probing = False
        if self._ended:
            raise RecordEnd("the replayed record has ended")
        return out

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


def _unambiguous(nodes: list[dict]) -> list[dict]:
    """Recorded ids share one namespace with root-slot cells ("root:<j>"). An id that looks like a slot (an imported
    ledger's, say) is renamed here, with its children's parent and cell, so a slot and an attempt are never the same
    cell; recorded ids never reach a policy, so nothing it sees changes."""
    ids = {str(n["id"]) for n in nodes}
    ren: dict[str, str] = {}
    for i in sorted(ids):
        if i.startswith(ROOT):
            new = "node:" + i
            while new in ids or new in ren.values():
                new = "node:" + new
            ren[i] = new
    if not ren:
        return nodes
    out = []
    for n in nodes:
        m = dict(n, id=ren.get(str(n["id"]), n["id"]))
        p = n.get("parent")
        if p is not None and str(p) in ren:
            m["parent"] = ren[str(p)]
            if n.get("cell") == p:  # opened from that parent (a root's own cell is its slot and stays)
                m["cell"] = ren[str(p)]
        out.append(m)
    return out


class ReplayQuestion(QuestionBase):
    """A frozen discovery tree used as a zero-cost simulator. Its public surface is the live question's: anything
    only replay could answer would let a policy act differently in replay than live."""

    def __init__(self, world: dict, max_parallelism: int, max_probes: int | None = None):
        self._world = world
        nodes = _unambiguous(world["nodes"])
        self._rec = {str(n["id"]): n for n in nodes}
        self._kids: dict[str | None, list[str]] = {}
        for n in nodes:
            p = n.get("parent")
            self._kids.setdefault(None if p is None else str(p), []).append(str(n["id"]))
        self._roots = self._kids.get(None, [])
        # A root is replayed in the slot it was opened in live (its recorded cell, "root:<j>"), so a policy replaying
        # its own round meets every attempt where it met it; a world without cells opens its roots in listed order.
        # A world with cells maps each root to its slot; a root with no slot cell (a pruned attempt's continuation,
        # re-rooted) was opened from no slot live, so no slot reaches it.
        slots: dict[int, str] = {}
        for r in self._roots:
            c = self._rec[r].get("cell")
            if isinstance(c, str) and c.startswith(ROOT) and c[len(ROOT):].isdigit():
                slots.setdefault(int(c[len(ROOT):]), r)
        self._has_cells = any(self._rec[n].get("cell") is not None for n in self._rec)
        self._slot = slots if self._has_cells else dict(enumerate(self._roots))
        self._off = 0  # probes past the record (evaluator-side: where replay stops being what happened)
        super().__init__(max_parallelism, world.get("baseline", 0.0), max_probes)

    def _root_capacity(self) -> int | None:
        """Unbounded under a budget, as live. With no budget the recorded roots are the limit, or a run that
        explores until nothing is legal would never end."""
        return None if self._max_probes is not None else (max(self._slot) + 1 if self._slot else 0)

    def _recorded(self, cell: str) -> dict | None:
        if cell.startswith(ROOT):
            nid = self._slot.get(int(cell[len(ROOT):]))
            return self._rec[nid] if nid is not None else None
        kids = [k for k in self._kids.get(cell, []) if k not in self._obs]
        if self._has_cells:  # the child opened from this leaf live; one opened elsewhere (re-parented) is not
            kids = [k for k in kids if self._rec[k].get("cell") == cell]
        return self._rec[kids[0]] if kids else None

    def _expand(self, cells):
        out = []
        for c in cells:
            node = self._recorded(c)
            if node is None and self._max_probes is not None:
                self._off += 1
                self._ended = True
            out.append(node)
        return out


class PolicyQuestion:
    """What a policy is handed, in replay, in the behaviour gate and in a live round alike. It is the question's API
    with every attempt shown under an id that says only when it was revealed ("a1", "a2", ...), and it prints as
    "<question>", so the recorded ids and the class doing the work (a replayed or a live question) never reach the
    policy: whatever a policy can observe, it observes the same way everywhere. The wrapped question enforces every
    rule; the checks here only keep refusals in the ids the policy was shown."""

    def __init__(self, question: QuestionBase):
        self._q = question
        self._shown: dict[str, str] = {}  # attempt id -> shown id
        self._real: dict[str, str] = {}  # shown id -> attempt id

    def __repr__(self) -> str:
        return "<question>"

    __str__ = __repr__

    @property
    def max_parallelism(self) -> int:
        return self._q.max_parallelism

    @property
    def baseline_score(self) -> float:
        return self._q.baseline_score

    @property
    def probes(self) -> int:
        return self._q.probes

    @property
    def rounds(self) -> int:
        return self._q.rounds

    @property
    def batch_sizes(self) -> list[int]:
        return self._q.batch_sizes

    @property
    def curve(self) -> list[tuple[int, float | None]]:
        return self._q.curve

    def reset(self) -> None:
        self._q.reset()
        self._shown.clear()
        self._real.clear()

    def best_score(self) -> float | None:
        return self._q.best_score()

    def opened_branches(self) -> list[int]:
        return self._q.opened_branches()

    def legal_roots(self) -> list[str]:
        return self._q.legal_roots()

    def legal_actions(self) -> list[str]:
        return [c if c.startswith(ROOT) else self._shown[c] for c in self._q.legal_actions()]

    def observed(self) -> dict[str, Observation]:
        return {self._shown[i]: self._view(o) for i, o in self._q.observed().items()}

    def meta(self, cell_id: str) -> CellMeta:
        if cell_id.startswith(ROOT):
            return self._q.meta(cell_id)
        if cell_id not in self._real:
            raise KeyError(cell_id)
        m = self._q.meta(self._real[cell_id])
        return CellMeta(branch=m.branch, attempt=m.attempt, parent_id=self._shown.get(m.parent_id) if m.parent_id
                        else None, seq=m.seq, tags=())

    def _view(self, o: Observation) -> Observation:
        # family is not shown: live it depends on when the classifier reached an attempt, and a world holds it as it
        # stood later, so a policy acting on it would act differently in replay than live
        return Observation(id=self._shown[o.id], parent_id=self._shown[o.parent_id] if o.parent_id else None,
                           branch=o.branch, attempt=o.attempt, seq=o.seq, score=o.score, valid=o.valid,
                           fail_class=o.fail_class, family=None)

    def probe_batch(self, cells, on_reveal=None) -> list[Observation | None]:
        q = self._q
        if q._ended:
            raise RecordEnd("the replayed record has ended")
        if q._probing:
            raise IllegalBatch("probe_batch cannot be called from inside on_reveal")
        cells = list(cells)
        if not cells:
            raise IllegalBatch("empty batch")
        if any(type(c) is not str for c in cells):
            raise IllegalBatch("cells must be plain strings")
        if q._max_probes is not None and q.probes >= q._max_probes:
            raise IllegalBatch(f"budget of {q._max_probes} probes is spent")
        if len(set(cells)) != len(cells):
            raise IllegalBatch(f"duplicate cells in batch: {cells}")
        if len(cells) > q.max_parallelism:
            raise IllegalBatch(f"batch of {len(cells)} exceeds max_parallelism {q.max_parallelism}")
        legal = set(self.legal_actions())
        bad = [c for c in cells if c not in legal]
        if bad:
            raise IllegalBatch(f"illegal cells: {bad}")

        def revealed(o: Observation) -> None:
            self._shown[o.id] = f"a{o.seq + 1}"  # reveal order is all a shown id says
            self._real[self._shown[o.id]] = o.id
            if on_reveal:
                on_reveal(self._view(o))
        out = q.probe_batch([c if c.startswith(ROOT) else self._real[c] for c in cells], on_reveal=revealed)
        return [None if o is None else self._view(o) for o in out]
