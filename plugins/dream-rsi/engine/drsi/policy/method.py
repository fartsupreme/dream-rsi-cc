"""Seed exploration policy (pi_1).

Parallel refinement, as in the paper's initial policy, plus two rules that keep a
search from circling: a branch closes when its recent attempts stop improving on
its own best (plateau rule), and closed branches are replaced by new root slots
so the search stays max_parallelism wide (coverage rule). `beta` in [0, 1] sets
patience: 0 closes branches and stops early, 1 keeps going longer.

Only the EVOLVE block is rewritten by the dream phase.
"""
from drsi.policy.api import LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result


class OptimalPolicy(LLMDesignedMethod):
    beta = 0.6

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        closed = set()
        state = {"stale_rounds": 0}
        while not _budget_done(question, budget):
            batch = self.select_batch(question, closed, state)
            if not batch:
                break
            before = question.best_score()
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
            after = question.best_score()
            if after is not None and (before is None or after > before):
                state["stale_rounds"] = 0
            else:
                state["stale_rounds"] += 1
        return finalize_result(question, res)

    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        W = question.max_parallelism
        patience = 1 + int(round(self.beta * 4))  # attempts without gain before a branch closes
        if state["stale_rounds"] >= 2 * patience:
            return []  # global plateau: stop spending
        by_branch = {}
        for o in sorted(question.observed().values(), key=lambda o: o.seq):
            by_branch.setdefault(o.branch, []).append(o)
        candidates = []
        for cell in question.legal_actions():
            if cell.startswith("root:") or cell in closed:
                continue
            hist = by_branch.get(question.meta(cell).branch, [])
            scores = [o.score if (o.valid and o.score is not None) else float("-inf") for o in hist]
            if len(scores) > patience and max(scores[-patience:]) <= max(scores[:-patience]):
                closed.add(cell)  # plateau rule
                continue
            candidates.append((max(scores) if scores else float("-inf"), -question.meta(cell).branch, cell))
        candidates.sort(reverse=True)
        batch = [c for _, _, c in candidates[:W]]
        for slot in question.legal_roots():  # coverage rule: keep W branches alive
            if len(batch) >= W:
                break
            batch.append(slot)
        return batch
    # EVOLVE-BLOCK-END
