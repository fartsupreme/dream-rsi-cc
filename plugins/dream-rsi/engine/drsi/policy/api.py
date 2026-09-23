"""Policy-side API (mirrors the names in the Dream-RSI paper's Listing 2).

Policies import from here:
    from drsi.policy.api import LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result
"""
from __future__ import annotations


class LLMDesignedMethod:
    """Base class for exploration policies. `beta` in [0, 1] is the one knob the
    evaluator sweeps: low beta should mean cheaper, high beta more thorough."""

    beta = 0.6

    def __init__(self, beta: float | None = None):
        if beta is not None:
            self.beta = float(beta)

    def solve(self, question, budget=None):  # pragma: no cover - abstract
        raise NotImplementedError


class SimResult:
    def __init__(self):
        self.curve: list[tuple[int, float | None]] = []


def _budget_done(question, budget) -> bool:
    if budget is not None and question.probes >= budget:
        return True
    return not question.legal_actions()


def _record_curve(res: SimResult, question) -> None:
    res.curve.append((question.probes, question.best_score()))


def finalize_result(question, res: SimResult) -> dict:
    return {"curve": list(res.curve), "probes": question.probes, "rounds": question.rounds,
            "best": question.best_score(), "batch_sizes": list(question.batch_sizes)}
