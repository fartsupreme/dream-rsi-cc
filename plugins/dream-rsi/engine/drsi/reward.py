"""Replay objectives.

Eq. 1 of the paper (per world):  V = max_v s_v - beta1*N + beta2*N/max(1,k)
  quality (here: attainment, the best revealed score normalised to the world),
  cost (N nodes revealed), and a parallelism bonus N/k: the mean number of nodes
  revealed per round (probes that reveal nothing do not count as parallel work).

Policy ranking (Appendix B.2): for each swept beta, the anytime attainment curve
(best attainment reached with at most x of the work) is averaged over the worlds;
the Pareto frontier over those per-beta mean curves is integrated over x in [0, 1]
(mean_curve_auc), minus lambda times the parallel penalty. One beta applies to every
world, so a policy cannot pick its best beta separately per world.
"""
from __future__ import annotations


def attainment(best: float | None, baseline: float, world_best: float | None) -> float:
    if best is None or world_best is None:
        return 0.0
    if world_best <= baseline:
        return 1.0 if best >= world_best else 0.0
    return min(1.0, max(0.0, (best - baseline) / (world_best - baseline)))


def eq1_value(quality: float, n: int, k: int, beta1: float, beta2: float) -> float:
    return quality - beta1 * n + beta2 * n / max(1, k)


def pareto_step_auc(points) -> float:
    """Area under the best-attainment-so-far step curve over work in [0, 1]."""
    pts = sorted((min(1.0, max(0.0, w)), a) for w, a in points)
    frontier: list[tuple[float, float]] = []
    for w, a in pts:
        if not frontier or a > frontier[-1][1]:
            frontier.append((w, a))
    area = 0.0
    for i, (w, a) in enumerate(frontier):
        nxt = frontier[i + 1][0] if i + 1 < len(frontier) else 1.0
        area += (nxt - w) * a
    return area


def parallel_penalty(batch_sizes, W: int) -> float:
    if not batch_sizes:
        return 1.0
    return max(0.0, 1.0 - (sum(batch_sizes) / len(batch_sizes)) / W)


def _step(points, x: float) -> float:
    """Best attainment reached with work <= x on one run's anytime curve."""
    best = 0.0
    for w, a in points:
        if w <= x + 1e-12 and a > best:
            best = a
    return best


def mean_curve_auc(curves_by_beta: dict) -> float:
    """curves_by_beta: beta -> list (one per world) of [(work, attainment), ...] anytime points.
    Returns the area under max over betas of the world-averaged step curves, over work in [0, 1]."""
    xs = {0.0, 1.0}
    for runs in curves_by_beta.values():
        for pts in runs:
            xs.update(min(1.0, max(0.0, w)) for w, _ in pts)
    xs = sorted(xs)
    area = 0.0
    for lo, hi in zip(xs, xs[1:]):
        best = 0.0
        for runs in curves_by_beta.values():
            if runs:
                best = max(best, sum(_step(p, lo) for p in runs) / len(runs))
        area += (hi - lo) * best
    return area
