"""The dream phase: improve the exploration policy by replay, then redeploy it.

M revisions per phase. Each revision is written by a policy-developer agent that
may edit only the EVOLVE block, is checked by the guard, and is scored by replay
over every frozen world. The best revision is deployed only if its reward is at
least the incumbent's (no regression). Every deployed version is archived.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from .guard import ALLOWED_MODULES
from .replay import _run_once, evaluate_policy, resampled_reward

SEED_POLICY = Path(__file__).resolve().parent / "policy" / "method.py"
START, END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"

API_NOTES = """Question API (all a policy may use):
- question.reset(); question.observed() -> {id: Observation(id, parent_id, branch, attempt, seq, score, valid,
  fail_class, family)}; question.best_score(); question.baseline_score; question.max_parallelism;
  question.probes (nodes revealed so far); question.rounds (batches probed so far)
- question.legal_actions() -> root slots ("root:<j>", open a new branch) + leaves of the revealed tree
- question.legal_roots() -> the next available root slots
- question.meta(cell) -> CellMeta(branch, attempt, parent_id, seq, tags)
- question.probe_batch(cells, on_reveal=...) -> reveals one child per cell. A batch must be non-empty,
  duplicate-free, at most max_parallelism long, and contain only legal actions.
Replay rules: a leaf reveals its recorded child; a leaf with no recorded child reveals nothing (None) and
stops being legal; a root slot reveals the next recorded root. Only leaves and root slots are actions.
"""


def split_evolve(src: str) -> tuple[str, str, str]:
    s = src.find(START)
    e = src.find(END)
    if s < 0 or e < 0 or e < s:
        raise ValueError("policy has no EVOLVE block")
    start = src.rfind("\n", 0, s) + 1
    nl = src.find("\n", e)
    end = len(src) if nl < 0 else nl + 1
    return src[:start], src[start:end], src[end:]


def block_problems(block: str) -> list[str]:
    """The EVOLVE block may only define methods; it may not replace solve() or define special methods."""
    import ast
    import textwrap
    code = "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("#"))
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError as e:
        return [f"EVOLVE block does not parse: {e}"]
    out = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.FunctionDef):
            out.append(f"line {stmt.lineno}: only method definitions belong in the EVOLVE block")
        elif stmt.name == "solve" or stmt.name.startswith("__"):
            out.append(f"line {stmt.lineno}: the EVOLVE block may not define {stmt.name}")
        elif stmt.decorator_list:
            out.append(f"line {stmt.lineno}: decorators not allowed in the EVOLVE block")
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):
            out.append(f"line {node.lineno}: assignment expressions (:=) not allowed in the EVOLVE block")
    return out


def _params(cfg: dict) -> dict:
    W = cfg["search"]["W"]
    d = cfg["dream"]
    return {"W": W, "betas": d["betas"], "budget": cfg["search"]["K1"] * W, "lam": d["lambda"],
            "beta1": d["beta1"], "beta2": d["beta2"], "score": d.get("score", "default"),
            "penalty": d.get("penalty", "support"), "curve": d.get("curve", "canonical")}


def gate_worlds(n: int, W: int, budget: int) -> list[dict]:
    """Synthetic live-like trees for the behaviour gate: more roots than a round can open and branches deeper
    than a round can go, so (as live) no probe ever comes back empty. Scores are seeded: half the trees are
    random walks, half independent draws, with a failure rate that varies per tree."""
    import random
    out = []
    for k in range(n):
        rng = random.Random(f"drsi-gate|{k}")
        p_valid, walk = 0.3 + 0.6 * rng.random(), k % 2 == 0
        nodes = []
        for r in range(budget + W):
            prev, s = None, rng.random()
            for dd in range(budget + 1):
                s = s + rng.gauss(0, 0.1) if walk else rng.random()
                valid = rng.random() < p_valid
                nid = f"g{k}r{r}d{dd}"
                nodes.append({"id": nid, "parent": prev, "score": s if valid else None, "valid": valid,
                              "fail_class": "ok" if valid or rng.random() < 0.5 else "agent_error",
                              "family": "ABCDEF"[rng.randrange(6)]})
                prev = nid
        out.append({"id": f"gate{k}", "baseline": 0.0, "nodes": nodes})
    return out


def behaviour_differs(cand_path, inc_path, W: int, budget: int, n: int = 32, timeout: int = 120) -> dict:
    """Run both policies at their default beta on live-like trees; True when any batch differs as a set. A
    revision that changes nothing live (only what happens when recorded roots or branches run out) must not
    be deployed on its replay gain."""
    worlds = gate_worlds(n, W, budget)
    traces = {}
    with tempfile.TemporaryDirectory(prefix="drsi-gate-") as tmp:
        for label, path in (("cand", cand_path), ("inc", inc_path)):
            res = _run_once({"policy": str(Path(path).resolve()), "worlds": worlds, "W": W, "betas": [],
                             "budget": budget}, Path(tmp), 1, timeout)
            if not res.get("ok"):
                return {"ok": False, "error": f"{label} fails on live-like trees: {res.get('error')}"}
            rows = res["runs"][str(float(res["default_beta"]))]
            traces[label] = [[sorted(b) for b in row["trace"]] for row in rows]
    same = sum(a == b for a, b in zip(traces["cand"], traces["inc"]))
    return {"ok": True, "differs": same < n, "identical_rounds": same, "rounds": n}


def deploy_checks(cand_path, inc_path, cand_rep: dict, inc_rep: dict, worlds: list[dict], params: dict,
                  d: dict) -> dict:
    """The candidate must beat the incumbent beyond resampling noise (paired bootstrap across worlds: the 5th
    percentile of the reward difference must exceed dream.margin) and must change live behaviour."""
    import random
    B, margin = int(d.get("bootstrap", 500)), float(d.get("margin", 0.0))
    kw = {k: params[k] for k in ("W", "lam", "beta1", "beta2", "score", "penalty", "curve")}
    out: dict = {"bootstrap": B, "margin": margin}
    if B > 0:
        n, rng = len(worlds), random.Random(0)
        diffs = []
        for _ in range(B):
            idx = [rng.randrange(n) for _ in range(n)]
            diffs.append(resampled_reward(cand_rep["measured"], worlds, idx, **kw)
                         - resampled_reward(inc_rep["measured"], worlds, idx, **kw))
        diffs.sort()
        out["bootstrap_p05"] = diffs[int(0.05 * B)]
        if not out["bootstrap_p05"] > margin:
            return out | {"ok": False, "why": f"gain within resampling noise (5th percentile {out['bootstrap_p05']:+.4f})"}
    if d.get("behaviour_gate", True):
        g = behaviour_differs(cand_path, inc_path, params["W"], params["budget"], int(d.get("gate_worlds", 32)))
        out["gate"] = g
        if not g.get("ok"):
            return out | {"ok": False, "why": g.get("error")}
        if not g["differs"]:
            return out | {"ok": False, "why": "no change in live behaviour (identical batches on live-like trees)"}
    return out | {"ok": True}


def _strip(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if k not in ("measured", "traces")}


def render_report(rep: dict, revisions: list[dict]) -> str:
    lines = ["# Replay report for the current best policy", ""]
    if rep.get("ok"):
        lines += [f"reward = {rep['reward']:.4f} (AUC {rep['auc']:.4f} at beta {', '.join(rep.get('scored_betas', []))}"
                  f" - lambda * parallel penalty {rep['parallel_penalty']:.4f})", "",
                  "Beta sweep (Pareto view; diagnostic, only the scored beta counts):", "",
                  "| beta | attainment | work | mean batch | eq1 |",
                  "|---|---|---|---|---|"]
        for b, r in sorted(rep["per_beta"].items(), key=lambda x: float(x[0])):
            lines.append(f"| {b} | {r['attainment']:.3f} | {r['work']:.3f} | {r['mean_batch']:.2f} | {r['eq1']:.3f} |")
    else:
        lines.append(f"The current policy failed evaluation: {rep.get('error')}")
    if revisions:
        lines += ["", "## Earlier revisions this phase", ""]
        lines += [f"- m={r['m']}: {r['stage']}" + (f", reward {r['reward']:.4f}" if r.get("reward") is not None else "")
                  + (f" — {r['error'][:200]}" if r.get("error") else "") for r in revisions]
    return "\n".join(lines) + "\n"


def build_prompt(cfg: dict) -> str:
    d = cfg["dream"]
    return f"""You are improving one prefix-only exploration policy. Edit only ./method.py, and only the lines
between `{START}` and `{END}`. Everything outside the block must stay byte-identical.

How the policy is scored: the evaluator replays it on frozen discovery trees at its class default beta (the
beta that runs live) and, for the report only, at beta in {d['betas']}. After every reveal it records (work so far,
attainment so far), crediting the cells of one batch in a fixed order (they run in parallel live, so the order
you list them in does not count); these curves are averaged over the trees and integrated over work in [0, 1]
(AUC), and
reward = AUC - {d['lambda']} * parallel_penalty. attainment = how close the best revealed score gets to that
tree's best (0..1); work = fraction of the tree's recorded attempts revealed; parallel_penalty = 1 - mean batch
fill, where a batch is full when it probes every cell the record can answer, up to max_parallelism. Running out
of recorded roots or reaching the end of a recorded branch is not charged (live has neither), and padding a
batch earns nothing. Higher is better: reveal the attempts that turn out best as early as possible, using full
parallel batches. `self.beta` in [0, 1] is your knob: low beta should mean cheap, high beta thorough. Each tree
gets a budget of {cfg['search']['K1']} x max_parallelism probes.

Stopping: returning [] ends the rollout. Stopping early earns nothing (the curve stays flat after the stop), and
the trees hold only what earlier policies explored, so their ceiling is not a live stopping signal. Stop only
after weighing every open branch, unopened root and repairable failure.
Deployment: a revision replaces the incumbent only if it beats it on a paired bootstrap across the trees and it
changes what the policy does in a live-like round (unbounded roots, no branch ends). A change that only acts
when recorded roots or branches run out is not deployed.

Prefix-only: decide only from what the question API reveals, `self.beta`, and your own bookkeeping.
Never use unrevealed scores, hardcoded cell ids, tree-specific constants or absolute score targets.
Imports: only `from <module> import <name>` with <module> one of {', '.join(sorted(ALLOWED_MODULES))} (never
`import <module>`). No attribute starting with an underscore or with co_, f_, tb_, gi_, cr_ or ag_, no
getattr/setattr/eval/exec/open/type/dir/vars/globals/format, no names starting with '__', no decorators, no
':=', no helper classes, no special methods, no bare `except:` or `except BaseException` (catch Exception or
narrower); only helper methods go in the EVOLVE block. The policy must be deterministic: identical
inputs must give identical batches.

{API_NOTES}
./REPORT.md has the current policy's reward, its per-beta sweep (a Pareto view, diagnostic only) and the
outcome of earlier revisions this phase.
Make one focused change you expect to raise the reward. Put a comment of at most three lines at the top of
the EVOLVE block saying what changed and why. When the file is saved, stop.
"""


def run_dream(policy_dir, worlds: list[dict], developer, cfg: dict, log_dir) -> dict:
    """One dream phase at a time per policy directory: a concurrent phase would compare against a stale
    incumbent and could overwrite a better policy deployed moments earlier."""
    import fcntl
    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    with open(policy_dir / "dream.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_dream(policy_dir, worlds, developer, cfg, log_dir)


def _run_dream(policy_dir, worlds: list[dict], developer, cfg: dict, log_dir) -> dict:
    policy_dir, log_dir = Path(policy_dir), Path(log_dir)
    method = policy_dir / "method.py"
    if not method.exists():
        policy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(SEED_POLICY, method)
    versions = policy_dir / "versions"
    versions.mkdir(exist_ok=True)
    if not any(versions.glob("v*.py")):
        shutil.copy(method, versions / "v0000.py")
    candidates = policy_dir / "candidates"
    candidates.mkdir(exist_ok=True)
    params = _params(cfg)
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{int(now * 1e6) % 1_000_000:06d}Z"

    inc_src = method.read_text()
    inc = evaluate_policy(method, worlds, **params)
    inc_reward = inc["reward"] if inc.get("ok") else float("-inf")
    best_src, best_rep, best_reward = inc_src, inc, inc_reward
    inc_parts = split_evolve(inc_src)
    revisions: list[dict] = []

    best_path = method
    for m in range(cfg["dream"]["M"]):
        with tempfile.TemporaryDirectory(prefix="drsi-dream-") as sb:
            sb = Path(sb)
            (sb / "method.py").write_text(best_src)
            (sb / "REPORT.md").write_text(render_report(best_rep, revisions))
            res = developer(sb, build_prompt(cfg))
            new_src = (sb / "method.py").read_text()
        if not res.ok:
            revisions.append({"m": m, "stage": "agent", "error": res.error})
            continue
        if new_src == best_src:
            revisions.append({"m": m, "stage": "unchanged"})
            continue
        try:
            before, block, after = split_evolve(new_src)
        except ValueError as e:
            revisions.append({"m": m, "stage": "scope", "error": str(e)})
            continue
        if (before, after) != (inc_parts[0], inc_parts[2]):
            revisions.append({"m": m, "stage": "scope", "error": "edited outside the EVOLVE block"})
            continue
        bad = block_problems(block)
        if bad:
            revisions.append({"m": m, "stage": "scope", "error": "; ".join(bad)})
            continue
        cand = candidates / f"{stamp}_m{m}.py"
        cand.write_text(new_src)
        rep = evaluate_policy(cand, worlds, **params)
        if not rep.get("ok"):
            revisions.append({"m": m, "stage": rep.get("stage", "run"), "error": rep.get("error", ""),
                              "path": str(cand)})
            continue
        revisions.append({"m": m, "stage": "scored", "reward": rep["reward"], "path": str(cand)})
        if rep["reward"] > best_reward + 1e-12:
            best_src, best_rep, best_reward, best_path = new_src, rep, rep["reward"], cand

    deployed = best_src != inc_src and best_reward > inc_reward  # a tie keeps the incumbent
    checks = None
    if deployed and inc.get("ok"):
        checks = deploy_checks(best_path, method, best_rep, inc, worlds, params, cfg["dream"])
        deployed = checks["ok"]
    version = None
    if deployed:
        n = 1 + max(int(p.stem[1:]) for p in versions.glob("v*.py"))
        version = f"v{n:04d}"
        (versions / f"{version}.py").write_text(best_src)
        method.write_text(best_src)
    report = {"stamp": stamp, "deployed": deployed, "version": version, "incumbent_reward": inc_reward,
              "incumbent_ok": bool(inc.get("ok")), "incumbent_error": inc.get("error"),
              "best_reward": best_reward, "revisions": revisions, "worlds": [w["id"] for w in worlds],
              "best_report": _strip(best_rep), "deploy_checks": checks}
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"dream-{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    return report
