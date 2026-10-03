"""The dream phase: improve the exploration policy by replay, then redeploy it.

M revisions per phase. Each revision is written by a policy-developer agent that
may edit only the EVOLVE block, is checked by the guard, and is scored by replay
over the frozen worlds on which the incumbent's replay stays on the record (there
its replay is exactly what it did live). The best revision is deployed only if its
reward beats the incumbent's beyond resampling noise and it changes live behaviour
without doing less work (deploy_checks). Every deployed version is archived.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from .agent import LIMIT_MAX_CUT_OFFS, limit_wait, stopped_by_limit, wait_unless_stopping
from .guard import ALLOWED_MODULES
from .question import POLICY_HASH_SEED
from .replay import CURVES, SCORES, _run_once, evaluate_policy, resampled_reward
from .worlds import comparable, informative

SEED_POLICY = Path(__file__).resolve().parent / "policy" / "method.py"
START, END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"

API_NOTES = """Question API (all a policy may use):
- question.reset(); question.observed() -> {id: Observation(id, parent_id, branch, attempt, seq, score, valid,
  fail_class, family)}; question.best_score(); question.baseline_score; question.max_parallelism;
  question.probes (probes spent so far; each revealed one attempt); question.rounds (batches probed so far).
  Attempt ids ("a1", "a2", ...) say only the order attempts were revealed in, and family is always None (live it
  depends on when an attempt was classified); the question is the same in replay and in a live round.
- question.legal_actions() -> root slots ("root:<j>", open a new branch) + leaves of the revealed tree
- question.legal_roots() -> the next available root slots
- question.meta(cell) -> CellMeta(branch, attempt, parent_id, seq, tags)
- question.probe_batch(cells, on_reveal=...) -> reveals one child per cell. A batch must be non-empty,
  duplicate-free, at most max_parallelism long, and contain only legal actions.
Replay rules: a root slot reveals the recorded root opened in that slot and a leaf its recorded child. Past the
record (a root slot beyond the recorded roots, a leaf beyond the end of its recorded branch) that tree's replay
ends: the probe reveals nothing, and the rest of the budget counts as empty batches, since replay cannot know what
that work would have found. Only leaves and root slots are actions.
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


REVISION_FLOOR_S = 120  # a revision's evaluation may take REVISION_SLOWDOWN x the incumbent's, and at least this
REVISION_SLOWDOWN = 4


def _params(cfg: dict) -> dict:
    """The replay ranking a campaign's config asks for. The parallel penalty is always live fill: a campaign created
    while "support" was the default (or set to "realized") still stores it, and ranks by "live" (see config_warnings)."""
    W = cfg["search"]["W"]
    d = cfg["dream"]
    score, curve = d.get("score", "default"), d.get("curve", "canonical")
    return {"W": W, "betas": d["betas"], "budget": cfg["search"]["K1"] * W, "lam": d["lambda"],
            "beta1": d["beta1"], "beta2": d["beta2"], "score": score if score in SCORES else "default",
            "penalty": "live", "curve": curve if curve in CURVES else "canonical"}


def config_warnings(cfg: dict) -> list[str]:
    d, out = cfg["dream"], []
    p = d.get("penalty", "live")
    if p != "live":
        out.append(f"dream.penalty = {p!r} is no longer supported (it paid for gains that do not exist live); ranking "
                   "by 'live'. Set it with: drsi config --set dream.penalty=live")
    for key, allowed, default in (("score", SCORES, "default"), ("curve", CURVES, "canonical")):
        v = d.get(key, default)
        if v not in allowed:
            out.append(f"dream.{key} = {v!r} is not one of {', '.join(allowed)}; ranking by {default!r}")
        elif v != default:
            out.append(f"dream.{key} = {v!r}: a deployed revision is shown to do better live on the same attempts only "
                       f"under the defaults (score 'default', curve 'canonical'); {v!r} can reward behaviour a live "
                       "round never shows (betas that never run, or the order a batch is listed in)")
    return out


def gate_worlds(n: int, W: int, budget: int, baseline: float = 0.0, lo: float = 0.0, hi: float = 1.0,
                families: list[str] | None = None, fail_classes: list[str] | None = None,
                valid_classes: list[str] | None = None) -> list[dict]:
    """Synthetic live-like trees for the behaviour gate: more roots than a round can open and branches deeper
    than a round can go, so (as live) the record always answers. Scores are seeded: half the trees are random
    walks (reflected at the bounds), half independent draws, with a failure rate that varies per tree, placed in
    [lo, hi] over the campaign's baseline, with the campaign's family names and the classes its valid and failed
    attempts carry, so they read like the campaign's own."""
    import random
    names = list(families) if families else [None]
    fails = list(fail_classes) if fail_classes else ["agent_error", "eval_error"]  # what a worker or scorer failure is
    oks = list(valid_classes) if valid_classes else ["ok"]
    out = []
    for k in range(n):
        rng = random.Random(f"drsi-gate|{k}")
        p_valid, walk = 0.3 + 0.6 * rng.random(), k % 2 == 0
        nodes = []
        for r in range(budget + W):
            prev, s = None, rng.random()
            for dd in range(budget + 1):
                s = s + rng.gauss(0, 0.1) if walk else rng.random()
                while not 0.0 <= s <= 1.0:
                    s = -s if s < 0 else 2.0 - s
                valid = rng.random() < p_valid
                nid = f"g{k}r{r}d{dd}"
                nodes.append({"id": nid, "parent": prev, "score": lo + (hi - lo) * s if valid else None, "valid": valid,
                              "fail_class": (oks[0] if len(oks) == 1 else rng.choice(oks)) if valid
                              else rng.choice(fails),
                              "family": rng.choice(names)})
                prev = nid
        out.append({"id": f"gate{k}", "baseline": baseline, "nodes": nodes})
    return out


def behaviour_differs(cand_path, inc_path, W: int, budget: int, n: int = 32, timeout: int = 120,
                      worlds: list[dict] | None = None) -> dict:
    """Run both policies at their default beta on live-like trees; True when any batch differs as a set. A
    revision that changes nothing live (only what happens when recorded roots or branches run out) must not
    be deployed on its replay gain. Also returns each policy's total probes there, since a revision that spends less
    of a live round than the incumbent cannot be justified by replay (see deploy_checks)."""
    if n < 1:
        return {"ok": False, "error": "dream.gate_worlds must be at least 1 while dream.behaviour_gate is on"}
    base, lo, hi = _campaign_scale(worlds or [])
    names = sorted({x["family"] for w in (worlds or []) for x in w["nodes"] if x.get("family")}) or None
    fails = sorted({x["fail_class"] for w in (worlds or []) for x in w["nodes"]
                    if not x.get("valid", True) and x.get("fail_class")}) or None
    oks = sorted({x["fail_class"] for w in (worlds or []) for x in w["nodes"]
                  if x.get("valid", True) and x.get("fail_class")}) or None
    worlds = gate_worlds(n, W, budget, base, lo, hi, families=names, fail_classes=fails, valid_classes=oks)
    traces, probes = {}, {}
    with tempfile.TemporaryDirectory(prefix="drsi-gate-") as tmp:
        for label, path in (("cand", cand_path), ("inc", inc_path)):
            res = _run_once({"policy": str(Path(path).resolve()), "worlds": worlds, "W": W, "betas": [],
                             "budget": budget}, Path(tmp), POLICY_HASH_SEED, timeout)
            if not res.get("ok"):
                return {"ok": False, "error": f"{label} fails on live-like trees: {res.get('error')}"}
            rows = res["runs"][str(float(res["default_beta"]))]
            traces[label] = [[sorted(b) for b in row["trace"]] for row in rows]
            probes[label] = sum(len(b) for row in rows for b in row["trace"])
    same = sum(a == b for a, b in zip(traces["cand"], traces["inc"]))
    return {"ok": True, "differs": same < n, "identical_rounds": same, "rounds": n,
            "probes": probes["cand"], "incumbent_probes": probes["inc"]}


def _campaign_scale(worlds: list[dict]) -> tuple[float, float, float]:
    """The replay worlds' most common baseline and the range of their valid scores, for the gate's trees."""
    from collections import Counter
    bases = Counter(w.get("baseline", 0.0) for w in worlds)
    vals = [n["score"] for w in worlds for n in w["nodes"] if n.get("valid") and n.get("score") is not None]
    base = bases.most_common(1)[0][0] if bases else 0.0
    return (base, min(vals), max(vals)) if len(vals) > 1 and min(vals) < max(vals) else (base, 0.0, 1.0)


def deploy_checks(cand_path, inc_path, cand_rep: dict, inc_rep: dict, worlds: list[dict], params: dict,
                  d: dict) -> dict:
    """The candidate must beat the incumbent beyond resampling noise (paired bootstrap across worlds: the 5th
    percentile of the reward difference must exceed dream.margin), must change live behaviour, and must not spend
    less of a live round than the incumbent: replay can only reveal what was recorded, so it cannot see what the
    work a revision gives up would have found."""
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
        g = behaviour_differs(cand_path, inc_path, params["W"], params["budget"], int(d.get("gate_worlds", 32)),
                              worlds=worlds)
        out["gate"] = g
        if not g.get("ok"):
            return out | {"ok": False, "why": g.get("error")}
        if not g["differs"]:
            return out | {"ok": False, "why": "no change in live behaviour (identical batches on live-like trees)"}
        if g["probes"] < g["incumbent_probes"]:
            return out | {"ok": False, "why": f"less work live: {g['probes']} probes against the incumbent's "
                                               f"{g['incumbent_probes']} on live-like trees, and replay cannot value "
                                               "work beyond the record"}
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


PENALTY_TEXT = ("parallel_penalty = 1 - mean batch fill, where a batch's fill is the cells it probes out of "
                "max_parallelism: live every probe is an attempt. Root slots never run out, as live; a probe past the "
                "tree's record (a root slot beyond its recorded roots, a leaf beyond the end of its recorded branch) "
                "ends that tree's replay, and the budget left counts as empty batches.")


def build_prompt(cfg: dict) -> str:
    d = cfg["dream"]
    return f"""You are improving one prefix-only exploration policy. Edit only ./method.py, and only the lines
between `{START}` and `{END}`. Everything outside the block must stay byte-identical.

How the policy is scored: the evaluator replays it on frozen discovery trees at its class default beta (the
beta that runs live) and, for the report only, at beta in {d['betas']}. After every reveal it records (work so far,
attainment so far), crediting the cells of one batch worst first, best last (they run in parallel live, so the
order you list them in does not count); these curves are averaged over the trees and integrated over work in [0, 1]
(AUC), and
reward = AUC - {d['lambda']} * parallel_penalty. attainment = how close the best revealed score gets to that
tree's best (0..1); work = probes spent over the tree's recorded attempts; {PENALTY_TEXT} Higher is better: reveal the attempts that turn out best as early as possible, using full
parallel batches. `self.beta` in [0, 1] is your knob: low beta should mean cheap, high beta thorough. Each tree
gets a budget of {cfg['search']['K1']} x max_parallelism probes.

Stopping: returning [] ends the rollout. Stopping early earns nothing (the curve stays flat after the stop), and
each batch the budget had left counts as an empty batch in parallel_penalty.
The trees hold only what earlier policies explored, so their ceiling is not a live stopping signal. Stop only
after weighing every open branch, unopened root and repairable failure.
Deployment: the trees are the rounds the incumbent's replay reproduces exactly (its own); a revision replaces the
incumbent only if it beats it on a paired bootstrap across them and it changes what the policy does in a live-like
round, without spending fewer probes there than the incumbent. Work the record lacks ends the replay, so a revision
earns reward by reaching the recorded good attempts sooner, not by exploring where nothing was recorded.

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
    worlds_total, on_record = len(worlds), None
    worlds = comparable(worlds)  # live rounds whose every attempt keeps its cell
    compared = len(worlds)
    started = time.monotonic()
    inc = evaluate_policy(method, worlds, **params) if worlds else {"ok": True, "reward": float("-inf"), "measured": {
        "runs": {}}, "default_beta": 0.0}
    inc_time = time.monotonic() - started
    if inc.get("ok"):
        # Replay is what happened only where the record answers. Where the incumbent's replay stays on the record
        # (every round it recorded itself, and any other whose record covers its whole path) its value is exact, and
        # a candidate's run, which ends at its first probe past the record, can only score below what it does live on
        # the same attempts. So policies are compared on those worlds alone: a candidate that beats the incumbent
        # there does better live on the same attempts.
        rows = inc["measured"]["runs"].get(str(float(inc["default_beta"])), [])
        worlds = [w for w, row in zip(worlds, rows) if row["off_record"] == 0]
        on_record = len(worlds)
        if worlds and on_record < compared:  # re-evaluate only if the on-record filter left a world out
            started = time.monotonic()
            inc = evaluate_policy(method, worlds, **params)
            inc_time = time.monotonic() - started
    inc_reward = inc["reward"] if inc.get("ok") else float("-inf")
    best_src, best_rep, best_reward = inc_src, inc, inc_reward
    inc_parts = split_evolve(inc_src)
    revisions: list[dict] = []

    best_path = method
    skipped = None
    need = max(1, int(cfg["dream"].get("min_worlds", 4)))  # no world, nothing to compare on
    if not inc.get("ok"):  # nothing can be compared with an incumbent that does not replay: keep it, call no one
        skipped = (f"the incumbent failed replay ({inc.get('stage', 'run')}: {inc.get('error', '')}); it is kept and "
                   "no revision is asked for")
    elif informative(worlds) < need:  # too few to tell policies apart: a dream would spend developer calls on noise
        skipped = (f"{informative(worlds)} of {worlds_total} world(s) recorded live, each attempt with its cell, can "
                   f"inform a dream with the incumbent's replay on the record throughout, fewer than dream.min_worlds "
                   f"= {need}; the incumbent is kept")
    for m in range(0 if skipped else cfg["dream"]["M"]):
        with tempfile.TemporaryDirectory(prefix="drsi-dream-") as sb:
            sb = Path(sb)
            def fresh():  # the sandbox as the developer is given it
                for x in sb.iterdir():
                    shutil.rmtree(x) if x.is_dir() and not x.is_symlink() else x.unlink()
                (sb / "method.py").write_text(best_src)
                (sb / "REPORT.md").write_text(render_report(best_rep, revisions))
            fresh()
            res = developer(sb, build_prompt(cfg))
            stopped_after_work = 0
            while stopped_by_limit(res):  # the limit's, not the revision's: wait, then ask again from a fresh sandbox
                if getattr(res, "cut_off", False):
                    stopped_after_work += 1
                    if stopped_after_work > LIMIT_MAX_CUT_OFFS:  # more than a usage limit does: the revision fails
                        break
                if not wait_unless_stopping(limit_wait(res)):
                    break
                if getattr(res, "cut_off", False):
                    fresh()
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
        # the loop waits for the dream: a revision gets a few times the incumbent's own time, never less than the floor
        rep = evaluate_policy(cand, worlds, **params, total_timeout=max(REVISION_FLOOR_S, REVISION_SLOWDOWN * inc_time))
        if not rep.get("ok"):
            revisions.append({"m": m, "stage": rep.get("stage", "run"), "error": rep.get("error", ""),
                              "path": str(cand)})
            continue
        revisions.append({"m": m, "stage": "scored", "reward": rep["reward"], "path": str(cand)})
        if rep["reward"] > best_reward + 1e-12:
            best_src, best_rep, best_reward, best_path = new_src, rep, rep["reward"], cand

    deployed = not skipped and best_src != inc_src and best_reward > inc_reward  # a tie keeps the incumbent
    checks = None
    if deployed:
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
              "worlds_total": worlds_total, "worlds_on_record": on_record,
              "best_report": _strip(best_rep), "deploy_checks": checks, "skipped": skipped,
              "warnings": config_warnings(cfg)}
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"dream-{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    return report
