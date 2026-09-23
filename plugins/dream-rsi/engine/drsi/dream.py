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
from .replay import evaluate_policy

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
            "beta1": d["beta1"], "beta2": d["beta2"]}


def render_report(rep: dict, revisions: list[dict]) -> str:
    lines = ["# Replay report for the current best policy", ""]
    if rep.get("ok"):
        lines += [f"reward = {rep['reward']:.4f} (pareto AUC {rep['auc']:.4f} - lambda * parallel penalty "
                  f"{rep['parallel_penalty']:.4f})", "", "| beta | attainment | work | mean batch | eq1 |",
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

How the policy is scored: the evaluator replays it on frozen discovery trees at beta in {d['betas']}. After
every reveal it records (work so far, attainment so far). For each beta the attainment-vs-work curve is averaged
over the trees; the Pareto frontier over those per-beta curves is integrated over work in [0, 1] (AUC), and
reward = AUC - {d['lambda']} * parallel_penalty. One beta applies to every tree, so beta cannot be tuned per tree.
attainment = how close the best revealed score gets to that tree's best (0..1); work = fraction of the tree's
recorded attempts revealed; parallel_penalty = 1 - mean batch size / max_parallelism. Higher is better:
reveal the attempts that turn out best as early as possible, using full parallel batches. `self.beta` in
[0, 1] is your knob: low beta should mean cheap, high beta thorough. Each tree gets a budget of
{cfg['search']['K1']} x max_parallelism probes.

Prefix-only: decide only from what the question API reveals, `self.beta`, and your own bookkeeping.
Never use unrevealed scores, hardcoded cell ids, tree-specific constants or absolute score targets.
Imports: only `from <module> import <name>` with <module> one of {', '.join(sorted(ALLOWED_MODULES))} (never
`import <module>`). No attribute starting with an underscore or with co_, f_, tb_, gi_, cr_ or ag_, no
getattr/setattr/eval/exec/open/type/dir/vars/globals/format, no names starting with '__', no decorators, no
':=', no helper classes, no special methods, no bare `except:` or `except BaseException` (catch Exception or
narrower); only helper methods go in the EVOLVE block. The policy must be deterministic: identical
inputs must give identical batches.

{API_NOTES}
./REPORT.md has the current policy's per-beta numbers and the outcome of earlier revisions this phase.
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
            best_src, best_rep, best_reward = new_src, rep, rep["reward"]

    deployed = best_src != inc_src and best_reward >= inc_reward
    version = None
    if deployed:
        n = 1 + max(int(p.stem[1:]) for p in versions.glob("v*.py"))
        version = f"v{n:04d}"
        (versions / f"{version}.py").write_text(best_src)
        method.write_text(best_src)
    report = {"stamp": stamp, "deployed": deployed, "version": version, "incumbent_reward": inc_reward,
              "incumbent_ok": bool(inc.get("ok")), "incumbent_error": inc.get("error"),
              "best_reward": best_reward, "revisions": revisions, "worlds": [w["id"] for w in worlds],
              "best_report": best_rep}
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"dream-{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    return report
