"""drsi — the command line for Dream-RSI campaigns.

Layer 1 (memory that survives compaction): init, import, fingerprint, families,
map, check, sync, status, list, config.
Layer 2 (the Dream-RSI loop): baseline, run, dream, replay.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .families import (assign_families, assign_new, build_frontier, build_taxonomy, family_stats,
                       load_families, rebuild_families, refresh_frontier)
from .fingerprint import CLASSIFIER_SYSTEM, fingerprint_nodes
from .importer import import_jsonl
from .llm import ClaudeCLI
from .checks import pending_checks, run_check  # noqa: F401 - pending_checks is part of the CLI API
from .novelty import render_check
from .agent import ClaudeAgent
from .dream import SEED_POLICY, run_dream
from .live import WORKER_REPORT_SCHEMA, WORKER_TOOLS, LiveRunner, run_cycles, write_map
from .replay import evaluate_policy
from .store import Campaign, default_home
from .worlds import load_worlds, world_from_tree

# Tests replace these. LLM_FACTORY(cfg, role) -> object with .json(prompt, schema);
# WORKER_FACTORY(camp) -> worker_fn(workspace, prompt, system);
# DEVELOPER_FACTORY(camp) -> developer(sandbox_dir, prompt).
LLM_FACTORY = None
WORKER_FACTORY = None
DEVELOPER_FACTORY = None
DRSI_BIN = str(Path(__file__).resolve().parents[2] / "bin" / "drsi")


def make_llm(cfg: dict, role: str = "classifier"):
    if LLM_FACTORY is not None:
        return LLM_FACTORY(cfg, role)
    model = cfg["llm"].get(f"{role}_model") or cfg["llm"]["model"]
    return ClaudeCLI(model=model, system_prompt=CLASSIFIER_SYSTEM)


def worker_agent(camp: Campaign, workspace, system: str) -> ClaudeAgent:
    """A headless worker confined by Claude Code's Bash sandbox: it can write only its own worktree and
    proposal directory, has no network unless live.allowed_domains names hosts, runs no hooks, and loads
    no user or project settings. Workers never run drsi: the orchestrator checks their proposals."""
    cfg = camp.config
    live = cfg["live"]
    workspace = Path(workspace)
    proposal_dir = camp.root / "work" / "_proposals" / workspace.name
    proposal_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "disableAllHooks": True,
        "sandbox": {
            "enabled": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False,
            # nothing under the clone's .git: git metadata is the orchestrator's (read-only git works)
            "filesystem": {"allowWrite": [str(workspace), str(proposal_dir)]},
            "network": {"allowedDomains": list(live.get("allowed_domains") or [])},
        },
    }
    return ClaudeAgent(model=cfg["llm"].get("worker_model") or cfg["llm"]["model"], tools=WORKER_TOOLS,
                       permission_mode=live["permission_mode"], allowed_tools=list(live.get("allowed_bash") or []),
                       append_system_prompt=system, json_schema=WORKER_REPORT_SCHEMA, timeout=live["timeout_s"],
                       env=cfg["workspace"].get("env") or None, settings=settings, setting_sources="local")


def make_worker(camp: Campaign):
    if WORKER_FACTORY is not None:
        return WORKER_FACTORY(camp)

    def worker(workspace, prompt, system):
        agent = worker_agent(camp, workspace, system)
        return agent.run(workspace, prompt, add_dirs=[camp.root / "work" / "_proposals" / Path(workspace).name])
    return worker


def make_checker(camp: Campaign):
    """The orchestrator's novelty check for a live attempt's proposal."""
    def checker(proposal: str, node: str) -> dict:
        return run_check(camp, proposal, make_llm(camp.config, "judge"), node=node)
    return checker


def developer_agent(camp: Campaign) -> ClaudeAgent:
    """The policy developer may read and edit only its sandbox directory (--restricted confines the file
    tools to the working directory and ignores every settings file), so it cannot read the replay worlds."""
    cfg = camp.config
    return ClaudeAgent(model=cfg["llm"]["model"], tools="Read,Edit,Write", timeout=3600,
                       extra_args=("--restricted",))


def make_developer(camp: Campaign):
    if DEVELOPER_FACTORY is not None:
        return DEVELOPER_FACTORY(camp)

    def developer(sandbox, prompt):
        return developer_agent(camp).run(sandbox, prompt)
    return developer


def make_indexer(camp: Campaign):
    def index(ids):
        cfg = camp.config
        if not ids:
            return
        llm = make_llm(cfg)
        fingerprint_nodes(camp.tree, llm, goal=cfg.get("goal", ""), ids=ids, batch=20, workers=4)
        if camp.families_path.exists():
            assign_new(camp.tree, camp.families_path, llm)
        _write_map(camp)
    return index


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def resolve_campaign(name: str | None) -> Campaign:
    name = name or os.environ.get("DRSI_CAMPAIGN")
    if name:
        return Campaign.open(name)
    root = default_home() / "campaigns"
    found = sorted(p.name for p in root.iterdir() if (p / "campaign.json").exists()) if root.exists() else []
    if len(found) == 1:
        return Campaign.open(found[0])
    raise SystemExit(f"drsi: name a campaign with -c (found: {', '.join(found) or 'none'})")


def _families(camp: Campaign) -> dict | None:
    return load_families(camp.families_path) if camp.families_path.exists() else None


def _write_map(camp: Campaign) -> str:
    return write_map(camp)


def _progress(done, total, err):
    msg = f"  fingerprinted {done}/{total}"
    if err:
        msg += f" (a batch failed: {str(err)[:120]})"
    _err(msg)


# ---- commands ---------------------------------------------------------------------------

def cmd_init(a) -> int:
    cfg = {"goal": a.goal or "", "direction": a.direction}
    if a.scorer:
        cfg["scorer"] = {"cmd": a.scorer}
    if a.repo:
        cfg["workspace"] = {"repo": str(Path(a.repo).expanduser().resolve()), "base": a.base}
    camp = Campaign.create(a.name, cfg)
    print(f"campaign {a.name} created at {camp.root}")
    return 0


def cmd_import(a) -> int:
    camp = resolve_campaign(a.campaign)
    src = str(Path(a.file).expanduser().resolve())
    field_map = json.loads(a.field_map) if a.field_map else None
    tree = camp.tree
    report = {}
    n = import_jsonl(tree, src, preset=a.preset, field_map=field_map, report=report)
    if report.get("skipped") or report.get("duplicates"):
        _err(f"  skipped {report['skipped']} malformed rows and {report['duplicates']} repeated ids")
    entry = {"path": src, "preset": a.preset, **({"field_map": field_map} if field_map else {})}

    def add_source(raw):
        sources = raw.setdefault("sources", [])
        if entry not in sources:
            sources.append(entry)
    camp.update_config(add_source)
    print(f"{n} new attempts imported (total {len(tree)})")
    return 0


def _fingerprint(camp: Campaign, batch: int, workers: int) -> dict:
    cfg = camp.config
    return fingerprint_nodes(camp.tree, make_llm(cfg), goal=cfg.get("goal", ""), batch=batch,
                             workers=workers, progress=_progress)


def cmd_fingerprint(a) -> int:
    if a.batch < 1 or a.workers < 1:
        _err("drsi fingerprint: --batch and --workers must be at least 1")
        return 2
    camp = resolve_campaign(a.campaign)
    st = _fingerprint(camp, a.batch, a.workers)
    print(f"{st['done']} fingerprinted, {st['failed']} failed")
    return 0 if st["failed"] == 0 else 1


def _print_family_table(camp: Campaign) -> None:
    fams = _families(camp)
    for s in family_stats(camp.tree, fams, camp.config["search"]["plateau"]):
        print(f"{s['id']}  {s['status']:<8} n={s['n']:<4} best={s['best'] or '-':<12} {s['name']}"
              f"  | stopped most by: {s['killed_by_top'] or '-'}")


def cmd_families(a) -> int:
    camp = resolve_campaign(a.campaign)
    if a.list:
        if not camp.families_path.exists():
            _err("no families yet; run `drsi families` first")
            return 1
        _print_family_table(camp)
        return 0
    cfg = camp.config
    llm = make_llm(cfg)
    tree = camp.tree
    goal = cfg.get("goal", "")
    if a.rebuild or not camp.families_path.exists():
        fams = rebuild_families(tree, llm, goal, camp.families_path, plateau=cfg["search"]["plateau"])
        assigned = sum(1 for n in camp.tree.nodes() if (n.get("fingerprint") or {}).get("family"))
        print(f"{len(fams['families']) - 1} families built; {assigned} attempts assigned")
    else:
        n = assign_new(tree, camp.families_path, llm)
        print(f"{n} new attempts assigned to existing families")
        if a.frontier:
            refresh_frontier(tree, llm, goal, camp.families_path, plateau=cfg["search"]["plateau"])
            print("frontier rebuilt")
    _write_map(camp)
    return 0


def cmd_map(a) -> int:
    camp = resolve_campaign(a.campaign)
    sys.stdout.write(_write_map(camp))
    return 0


def cmd_check(a) -> int:
    camp = resolve_campaign(a.campaign)
    proposal = Path(a.file).read_text() if a.file else " ".join(a.proposal)
    if not proposal.strip():
        _err("drsi check: give the proposal as text or --file")
        return 2
    result = run_check(camp, proposal, make_llm(camp.config, "judge"), node=a.node)
    print(json.dumps(result, indent=1, ensure_ascii=False) if a.json else render_check(result))
    return result["exit_code"]


def cmd_sync(a) -> int:
    camp = resolve_campaign(a.campaign)
    cfg = camp.config
    added = 0
    for src in cfg.get("sources", []):
        added += import_jsonl(camp.tree, src["path"], preset=src["preset"], field_map=src.get("field_map"))
    st = _fingerprint(camp, 20, 6)  # also retries anything an earlier pass left unfingerprinted
    assigned = 0
    if camp.families_path.exists():
        assigned = assign_new(camp.tree, camp.families_path, make_llm(cfg))
    _write_map(camp)
    print(f"{added} new attempts imported, {st['done']} fingerprinted, {assigned} assigned; map rewritten")
    return 0 if st["failed"] == 0 else 1


def cmd_status(a) -> int:
    camp = resolve_campaign(a.campaign)
    cfg = camp.config
    nodes = camp.tree.nodes()
    fp = sum(1 for n in nodes if (n.get("fingerprint") or {}).get("mechanism"))
    print(f"campaign: {cfg.get('name')}  ({camp.root})")
    print(f"goal: {cfg.get('goal') or '(not stated)'}")
    print(f"attempts: {len(nodes)}  fingerprinted: {fp}")
    if camp.families_path.exists():
        stats = family_stats(camp.tree, load_families(camp.families_path), cfg["search"]["plateau"])
        by = {}
        for s in stats:
            by[s["status"]] = by.get(s["status"], 0) + 1
        print(f"families: {len(stats) - 1}  " + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
    checks = [ln for ln in (camp.checks_path.read_text(errors="replace").splitlines() if camp.checks_path.exists() else [])
              if '"verdict": "claim"' not in ln]
    print(f"novelty checks run: {len(checks)}")
    if nodes:
        print(f"last attempt: #{nodes[-1]['id']} ({(nodes[-1].get('created') or '')[:10]})")
    return 0


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_config(a) -> int:
    camp = resolve_campaign(a.campaign)
    changes = []
    for item in a.set or []:
        key, eq, value = item.partition("=")
        if not eq or not key:
            _err(f"drsi config: --set needs key=value (got {item!r})")
            return 2
        changes.append((key.split("."), _parse_value(value)))

    def apply(raw):
        for parts, value in changes:
            node = raw
            for part in parts[:-1]:
                child = node.setdefault(part, {})
                if not isinstance(child, dict):
                    raise SystemExit(f"drsi config: {part} is a value, not a section")
                node = child
            node[parts[-1]] = value
    if changes:
        camp.update_config(apply)
    print(json.dumps(camp.config, indent=1))
    return 0


def _history_world(camp: Campaign) -> dict:
    ids = {n["id"] for n in camp.tree.nodes() if n.get("source") == "import"}
    return world_from_tree(camp.tree, "history", ids=ids)


def _worlds(camp: Campaign, history: bool) -> list[dict]:
    worlds = load_worlds(camp.root / "trace_pool")
    if history:
        worlds.append(_history_world(camp))
    return [w for w in worlds if w["nodes"]]


def _policy_path(camp: Campaign) -> Path:
    path = camp.root / "policy" / "method.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SEED_POLICY.read_text())
    return path


def _live_ready(camp: Campaign) -> str | None:
    cfg = camp.config
    if not cfg["scorer"].get("cmd"):
        return "scorer.cmd is not set (drsi config -c NAME --set scorer.cmd='\"...\"')"
    if not cfg["workspace"].get("repo"):
        return "workspace.repo is not set (the repository the campaign clones)"
    if not cfg["workspace"].get("mutable"):
        return "workspace.mutable is empty: list the paths workers may edit (e.g. --set workspace.mutable='[\"src/**\"]')"
    return None


def cmd_baseline(a) -> int:
    camp = resolve_campaign(a.campaign)
    problem = _live_ready(camp)
    if problem:
        _err(f"drsi baseline: {problem}")
        return 2
    lock = _run_lock(camp)  # noqa: F841 - held for the life of the command
    if a.remeasure:
        camp.update_config(lambda raw: (raw.pop("baseline", None), raw.pop("baseline_raw", None)))
    runner = LiveRunner(camp, None, None, "baseline")
    print(f"baseline: {runner.ensure_baseline()}")
    return 0


def cmd_replay(a) -> int:
    camp = resolve_campaign(a.campaign)
    worlds = _worlds(camp, a.history)
    if not worlds:
        print("no replay worlds yet: run `drsi run` first (or pass --history to replay the imported record)")
        return 1
    cfg = camp.config
    W = cfg["search"]["W"]
    rep = evaluate_policy(Path(a.policy) if a.policy else _policy_path(camp), worlds, W=W,
                          betas=cfg["dream"]["betas"], budget=cfg["search"]["K1"] * W, lam=cfg["dream"]["lambda"],
                          beta1=cfg["dream"]["beta1"], beta2=cfg["dream"]["beta2"])
    if not rep.get("ok"):
        print(f"policy failed at {rep.get('stage')}: {rep.get('error')}")
        return 1
    print(f"worlds: {len(worlds)}  reward: {rep['reward']:.4f}  (AUC {rep['auc']:.4f}, "
          f"parallel penalty {rep['parallel_penalty']:.4f})")
    for b, r in sorted(rep["per_beta"].items(), key=lambda x: float(x[0])):
        print(f"  beta {b}: attainment {r['attainment']:.3f}  work {r['work']:.3f}  mean batch {r['mean_batch']:.2f}")
    return 0


def cmd_dream(a) -> int:
    camp = resolve_campaign(a.campaign)
    worlds = _worlds(camp, a.history)
    if not worlds:
        print("no replay worlds yet: run `drsi run` first (or pass --history)")
        return 1
    _policy_path(camp)
    d = run_dream(camp.root / "policy", worlds, make_developer(camp), camp.config, camp.root / "logs")
    for r in d["revisions"]:
        extra = f" reward {r['reward']:.4f}" if r.get("reward") is not None else ""
        print(f"  revision {r['m']}: {r['stage']}{extra}{' — ' + r['error'][:160] if r.get('error') else ''}")
    verdict = f"deployed {d['version']}" if d["deployed"] else "kept the incumbent"
    print(f"dream {verdict} (reward {d['incumbent_reward']:.4f} -> {d['best_reward']:.4f})")
    return 0


def _run_lock(camp: Campaign):
    """One live run (or baseline) per campaign at a time: they share one clone and one tree."""
    import fcntl
    path = camp.root / "logs" / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit("drsi: another `drsi run` or `drsi baseline` is active on this campaign")
    return fh


def cmd_run(a) -> int:
    camp = resolve_campaign(a.campaign)
    lock = _run_lock(camp)  # noqa: F841 - held for the life of the command
    problem = _live_ready(camp)
    if problem:
        _err(f"drsi run: {problem}")
        return 2
    rep = run_cycles(camp, a.rounds, worker_fn=make_worker(camp), developer=make_developer(camp),
                     indexer=make_indexer(camp), checker=make_checker(camp),
                     history_world=_history_world(camp) if a.history else None, progress=print)
    for r in rep["rounds"]:
        print(f"{r['round_id']}: {r['attempts']} attempts ({r['valid']} valid) best {r['best_score']} "
              f"baseline {r['baseline']}; dream {'deployed ' + r['dream']['version'] if r['dream']['deployed'] else 'kept the incumbent'}")
    return 0


def cmd_gate(a) -> int:
    """Hook helper: exit 0 only if a non-duplicate novelty check ran within the window."""
    import calendar
    import time as _time
    camp = resolve_campaign(a.campaign)
    now = _time.time()
    ok = False
    if camp.checks_path.exists():
        for line in camp.checks_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
                when = calendar.timegm(_time.strptime(row.get("checked", ""), "%Y-%m-%dT%H:%M:%SZ"))
            except (json.JSONDecodeError, ValueError):
                continue
            if row.get("verdict") in ("novel", "variant") and now - when <= a.max_age_hours * 3600:
                ok = True
    if not ok:
        print(f"drsi gate: no passing novelty check in the last {a.max_age_hours:g}h for campaign "
              f"{camp.config.get('name')}. Run `drsi check -c {camp.config.get('name')} --file <proposal>` first.",
              file=sys.stderr)
        return 2
    return 0


def cmd_list(a) -> int:
    root = default_home() / "campaigns"
    for p in sorted(root.iterdir()) if root.exists() else []:
        if (p / "campaign.json").exists():
            print(p.name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drsi", description="Dream-RSI campaigns for Claude Code")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_c(sp):
        sp.add_argument("-c", "--campaign", help="campaign name or path (default: $DRSI_CAMPAIGN or the only one)")
        return sp

    s = sub.add_parser("init", help="create a campaign")
    s.add_argument("name")
    s.add_argument("--goal", default="")
    s.add_argument("--direction", choices=["max", "min"], default="max")
    s.add_argument("--scorer", help="scorer command (Layer 2)")
    s.add_argument("--repo", help="code repository the live loop clones (Layer 2)")
    s.add_argument("--base", help="base revision for the clone (Layer 2)")
    s.set_defaults(fn=cmd_init)

    s = with_c(sub.add_parser("import", help="import an attempt log (incremental)"))
    s.add_argument("file")
    s.add_argument("--preset", default="attempt-ledger", choices=["attempt-ledger", "generic"])
    s.add_argument("--field-map", help='JSON, for --preset generic: {"id":..,"parent":..,"proposal":..,"text":[..]}')
    s.set_defaults(fn=cmd_import)

    s = with_c(sub.add_parser("fingerprint", help="fingerprint attempts that lack one"))
    s.add_argument("--batch", type=int, default=20)
    s.add_argument("--workers", type=int, default=6)
    s.set_defaults(fn=cmd_fingerprint)

    s = with_c(sub.add_parser("families", help="build or update approach families"))
    s.add_argument("--rebuild", action="store_true", help="rebuild the taxonomy from scratch")
    s.add_argument("--frontier", action="store_true", help="regenerate untried-direction suggestions")
    s.add_argument("--list", action="store_true", help="print the family table")
    s.set_defaults(fn=cmd_families)

    s = with_c(sub.add_parser("map", help="print (and rewrite) the search map"))
    s.set_defaults(fn=cmd_map)

    s = with_c(sub.add_parser("check", help="novelty check a proposal; exit 0 novel, 3 variant, 4 duplicate"))
    s.add_argument("proposal", nargs="*")
    s.add_argument("--file")
    s.add_argument("--json", action="store_true")
    s.add_argument("--node", help="live attempt id this check is for (binds the ticket to that attempt)")
    s.set_defaults(fn=cmd_check)

    s = with_c(sub.add_parser("sync", help="re-import sources, index new attempts, rewrite the map"))
    s.set_defaults(fn=cmd_sync)

    s = with_c(sub.add_parser("status", help="campaign summary"))
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("list", help="list campaigns")
    s.set_defaults(fn=cmd_list)

    s = with_c(sub.add_parser("config", help="show or change campaign settings (--set a.b=JSON)"))
    s.add_argument("--set", action="append", help='e.g. --set search.W=4 --set workspace.mutable=\'["src/**"]\'')
    s.set_defaults(fn=cmd_config)

    s = with_c(sub.add_parser("baseline", help="score the untouched base once (stored in the campaign)"))
    s.add_argument("--remeasure", action="store_true")
    s.set_defaults(fn=cmd_baseline)

    s = with_c(sub.add_parser("run", help="Dream-RSI cycles: live round, then dream, repeated"))
    s.add_argument("--rounds", type=int, default=1)
    s.add_argument("--history", action="store_true", help="also dream over the imported record")
    s.set_defaults(fn=cmd_run)

    s = with_c(sub.add_parser("dream", help="improve the policy by replay over the frozen worlds"))
    s.add_argument("--history", action="store_true")
    s.set_defaults(fn=cmd_dream)

    s = with_c(sub.add_parser("gate", help="hook helper: exit 2 unless a passing novelty check ran recently"))
    s.add_argument("--max-age-hours", type=float, default=6.0)
    s.set_defaults(fn=cmd_gate)

    s = with_c(sub.add_parser("replay", help="score a policy by replay (default: the deployed one)"))
    s.add_argument("--policy")
    s.add_argument("--history", action="store_true")
    s.set_defaults(fn=cmd_replay)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
