"""The online phase: the deployed policy drives real attempts, and the tree grows.

Each probe is one attempt in its own git worktree, in two phases:
  propose   - a fresh worker, briefed with the map and its parent attempt, writes a proposal;
              the orchestrator itself checks it against everything tried (up to max_proposals
              tries, feeding the judge's reasons back after a duplicate);
  implement - only an accepted proposal is built.
The orchestrator then commits the workspace, checks that nothing outside
`workspace.mutable` differs from the campaign base, and runs the campaign scorer on
a clean checkout of the commit. Workers never run the check themselves, so it can
be neither skipped nor forged, and the proposal recorded is the one that was judged.
A finished round is frozen into the trace pool as a replay world.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from .agent import (LIMIT_MAX_CUT_OFFS, LIMIT_MIN_WAIT_S, LIMIT_POLL_S, AgentResult, kill_all_children, limit_wait, refuse_if_stopping,
                    register_child, stopped_by_limit, unregister_child, wait_unless_stopping)
from .llm import LLMLimited
from .dream import SEED_POLICY, run_dream
from .families import load_families
from .guard import check_policy_source, safe_builtins
from .mapview import render_map
from . import offload
from .novelty import render_check, text_hash
from .question import POLICY_HASH_SEED, ROOT, IllegalBatch, QuestionBase
from .scorer import run_scorer
from .store import Campaign, _atomic_write, make_node
from .workspace import Workspaces, out_of_scope
from .worlds import comparable, freeze_world, informative, load_worlds, with_cells, world_from_tree

WORKER_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "proposal": {"type": "string"},
        "summary": {"type": "string"},
        "self_reported_score": {"type": ["number", "null"]},
        "notes": {"type": "string"},
    },
    "required": ["proposal", "summary", "self_reported_score", "notes"],
    "additionalProperties": False,
}

WORKER_TOOLS = "Read,Edit,Write,Bash,Glob,Grep"
WORKER_PROMPT = "Begin. Follow the instructions in your system prompt, then return the report."
PROPOSE, IMPLEMENT = "PHASE: PROPOSE", "PHASE: IMPLEMENT"
# fail classes that say nothing about whether the idea works
PROCEDURAL = {"not_novel", "out_of_scope", "agent_error", "orchestrator_error"}


class PolicySource:
    """A deployed policy, checked by the guard. It runs only in a child process (drsi/live_runner.py)."""

    def __init__(self, path, source: str):
        self.path, self.source = str(path), source


def load_policy(path) -> PolicySource:
    src = Path(path).read_text()
    problems = check_policy_source(src)
    if problems:
        raise ValueError(f"policy {path} fails the guard: {problems}")
    return PolicySource(path, src)  # the child runs exactly the text that was checked


def _read_regular(path: Path, cap: int) -> str:
    """The file's text if it is a regular file; a worker-made symlink, FIFO or device is never opened
    in a way that follows or blocks on it."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return ""
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return ""
        return fh.read(cap).decode("utf-8", "replace")


_limit_wait = limit_wait  # the wait for calls refused for usage limits (agent.limit_wait)


def _clear(path: Path) -> None:
    if path.is_symlink() or path.is_file() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _clear_dir(path: Path, keep: tuple = ()) -> None:
    """Empty a directory the worker may write to, keeping the names in `keep`; links are removed, never followed."""
    try:
        entries = list(Path(path).iterdir())
    except FileNotFoundError:
        return
    for x in entries:
        if x.name in keep:
            continue
        if x.is_dir() and not x.is_symlink():
            shutil.rmtree(x, ignore_errors=True)
        else:
            x.unlink(missing_ok=True)


def _clip(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + "…"


def write_map(camp: Campaign) -> str:
    cfg = camp.config
    fams = load_families(camp.families_path) if camp.families_path.exists() else None
    text = render_map(camp.tree, fams, goal=cfg.get("goal", ""), max_chars=cfg["map"]["max_chars"],
                      plateau=cfg["search"]["plateau"])
    _atomic_write(camp.map_path, text)
    return text


@contextmanager
def _file_lock(path: Path, enabled: bool = True):
    if not enabled:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def live_outcome(node: dict, reference: float | None, margin: float = 0.0) -> tuple[str, str]:
    """Outcome from the scorer, never from the worker's report. A pass must beat the reference (the parent's
    score, else the baseline) by more than `margin` (live.pass_margin): a re-measurement of the same code lands
    above its parent about half the time."""
    if not node["valid"]:
        return ("inconclusive" if node["fail_class"] in PROCEDURAL else "killed"), node["fail_class"] or "invalid"
    if reference is None or node["score"] > reference + margin:
        return "pass", ""
    return "measured", ""


class LiveQuestion(QuestionBase):
    """The orchestrator's copy of the round's question: the one whose metrics count."""

    def __init__(self, runner: "LiveRunner", max_parallelism: int, baseline: float, max_probes: int | None = None):
        self._runner = runner
        super().__init__(max_parallelism, baseline, max_probes)

    def _expand(self, cells):
        return self._runner.run_batch(cells)


class LiveRunner:
    def __init__(self, camp: Campaign, worker_fn, indexer, round_id: str,
                 workspaces: Workspaces | None = None, log=None, checker=None):
        """checker(proposal, node_id) -> novelty result; required when live.require_check is on."""
        self.camp, self.worker_fn, self.indexer = camp, worker_fn, indexer
        self.round_id, self.checker = round_id, checker
        self.log = log or (lambda msg: None)
        cfg = camp.config
        offload.check(cfg, camp.root)  # bad live.offload or live.worker_mem_gb settings stop the run at its start
        self.ws = workspaces or Workspaces(camp.root, cfg["workspace"]["repo"], cfg["workspace"].get("base"),
                                           ignore=cfg["workspace"].get("ignore"))
        self.ws.ensure_clone()
        models = cfg["llm"].get("worker_models")
        if models is not None and (not isinstance(models, list) or not models
                                   or not all(isinstance(m, str) and m.strip() for m in models)):
            raise ValueError(f"llm.worker_models must be a non-empty list of model names, not {models!r}")
        self.worker_models = models  # slot i of a batch runs models[i % len(models)]; None: every worker on worker_model
        fallback = cfg["llm"].get("worker_fallback")
        if fallback is not None and (not isinstance(fallback, str) or not fallback.strip()):
            raise ValueError(f"llm.worker_fallback must be a model name, not {fallback!r}")
        self.worker_fallback = fallback  # a call refused for its model's usage limit is made again on this model
        self._fell_back: dict[str, tuple[str, str]] = {}  # attempt id -> (the model it fell back from, the phase)
        self._fallback_refused: dict[str, str] = {}  # attempt id -> a fallback that was refused as well
        self._limit_waits: dict[str, float] = {}  # attempt id -> seconds it waited out the account's usage limit
        self._cut_offs: dict[str, int] = {}  # attempt id -> its calls a usage limit stopped after the model had worked
        self._stopped_waiting: set[str] = set()  # attempts the run's stop found waiting out a usage limit
        self._last: dict[str, AgentResult] = {}  # attempt id -> the result of its latest call, while its calls run
        objective = cfg["live"].get("objective")
        if objective is not None and (not isinstance(objective, str) or not objective.strip()):
            raise ValueError(f"live.objective must be text saying what the workspace scores, not {objective!r}")
        self.objective = objective  # what the scorer measures, when it is narrower than the goal
        self.default_model = cfg["llm"].get("worker_model") or cfg["llm"]["model"]
        self._models: dict[str, str | None] = {}
        self._batches = 0  # model slots rotate by one each batch, so no model always gets the last-ranked cell
        self.ids: list[str] = []
        self._last_call: dict[str, AgentResult] = {}
        self._mem_kills: dict[str, list] = {}  # an attempt's memory-cap kills, across all its calls
        self.shown_family: dict[str, str | None] = {}  # the family each attempt was shown with at its reveal
        self._seq = 0
        self._git_lock = threading.Lock()
        self.proposals = camp.root / "work" / "_proposals"
        self.proposals.mkdir(parents=True, exist_ok=True)

    # -- scoring --------------------------------------------------------------------------
    def _score(self, commit: str, label: str) -> dict:
        cfg = self.camp.config
        with _file_lock(self.camp.root / "logs" / "score.lock", cfg["scorer"].get("serial", True)):
            with self._git_lock:
                path = self.ws.create_detached(f"_score-{label}", commit)
            try:
                return run_scorer(cfg["scorer"]["cmd"], path, cfg["scorer"]["timeout_s"], cfg.get("direction", "max"),
                                  env=cfg["workspace"].get("env"), sandbox=cfg["scorer"].get("sandbox", "auto"),
                                  allow_write=cfg["scorer"].get("allow_write") or [],
                                  network=bool(cfg["scorer"].get("network", False)))
            finally:
                with self._git_lock:
                    self.ws.remove(path)

    def ensure_baseline(self) -> float:
        cfg = self.camp.config
        if cfg.get("baseline") is not None:
            return cfg["baseline"]
        sc = self._score(self.ws.base_commit, "base")
        if not sc["valid"]:
            raise RuntimeError(f"the scorer fails on the untouched base: {sc['error'] or sc['fail_class']}")
        self.camp.update_config(lambda raw: raw.__setitem__("baseline", sc["score"]))
        self.camp.update_config(lambda raw: raw.__setitem__("baseline_raw", sc["raw_score"]))
        return sc["score"]

    # -- briefing ------------------------------------------------------------------------
    def proposal_dir(self, node_id: str) -> Path:
        return self.proposals / node_id

    def _context(self, parent: dict | None, branch: int, workspace: Path) -> list[str]:
        cfg = self.camp.config
        parts = ["You are one worker in a Dream-RSI research campaign. You make exactly one attempt.", "",
                 "GOAL", cfg.get("goal") or "(not stated)", ""]
        if self.objective:
            parts += ["WHAT THIS WORKSPACE SCORES", self.objective,
                      "The scorer records an attempt that does not serve this as invalid, however good its idea. The "
                      "goal above is what it serves: take a direction toward the goal only in a form the scorer "
                      "measures.", ""]
        parts += ["WORKSPACE",
                 f"{workspace} is a git worktree made for this attempt only. Edit only files matching "
                 f"{cfg['workspace']['mutable']}. Do not commit; the orchestrator commits when you finish. "
                 f"After you finish, the campaign scorer (`{cfg['scorer']['cmd']}`) runs on a clean checkout of "
                 "what you committed, and its verdict, not your report, is what gets recorded. Your shell runs "
                 "in a sandbox: you can write only inside the workspace and your proposal directory.", ""]
        parts += offload.brief(self.camp)
        cap = cfg["live"].get("worker_mem_gb")
        if cap:
            parts += [f"When the processes you start here together go over {cap} GB of memory, they are killed, "
                      "largest first, until they are under it.", ""]
        if parent:
            fp = parent.get("fingerprint") or {}
            raw = (parent.get("artifacts") or {}).get("raw_score")
            parts += ["PARENT ATTEMPT (you continue from its committed state)",
                      f"#{parent['id']} score={raw if raw is not None else parent.get('score')} "
                      f"valid={parent.get('valid')} fail_class={parent.get('fail_class')}",
                      f"proposal: {parent.get('proposal', '')}",
                      f"summary: {(parent.get('text') or {}).get('summary', '')}",
                      f"mechanism: {fp.get('mechanism', '')}  why: {fp.get('why', '')}"]
            anc = self.camp.tree.ancestors(parent["id"])[:3]
            if anc:
                parts.append("EARLIER ANCESTORS: " + " | ".join(
                    f"#{a['id']}: {(a.get('fingerprint') or {}).get('mechanism') or a.get('proposal', '')[:120]}"
                    f" ({(a.get('fingerprint') or {}).get('outcome', a.get('fail_class'))})" for a in anc))
            parts.append("")
        else:
            parts += ["NEW BRANCH", "You start from the campaign base. Open a direction that is not a variant of "
                      "a dead or plateaued family on the map."]
            fams = load_families(self.camp.families_path) if self.camp.families_path.exists() else {}
            frontier = fams.get("frontier") or []
            if branch < len(frontier):  # each suggestion to one branch of a round; the rest choose from the map
                d = frontier[branch]
                near = f" (nearest tried: {', '.join('#' + str(i) for i in d['near'])})" if d.get("near") else ""
                parts.append(f"Suggested untried direction for this branch (unverified; use it or beat it): "
                             f"{d['direction']}{near}")
            parts.append("")
        return parts

    def proposal_file(self, node_id: str) -> Path:
        return self.proposal_dir(node_id) / "proposal.txt"

    def brief_propose(self, node_id, parent, branch, workspace, map_text, feedback: str | None) -> str:
        parts = self._context(parent, branch, workspace)
        parts += ["MAP OF EVERYTHING TRIED SO FAR", map_text, "", PROPOSE,
                  "Read the map and the parent record. For every failure, decide why: a flawed core idea, or a "
                  "good idea let down by a bug, bad parameters or an implementation slip. If attempts cluster "
                  "around small variations of one mechanism with flattening returns, that is a local optimum: "
                  "leave it.",
                  f"Write your proposal (one paragraph: the mechanism, and why it could get past what stopped the "
                  f"nearest attempts) to {self.proposal_file(node_id)}, and put the same text in the report's "
                  "proposal field. Do not implement anything yet: the orchestrator checks the proposal against "
                  "everything tried first."]
        if feedback and feedback.startswith("VERDICT: OFF_TARGET"):
            parts += ["", "YOUR PREVIOUS PROPOSAL DOES NOT TARGET WHAT STOPPED ITS FAMILY. The check said:", feedback,
                      "Aim the difference at that stopper, or propose a different mechanism."]
        elif feedback:
            parts += ["", "YOUR PREVIOUS PROPOSAL REPEATS HISTORY. The check said:", feedback,
                      "Propose a different mechanism."]
        return "\n".join(parts)

    def brief_implement(self, node_id, parent, branch, workspace, check: dict | None, map_text: str) -> str:
        parts = self._context(parent, branch, workspace)
        if check is None:
            parts += ["MAP OF EVERYTHING TRIED SO FAR", map_text, "", IMPLEMENT,
                      "Choose an attempt that is not a repeat of the map, then implement it in the workspace."]
        else:
            parts += [IMPLEMENT, f"Your proposal was accepted ({check['verdict']}):", check["proposal"], ""]
            if check.get("what_differs"):
                parts.append(f"What the check found new: {check['what_differs']}")
            for w in check.get("warnings") or []:
                parts.append(f"Warning: {w}")
            if check.get("doubts"):
                parts.append(f"The check's doubts (advice, not a veto): {check['doubts']}")
            parts += ["", "Implement exactly this proposal in the workspace."]
        parts.append("You may run the scorer yourself while you work. Finish with the structured report: proposal "
                     "(what you built), summary (what you did and what happened), self_reported_score (or null), "
                     "notes.")
        return "\n".join(parts)

    # -- one attempt -----------------------------------------------------------------------
    def _run_worker(self, path, system, model: str | None) -> AgentResult:
        try:
            with offload.serve(self.camp, path, Path(path).name, log=self.log):  # the worker's heavy runs, elsewhere
                if model:
                    return self.worker_fn(path, WORKER_PROMPT, system, model=model)
                return self.worker_fn(path, WORKER_PROMPT, system)
        except OSError:  # the call could not be started (no `claude` to run): the loop's failure, not the model's
            raise
        except Exception as e:  # noqa: BLE001 - a crashed worker is a recorded attempt, not a lost round
            return AgentResult(ok=False, error=f"{type(e).__name__}: {e}")

    def _call(self, path, system, nid: str | None = None, reset=None) -> AgentResult:
        """One worker call of an attempt. A call refused for a usage limit before the model ran is never a failed
        attempt: it runs on llm.worker_fallback when that can, and when no model the attempt may use can run (no
        fallback, the same model, or the fallback refused as well) the attempt waits for the earliest reset among the
        refusals, whatever their kind, and the call is made again, its own model first. A refusal's kind cannot say
        which models it binds (Claude Code reports the exceeded window that resets last), so none is trusted to. A
        call the limit cut off after the model had worked is treated the same, and `reset` (the caller's) undoes what
        it did before the call is made again, at most LIMIT_MAX_CUT_OFFS times: a usage limit stops a call once a window,
        so a 429 after work that keeps coming is something else, and the call's failure then stands. A run that begins
        stopping ends the wait."""
        kills: list = []
        if nid is not None:
            self._last.pop(nid, None)
        try:
            return self._call_until_run(path, system, nid, reset, kills)
        finally:  # kept even if a reset fails, so the record still names the calls and what the memory cap killed
            if nid is not None:
                last = self._last.pop(nid, None)
                if last is not None:
                    self._last_call[nid] = last
                self._mem_kills.setdefault(nid, []).extend(kills)

    def _call_until_run(self, path, system, nid, reset, kills: list) -> AgentResult:
        stopped_after_work = 0
        while True:
            res, part, refusals, other = self._try_call(path, system, nid, reset)
            kills += part
            if nid is not None:
                self._last[nid] = res
            if not refusals:
                break
            cut_off = any(getattr(r, "cut_off", False) for r in refusals)
            if cut_off:
                stopped_after_work += 1
                if stopped_after_work > LIMIT_MAX_CUT_OFFS:
                    self.log(f"{nid}: a 429 stopped the call after work {stopped_after_work} times, more than a usage "
                             "limit does; its failure stands")
                    break
            secs = limit_wait(*refusals)
            kinds = ", ".join(sorted({str(getattr(r, "limit_type", None) or "unnamed") for r in refusals}))
            until = time.strftime("%H:%M", time.gmtime(time.time() + secs))
            self.log(f"{nid}: no model it may use can run ({kinds} usage limit); waiting until {until} UTC, then "
                     "trying again")
            cut = cut_off and reset is not None
            if not wait_unless_stopping(secs):
                if other is not None and other == self.worker_fallback:  # stopped while the fallback was refused too
                    self._fallback_refused[nid] = other
                if cut:
                    reset()  # a stopped run keeps none of the cut-off call's half work: it is no attempt's result
                if nid is not None:
                    self._stopped_waiting.add(nid)
                break
            self._limit_waits[nid] = self._limit_waits.get(nid, 0.0) + secs
            if cut:
                reset()  # the last call made worked before the limit stopped it
        return res

    def _try_call(self, path, system, nid: str | None, reset=None) -> tuple[AgentResult, list, tuple, str | None]:
        """(the call's result, the memory-cap kills of every call made, the refusals when no model could run, the
        other model tried)."""
        model = self._models.get(nid)
        res = self._run_worker(path, system, model)
        kills = list(getattr(res, "mem_kills", None) or [])
        if nid is None or not stopped_by_limit(res):
            return res, kills, (), None
        if getattr(res, "cut_off", False):
            self._cut_offs[nid] = self._cut_offs.get(nid, 0) + 1
        current = model or self.default_model
        own = self._fell_back[nid][0] if nid in self._fell_back else current  # the model the attempt was given
        other = own if current != own else self.worker_fallback  # on the fallback, its own model is the other one
        if not other or other == current:
            return res, kills, (res,), None
        # stopped by the limit, before the model ran or after (what it did is undone first): this call, and the rest of
        # the attempt, run on the other model. The next attempt tries its own model again, so the model comes back
        # when its limit resets or the account changes.
        phase = "propose" if PROPOSE in system else "implement"
        how = "cut off by" if getattr(res, "cut_off", False) else "refused for"
        self.log(f"{nid}: {current} {how} its usage limit; the attempt continues on {other}")
        if getattr(res, "cut_off", False) and reset is not None:
            try:
                reset()
            except BaseException:  # the record still names the cut-off call and what the memory cap killed in it
                self._last[nid] = res
                self._mem_kills.setdefault(nid, []).extend(kills)
                raise
        if phase == "propose":
            _clear(self.proposal_file(nid))  # as before any proposal call: a stale one is never judged
        retry = self._run_worker(path, system, other)
        kills += getattr(retry, "mem_kills", None) or []
        if getattr(retry, "cut_off", False):
            self._cut_offs[nid] = self._cut_offs.get(nid, 0) + 1
        if not stopped_by_limit(retry):
            if other != own:
                self._fell_back[nid] = (current, phase)
            self._models[nid] = other  # back on its own model, the record keeps where the fallback took over
            return retry, kills, (), None
        # refused as well: the attempt keeps its current model and this call's session, transcript and error
        return res, kills, (res, retry), other

    def _check(self, proposal: str, nid: str) -> dict:
        """The novelty check, waiting while its calls are stopped by a usage limit (a stopped check is no verdict): however
        long for a refusal before any work, at most LIMIT_MAX_CUT_OFFS times for a call stopped after work."""
        stopped_after_work = 0
        while True:
            try:
                return self.checker(proposal, nid)
            except LLMLimited as e:
                if getattr(e, "worked", False):
                    stopped_after_work += 1
                    if stopped_after_work > LIMIT_MAX_CUT_OFFS:
                        raise
                until = time.strftime("%H:%M", time.gmtime(time.time() + LIMIT_POLL_S))
                self.log(f"{nid}: the novelty check was stopped by a usage limit ({str(e)[:160]}); waiting until {until} "
                         "UTC, then checking again")
                if not wait_unless_stopping(LIMIT_POLL_S):
                    raise
                self._limit_waits[nid] = self._limit_waits.get(nid, 0.0) + LIMIT_POLL_S

    def _read_proposal(self, node_id: str, res: AgentResult) -> str:
        text = _read_regular(self.proposal_file(node_id), 256_000)
        return (text or str((res.structured or {}).get("proposal") or "")).strip()

    def _attempt(self, job) -> dict:
        node = self._attempt_node(job)
        self.camp.tree.add(node)  # recorded the moment it is done: an interrupted batch keeps finished work
        return node

    def _attempt_node(self, job) -> dict:
        cell, nid, parent_id, path, branch, map_text = job
        parent = self.camp.tree.get(parent_id) if parent_id else None
        cfg = self.camp.config
        check, checks = None, []
        start = self.ws.start_for((parent or {}).get("artifacts", {}).get("commit"))
        try:
            if cfg["live"].get("require_check", True):
                if self.checker is None:
                    raise RuntimeError("live.require_check is on but no checker was given")
                feedback = None
                def fresh_proposal():  # a proposal call the limit cut off: nothing it did survives
                    with self._git_lock:
                        self.ws.recreate(nid, start)
                    _clear_dir(self.proposal_dir(nid))
                for _ in range(max(1, int(cfg["live"].get("max_proposals", 3)))):
                    _clear(self.proposal_file(nid))  # a stale proposal must never be judged again
                    res = self._call(path, self.brief_propose(nid, parent, branch, path, map_text, feedback), nid,
                                     reset=fresh_proposal)
                    with self._git_lock:
                        self.ws.recreate(nid, start)  # proposing builds nothing: nothing it did survives
                    proposal = self._read_proposal(nid, res) if res.ok else ""
                    if not res.ok or not proposal:
                        return self._finish(job, res if not res.ok else AgentResult(
                            ok=False, error="the worker wrote no proposal", session_id=res.session_id, secs=res.secs,
                            transcript=getattr(res, "transcript", None)), None, checks, no_commit=start)
                    check = self._check(proposal, nid)
                    checks.append({k: check.get(k) for k in ("verdict", "ticket", "rule", "family", "rationale")})
                    if check["verdict"] not in ("duplicate", "off_target"):
                        break
                    feedback = render_check(check)
                if check["verdict"] in ("duplicate", "off_target"):
                    return self._finish(job, AgentResult(ok=True, session_id=res.session_id, secs=res.secs,
                                                         transcript=getattr(res, "transcript", None)), check,
                                        checks, not_novel=True, no_commit=start)
            def fresh_build():  # a build the limit cut off: the build is made again from the attempt's start
                with self._git_lock:
                    self.ws.recreate(nid, start)
                _clear_dir(self.proposal_dir(nid), keep=("proposal.txt",))
            res = self._call(path, self.brief_implement(nid, parent, branch, path, check, map_text), nid,
                             reset=fresh_build)
            return self._finish(job, res, check, checks)
        except Exception as e:  # noqa: BLE001 - an orchestration failure says nothing about the idea
            try:
                with self._git_lock:
                    self.ws.remove(path)
            except Exception:  # noqa: BLE001
                pass
            proposal = (check or {}).get("proposal", "")
            return make_node(id=nid, parent=parent_id, source="live", valid=False, fail_class="orchestrator_error",
                             proposal=_clip(proposal, 8000), ext={"round": self.round_id, "cell": cell,
                                                                 "proposal_sha": text_hash(proposal)},
                             text={"orchestrator_error": f"{type(e).__name__}: {e}"},
                             artifacts={"commit": start, "changed": [], "checks": checks,
                                        "outcome": "inconclusive", "killed_by": "orchestrator_error"},
                             fingerprint={"outcome": "inconclusive", "killed_by": "orchestrator_error"},
                             worker=self._worker_record(nid, self._last_call.get(nid)))

    def _finish(self, job, res: AgentResult, check: dict | None, checks: list, not_novel: bool = False,
                no_commit: str | None = None) -> dict:
        cell, nid, parent_id, path, branch, map_text = job
        cfg = self.camp.config
        parent = self.camp.tree.get(parent_id) if parent_id else None
        start = self.ws.start_for((parent or {}).get("artifacts", {}).get("commit"))
        nested: list[str] = []
        if no_commit is not None:  # nothing was built: the attempt owns no code of its own
            commit, changed, links, gitlinks = no_commit, [], [], []
        else:
            with self._git_lock:
                nested = self.ws.nested_repos(nid)
                if nested:  # never let git read a repository the worker made
                    commit, changed, links, gitlinks = start, [], [], []
                else:
                    commit = self.ws.snapshot(nid, start)
                    changed = self.ws.changed_since_base(commit)
                    links = self.ws.symlinks_since_base(commit)
                    gitlinks = self.ws.gitlinks_since_base(commit)
        report = res.structured or {}
        # a failed call's text is its failure (a usage-limit notice, say), never the attempt's proposal
        proposal = check["proposal"] if check else str(report.get("proposal") or (res.result_text if res.ok else ""))
        fields = {"score": None, "valid": False, "gates": {}, "fail_class": None}
        # symbolic links can point the scorer at files that are not in the commit
        oos = (out_of_scope(changed, cfg["workspace"]["mutable"]) + [f"symlink:{p}" for p in links]
               + [f"nested-repo:{p}" for p in nested] + [f"gitlink:{p}" for p in gitlinks])
        if not cfg["workspace"].get("keep_worktrees"):
            with self._git_lock:
                self.ws.remove(path)  # nothing on disk but the commit itself can reach the scorer
        sc = {}
        if not_novel:
            fields["fail_class"] = "not_novel"
        elif not res.ok:
            fields["fail_class"] = "agent_error"
        elif oos:
            fields["fail_class"] = "out_of_scope"
        else:
            sc = self._score(commit, nid)
            fields = {"score": sc["score"], "valid": sc["valid"], "gates": sc["gates"],
                      "fail_class": sc["fail_class"]}
        if no_commit is not None and not cfg["workspace"].get("keep_worktrees") and path.exists():
            with self._git_lock:
                self.ws.remove(path)
        ref = parent["score"] if parent and parent.get("valid") else cfg.get("baseline")
        outcome, killed_by = live_outcome(dict(fields), ref, float(cfg["live"].get("pass_margin", 0.0)))
        return make_node(
            id=nid, parent=parent_id, source="live", proposal=_clip(proposal, 8000),
            fingerprint={"outcome": outcome, "killed_by": killed_by},
            text={"summary": str(report.get("summary", ""))[:1500], "notes": str(report.get("notes", ""))[:800],
                  "worker_error": res.error[:500]},
            artifacts={"workspace": str(path), "branch": f"drsi/{nid}", "commit": commit, "changed": changed,
                       "out_of_scope": oos, "checks": checks, "raw_score": sc.get("raw_score"),
                       "outcome": outcome, "killed_by": killed_by,
                       "scorer_summary": str(sc.get("summary") or sc.get("error") or "")[:4000],
                       "self_reported_score": report.get("self_reported_score")},
            worker=self._worker_record(nid, res),
            ext={"round": self.round_id, "cell": cell, "proposal_sha": text_hash(proposal)}, **fields)

    def _worker_record(self, nid: str, res: AgentResult | None) -> dict:
        """What the attempt records of its worker: the last call's session, time and transcript (logs/workers/<id>/
        holds every call's), and the model."""
        rec = {"session": getattr(res, "session_id", None), "secs": round(getattr(res, "secs", 0.0) or 0.0, 1),
               "model": self._models.get(nid) or self.default_model, "transcript": getattr(res, "transcript", None)}
        if nid in self._fell_back:  # the model the attempt was given, refused for its usage limit, and where
            rec["fell_back_from"], rec["fell_back_in"] = self._fell_back[nid]
        if nid in self._fallback_refused:
            rec["fallback_refused"] = self._fallback_refused[nid]
        if self._limit_waits.get(nid):  # the time its calls waited for the account's usage limit to reset
            rec["waited_for_limit_s"] = round(self._limit_waits[nid])
        if self._cut_offs.get(nid):  # calls a 429 stopped after the model had worked
            rec["cut_off_by_limit"] = self._cut_offs[nid]
        if nid in self._stopped_waiting:  # the run's stop found it waiting out a usage limit: its end is the stop's
            rec["stopped_in_limit_wait"] = True
        if self._mem_kills.get(nid):  # the processes the memory cap killed in any of the attempt's calls
            rec["mem_kills"] = list(self._mem_kills[nid])
        return rec

    def _setup_failed(self, cell, nid, parent_id, parent_commit, error: Exception) -> dict:
        """A cell whose workspace could not be made is recorded as an orchestration failure under its parent,
        so the parent is no longer a leaf a policy would pick again."""
        try:
            with self._git_lock:
                self.ws.remove(self.ws.path(nid))
        except Exception:  # noqa: BLE001
            pass
        node = make_node(id=nid, parent=parent_id, source="live", valid=False, fail_class="orchestrator_error",
                         ext={"round": self.round_id, "cell": cell, "proposal_sha": text_hash("")},
                         text={"orchestrator_error": f"setting up the workspace: {type(error).__name__}: {error}"},
                         artifacts={"commit": parent_commit, "changed": [], "checks": [],
                                    "outcome": "inconclusive", "killed_by": "orchestrator_error"},
                         fingerprint={"outcome": "inconclusive", "killed_by": "orchestrator_error"})
        self.camp.tree.add(node)
        self.log(f"{nid}: workspace setup failed ({error}); recorded, the batch goes on")
        return node

    def run_batch(self, cells) -> list[dict]:
        tree = self.camp.tree
        map_text = write_map(self.camp)  # once per batch: every worker in it sees the same map
        slots = []  # per cell, in order: its job, or the node recorded when its setup failed
        turn, self._batches = self._batches, self._batches + 1
        for slot, cell in enumerate(cells):
            self._seq += 1
            nid = f"{self.round_id}-{self._seq:03d}"
            if self.worker_models:
                self._models[nid] = self.worker_models[(slot + turn) % len(self.worker_models)]
            if cell.startswith(ROOT):
                parent_id, parent_commit, branch = None, None, int(cell[len(ROOT):])
            else:
                parent_id = cell
                parent_commit = tree.get(cell)["artifacts"].get("commit")
                branch = 0
            self.proposal_dir(nid).mkdir(parents=True, exist_ok=True)
            try:
                with self._git_lock:
                    path = self.ws.create(nid, self.ws.start_for(parent_commit))
            except Exception as e:  # noqa: BLE001 - one cell that cannot be set up must not stop the batch
                slots.append(self._setup_failed(cell, nid, parent_id, parent_commit, e))
                continue
            slots.append((cell, nid, parent_id, path, branch, map_text))
        jobs = [s for s in slots if isinstance(s, tuple)]
        pool = ThreadPoolExecutor(max_workers=max(1, len(jobs)))
        try:
            done = iter(list(pool.map(self._attempt, jobs)))
            nodes = [next(done) if isinstance(s, tuple) else s for s in slots]  # probe_batch pairs by position
        except BaseException:
            # Ctrl-C (or any abort): workers and scorers run in their own process groups, so stop them
            # explicitly instead of waiting hours for them to finish.
            kill_all_children()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        for node in nodes:  # each was recorded as soon as it finished
            self.ids.append(node["id"])
        try:
            self.indexer([n["id"] for n in nodes])
        except Exception as e:  # noqa: BLE001 - the map can be rebuilt later; the attempts are safe
            self.log(f"indexing failed ({e}); run `drsi sync` later")
        tree = self.camp.tree
        out = [{"id": n["id"], "score": n["score"], "valid": n["valid"], "fail_class": n["fail_class"],
                "family": ((tree.get(n["id"]).get("fingerprint") or {}).get("family"))} for n in nodes]
        self.shown_family.update({n["id"]: n["family"] for n in out})
        return out


RUNNER = Path(__file__).resolve().parent / "live_runner.py"
ENGINE_DIR = Path(__file__).resolve().parents[1]


class _Lines:
    """Newline-framed messages from a pipe, with a deadline per message."""

    def __init__(self, fd: int):
        self.fd, self.buf = fd, b""

    def read(self, timeout: float | None) -> bytes | None:
        """One line; b"" at end of stream; None when `timeout` seconds pass first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return None
            ready, _, _ = select.select([self.fd], [], [], left)
            if not ready:
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return b""
            self.buf += chunk
            if len(self.buf) > 4_000_000:
                return b""  # no legitimate request is this large
        line, self.buf = self.buf.split(b"\n", 1)
        return line


def _drive(policy: PolicySource, q: LiveQuestion, budget: int, think: float | None, log) -> str:
    """Run the policy in a child process against the orchestrator's question. Returns why it stopped."""
    refuse_if_stopping()
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    with tempfile.TemporaryDirectory(prefix="drsi-policy-") as d:
        job = Path(d) / "job.json"
        job.write_text(json.dumps({"source": policy.source, "label": policy.path, "W": q.max_parallelism,
                                   "baseline": q.baseline_score, "budget": budget}))
        proc = subprocess.Popen([sys.executable, "-s", "-P", str(RUNNER), str(ENGINE_DIR), str(job),
                                 str(req_w), str(resp_r)], pass_fds=(req_w, resp_r), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                                env={"PYTHONHASHSEED": str(POLICY_HASH_SEED), "PATH": "/usr/bin:/bin"})
        os.close(req_w)
        os.close(resp_r)
        register_child(proc)
        lines = _Lines(req_r)
        try:
            while True:
                line = lines.read(think)  # the clock runs only while the policy itself is thinking
                if line is None:
                    return f"the policy exceeded live.think_timeout_s ({think}s) without asking for a batch"
                if not line:
                    return "the policy process ended without finishing"
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    return "the policy process sent a malformed request"
                if not isinstance(msg, dict) or msg.get("op") == "done":
                    return "done" if isinstance(msg, dict) else "the policy process sent a malformed request"
                if msg.get("op") == "error":
                    return f"the policy raised {str(msg.get('error'))[:500]}"
                cells = msg.get("cells")
                if msg.get("op") != "expand" or not isinstance(cells, list):
                    return "the policy process sent a malformed request"
                try:
                    revealed = q.probe_batch(cells)  # validated on the orchestrator's own copy
                except IllegalBatch as e:
                    return f"illegal batch: {e}"
                nodes = [None if o is None else {"id": o.id, "score": o.score, "valid": o.valid,
                                                 "fail_class": o.fail_class, "family": o.family} for o in revealed]
                try:
                    os.write(resp_w, (json.dumps({"nodes": nodes}) + "\n").encode())
                except OSError:
                    return "the policy process ended without finishing"
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            unregister_child(proc)
            os.close(req_r)
            os.close(resp_w)


def live_round(camp: Campaign, policy: PolicySource, runner: LiveRunner) -> dict:
    baseline = runner.ensure_baseline()
    cfg = camp.config
    W = cfg["search"]["W"]
    budget = cfg["search"]["K1"] * W
    q = LiveQuestion(runner, W, baseline, max_probes=budget)
    why = _drive(policy, q, budget, cfg["live"].get("think_timeout_s") or None, runner.log)
    if why != "done":  # the budget, batch and time rules hold whatever the policy does
        runner.log(f"{runner.round_id}: policy stopped: {why}")
    # the world keeps what the policy was shown, not the families assigned since (replay must see what live saw)
    world = world_from_tree(camp.tree, runner.round_id, baseline, ids=set(runner.ids), shown=runner.shown_family)
    freeze_world(camp.root / "trace_pool", world)
    write_map(camp)
    valid = [camp.tree.get(i) for i in runner.ids]
    return {"round_id": runner.round_id, "attempts": len(runner.ids), "rounds": q.rounds,
            "valid": sum(1 for n in valid if n["valid"]), "best_score": q.best_score(), "baseline": baseline,
            "stopped": why}


def next_round_id(camp: Campaign) -> str:
    """One past every round id already used anywhere: frozen worlds, recorded attempts, worktrees, pruned attempts.
    A round interrupted before freezing must not have its id (and branch names) reused."""
    used = [0]
    pat = re.compile(r"^iter(\d{4,})")
    pool = camp.root / "trace_pool"
    for d in (pool.glob("iter*") if pool.exists() else []):
        m = pat.match(d.name)
        used.append(int(m.group(1)) if m else 0)
    for n in camp.tree.nodes():
        m = pat.match(str((n.get("ext") or {}).get("round") or ""))
        used.append(int(m.group(1)) if m else 0)
    work = camp.root / "work"
    for d in (work.iterdir() if work.exists() else []):
        m = pat.match(d.name)
        used.append(int(m.group(1)) if m else 0)
    log = camp.root / "logs" / "prune.jsonl"  # a round whose every attempt was pruned leaves only this
    for line in (log.read_text(errors="replace").splitlines() if log.exists() else []):
        try:
            ids = json.loads(line).get("ids") or []
        except (json.JSONDecodeError, AttributeError):
            continue
        for i in ids:
            m = pat.match(str(i))
            used.append(int(m.group(1)) if m else 0)
    if (camp.root / "repo" / ".git").exists():  # an interrupted round may have left only branches behind
        from .workspace import _git
        out = _git(camp.root / "repo", "branch", "--list", "drsi/iter*", "--format=%(refname:short)", check=False)
        for b in out.split():
            m = pat.match(b.split("/", 1)[-1])
            used.append(int(m.group(1)) if m else 0)
    return f"iter{max(used) + 1:04d}"


def _has_signal(worlds: list[dict]) -> bool:
    return any(any(n.get("valid") and n.get("score") is not None for n in w["nodes"]) for w in worlds)


def run_cycles(camp: Campaign, n: int, worker_fn, developer, indexer, checker=None, progress=None) -> dict:
    say = progress or (lambda msg: None)
    policy_dir = camp.root / "policy"
    policy_dir.mkdir(exist_ok=True)
    if not (policy_dir / "method.py").exists():
        (policy_dir / "method.py").write_text(SEED_POLICY.read_text())
    rounds = []
    for _ in range(n):
        round_id = next_round_id(camp)
        (camp.root / "logs").mkdir(parents=True, exist_ok=True)
        (camp.root / "logs" / "current_round").write_text(round_id)  # prune leaves this round alone while it runs
        runner = LiveRunner(camp, worker_fn, indexer, round_id, log=say, checker=checker)
        summary = live_round(camp, load_policy(policy_dir / "method.py"), runner)
        say(f"{round_id}: {summary['attempts']} attempts, {summary['valid']} valid, best {summary['best_score']}")
        worlds = with_cells(load_worlds(camp.root / "trace_pool"), camp.tree)
        need, k = max(1, int(camp.config["dream"].get("min_worlds", 4))), informative(comparable(worlds))
        if k < need:  # too few worlds to tell policies apart: a dream would spend developer calls on noise
            say(f"{round_id}: {k} world(s) can inform a dream, fewer than dream.min_worlds = {need}; dream skipped")
            rounds.append(summary | {"dream": {"deployed": False, "version": None, "skipped": True,
                                               "incumbent_reward": None, "best_reward": None}})
            continue
        d = run_dream(policy_dir, worlds, developer, camp.config, camp.root / "logs")
        for w in d.get("warnings", []):
            say(f"{round_id}: warning: {w}")
        if d.get("skipped"):
            say(f"{round_id}: dream skipped: {d['skipped']}")
        else:
            say(f"{round_id}: dream {'deployed ' + d['version'] if d['deployed'] else 'kept the incumbent'} "
                f"(reward {d['incumbent_reward']:.4f} -> {d['best_reward']:.4f})")
        rounds.append(summary | {"dream": {"deployed": d["deployed"], "version": d["version"],
                                           "incumbent_reward": d["incumbent_reward"], "best_reward": d["best_reward"]}})
    return {"rounds": rounds}
