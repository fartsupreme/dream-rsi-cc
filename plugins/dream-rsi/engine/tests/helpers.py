"""Test doubles. A ScriptedLLM answers .json(prompt, schema) with a function."""
import json
import re
import threading


class ScriptedLLM:
    def __init__(self, fn):
        self.fn = fn
        self.prompts = []
        self._lock = threading.Lock()

    def json(self, prompt, schema):
        with self._lock:
            self.prompts.append(prompt)
        return self.fn(prompt, schema)


def ids_in_block(prompt, tag="ATTEMPT"):
    """Extract ids from lines like '<ATTEMPT id="12" ...>' in a prompt."""
    return re.findall(rf'<{tag} id="([^"]+)"', prompt)


def fp_for(i, family_hint="hint", outcome="refuted", killed_by="G-X"):
    return {"id": i, "mechanism": f"mechanism of {i}", "object": "obj", "key_move": "move",
            "kind": "construction", "outcome": outcome, "killed_by": killed_by,
            "why": "because", "family_hint": family_hint}


def dumps(o):
    return json.dumps(o, sort_keys=True)


# Ground truth and recorded rounds (round 26): the dream compares policies only on worlds where the incumbent's replay
# stays on the record, as it does on every round it recorded itself.
def truth_world(i, roots=30, depth=30, plateau=False):
    """Ground truth deeper and wider than any round: root r scores s_r, each continuation 0.3 more (with plateau,
    only the first continuation gains, then the branch is flat)."""
    nodes = []
    for r in range(roots):
        s = ((r * 7 + i * 5) % roots) / roots
        prev = None
        for d in range(depth):
            nid = f"t{i}r{r}d{d}"
            gain = 0.3 * (min(d, 1) if plateau else d)
            nodes.append({"id": nid, "parent": prev, "score": s + gain, "valid": True, "fail_class": "ok"})
            prev = nid
    return {"id": f"truth{i}", "baseline": 0.0, "nodes": nodes}


def record(policy_path, truth, W, budget, world_id):
    """What a live round of this policy records on the ground truth: the attempts it revealed, under the ids live gives
    them (<round>-<seq>, numbered in the order the policy listed each batch), each with the cell that opened it (its
    root slot, or its parent's id), listed in the order the workers finished (reversed within each batch)."""
    from drsi.question import ROOT, ReplayQuestion
    from drsi.replay import evaluate_policy
    rep = evaluate_policy(policy_path, [truth], W=W, betas=[], budget=budget, lam=0.25, beta1=0.01, beta2=0.01)
    assert rep["ok"], rep
    trace = rep["traces"]["runs"][str(float(rep["default_beta"]))][0]["trace"]
    q = ReplayQuestion(truth, W, max_probes=budget)
    live, nodes = {}, []
    for batch in trace:
        seen = len(q._order)
        q.probe_batch(batch)
        assert len(q._order) - seen == len(batch), "the ground truth must answer every probe"
        done = []
        for cell, nid in zip(batch, q._order[seen:]):
            live[nid] = f"{world_id}-{len(live) + 1:03d}"
            o = q._obs[nid]
            done.append({"id": live[nid], "parent": live.get(o.parent_id), "score": o.score, "valid": o.valid,
                         "fail_class": o.fail_class, "cell": cell if cell.startswith(ROOT) else live[cell]})
        nodes += reversed(done)
    return {"id": world_id, "baseline": 0.0, "nodes": nodes}


def with_block(block):
    """An edit that puts this EVOLVE block into a policy source."""
    def edit(src):
        from drsi.dream import split_evolve
        before, _, after = split_evolve(src)
        return before + block + after
    return edit


# Two policies over the same eight attempts per world: open four roots, then continue each root once, the incumbent
# from its worst root up, the candidate from its best down. The best attempts are continuations of the best roots.
ORDERED = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        W = question.max_parallelism
        obs = question.observed()
        if len([o for o in obs.values() if o.parent_id is None]) < 4:
            return question.legal_roots()[:W]
        firsts = [c for c in question.legal_actions() if not c.startswith("root:") and obs[c].parent_id is None]
        firsts.sort(key=lambda c: (obs[c].score if obs[c].score is not None else -1.0, c), reverse=REVERSE)
        return firsts[:W]
    # EVOLVE-BLOCK-END
"""
WORST_FIRST, BEST_FIRST = ORDERED.replace("REVERSE", "False"), ORDERED.replace("REVERSE", "True")


def own_worlds(policy_src: str, W: int, budget: int, n: int = 6, plateau: bool = False) -> list[dict]:
    """n rounds this policy recorded itself on ground truth: worlds on which its replay stays on the record, so the
    dream compares policies there (round 26)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "policy.py"
        path.write_text(policy_src)
        return [record(path, truth_world(i, plateau=plateau), W, budget, f"iter{i:04d}") for i in range(n)]
