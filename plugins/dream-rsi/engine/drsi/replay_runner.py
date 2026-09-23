"""Runs one policy against replay worlds. Invoked only by drsi.replay in a clean subprocess.

Usage: python3 -s -P replay_runner.py <engine_dir> <job.json>
Every run gets a freshly executed policy module with restricted builtins. The runner records only the
batches each run asked the question to reveal (its action trace); the parent process replays those
traces on its own copy of the world and computes every metric itself, so nothing a policy does to the
objects in this process can change its score. The result goes to the job's private output path and the
process then hard-exits, so printed output, SystemExit or finalizers cannot stand in for it.
"""
import json
import os
import sys


def main() -> None:
    engine, job_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, engine)
    from drsi.guard import safe_builtins
    from drsi.question import ReplayQuestion

    class TracingQuestion(ReplayQuestion):
        def __init__(self, *a, **kw):
            self._trace = []
            super().__init__(*a, **kw)

        def _expand(self, cells):
            self._trace.append(list(cells))  # every batch that passed validation, whatever happens next
            return super()._expand(cells)

    with open(job_path) as fh:
        job = json.load(fh)
    out_path = job.pop("out")
    try:
        with open(job["policy"]) as fh:
            code = compile(fh.read(), job["policy"], "exec")

        def fresh_policy():
            ns = {"__name__": "drsi_policy", "__builtins__": safe_builtins()}
            exec(code, ns)  # source passed drsi.guard before this process started
            return ns["OptimalPolicy"]

        probe_cls = fresh_policy()
        default_beta = float(probe_cls.__dict__.get("beta", 0.6)) if isinstance(probe_cls, type) else 0.6
        betas = list(job["betas"])
        if default_beta not in betas:
            betas.append(default_beta)
        runs = {}
        for beta in betas:
            rows = []
            for world in job["worlds"]:
                q = TracingQuestion(world, job["W"], max_probes=job["budget"])
                fresh_policy()(beta=beta).solve(q, job["budget"])
                rows.append({"trace": q._trace})
            runs[str(float(beta))] = rows
        result = {"ok": True, "default_beta": default_beta, "runs": runs}
    except BaseException as e:  # noqa: BLE001 - any policy failure (even SystemExit) is a result
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with open(out_path, "w") as fh:
        fh.write(json.dumps(result, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
