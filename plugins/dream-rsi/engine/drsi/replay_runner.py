"""Runs one policy on one replay world at one beta. Invoked only by drsi.replay, in a clean subprocess per run.

Usage: python3 -s -P replay_runner.py <engine_dir> <job.json>
A job with no world only reports the policy's default beta. Each run has a process of its own, as a live round's
policy has: nothing a run leaves behind in its process (a class attribute, module state) can reach another run.
The runner records only the batches the run asked the question to reveal (its action trace); the parent process
replays that trace on its own copy of the world and computes every metric itself, so nothing a policy does to the
objects in this process can change its score. The result goes to the job's private output path and the process
then hard-exits, so printed output, SystemExit or finalizers cannot stand in for it.
"""
import json
import os
import sys


def main() -> None:
    engine, job_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, engine)
    from drsi.guard import safe_builtins
    from drsi.question import PolicyQuestion, RecordEnd, ReplayQuestion

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

        ns = {"__name__": "drsi_policy", "__builtins__": safe_builtins()}
        exec(code, ns)  # source passed drsi.guard before this process started
        cls = ns["OptimalPolicy"]
        default_beta = float(cls.__dict__.get("beta", 0.6)) if isinstance(cls, type) else 0.6
        if job.get("world") is None:
            result = {"ok": True, "default_beta": default_beta}
        else:
            q = TracingQuestion(job["world"], job["W"], max_probes=job["budget"])
            # at its default beta (beta None) a policy is built as the live round builds it, with no argument
            policy = cls() if job.get("beta") is None else cls(beta=job["beta"])
            try:
                policy.solve(PolicyQuestion(q), job["budget"])  # the view every environment hands a policy
            except RecordEnd:  # the run reached the end of its record: what it did up to there is its trace
                pass
            result = {"ok": True, "default_beta": default_beta, "trace": q._trace}
    except BaseException as e:  # noqa: BLE001 - any policy failure (even SystemExit) is a result
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with open(out_path, "w") as fh:
        fh.write(json.dumps(result, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
