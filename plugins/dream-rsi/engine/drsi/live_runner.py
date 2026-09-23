"""Runs the deployed policy for one live round. Invoked only by drsi.live, in its own process group.

Usage: python3 -s -P live_runner.py <engine_dir> <job.json> <request_fd> <response_fd>
The policy's question keeps the same state as the orchestrator's: every batch it asks for is sent to
the orchestrator, which validates it on its own copy, runs the attempts, and answers with what was
revealed. The orchestrator also times the policy between batches and kills this process group when it
thinks too long, so no handler or finally block in the policy can keep a round alive.
"""
import json
import os
import sys


def main() -> None:
    engine, job_path, req_fd, resp_fd = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    sys.path.insert(0, engine)
    from drsi.guard import safe_builtins
    from drsi.question import QuestionBase

    requests = os.fdopen(req_fd, "wb", buffering=0)
    answers = os.fdopen(resp_fd, "rb")

    def send(msg: dict) -> None:
        requests.write((json.dumps(msg) + "\n").encode())

    class RemoteQuestion(QuestionBase):
        def _expand(self, cells):
            send({"op": "expand", "cells": list(cells)})
            line = answers.readline()
            if not line:
                os._exit(0)  # the orchestrator ended the round
            return json.loads(line)["nodes"]

    with open(job_path) as fh:
        job = json.load(fh)
    try:
        ns = {"__name__": "drsi_policy", "__builtins__": safe_builtins()}
        exec(compile(job["source"], job["label"], "exec"), ns)  # the orchestrator guard-checked this source
        q = RemoteQuestion(job["W"], job["baseline"], max_probes=job["budget"])
        ns["OptimalPolicy"]().solve(q, budget=job["budget"])
        send({"op": "done"})
    except BaseException as e:  # noqa: BLE001 - any policy failure (even SystemExit) ends the round
        send({"op": "error", "error": f"{type(e).__name__}: {e}"[:2000]})
    os._exit(0)


if __name__ == "__main__":
    main()
