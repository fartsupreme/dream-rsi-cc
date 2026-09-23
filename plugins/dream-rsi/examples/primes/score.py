"""Scorer for the primes example: the pattern a drsi scorer should follow.

- The candidate never runs in this process: each measurement is a separate child
  interpreter, so candidate code cannot patch our timer or print our final line.
- Inputs are CALLS distinct values drawn fresh every run and checked against our own
  reference, so hardcoding or caching answers for known inputs does not pay.
- The measured cost is the whole child (import + every call) minus an empty
  interpreter's startup, so work moved to import time still counts.
- primes.py may be at most MAX_SOURCE bytes: an honest implementation is well under
  1 KB, and tabulating answers for the input range is not the task. (Workers found
  each of these loopholes in testing: timing noise, import-time work, answer tables.)
Last stdout line: {"score": ms per call, "valid": bool, ...}. Campaign direction: "min".
"""
import json
import os
import random
import subprocess
import sys
import time

CALLS = 60
LO, HI = 5_000, 40_000
MAX_SOURCE = 3_000


def reference(n: int) -> int:
    if n < 3:
        return 0
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n, i)))
    return sum(i for i in range(n) if sieve[i])


CHILD = ("import sys; sys.path.insert(0, sys.argv[1]); from primes import prime_sum; "
         "print(' '.join(str(prime_sum(int(x))) for x in sys.argv[2:]))")
EMPTY = "pass"


def run_child(code, args, timeout=120):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}  # nothing of ours leaks into the candidate
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-I", "-c", code, *args], capture_output=True, text=True,
                          timeout=timeout, env=env)
    return time.perf_counter() - t0, proc


def main():
    ws = os.environ.get("DRSI_WORKSPACE", ".")
    size = os.path.getsize(os.path.join(ws, "primes.py"))
    if size > MAX_SOURCE:
        print(json.dumps({"score": None, "valid": False, "fail_class": "out_of_spec",
                          "error": f"primes.py is {size} bytes (max {MAX_SOURCE}): answers must be computed"}))
        return 0
    rng = random.SystemRandom()
    try:
        timings = []
        for _ in range(3):
            ns = rng.sample(range(LO, HI), CALLS)
            secs, proc = run_child(CHILD, [ws, *map(str, ns)])
            got = proc.stdout.strip().split()
            if proc.returncode != 0 or got != [str(reference(n)) for n in ns]:
                print(json.dumps({"score": None, "valid": False, "fail_class": "eval_error",
                                  "error": f"wrong or crashed: {proc.stderr[-300:]}"}))
                return 0
            timings.append(secs)
        startup = min(run_child(EMPTY, [])[0] for _ in range(3))
    except subprocess.TimeoutExpired:
        print(json.dumps({"score": None, "valid": False, "fail_class": "timeout", "error": "candidate timed out"}))
        return 0
    ms = max(0.0, (min(timings) - startup) / CALLS * 1000)
    print(json.dumps({"score": round(ms, 4), "valid": True, "fail_class": "ok", "gates": {"exact": {"pass": True}},
                      "summary": f"{ms:.4f} ms per call over {CALLS} fresh inputs in [{LO}, {HI})"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
