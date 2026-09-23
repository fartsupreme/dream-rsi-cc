---
name: init
description: Set up a Dream-RSI campaign — import an existing attempt log, build the families and map, and optionally configure the live loop.
disable-model-invocation: true
argument-hint: "[campaign-name]"
allowed-tools:
  - Bash("${CLAUDE_PLUGIN_ROOT}/bin/drsi" *)
  - Read
  - AskUserQuestion
---

# /dream-rsi:init

`drsi` is `"${CLAUDE_PLUGIN_ROOT}/bin/drsi"`. Campaign state lives in `~/.dream-rsi/campaigns/<name>/`,
outside every repository; `drsi` never writes to the project it reads.

1. Ask for anything not given: the campaign name ($ARGUMENTS if present), a one-paragraph goal that names
   the success criteria, and the path of an existing attempt log if there is one.
2. `drsi init <name> --goal "<goal>"`.
3. Import the log: `drsi import -c <name> <file>` (the `attempt-ledger` preset is the default: one JSON object per attempt with `candidate`, `falsifiable`,
   `verdict`, `next`, `check_cmd` and `supersedes`/`refutes` links). Any other
   JSONL: `--preset generic --field-map '{"id":"...","parent":"...","proposal":"...","text":["..."]}'`.
4. Fingerprint and group (long: run in the background and wait for completion):
   `drsi fingerprint -c <name>` then `drsi families -c <name>`.
5. Show `drsi map -c <name>` and the family table (`drsi families -c <name> --list`).
6. For the live loop, collect: the code repo to clone (never modified), the scorer command, and the paths
   workers may edit. Rules that keep scores honest:
   - The scorer must exit 0 and print `{"score": n, "valid": bool, ...}` as its LAST stdout line, and should
     run candidate code in a child process (see `${CLAUDE_PLUGIN_ROOT}/examples/primes/score.py`).
   - Nothing the scorer executes to judge an attempt (its script, tests, fixtures, build files) may be
     inside `workspace.mutable`; otherwise a worker can edit its own grader.
   - If workers need the network (crate or package downloads), list the hosts in `live.allowed_domains`.
   Then `drsi config -c <name> --set scorer.cmd='"..."' --set workspace.repo='"..."' --set workspace.mutable='["src/**"]'`
   and `drsi baseline -c <name>`.
