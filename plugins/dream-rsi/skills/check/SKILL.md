---
name: check
description: Novelty-check a proposed attempt against everything a Dream-RSI campaign has tried.
disable-model-invocation: true
argument-hint: "<proposal text>"
allowed-tools:
  - Bash("${CLAUDE_PLUGIN_ROOT}/bin/drsi" *)
  - Write
---

Write the proposal ($ARGUMENTS) to a temporary file, then run
`"${CLAUDE_PLUGIN_ROOT}/bin/drsi" check -c <campaign> --file <that file>` and report the verdict, the rule
applied (if any), and the nearest prior attempts with what stopped them. Exit 0 novel, 3 variant, 4 duplicate.
