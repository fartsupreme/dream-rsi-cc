---
name: status
description: Summarise a Dream-RSI campaign — attempts, families by status, checks run, policy version.
disable-model-invocation: true
argument-hint: "[campaign]"
allowed-tools:
  - Bash("${CLAUDE_PLUGIN_ROOT}/bin/drsi" *)
---

Run `"${CLAUDE_PLUGIN_ROOT}/bin/drsi" status -c <campaign>` (or `drsi list` if no campaign is named and
several exist) and relay the output.
