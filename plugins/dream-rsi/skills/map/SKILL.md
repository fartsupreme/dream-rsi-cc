---
name: map
description: Show the Dream-RSI map of everything tried in a campaign (syncs the source ledger first).
disable-model-invocation: true
argument-hint: "[campaign]"
allowed-tools:
  - Bash("${CLAUDE_PLUGIN_ROOT}/bin/drsi" *)
---

Run `"${CLAUDE_PLUGIN_ROOT}/bin/drsi" sync -c <campaign>` then `"${CLAUDE_PLUGIN_ROOT}/bin/drsi" map -c <campaign>`
(campaign = $ARGUMENTS, or the only campaign if blank). Show the map as printed, then state in two lines
which families are open and which untried directions look strongest.
