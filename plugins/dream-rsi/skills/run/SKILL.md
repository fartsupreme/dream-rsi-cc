---
name: run
description: Run Dream-RSI cycles for a campaign — parallel fresh workers explore, the scorer grades, then the policy is improved by replay.
disable-model-invocation: true
argument-hint: "[campaign] [rounds]"
allowed-tools:
  - Bash("${CLAUDE_PLUGIN_ROOT}/bin/drsi" *)
  - Read
---

1. `"${CLAUDE_PLUGIN_ROOT}/bin/drsi" config -c <campaign>`: confirm `scorer.cmd`, `workspace.repo`,
   `workspace.mutable`, `live.allowed_bash` and `search.W`/`K1` suit the task; if the scorer or repo is
   missing, run `/dream-rsi:init` steps 6 first.
2. `"${CLAUDE_PLUGIN_ROOT}/bin/drsi" baseline -c <campaign>` (scores the untouched base once).
3. Run in the background (rounds default 1): `"${CLAUDE_PLUGIN_ROOT}/bin/drsi" run -c <campaign> --rounds <n>`.
   Each batch runs W attempts, each in its own git worktree of a campaign-owned clone, with fresh workers
   inside Claude Code's Bash sandbox (writes only to the worktree and proposal directory; no network unless
   `live.allowed_domains`; no hooks). An attempt is: propose -> the orchestrator novelty-checks the proposal
   against everything tried (a duplicate goes back with the judge's reasons, up to `live.max_proposals`
   tries) -> implement the accepted proposal. The orchestrator then rejects any change outside
   `workspace.mutable` (measured from the campaign base) and scores a clean checkout of the commit with the
   campaign scorer, one scorer at a time. The dream phase rewrites only the policy's EVOLVE block and
   deploys a revision only if replay reward does not drop.
4. When it finishes, report per round: attempts, valid, best score vs baseline, and whether a new policy was
   deployed. Each attempt's code is on branch `drsi/<node>` in `~/.dream-rsi/campaigns/<campaign>/repo`.
   Nothing is applied to the real repository; applying a result is the user's call.
