---
name: dream-rsi
description: Use when working a long research or optimization campaign with many attempts (constructions, proofs, performance work), when resuming such a campaign after a compaction, restart or handoff, before starting any new attempt in it, or when the user says the agent keeps re-trying things, going in circles, or "we already tried that".
allowed-tools:
  - Bash("${CLAUDE_PLUGIN_ROOT}/bin/drsi" *)
---

# Dream-RSI: search memory that survives compaction

What was tried lives on disk as an attempt tree, grouped into approach families. You never rely on
remembering it. `drsi` is `"${CLAUDE_PLUGIN_ROOT}/bin/drsi"`; `drsi list` names the campaigns.

## Every attempt, in order

1. `drsi sync -c <campaign>` then `drsi map -c <campaign>`. Read the whole map: families (open, plateau,
   dead, untried), what stopped each, the last 10 attempts, untried directions.
2. Write the proposal to a file: one paragraph naming the mechanism and why it could get past what stopped
   the nearest attempts.
3. `drsi check -c <campaign> --file <proposal>` — exit 0 novel, 3 variant (allowed), 4 duplicate.
   On 4, choose a different mechanism. The judge compares mechanisms, not wording. Duplicates come only
   from the record; the judge's doubts and family warnings are advice to weigh, not vetoes.
4. Do the attempt and record it in the project's own ledger as usual.
5. `drsi sync -c <campaign>` so the next attempt (or the next session) sees it.

## Quick reference

| Need | Command |
|---|---|
| What has been tried | `drsi map -c C` |
| Is this idea new | `drsi check -c C --file P` (`--json` for machine use) |
| Family table | `drsi families -c C --list` |
| New untried-direction suggestions | `drsi families -c C --frontier` |
| Campaign summary | `drsi status -c C` |
| Run the Dream-RSI loop (workers + dreaming) | `/dream-rsi:run` |

## Common mistakes

- Rewording a duplicate until the check passes: it stays the same attempt, and it is recorded as a check.
- Treating a `variant` verdict as permission to repeat the family: a variant must target what stopped it.
- Ignoring a `warning:` line: a family with many attempts since its best is where circling happens.
- Skipping `drsi sync` after landing an attempt: the next check then cannot see it.
