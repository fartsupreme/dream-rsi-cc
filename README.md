# dream-rsi-cc

A Claude Code plugin for long research campaigns that keep re-trying what they already tried.

It has two layers:

1. **Search memory that survives compaction.** Every attempt becomes a node in an attempt tree on disk.
   A classifier fingerprints each attempt's *mechanism* (not its title), groups attempts into approach
   families, and renders a hard-capped map (~4k tokens) of what was tried, what stopped it, and what is
   untried. `drsi check` judges a new proposal against that history before any work starts:
   novel / variant / duplicate.
2. **The Dream-RSI loop** (Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*,
   arXiv 2609.14858). An exploration policy (Python code) chooses which attempts to extend and how many
   to run in parallel. Fresh headless Claude workers run the attempts in git worktrees, and a campaign
   scorer grades them. Each finished round is frozen as a replay world. In the dream phase an agent
   rewrites the policy, each revision is scored by replaying it over every frozen world at zero execution
   cost, and the best one is redeployed only if it does not regress.

## What it does not do

- It does not make the model smarter or add ideas the model cannot have. The paper's gains are mostly
  efficiency (the same quality for fewer calls).
- The paper shows no transfer of a learned policy across tasks. Policies here are per campaign.
- Replay can only evaluate a policy on branches that were actually recorded.
- If a goal is impossible, this shows that faster. It does not change it.

## Install

```
/plugin marketplace add fartsupreme/dream-rsi-cc
/plugin install dream-rsi@dream-rsi-local
```

or, from a local clone, `/plugin marketplace add /path/to/dream-rsi-cc`. The installed copy lives in
Claude Code's plugin cache; after changing a local clone, bump the version in both manifests and run
`claude plugin update dream-rsi@dream-rsi-local`.

The CLI is `plugins/dream-rsi/bin/drsi`. It needs only Python 3.11+ (stdlib) and the `claude` CLI.
Campaign state lives in `~/.dream-rsi/campaigns/<name>/` (override with `DRSI_HOME`), outside every
repository. `drsi` only reads the projects it indexes, and the live loop works in a campaign-owned clone.

## Commands

| Command | What it does |
|---|---|
| `drsi init NAME --goal "..."` | create a campaign |
| `drsi import -c NAME FILE [--preset attempt-ledger\|generic]` | import an attempt log (incremental) |
| `drsi fingerprint -c NAME [--stale]` | classify attempts that lack a fingerprint; `--stale` also re-reads those read under an earlier goal |
| `drsi families -c NAME [--rebuild\|--frontier\|--list]` | build/update families and untried directions |
| `drsi map -c NAME` | print the map |
| `drsi check -c NAME --file P` | novelty verdict for an in-session attempt; exit 0 novel, 3 variant, 4 duplicate |
| `drsi sync -c NAME` | re-import sources, index new attempts and rows corrected since import, rewrite the map |
| `drsi config -c NAME --set a.b=JSON` | change settings |
| `drsi baseline / run / dream / replay -c NAME` | the Dream-RSI loop |
| `drsi gate -c NAME` | hook helper (see `hooks/novelty-gate.example.json`) |

Slash commands: `/dream-rsi:init`, `/dream-rsi:map`, `/dream-rsi:check`, `/dream-rsi:run`,
`/dream-rsi:status`. The model-invoked `dream-rsi` skill carries the per-attempt protocol.

## Scorer contract (live loop)

The scorer runs with cwd set to a **clean checkout of the attempt's commit** and `DRSI_WORKSPACE` exported.
It must exit 0, and its last non-empty stdout line must be
`{"score": number|null, "valid": bool, "fail_class": "ok"|..., "gates": {...}, "summary": "..."}`.
A non-zero exit, a last line that isn't that object, and non-finite or boolean scores are all `eval_error`.
Write the scorer so that:
- Candidate code runs in a child process, and the scorer prints the final line itself, so candidate output
  can never be the last line.
- Inputs are drawn fresh and checked against the scorer's own reference, so hardcoding known answers
  doesn't pay.
- Everything the candidate costs is measured, including work done at import time.

`examples/primes/score.py` follows this pattern. On macOS the scorer runs under `sandbox-exec`. It can read
anything, but write only to its checkout, a private temp directory (exported as `TMPDIR`),
`scorer.allow_write` and the null and descriptor devices (not terminals). It has no network unless
`scorer.network` is `true`. Candidate code it runs therefore can't alter the scorer, the campaign, or shared
temp files. Add build output directories such as a shared `CARGO_TARGET_DIR` to `scorer.allow_write`.

When the scorer finishes, its process group is killed. So is every descendant it was seen to start, and
every process still working inside its checkout or temp dir. The profile also stops the scorer opening
applications through LaunchServices. See **Limits** below for what it does not stop.

**Keep the grader out of reach.** Nothing the scorer runs to judge an attempt (its script, tests, fixtures,
build files) may match `workspace.mutable`. Any change outside `workspace.mutable`, measured from the campaign
base, makes the attempt `out_of_scope`. An attempt is the files its worker leaves in the worktree; what the
worker did with git (commits, the index, `.git` itself) is ignored. Renames, a parent's edits, symbolic links
and nested git repositories all count. `workspace.mutable` is empty by default, and `drsi run` refuses until you set it.

**The scorer is the goal.** Workers optimise what the scorer measures, loopholes included. On the primes
demo, three rounds of real workers found three distinct loopholes:
1. Exploiting timing noise.
2. Moving work to import time, which the scorer then subtracted.
3. Embedding a table of every answer in the tested input range.

Each loophole was closed in the scorer, not the engine. Make the scorer measure exactly what you want, and
expect a campaign's first rounds to find its loopholes.

## Security model (what a worker or a generated policy cannot do)

- **Workers** run headless in Claude Code's Bash sandbox. They can write only to their own worktree and
  proposal directory (nothing under the clone's `.git`), have no network unless `live.allowed_domains` lists hosts, and run with hooks off and no
  user or project settings. They never run drsi themselves. A worker's score comes only from the scorer,
  never from its report.
- **Novelty checks are the orchestrator's, not the worker's.** Each live attempt goes: propose; the
  orchestrator checks the proposal against everything tried; implement. A duplicate goes back with the
  judge's reasons, up to `live.max_proposals` tries, and is recorded as `not_novel` if it never passes.
  The worktree is deleted and recreated after proposing, so nothing a worker does then survives. An attempt
  that never passes the check owns no code, so nothing unchecked can reach its descendants. The
  check can't be skipped or forged, and the proposal recorded is the one that was judged. Parallel attempts
  in a round make two-phase claims, so each check sees the claims made before it. Text identical to an
  earlier attempt or claim is a duplicate without asking the judge. The match uses a hash of the full text,
  so a long shared preamble never counts as identical.
- **The orchestrator never runs git inside a worktree.** A worker can rewrite its worktree's `.git`, and git
  run there would read a repository the worker chose, whose config can name commands git runs. Every git
  command runs in the campaign clone with its own git directory and only the clone's own config, so filter
  drivers in your global git config can't be triggered by an attempt's `.gitattributes`. An attempt is
  committed by snapshotting the worktree's files through a private index. A nested repository (any
  `.git`, in any letter case) or a submodule entry makes the attempt `out_of_scope`. Worktrees are deleted
  as plain files before git forgets them. A git admin dir holding anything but plain files is removed
  before git can block on it, and every git call has a timeout. The proposal file is read only if it is a
  regular file (never a symlink or FIFO). When a worker exits, its detached descendants and any process
  still working in its worktree are killed.
- **Generated policies:**
  - They pass an AST guard. It allows `from <module> import <name>` only, never a module object, and blocks
    private attributes, reflection, format-string and match-pattern attribute access, `:=`, decorators,
    helper classes, special methods, names starting with `__`, catch-all exception handlers, and attribute
    writes to anything but `self`.
  - They then run with restricted builtins (no `open`, `getattr`, `setattr`, `eval`, `exec`, or unrestricted
    `__import__`), freshly executed for every (beta, world) run, in a clean subprocess.
  - That subprocess records only the batches each run requested. The parent process replays those traces on
    its own copy of each world and computes every metric itself, so nothing a policy does to objects in its
    own process can change its score.
  - The question enforces the probe budget exactly, forbids re-entrant probes and `reset()` after probing,
    and counts parallel work only for probes that revealed something.
  - In a live round the policy runs in a child process. The orchestrator validates every batch on its own
    copy of the question and runs the attempts. If the policy thinks longer than `live.think_timeout_s`
    between batches, the orchestrator kills the child's process group, and no handler or `finally` block in
    the policy can prevent it. The attempts already made are kept.
- **The policy developer** runs with `--restricted` in a sandbox directory holding only the policy and its
  replay report. It can't read the replay worlds.
- **Classifier and judge calls** run from a neutral empty directory, so a project's own hooks (for example
  a stop gate) never load inside them.
- **Concurrency:**
  - Tree mutations reload under a file lock, so no process can overwrite another's attempts.
  - Family rebuilds are transactional: a failed rebuild leaves the old families in place. If one is killed
    mid-swap, the next families operation finishes or discards the swap, as the tree shows.
  - Only one `drsi run` or `drsi baseline` runs per campaign at a time.
  - Each attempt is recorded the moment it finishes, so an interrupted batch keeps its finished work.
  - Ctrl-C kills the workers' and scorers' process groups.
  - A judge or classifier outage is recorded as `orchestrator_error`, never as a failed idea.

## Limits

- **The macOS sandboxes are containment, not a boundary against a determined adversary.** Both the scorer's
  `sandbox-exec` profile and Claude Code's worker sandbox start from "allow by default" and deny writes and
  network. They stop accidents and the reward hacking seen so far. They do not stop every form of
  inter-process communication; for example, the scorer profile does not block Apple Events. A candidate
  built to escape could ask another, unsandboxed program to act for it. If you expect adversarial
  attempts, run the campaign in a VM or under a separate user account.
- **A process built to evade cleanup can outlive its worker or scorer.** It has to detach at once and
  leave its directory. It keeps the sandbox it started in, so it still can't write outside its own
  worktree or checkout.
- **The policy guard is load-bearing for dreaming.** A replay run's policy process receives every world
  it is scored on. A policy that got past both the AST guard and the restricted builtins could read
  unrevealed scores. No reviewer has found such a bypass. Live rounds don't share this limit: there the
  policy process only ever sees what was revealed.
- **Re-importing an old tree backfills full-text hashes from the ledger as it is now.** If a ledger
  rewrote a row past its first 600 characters and kept the id, the hash follows the rewritten text.

## Choices the paper leaves open (made here, all configurable)

- **Replay:** a leaf reveals its first recorded child, and a leaf with no recorded child reveals nothing
  and is exhausted. Only leaves and root slots are actions (A(T) = {r} ∪ leaves), so siblings under an
  interior node are unreachable in replay. A world's target and its work normalisation are therefore
  computed over reachable attempts only. Probes that reveal nothing don't count as parallel work.
- **Reward:** Eq. 1 `V = attainment − β1·N + β2·N/max(1,k)` (β1 = β2 = 0.01) is reported per beta, as a
  mean over worlds. Policies are ranked by B.2's Pareto AUC. For each swept beta (`[0, .2, .4, .6, .8, 1]`),
  every run's anytime curve (attainment reached with at most x of the work) is averaged over the worlds that
  carry signal. The frontier over those per-beta mean curves is integrated over work in [0, 1], minus
  λ = 0.25 × the parallel penalty.
  - Reaching good attempts sooner scores higher even when every run explores the whole world.
  - One beta applies to every world, so a policy can't pick its best beta per world.
  - Worlds without a single valid reachable score can't favour any policy.
- **Parallel attempts:** the orchestrator's checks within a round are two-phase claims. Each claim is
  recorded under a lock and then judged without the lock, so checks run concurrently, but each one sees every
  claim before it. Claims from an interrupted earlier round are not treated as in flight.
- **Workspaces:** build and runtime artifacts (`__pycache__/`, `*.pyc`, `target/`, …, plus
  `workspace.ignore`) are excluded in the clone and never count as edits.
- **Defaults** follow pi-dream-rsi (juanmackie, MIT): W = 4, K1 = 6, M = 3, root slots, and the
  determinism rerun (every policy is replayed twice under different hash seeds).
- **Seed policy π1:** the paper's parallel refinement, plus a plateau rule and a coverage rule.
- **Policy safety:** an AST guard (prefix-only: no private attributes, reflection, I/O or
  non-allowlisted imports), then a clean subprocess with a timeout.

## Tests

```
cd plugins/dream-rsi/engine && python3 -m unittest discover -s tests -t .
```
