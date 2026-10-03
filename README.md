# dream-rsi-cc

A Claude Code plugin for long research campaigns that keep re-trying what they already tried.

It has two layers:

1. **Search memory that survives compaction.** Every attempt becomes a node in an attempt tree on disk.
   A classifier fingerprints each attempt's *mechanism* (not its title), groups attempts into approach
   families, and renders a hard-capped map (~4k tokens) of what was tried, what stopped it, and what is
   untried. `drsi check` judges a new proposal against that history before any work starts:
   novel / variant / duplicate, plus off_target (a new variant not aimed at what stopped its family: revise it)
   and retry (a located fix to an attempt a bug stopped before its mechanism was measured).
2. **The Dream-RSI loop** (Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*,
   arXiv 2609.14858). An exploration policy (Python code) chooses which attempts to extend and how many
   to run in parallel. Fresh headless Claude workers run the attempts in git worktrees, and a campaign
   scorer grades them. Each finished round is frozen as a replay world. In the dream phase an agent
   rewrites the policy, each revision is scored by replaying it over the frozen worlds at zero execution
   cost, and the best one is redeployed only if the record shows it doing better than the incumbent (see
   **Deploying a revision**).

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
| `drsi check -c NAME --file P` | novelty verdict for an in-session attempt; exit 0 novel, 3 variant, 4 duplicate, 5 off target, 6 retry, 7 no verdict (a usage limit stopped the check) |
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
  settings source at all (`--setting-sources ""`: no user, project or local settings, and no CLAUDE.md or
  CLAUDE.local.md, which a worker could otherwise leave in its checkout for a continuation of it to load); what they
  need comes in `--settings` and the call's flags. They never run drsi themselves. A worker's score comes only from the scorer,
  never from its report.
- **No Claude Code memory.** Claude Code's auto memory loads a project's `MEMORY.md` into every session and lets
  the file tools write notes beside it, outside the Bash sandbox; workers of one campaign share a project (their
  checkouts are worktrees of one clone), so it would be a channel between them that the orchestrator neither sees
  nor checks. Every call the engine makes (workers, the policy developer, the classifier and the judge) loads no
  settings source and runs with it off (`autoMemoryEnabled: false` where the call takes settings,
  `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` and `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` always), and a worker may neither write
  nor read `~/.claude` (or `CLAUDE_CONFIG_DIR`, when set): a session may write its own memory folder even with auto
  memory off, and every session's notes and transcripts are there. Notes a campaign's workers left before 0.4.3 stay
  in `~/.claude/projects/<its repo>/memory/`; no call loads them and no worker can read them.
- **Heavy runs elsewhere (`live.offload`) never open the worker's sandbox.** A worker asks for a run by writing a
  request file into its own proposal directory; the orchestrator, outside the sandbox, checks it (an argument list,
  a directory inside the checkout, limits within the configured maximum) and runs the one configured command with
  it, never a shell. The directory is the worker's to write and the orchestrator is outside the sandbox, so it
  reaches every file there through a handle on a directory it made itself for the call (whatever stood in its place,
  a link included, is moved aside, never followed, then removed within bounds, so requests an earlier call left are
  not run), reads a request only from a plain file with one link, creates the output file anew, puts the status file
  in place by a rename, and never repeats a request's values back. A request runs only while a process holds the
  lock on its `<id>.lock` (a plain file with one link; the helper holds it for its life), a run whose output passes
  64 MB is stopped, and a directory flooded with entries is no longer served. The checkout and the directory the command is given are the worker's to write, so
  the configured command must treat them as such (follow no links there, pass them to no shell unquoted); what it
  runs, and where, is the campaign's to contain.
- **Novelty checks are the orchestrator's, not the worker's.** Each live attempt goes: propose; the
  orchestrator checks the proposal against everything tried; implement. A duplicate or an off-target variant goes
  back with the judge's reasons, up to `live.max_proposals` tries, and is recorded as `not_novel` if it never passes.
  How the check decides: BM25 over fingerprints retrieves the nearest attempts, with the proposal's own fingerprint
  added to the query (a renamed repeat then shares the record's plain words), and any `#id` the proposal cites
  shown first, next to the search hits rather than in place of them; an LLM judge compares the proposal with only
  what it was shown, and a citation counts only if the judge was shown that attempt. A variant not aimed at what
  stopped its family is off target, not a duplicate: it may be revised. A located fix (where the bug is and what the
  fix is) to an attempt a bug stopped before its mechanism was measured is a retry, allowed like a variant; a retry
  in a dead family, or of an attempt whose mechanism was measured, is a duplicate. Before any other duplicate verdict
  is final, a second pass reads the full cited records, the other attempts the judge was shown and the in-flight
  proposals (one in flight counts as tried), every field and id flattened to one capped line with look-alike fence
  markers broken. It may confirm a repeat of any of them, overturn the duplicate only by naming a concrete difference
  and no repeated attempt, and to
  a retry only by naming the exact attempt, an unmeasured one, and the located fix. A duplicate that a rule made (a
  variant with no difference, a retry of an attempt not shown or barred) is final and not sent to it.
  Every attempt and in-flight proposal the judge sees carries a label of its own, so ids that flatten alike still
  lead a citation to the record it names, and a "difference" must hold a visible letter or digit.
  On a planted test set (88 recorded attempts, 65 labelled probes, a real judge) this took non-repeats wrongly made
  final duplicates from 3 of 28 to 0 in each of five runs of the last versions of the check; true repeats let through
  went from 1 of 37 to 0 or 1, as the judge varies between runs (twice a first judge took a schedule tweak or a
  constant change for a variant aimed at the stopper). One or two variants not aimed at a stopper were sent back
  as off target.
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
  - The question enforces the probe budget exactly and forbids re-entrant probes and `reset()` after probing.
  - Replay, the behaviour gate and a live round hand the policy the same view: attempts are shown under ids that
    say only when they were revealed (`a1`, `a2`, ...), no family is shown (live, it depends on when the classifier
    reached an attempt), the question prints as `<question>`, the policy is built
    with no argument at its default beta, as a live round builds it, and every policy process runs at one hash seed
    (a policy can read the seed through set order; the second replay run uses another seed, and a policy whose
    traces differ between the two runs is refused).
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
  - Ctrl-C or `drsi stop` ends everything the run started (see **Stopping a run** below).
  - A judge or classifier outage is recorded as `orchestrator_error`, never as a failed idea.

## Limits

- **The macOS sandboxes are containment, not a boundary against a determined adversary.** Both the scorer's
  `sandbox-exec` profile and Claude Code's worker sandbox start from "allow by default" and deny writes and
  network. They stop accidents and the reward hacking seen so far. They do not stop every form of
  inter-process communication; for example, the scorer profile does not block Apple Events. A candidate
  built to escape could ask another, unsandboxed program to act for it. If you expect adversarial
  attempts, run the campaign in a VM or under a separate user account.
- **Claude Code's own temporary folder is writable.** Its sandbox lets every session write `/tmp/claude-<uid>` (where
  it keeps background tasks' output), so a worker can write there besides its checkout and proposal directory.
  Nothing a later worker or the orchestrator loads lives there.
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

- **Replay:** a root slot reveals the root opened in that slot live, and a leaf reveals the child opened from it
  (in a world recorded before attempts kept their cells, the roots in order and each first child). Only leaves and
  root slots are actions (A(T) = {r} ∪ leaves), so siblings under an interior node, and an attempt a prune moved
  (re-rooted, or re-parented), are unreachable in replay. A world's target and its work normalisation are therefore
  computed over reachable attempts only. Replay shows a policy exactly what a live round
  would, and root slots never run out. Replay can value only what was recorded, in breadth and in depth alike, so
  the first probe past the record (a root slot beyond the recorded roots, a leaf beyond the end of its recorded
  branch) ends that run: the batch's recorded cells count, the probes past the record cost budget and reveal
  nothing, and the rest of the budget counts as empty batches. The policy cannot catch the end, and nothing it
  could learn past the record counts, so a replayed run never scores above what the same policy does live on the
  same attempts. Recorded roots open in the slots they were opened in live (each attempt keeps its cell), so a
  policy replaying a round it ran meets every attempt where it met it. Ranking always runs under a budget
  (K1 × W probes per world, as a live round).
- **Reward:** the paper states Eq. 1 `V = max s_v − β1·N + β2·N/max(1,k)` in its method section but ranks
  policies by `pareto.auc − λ · parallel_penalty` over a beta sweep in the prompt it ran (Appendix B.2); the two
  disagree. Here Eq. 1 (with attainment for s_v, β1 = β2 = 0.01) is reported, and policies are ranked by an
  anytime AUC: a run's curve of attainment reached with at most x of the work, averaged over the worlds that
  carry signal and integrated over work in [0, 1], minus λ = 0.25 × the parallel penalty.
  - The policy is scored at its own default beta, the one that runs live; the sweep over
    `[0, .2, .4, .6, .8, 1]` is reported but earns nothing, so behaviour at betas that never run cannot win.
  - The parallel penalty is 1 − the mean batch fill as live counts it: the cells a batch probes out of W. A run that
    stops before its budget counts each batch it left as empty, so stopping never escapes the charge an under-filled
    continuation pays. Rounds 19 to 23 measured fill against what the record could answer and capped root slots at
    the recorded roots; three reviews found revisions that gained in replay only through that cap (stopping after
    the roots, dropping the plateau rule, opening every root first), and later reviews found revisions that told
    replay from live by an empty probe, by ids, by the question's class or by how the policy was built. One view in
    every environment removes what they all used. The older penalties are gone: a campaign whose `dream.penalty`
    still says `"support"` or `"realized"` ranks by `"live"` and warns; an unknown `dream.score` or `dream.curve`
    ranks by the default and warns.
  - The cells of one batch are credited worst first, best last, by score: they run in parallel live, so the order
    a policy lists them in earns nothing (nor do the ids, which live numbers in that order).
  - Reaching good attempts sooner scores higher even when every run explores the whole world.
  - Worlds without a single valid reachable score can't favour any policy.
- **Deploying a revision:** replay is exact only where the record answers. Policies are compared on the rounds
  recorded live whose cells are exactly what live records: each root in a `root:<n>` slot of its own and every other
  attempt opened from its parent (without cells a world cannot say which child was opened from which leaf, and a
  prune re-parents an attempt but keeps its cell, so replay cannot reach it and the world's best drops; older worlds
  get their cells from the tree), and among those, on the worlds where the incumbent's replay stays on the record
  (every round it recorded itself, and any other whose record covers its whole path): there its value is exact, and a candidate's run, which ends at its first probe past the
  record, can only score below what it does live on the same attempts, so a candidate that wins there does better
  live on them. On a world the incumbent leaves, its own replay would be cut short and any candidate resembling the
  world's recorder would look better. A candidate that probes past the
  record ends its run there, which errs toward keeping the incumbent: a revision that explores where the record is
  thin looks no better than the incumbent, and a revision the record can vouch for (one that reaches the
  incumbent's own best attempts sooner, say) deploys. Strictly better replay reward is not enough: the 5th
  percentile of a paired bootstrap of the reward difference across those worlds must exceed `dream.margin`, and
  the revision must change what the policy does on live-like trees without spending fewer probes there than the
  incumbent. A tie keeps the incumbent, and so does an incumbent that fails replay (nothing can be compared with
  it; no revision is asked for). No dream runs, in the loop or in `drsi dream`, until `dream.min_worlds`
  (default 4, at least 1) of those worlds can separate policies (any with a valid score replay can reveal: even
  roots alone do, by which slots a policy opens first); the reason is printed. Worlds an earlier policy recorded count
  only where the deployed policy's replay stays on their record, so after a deploy (or a change of W or K1) the
  comparison is rebuilt from the rounds the deployed policy runs.
  The old ranking's score and curve stay available as `dream.score = "sweep"` and `dream.curve = "reveal"`
  (or `"batch"`, `"clock"`), but the guarantee above holds only for the defaults: those options can reward
  behaviour a live round never shows, and the dream warns when one is set. `drsi replay` ranks with the dream
  step's formula over every world and says in how many the deployed policy stays on the record (the dream compares
  on those alone). The history world (`--history`, imported rows, some scored by the classifier's outcome) was
  never a live round, so the dream does not compare on it. The bound errs one way: a revision that goes past the
  record is valued as if every probe it had left were an empty batch, so the dream can pass over a revision that
  would do better live by exploring where nothing was recorded, and never deploys one that does worse.
- **Parallel attempts:** the orchestrator's checks within a round are two-phase claims. Each claim is
  recorded under a lock and then judged without the lock, so checks run concurrently, but each one sees every
  claim before it. Claims from an interrupted earlier round are not treated as in flight.
- **Workspaces:** build and runtime artifacts (`__pycache__/`, `*.pyc`, `target/`, …, plus
  `workspace.ignore`) are excluded in the clone and never count as edits.
- **Stopping a run:** `drsi stop -c NAME` ends a running `drsi run` and everything it started: it asks the run to end
  through its own cleanup, and sends SIGKILL after `--grace` seconds. A run holds `logs/run.lock` for its whole life,
  records every process group it starts (workers, scorers, the policy, the developer) in `logs/run-children.json`
  with its own pid and start time, and starts a guardian in a session of its own. The guardian notes every process in
  the run's tree as it goes and keeps that list on disk, since a worker's shell commands run in sessions of their own
  and can leave the workspaces. If the lock comes free while that record still exists (Ctrl-C, a crash, a kill), the
  guardian stops the recorded groups and every process on its list that is still the same process (pid and start
  time), with everything descended from them, then kills them all, then every headless process of yours still
  working in the run's workspaces, and exits. A terminal or an editor you opened in a workspace is never touched. A
  run that ends by itself sweeps its own tree the same way and closes the record; an interrupted one sweeps, starts no
  new process, and leaves the record for the guardian. `ps` only identifies processes: a process table or a
  workspace that cannot be read counts as "unknown", never "dead" or "empty", and the record stays for the next
  attempt. A new `drsi run` also finishes a dead run's cleanup first, and will not start over a record it could not
  finish.
- **Stopping at a round boundary:** `drsi stop -c NAME --after-round` asks the run to end without cutting an attempt
  short: the round under way finishes, its world is frozen and its dream step is taken (a dream, or the usual skip
  while too few worlds can inform one), and the run exits before it starts another. Use it to restart onto a new
  release or new settings. The request (`logs/stop_after_round`) names the run, so no later run takes it, and a run
  started by a drsi without this option is refused (it would never see the request: stop it with `drsi stop` once its
  round is over). `--wait` follows that run until it has ended, however long the round takes, and reports what the
  run itself recorded (`logs/stopped_after_round`): exit 0 if it stopped at the boundary, 1 if it ended some other
  way first (the request is then withdrawn). A run that is stopped mid-round anyway records each attempt the stop cut
  short as the loop's failure (`orchestrator_error`, `worker.stopped_by_run`), never its model's: a call the stop
  ended commits no half build, a novelty check it interrupted gives no verdict, and a finished build it left unscored
  stays recorded for `drsi rescore`. A call that failed on its own before the stop, and a verdict reached before it
  (valid, not novel, out of scope), keep their records.
- **Rescoring:** when the scorer is corrected mid-campaign, `drsi rescore -c NAME --all` (or `--ids a,b`) runs the
  current scorer on each live attempt's own commit, as the loop scores it, keeps the old reading on the node
  (`artifacts.rescored`), judges every live outcome again against its parent's score, and updates the frozen round
  worlds and the map. Attempts that never reached the scorer (not novel, out of scope, a failed worker) are left alone.
- **Pruning:** `drsi prune -c NAME --error-match TEXT` (or `--ids a,b,...`, `--dry-run` first) removes recorded
  attempts that did no work: a live attempt that failed as a worker or orchestration error, with no score and no
  changed files. It moves anything that continued from one to its nearest kept ancestor, drops them from the frozen round worlds (a world
  left empty goes), deletes their branches, worktrees and proposal directories, withdraws their novelty claims, and
  logs each removal to `logs/prune.jsonl`. An attempt that did work is refused whatever it matches, and while a run
  is live the round it is running (which the run notes in `logs/current_round`) is left alone. The tree is changed
  last, after the log line (written so a torn earlier line cannot swallow it), so a prune that
  fails midway leaves nothing pointing at missing attempts and can be run again to finish; prune and rescore rewrite
  worlds under one lock; a pruned round's id is never handed out again.
- **Worker models:** workers run on `llm.worker_model` (default `opus`). To draw ideas from more than one model at once,
  set `llm.worker_models` to a list: slot i of each batch runs `worker_models[i % len]`, so `["opus", "fable"]` with
  `search.W = 6` runs three of each. The assignment rotates by one slot each batch: a policy lists a batch best cell
  first, so a fixed assignment would always give the last model the least promising cell. One model proposes and
  builds an attempt, every attempt records its model, and so does each frozen world.
- **A model at its usage limit:** with `llm.worker_fallback` set to a model, a worker call refused for its model's
  usage limit before the model ran (its result shows no turn of work, one turn and no API time, and the stream's last
  usage event says `rejected`; it comes back within seconds) is made again on the fallback, and the rest of that
  attempt runs on it. A call the limit cuts off after work goes to the fallback the same way, once what it did is
  undone (see the next point), and a short-term rate limit Claude Code retries away does not count. The attempt
  records the model that did the rest of the work, the one it was given (`worker.fell_back_from`) and the phase the
  fallback took over (`worker.fell_back_in`: after a proposal on its own model, the build may be the fallback's); a
  fallback refused as well is not retried: the attempt keeps its own model and its own refused call (session,
  transcript, error), and names the refused fallback in `worker.fallback_refused`. Every attempt tries its own model
  first, so a model whose limit has reset, or whose account has changed, is back at once. A call on the fallback that
  is refused in turn tries the attempt's own model; if that runs, the attempt goes on there (`worker.model` then
  equals `fell_back_from`, and `fell_back_in` names the phase the fallback took). The fallback covers worker calls
  only (not the classifier, the judge or the policy developer) and compares model names as written: give it in the
  same form as the workers' models (`worker_models`, or `worker_model` or `model` when that list is unset). Every
  call keeps its own transcript in the attempt's folder, the refused one included.
- **The account at its usage limit:** a call refused for a usage limit before the model ran is never a failed
  attempt. When no model the attempt may use can run (no fallback, the fallback is the same model, or it is refused
  as well), the attempt waits for the earliest reset the refusals name, checking again at least every 10 minutes (so
  switching the account ends the wait sooner), then makes the call again, its own model first; the run log says when
  it will try, and the attempt records the time in `worker.waited_for_limit_s`. The limit's kind does not decide
  this: Claude Code names one of `five_hour`, `seven_day`, `seven_day_opus`, `seven_day_sonnet`,
  `seven_day_overage_included` and `overage`, or none, and reports the exceeded window that resets last, so a kind
  cannot say which models a refusal binds. A model with a limit of its own and no fallback therefore holds its batch
  until the limit resets; set `llm.worker_fallback` to keep the other seats working meanwhile. The novelty check
  waits the same way when its calls are refused, and so does the dream's policy developer. A run that is stopped ends
  every wait. Without this, a limit turned every call of a round into a recorded failure within seconds, and the loop
  ran through rounds doing no work. A call the limit cuts off after the model has worked (an error result of status
  429, with the stream's last usage event rejected or Claude Code's synthetic `rate_limit` message) is treated the
  same, once what it did is undone: a build is made again on a fresh checkout at the attempt's start, a proposal on a
  fresh checkout, and what the call left in the attempt's proposal directory is cleared (the accepted proposal kept
  for a build); the policy developer is asked again in a sandbox reset to the policy it was given. The attempt counts
  these calls in `worker.cut_off_by_limit`. Claude Code retries a 429 that clears within a minute itself and ends a
  call on any other with the same message, a usage limit's or not, so the attempt waits out a call stopped after work
  at most three times (with a fallback, each wait follows a try on both models) and then its failure stands: a usage
  limit stops a call once a window, and a 429 that keeps coming is something else. A refusal before any work is
  waited out however long the limit lasts. A run stopped during such a wait undoes the cut-off call too, so its half
  work is never recorded as the attempt's, and the attempt records `worker.stopped_in_limit_wait`. The novelty
  check's calls are single prompts that change nothing, so a usage limit there is waited out the same way (a call
  stopped after work at most three times), and `drsi check` stopped by one gives no verdict and exits 7.
- **Claude Code updating itself:** Claude Code updates itself through npm, which renames its installed package and
  the `claude` link away, links a placeholder, then puts the new binary in place; a call started in that window, a
  few seconds long, finds no binary or one it cannot execute. Such a call (a worker's, the policy developer's, the
  judge's or the classifier's) waits until `claude` is found on the call's own `PATH` twice 2 seconds apart and is
  started again, until it starts or 5 minutes have passed for that call; a run that is stopped ends the wait. A call
  that never starts fails as before, except that a worker's attempt is recorded as the loop's failure
  (`orchestrator_error`), not the model's, and a policy revision fails without stopping the run. Any other missing
  file, such as the working directory, fails the call at once.
- **Transcripts:** every worker call (each proposal and the build) and every policy-developer call writes its whole
  session to `logs/workers/<attempt>/<UTC time>-<random>.jsonl` or `logs/developer/<UTC time>-<random>.jsonl` in the
  campaign. The first
  line is the call itself (model, prompt, the brief appended to the system prompt); after it, Claude Code's
  `stream-json` events as they happen: each message, tool call and result, usage events and the final result. A
  call killed at `live.timeout_s` keeps everything up to the kill, and its recorded error names the transcript, its
  last event and the last usage status it saw; anything the call wrote to stderr is kept beside it
  (`<name>.stderr.txt`). Transcripts are streamed, never read back whole, and no file is overwritten. A call's result is the
  worker's own last result; when a background task ends, Claude Code runs a turn of its own, whose result replaces
  it only if that turn succeeded with a report (one still running leaves system events after the last result). The
  call fails if an earlier result of its own turns was an error or anything but whole system events follows the
  last result. A call that times out
  keeps its stderr too. An attempt's
  `worker.transcript` is its last call's. Prunes leave transcripts in place (they are how a failed worker is
  diagnosed), and they are kept until you delete them.
- **Heavy computation off the workers' machine:** workers run where the loop runs, and so does everything they
  start. A campaign whose experiments can outgrow that machine sets `live.offload` = `{"cmd": PATH, "mem_gb": 8,
  "secs": 1800, "max_mem_gb": 24, "max_secs": 3500, "note": "what the host has"}`. Each worker is told to run
  anything more than a quick check with `python3 <engine>/drsi/offload.py [--mem GB] [--secs S] -- COMMAND
  [ARGS...]` from its checkout: the orchestrator runs `PATH CHECKOUT MEM_GB SECONDS DIR -- COMMAND [ARGS...]` (DIR
  relative to the checkout) while the worker's call lasts, and the helper streams its output and exits with its
  status. `PATH` is the campaign's, an absolute path outside the workers' checkouts (it runs with the checkout as its
  directory): ship the checkout to a compute host and run the command there in a sandbox with those limits, say. A headless session has no later turn (when a worker stops, its session
  ends and its runs are stopped), so the brief tells workers to wait for what they need: a run longer than the shell's
  command limit (10 minutes at most) starts in the background, and `offload.py --wait` (with no ID, the worker's one running request; else
  `--wait ID`, the request ID the helper prints first) attaches to it, printing its output and exiting with its
  status, or printing `offload: the run is still going` after `--for` seconds (default 540) while it goes on. A run is stopped when its helper ends or the worker's call does, and when it ends on
  its own its process group goes with it: the group and the processes the tracker saw it start are killed (one that
  detaches before the tracker's first look, every 0.5 s, and leaves the checkout escapes, as for workers), and a
  command that starts work on another machine must stop that work when it is killed. What a run writes stays where it
  ran. A maximum set alone lowers its default to it; bad settings (a cmd that is not an executable file, say) stop
  `drsi run` and `drsi baseline` before they start anything.
- **Workers' memory:** `live.worker_mem_gb` caps what a worker's processes on the loop's machine hold together:
  while their total is over it, the largest is killed (never the worker's Claude Code process itself). On macOS a
  process counts its physical footprint (the memory it owns, resident, compressed or swapped out; the resident size
  alone reads low exactly when memory is short, though it also counts clean pages of mapped files, which the
  footprint does not), elsewhere its resident size. The worker is told the cap, and the attempt records
  each kill, from any of its calls, in `worker.mem_kills`. A process that detaches before the tracker's first look
  (every 0.5 s) escapes the cap; the cleanup at the end of the call still finds it while it works in the checkout. It is a guard, not a place for
  heavy work: with `live.offload` set, a worker has somewhere to run what the cap stops.
- **What counts as a pass:** a live attempt's outcome is `pass` when its score beats its parent's (or, for a new
  branch, the baseline) by more than `live.pass_margin` (default 0). Set it to about twice the scorer's
  test-retest noise, or re-measuring the same code will pass about half the time and keep a stalled family open.
- **Workspace objective:** when the workspace can build and score only part of the goal, say which part in
  `live.objective`. Every worker's brief carries it after the goal, and `drsi families` asks the frontier for
  directions a worker can build there. Each suggested direction goes to one new branch of a round; branches past
  the frontier's length choose from the map. Each suggestion is checked against the record before anyone sees it:
  a duplicate or off-target one is dropped (listed under `frontier_dropped` in families.json), and one judged a
  variant names its nearest attempts. If the check itself fails, the suggestion is kept, recorded as `checked: false`
  in families.json (a worker's own proposal is still checked before it builds).
- **Moving the base:** to correct the fixed files workers read (a brief, a README, the scorer) mid-campaign,
  commit the change in the source repository and set `workspace.base` to that commit. The next round takes it
  up if it descends from the pinned base. New branches start on it; a continuation starts on it with its
  parent's own edits laid on top, so it sees the corrected files and its scope is measured as before. A base
  that does not descend from the pinned one, or a different source repository, still needs a new campaign.
- **Defaults** follow pi-dream-rsi (juanmackie, MIT): W = 4, K1 = 6, M = 3, root slots, and the
  determinism rerun (every policy is replayed twice under different hash seeds).
- **Seed policy π1:** the paper's parallel refinement, plus a plateau rule and a coverage rule.
- **Policy safety:** an AST guard (prefix-only: no private attributes, reflection, I/O or
  non-allowlisted imports), then a clean subprocess per replay run, each with a timeout (120 s). A revision's
  whole evaluation may take 4× the incumbent's (at least 120 s): the loop waits for the dream.

## Tests

```
cd plugins/dream-rsi/engine && python3 -m unittest discover -s tests -t .
```
