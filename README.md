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
| `drsi check -c NAME --file P` | novelty verdict for an in-session attempt; exit 0 novel, 3 variant, 4 duplicate, 5 off target, 6 retry |
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
  recorded live whose every attempt keeps its cell (without cells a world cannot say which child was opened from
  which leaf, and an attempt a prune re-parented would read as a continuation live never ran; older worlds get
  their cells from the tree), and among those, on the worlds where the incumbent's replay stays on the record
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
- **Transcripts:** every worker call (each proposal and the build) and every policy-developer call writes its whole
  session to `logs/workers/<attempt>/<UTC time>.jsonl` or `logs/developer/<UTC time>.jsonl` in the campaign. The first
  line is the call itself (model, prompt, the brief appended to the system prompt); after it, Claude Code's
  `stream-json` events as they happen: each message, tool call and result, usage events and the final result. A
  call killed at `live.timeout_s` keeps everything up to the kill, and its recorded error names the transcript, its
  last event and the last usage status it saw; anything the call wrote to stderr is kept beside it
  (`<time>.stderr.txt`). Files are streamed, never read back whole, and never overwritten. An attempt's
  `worker.transcript` is its last call's. Prunes leave transcripts in place (they are how a failed worker is
  diagnosed), and they are kept until you delete them.
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
