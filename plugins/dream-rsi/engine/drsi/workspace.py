"""Per-attempt workspaces as git worktrees of a campaign-owned clone.

Each attempt gets branch drsi/<node> started from its parent attempt's commit (or
the campaign base for a new branch), so the attempt tree and the git tree match.
The scope of an attempt is everything that differs from the campaign base. Scoring
uses a fresh detached checkout of the committed attempt. The source repository is
only ever cloned from, never written to.

The base may move forward (`workspace.base` set to a descendant of the pinned
commit), so the fixed files workers read can be corrected mid-campaign. A parent
built on an earlier base is then continued from the current base with the parent's
own edits laid on top.

A worker can write anything inside its worktree, including the `.git` file that
tells git where the repository is. Git run there would read a repository the
worker chose, whose config can name commands git runs (core.fsmonitor,
diff.external, ...). So the orchestrator never runs git inside a worktree: every
command runs in the clone with its own git directory, and an attempt is committed
by snapshotting the worktree's files through a private index.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unicodedata
from pathlib import Path

DEFAULT_IGNORE = ["__pycache__/", "*.pyc", "*.pyo", ".DS_Store", "target/", ".pytest_cache/", "*.egg-info/",
                  "node_modules/", ".mypy_cache/"]


GIT_TIMEOUT = 1800


def _git(cwd, *args, check=True, env=None, binary=False):
    """git's output as text, or as bytes with binary=True (paths exactly as stored: text mode would turn a
    carriage return in a file name into a newline)."""
    text = {} if binary else {"text": True, "errors": "surrogateescape"}
    try:
        proc = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.quotepath=off",
                               "-c", "core.fsmonitor=false", *args], cwd=cwd, capture_output=True, env=env,
                              timeout=GIT_TIMEOUT, **text)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {' '.join(map(str, args))} timed out in {cwd}") from None
    if check and proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace") if binary else proc.stderr
        raise RuntimeError(f"git {' '.join(map(str, args))} failed in {cwd}: {err.strip()[-400:]}")
    return proc.stdout


def out_of_scope(files: list[str], globs: list[str]) -> list[str]:
    return [f for f in files if not any(fnmatch.fnmatch(f, g) for g in globs)]


def _is_dot_git(name: str) -> bool:
    # case-insensitive and normalisation-insensitive filesystems (macOS) resolve ".GIT" to ".git"
    return unicodedata.normalize("NFC", name).casefold() == ".git"


def _rmtree(path) -> None:
    """Remove a tree a worker may have made hard to remove (read-only dirs, immutable flags)."""
    def fix(func, p, _exc):
        for q in (p, os.path.dirname(p)):
            try:
                if hasattr(os, "lchflags"):
                    os.lchflags(q, 0)
                if not os.path.islink(q):
                    os.chmod(q, 0o700)
            except OSError:
                pass
        try:
            func(p)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=fix)
    else:
        shutil.rmtree(path, onerror=lambda f, p, ei: fix(f, p, ei[1]))


class Workspaces:
    def __init__(self, camp_root, source_repo, base: str | None = None, ignore: list[str] | None = None):
        self.root = Path(camp_root)
        self.source = Path(source_repo).expanduser()
        self.base = base
        self.ignore = list(DEFAULT_IGNORE) + list(ignore or [])
        self.repo = self.root / "repo"
        self.work = self.root / "work"
        self.base_commit: str | None = None
        self._starts: dict[str, str] = {}
        self._starts_lock = threading.Lock()

    def ensure_clone(self) -> None:
        pin = self.root / "base_commit.json"
        if not (self.repo / ".git").exists():
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / "repo.partial"
            if tmp.exists():
                shutil.rmtree(tmp)
            _git(self.root, "clone", "--quiet", "--no-hardlinks", str(self.source), str(tmp))
            if self.base:
                _git(tmp, "checkout", "--quiet", self.base)
            _git(tmp, "config", "user.email", "dream-rsi@localhost")
            _git(tmp, "config", "user.name", "dream-rsi")
            tmp.rename(self.repo)  # only a clone at the right base ever becomes the campaign clone
        if pin.exists():
            recorded = json.loads(pin.read_text())
            if (recorded["source"], recorded["base"]) != (str(self.source), self.base):
                recorded = self._move_base(pin, recorded)
            self.base_commit = recorded["commit"]
        else:
            self.base_commit = _git(self.repo, "rev-parse", "HEAD").strip()
            pin.write_text(json.dumps({"source": str(self.source), "base": self.base, "commit": self.base_commit}))
        # Build and runtime artifacts (bytecode, target/, ...) are never part of an attempt.
        exclude = self.repo / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text().splitlines() if exclude.exists() else []
        missing = [p for p in self.ignore if p not in existing]
        if missing:
            exclude.write_text("\n".join(existing + missing) + "\n")

    def _move_base(self, pin: Path, recorded: dict) -> dict:
        """The config names a new base. A descendant of the pinned commit in the same source is taken up;
        anything else would change what earlier attempts were measured against, so it needs a new campaign."""
        refuse = RuntimeError(f"the campaign clone was made from {recorded['source']} at {recorded['base']}; "
                              f"the config now says {self.source} at {self.base}. Start a new campaign.")
        if recorded["source"] != str(self.source) or not self.base:
            raise refuse
        try:
            commit = _git(self.source, "rev-parse", "--verify", "--quiet", f"{self.base}^{{commit}}").strip()
        except RuntimeError:
            raise refuse from None
        self._in_repo("fetch", "--quiet", "--no-tags", str(self.source),
                      "+refs/heads/*:refs/drsi-source/heads/*", "+refs/tags/*:refs/drsi-source/tags/*")
        if not self._in_repo("cat-file", "-t", commit, check=False).strip() == "commit":
            raise RuntimeError(f"the new base {self.base} is on no branch or tag of {self.source}")
        if self._in_repo("merge-base", recorded["commit"], commit, check=False).strip() != recorded["commit"]:
            raise refuse
        history = list(recorded.get("history", []))
        if commit != recorded["commit"]:  # another name for the pinned commit is not a move
            history.append({"base": recorded["base"], "commit": recorded["commit"]})
        moved = {"source": str(self.source), "base": self.base, "commit": commit, "history": history}
        tmp = pin.with_name(pin.name + ".new")
        tmp.write_text(json.dumps(moved))
        os.replace(tmp, pin)
        return moved

    def start_for(self, parent_commit: str | None) -> str:
        """The commit an attempt starts from: the base for a new branch; its parent's commit when that already
        stands on the current base; otherwise (the base moved after the parent was built) the current base
        with the parent's own edits laid on top, so the attempt sees the new fixed files and keeps its
        parent's work, and its scope is measured as before."""
        if not parent_commit:
            return self.base_commit
        with self._starts_lock:
            if parent_commit not in self._starts:
                old = self._in_repo("merge-base", parent_commit, self.base_commit, check=False).strip()
                self._starts[parent_commit] = (parent_commit if old in ("", self.base_commit)
                                               else self._onto_base(parent_commit, old))
            return self._starts[parent_commit]

    def _onto_base(self, parent_commit: str, old_base: str) -> str:
        """The new base's tree with the parent's own changes (its diff from the base it was built on) laid on
        top, as a rebase would: a file the parent never changed follows the new base, deletions included. A
        changed shape (a directory the parent made a file, or the base did) goes the parent's way, and the
        result is judged by the scope check like any attempt, never refused here."""
        out = self._in_repo("diff", "--no-renames", "--no-abbrev", "--raw", "-z", old_base, parent_commit,
                            binary=True)
        parts = out.split(b"\0")
        changes = [(parts[i].lstrip(b":").split(), parts[i + 1]) for i in range(0, len(parts) - 1, 2)]
        with tempfile.TemporaryDirectory(prefix="drsi-index-") as d:
            index = Path(d) / "index"
            self._in_repo("read-tree", self.base_commit, index=index)
            for fields, path in sorted(changes, key=lambda c: not c[0][4].startswith(b"D")):  # removals first
                if fields[4].startswith(b"D"):
                    self._in_repo("update-index", "--force-remove", "--", path, index=index)
                else:
                    self._in_repo("update-index", "--add", "--replace", "--cacheinfo",
                                  fields[1] + b"," + fields[3] + b"," + path, index=index)
            tree = self._in_repo("write-tree", index=index).strip()
        return self._in_repo("commit-tree", tree, "-p", parent_commit, "-p", self.base_commit,
                             "-m", f"dream-rsi: {parent_commit[:12]} on the moved base").strip()

    def path(self, node_id: str) -> Path:
        return self.work / node_id

    def _in_repo(self, *args, check=True, work_tree=None, index=None, binary=False):
        """git on the clone's own metadata, run from the clone; never from a worktree. Only the clone's own
        config applies: filter drivers in the user's global or system config could otherwise be named by
        an attempt's .gitattributes and run on its files."""
        env = dict(os.environ)
        for k in [k for k in env if k.startswith("GIT_")]:
            del env[k]
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        self._scrub_admin()
        flags = ["--git-dir", str(self.repo / ".git")] + (["--work-tree", str(work_tree)] if work_tree else [])
        return _git(self.repo, *flags, *args, check=check, env=env, binary=binary)

    def _scrub_admin(self) -> None:
        """Git opens every registered worktree's admin files (HEAD, gitdir, ...) when it lists worktrees.
        An admin dir holding anything but plain files and directories (a FIFO would block git forever) is
        deleted, which only unregisters that worktree: attempts are committed from their files, not their
        git metadata."""
        admin = self.repo / ".git" / "worktrees"
        if not admin.is_dir():
            return
        for entry in list(admin.iterdir()):
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
                continue
            for dirpath, dirnames, filenames in os.walk(entry):
                bad = False
                for name in dirnames + filenames:
                    mode = os.lstat(os.path.join(dirpath, name)).st_mode
                    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                        bad = True
                        break
                if bad:
                    _rmtree(entry)
                    break

    def create(self, node_id: str, parent_commit: str | None) -> Path:
        path = self.path(node_id)
        self.work.mkdir(parents=True, exist_ok=True)
        self._in_repo("worktree", "add", "--quiet", "-b", f"drsi/{node_id}", str(path),
                      parent_commit or self.base_commit)
        return path

    def recreate(self, node_id: str, commit: str) -> Path:
        """A brand-new worktree at `commit`: nothing a worker did to the old one (files, git metadata, index
        flags, nested repositories) survives, which resetting it in place cannot promise."""
        self.remove(self.path(node_id))
        self._in_repo("branch", "-D", f"drsi/{node_id}", check=False)
        return self.create(node_id, commit)

    def branches(self) -> list[str]:
        out = self._in_repo("branch", "--list", "drsi/*", "--format=%(refname:short)", check=False)
        return [b.strip() for b in out.splitlines() if b.strip()]

    def create_detached(self, name: str, commit: str) -> Path:
        path = self.work / name
        if path.exists():
            self.remove(path)
        self.work.mkdir(parents=True, exist_ok=True)
        self._in_repo("worktree", "add", "--quiet", "--detach", str(path), commit)
        return path

    def remove(self, path) -> None:
        """Delete the files first and let git forget the worktree afterwards: git is never pointed at a
        directory a worker or a scorer could have rewritten."""
        if Path(path).exists() or Path(path).is_symlink():
            _rmtree(path)
        self._in_repo("worktree", "prune", check=False)

    def nested_repos(self, node_id: str) -> list[str]:
        """Paths of `.git` entries below the worktree's top level. Git would read those repositories when
        adding them, so an attempt that contains one is never committed."""
        root = self.path(node_id)
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            for name in dirnames + filenames:
                if _is_dot_git(name) and rel != ".":
                    found.append(os.path.join(rel, name))
            dirnames[:] = [d for d in dirnames if not _is_dot_git(d)]  # never descend into a repository
        return sorted(found)

    def snapshot(self, node_id: str, parent_commit: str) -> str:
        """Commit the files the worktree holds now as a child of `parent_commit`, through a private index
        and the clone's own git directory. What the worker did with git itself is ignored."""
        with tempfile.TemporaryDirectory(prefix="drsi-index-") as d:
            index = Path(d) / "index"
            wt = self.path(node_id)
            self._in_repo("read-tree", parent_commit, index=index)
            self._in_repo("add", "-A", work_tree=wt, index=index)
            tree = self._in_repo("write-tree", index=index).strip()
        commit = self._in_repo("commit-tree", tree, "-p", parent_commit, "-m", f"dream-rsi attempt {node_id}").strip()
        self._in_repo("update-ref", f"refs/heads/drsi/{node_id}", commit)
        return commit

    def changed_since_base(self, commit: str) -> list[str]:
        out = self._in_repo("diff", "--no-renames", "--name-only", "-z", self.base_commit, commit)
        return [f for f in out.split("\0") if f]

    def _modes_since_base(self, commit: str, mode: str) -> list[str]:
        out = self._in_repo("diff", "--no-renames", "--raw", "-z", self.base_commit, commit)
        parts, found = out.split("\0"), []
        for i in range(0, len(parts) - 1, 2):
            meta, path = parts[i], parts[i + 1]
            fields = meta.lstrip(":").split()
            if len(fields) >= 2 and fields[1] == mode:
                found.append(path)
        return found

    def symlinks_since_base(self, commit: str) -> list[str]:
        return self._modes_since_base(commit, "120000")

    def gitlinks_since_base(self, commit: str) -> list[str]:
        """Submodule entries: a pointer into another repository, never the attempt's own code."""
        return self._modes_since_base(commit, "160000")
