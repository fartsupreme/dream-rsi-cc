"""Campaign directory and the attempt tree (one JSONL node per attempt).

The tree is the search state. It lives on disk so nothing about what was tried
depends on any agent's context window.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

def default_home() -> Path:
    """Read at call time so DRSI_HOME can be set per process (tests, sandboxes)."""
    return Path(os.environ.get("DRSI_HOME", Path.home() / ".dream-rsi"))

NODE_FIELDS = {
    "id": None,
    "parent": None,
    "branch": None,          # int; inherited from parent when unset
    "seq": None,             # depth below the root of its branch; computed when unset
    "source": "live",        # import | live
    "created": None,         # ISO-8601 UTC
    "proposal": "",          # one-paragraph statement of the attempt
    "text": {},              # longer trimmed fields (candidate, verdict, next, ...)
    "fingerprint": None,     # {family, mechanism, key_move, outcome, killed_by, why, ...}
    "gates": {},             # {gate: {pass: bool, slack: float|None}}
    "score": None,           # campaign-normalised, higher is better
    "valid": None,
    "fail_class": None,      # ok | eval_error | timeout | agent_error | not_novel | out_of_scope
    "artifacts": {},
    "worker": {},            # {model, session, secs}
    "ext": {},               # source-specific extras (e.g. ledger line/id)
}

DEFAULT_CONFIG = {
    "goal": "",
    "direction": "max",                  # max | min for the raw score
    "search": {"W": 4, "K1": 6, "K2": 8, "plateau": 3},
    "dream": {"M": 3, "betas": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "lambda": 0.25,
               "beta1": 0.01, "beta2": 0.01},
    # Spending is neither capped nor tracked (operator directive, 2026-09-22).
    "llm": {"model": "opus", "classifier_model": "opus", "worker_model": "opus"},
    "map": {"max_chars": 16000},
    # serial: one scorer at a time, so timing-based scores within a round are comparable.
    # sandbox: "auto" runs the scorer under macOS sandbox-exec (writes only to its checkout, temp dirs and
    # allow_write), so candidate code the scorer runs cannot alter the scorer, the campaign or anything else.
    "scorer": {"cmd": None, "timeout_s": 3600, "serial": True, "sandbox": "auto", "allow_write": [],
               "network": False},
    # mutable: the paths workers may change, relative to the repo root. Empty means `drsi run` refuses.
    "workspace": {"repo": None, "base": None, "mutable": [], "env": {}, "ignore": [], "keep_worktrees": False},
    # Live workers run headless in Claude Code's Bash sandbox: writes only to their own worktree and
    # proposal directory, no network unless allowed_domains lists it, hooks off. With require_check, each
    # attempt is propose -> orchestrator's novelty check (up to max_proposals tries) -> implement.
    "live": {"require_check": True, "max_proposals": 3, "permission_mode": "acceptEdits",
             "timeout_s": 6 * 3600, "think_timeout_s": 300, "allowed_bash": [], "allowed_domains": []},
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_node(**fields) -> dict:
    unknown = set(fields) - set(NODE_FIELDS)
    if unknown:
        raise TypeError(f"unknown node fields: {sorted(unknown)}")
    node = copy.deepcopy(NODE_FIELDS)
    node.update(fields)
    if node["id"] is None:
        raise TypeError("node id is required")
    node["id"] = str(node["id"])
    if node["parent"] is not None:
        node["parent"] = str(node["parent"])
    if node["created"] is None:
        node["created"] = utcnow()
    return node


@contextmanager
def _locked(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _dump(node: dict) -> str:
    # ASCII escapes keep any text (even lone surrogates from odd ledgers) writable as UTF-8
    return json.dumps(node, ensure_ascii=True)


class Tree:
    """Append-mostly JSONL tree. Node order is insertion order.

    Several processes may share one tree (a live run, `drsi sync`, workers running
    `drsi check`). Every mutation therefore reloads the file under an exclusive
    file lock, applies only its own change, and writes, so no process can
    overwrite another's attempts with a stale in-memory copy.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._torn = False
        self._load(self._read())

    def _read(self) -> list[dict]:
        self._torn = False
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            raw = fh.read()
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        out = []
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    self._torn = True  # a writer died mid-append; that line was never committed
                    break
                raise
        if raw and not raw.endswith("\n"):
            self._torn = True
        return out

    def _load(self, nodes: list[dict]) -> None:
        self._nodes: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._children: dict[str | None, list[str]] = {}
        for n in nodes:
            self._index(n)

    def _index(self, node: dict) -> None:
        self._nodes.append(node)
        self._by_id[node["id"]] = node
        self._children.setdefault(node["parent"], []).append(node["id"])

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id) -> bool:
        return str(node_id) in self._by_id

    def nodes(self) -> list[dict]:
        return list(self._nodes)

    def get(self, node_id) -> dict:
        return self._by_id[str(node_id)]

    def children(self, node_id) -> list[dict]:
        return [self._by_id[c] for c in self._children.get(str(node_id), [])]

    def roots(self) -> list[dict]:
        return [self._by_id[c] for c in self._children.get(None, [])]

    def leaves(self) -> list[dict]:
        return [n for n in self._nodes if not self._children.get(n["id"])]

    def ancestors(self, node_id) -> list[dict]:
        out, cur = [], self.get(node_id)["parent"]
        while cur is not None:
            node = self.get(cur)
            out.append(node)
            cur = node["parent"]
        return out

    def _place(self, node: dict) -> None:
        """Validate against the current state, fill branch/seq, and index."""
        if node["id"] in self._by_id:
            raise ValueError(f"duplicate node id {node['id']!r}")
        parent = node["parent"]
        if parent is not None and parent not in self._by_id:
            raise ValueError(f"unknown parent {parent!r} for node {node['id']!r}")
        if parent is not None:
            p = self._by_id[parent]
            if node["branch"] is None:
                node["branch"] = p["branch"]
            if node["seq"] is None:
                node["seq"] = (p["seq"] or 0) + 1
        else:
            if node["branch"] is None:
                node["branch"] = len(self._children.get(None, []))
            if node["seq"] is None:
                node["seq"] = 0
        self._index(node)

    def add(self, node: dict) -> dict:
        with _locked(self.path):
            self._load(self._read())
            self._place(node)
            if self._torn:
                self._write_locked()  # drops the torn tail and writes the new node with the rest
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as fh:
                    fh.write(_dump(node) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        return node

    def add_many(self, nodes: list[dict], skip_existing: bool = False) -> int:
        """Bulk insert with a single rewrite (used by importers). With skip_existing, nodes whose id is
        already present (another process got there first) are skipped. Returns the number added."""
        with _locked(self.path):
            fresh = self._read()
            self._load(fresh)
            added = 0
            try:
                for node in nodes:
                    if skip_existing and node["id"] in self._by_id:
                        continue
                    self._place(node)
                    added += 1
            except ValueError:
                self._load(fresh)
                raise
            if added or self._torn:
                self._write_locked()
            return added

    def update(self, node_id, **fields) -> dict:
        self.update_many({str(node_id): fields})
        return self.get(node_id)

    def update_many(self, updates: dict) -> None:
        for fields in updates.values():
            unknown = set(fields) - set(NODE_FIELDS)
            if unknown:
                raise TypeError(f"unknown node fields: {sorted(unknown)}")
            if "id" in fields or "parent" in fields:
                raise TypeError("id and parent are immutable")
        with _locked(self.path):
            self._load(self._read())
            for node_id, fields in updates.items():
                self.get(node_id).update(fields)
            self._write_locked()

    def modify(self, fns: dict) -> None:
        """Apply fn(node) for each id under the lock, on the node as it is on disk now."""
        with _locked(self.path):
            self._load(self._read())
            for node_id, fn in fns.items():
                if node_id in self._by_id:
                    node = self.get(node_id)
                    parent = node["parent"]
                    fn(node)
                    if node["id"] != node_id or node["parent"] != parent:
                        raise TypeError("id and parent are immutable")
            self._write_locked()

    def merge_many(self, updates: dict) -> None:
        """Like update_many, but dict-valued fields are merged key by key into the value read under the
        lock, so a caller holding an older snapshot cannot clobber keys another process wrote since."""
        with _locked(self.path):
            self._load(self._read())
            for node_id, fields in updates.items():
                if node_id not in self._by_id:
                    continue
                node = self.get(node_id)
                for k, v in fields.items():
                    if k in ("id", "parent") or k not in NODE_FIELDS:
                        raise TypeError(f"cannot merge field {k!r}")
                    if isinstance(v, dict) and isinstance(node.get(k), dict):
                        node[k] = {**node[k], **v}
                    else:
                        node[k] = v
            self._write_locked()

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, "".join(_dump(n) + "\n" for n in self._nodes))


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Campaign:
    SUBDIRS = ("policy", "trace_pool", "logs", "work")

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def config_path(self) -> Path:
        return self.root / "campaign.json"

    @property
    def config(self) -> dict:
        cfg = _deep_merge(DEFAULT_CONFIG, json.loads(self.config_path.read_text()))
        for section, key in (("workspace", "mutable"), ("workspace", "ignore"), ("live", "allowed_bash"),
                             ("live", "allowed_domains")):
            if isinstance(cfg[section].get(key), str):  # `--set workspace.mutable=src/**` without brackets
                cfg[section][key] = [cfg[section][key]]
        return cfg

    def save_config(self, cfg: dict) -> None:
        _atomic_write(self.config_path, json.dumps(cfg, indent=2) + "\n")

    def update_config(self, fn) -> dict:
        """Read-modify-write campaign.json under a lock, so concurrent writers do not lose updates."""
        with _locked(self.config_path):
            raw = json.loads(self.config_path.read_text())
            fn(raw)
            self.save_config(raw)
            return raw

    @property
    def tree(self) -> Tree:
        return Tree(self.root / "tree.jsonl")

    @property
    def families_path(self) -> Path:
        return self.root / "families.json"

    @property
    def map_path(self) -> Path:
        return self.root / "map.md"

    @property
    def checks_path(self) -> Path:
        return self.root / "logs" / "checks.jsonl"

    @classmethod
    def create(cls, name: str, config: dict, home: Path | None = None) -> "Campaign":
        if not _SAFE_NAME.match(name):
            raise ValueError(f"unsafe campaign name {name!r}")
        root = Path(home or default_home()) / "campaigns" / name
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        for sub in cls.SUBDIRS:
            (root / sub).mkdir()
        cfg = _deep_merge(DEFAULT_CONFIG, config)
        cfg["name"] = name
        cfg["created"] = utcnow()
        camp = cls(root)
        camp.save_config(cfg)
        return camp

    @classmethod
    def open(cls, name_or_path: str, home: Path | None = None) -> "Campaign":
        p = Path(name_or_path).expanduser()
        if p.is_absolute() or os.sep in name_or_path:
            root = p
        else:
            root = Path(home or default_home()) / "campaigns" / name_or_path
        if not (root / "campaign.json").exists():
            raise FileNotFoundError(f"no campaign at {root}")
        return cls(root)
