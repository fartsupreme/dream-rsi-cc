"""Headless Claude calls that return schema-validated JSON.

Classifier-style calls run lean: a short replacement system prompt, no tools,
no MCP servers, and no user-level settings (so the operator's own hooks and
plugins never load inside these calls). They run from a neutral empty directory, so a
project's own settings and hooks (for example a stop gate) never load either, whatever
directory drsi was called from. The prompt travels on stdin.
"""
from __future__ import annotations

import json
import subprocess

from .agent import run_group
from .store import default_home


class LLMError(RuntimeError):
    pass


class ClaudeCLI:
    def __init__(self, model: str, system_prompt: str | None = None, tools: str = "",
                 binary: str = "claude", runner=run_group, timeout: int = 900,
                 cwd: str | None = None, extra_args: tuple = ()):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.binary = binary
        self.runner = runner
        self.timeout = timeout
        self.cwd = cwd
        self.extra_args = tuple(extra_args)

    def build_args(self, schema: dict) -> list[str]:
        args = [self.binary, "-p", "--model", self.model, "--output-format", "json",
                "--no-session-persistence", "--setting-sources", "project,local",
                "--strict-mcp-config", "--tools", self.tools,
                "--json-schema", json.dumps(schema)]
        if self.system_prompt:
            args += ["--system-prompt", self.system_prompt]
        return args + list(self.extra_args)

    def _once(self, prompt: str, schema: dict) -> dict:
        try:
            cwd = self.cwd
            if cwd is None:
                neutral = default_home() / "_neutral"
                neutral.mkdir(parents=True, exist_ok=True)
                cwd = str(neutral)
            proc = self.runner(self.build_args(schema), input=prompt, capture_output=True,
                               text=True, timeout=self.timeout, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude -p timed out after {self.timeout}s") from e
        if proc.returncode != 0:
            raise LLMError(f"claude -p exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
        try:
            env = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude -p returned non-JSON: {proc.stdout[:300]!r}") from e
        if env.get("is_error") or env.get("subtype") != "success":
            raise LLMError(f"claude -p failed: subtype={env.get('subtype')} result={str(env.get('result'))[:300]}")
        out = env.get("structured_output")
        if not isinstance(out, dict):
            raise LLMError("claude -p returned no structured_output")
        return out

    def json(self, prompt: str, schema: dict) -> dict:
        try:
            return self._once(prompt, schema)
        except LLMError as first:
            try:
                return self._once(prompt, schema)
            except LLMError as second:
                raise LLMError(f"two attempts failed: {first} | {second}") from second
