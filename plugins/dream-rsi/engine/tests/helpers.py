"""Test doubles. A ScriptedLLM answers .json(prompt, schema) with a function."""
import json
import re
import threading


class ScriptedLLM:
    def __init__(self, fn):
        self.fn = fn
        self.prompts = []
        self._lock = threading.Lock()

    def json(self, prompt, schema):
        with self._lock:
            self.prompts.append(prompt)
        return self.fn(prompt, schema)


def ids_in_block(prompt, tag="ATTEMPT"):
    """Extract ids from lines like '<ATTEMPT id="12" ...>' in a prompt."""
    return re.findall(rf'<{tag} id="([^"]+)"', prompt)


def fp_for(i, family_hint="hint", outcome="refuted", killed_by="G-X"):
    return {"id": i, "mechanism": f"mechanism of {i}", "object": "obj", "key_move": "move",
            "kind": "construction", "outcome": outcome, "killed_by": killed_by,
            "why": "because", "family_hint": family_hint}


def dumps(o):
    return json.dumps(o, sort_keys=True)
