"""Static checks on LLM-written policy code before it is ever run.

A policy may read only what the question API reveals (prefix-only), and it must not
be able to change the metrics the evaluator reads. The guard blocks the ways Python
code reaches around that: private attributes, reflection builtins, attribute access
through format strings or match patterns, module objects reachable through allowed
modules, attribute writes to anything but `self`, and non-allowlisted imports.
The replay evaluator also runs policies in an isolated subprocess with a timeout;
this guard is the first line, not the only one.
"""
from __future__ import annotations

import ast
import importlib
import inspect

ALLOWED_MODULES = {"math", "itertools", "collections", "heapq", "bisect", "statistics"}
ALLOWED_FROM = ALLOWED_MODULES | {"drsi.policy.api"}
API_NAMES = {"LLMDesignedMethod", "SimResult", "_budget_done", "_record_curve", "finalize_result"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "hasattr", "input", "breakpoint", "help", "memoryview", "exit", "quit", "type", "dir",
    "__builtins__", "__loader__", "__spec__", "__file__", "object", "format", "__build_class__",
}
# handlers that would also catch interrupts and exits
CATCH_ALL = {"BaseException", "KeyboardInterrupt", "SystemExit", "GeneratorExit"}
BANNED_ATTRS = {"world", "f_globals", "f_locals", "f_back", "gi_frame", "cr_frame", "tb_frame", "tb_next",
                "format", "format_map", "mro"}
# code, frame, traceback and generator/coroutine internals: reflection by another name
BANNED_ATTR_PREFIXES = ("co_", "f_", "tb_", "gi_", "cr_", "ag_")
MAX_CHARS = 40_000


def _module_attr_problem(module: str, attr: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return f"module {module} cannot be imported"
    if not hasattr(mod, attr):
        return f"{module}.{attr} does not exist"
    if inspect.ismodule(getattr(mod, attr)):
        return f"{module}.{attr} is a module"
    return None


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def check_policy_source(src: str) -> list[str]:
    problems: list[str] = []
    if len(src) > MAX_CHARS:
        problems.append(f"policy source is {len(src)} chars (max {MAX_CHARS})")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # a policy never holds a module object: modules lead to sys, builtins and everything else
            problems.append(f"line {node.lineno}: use `from <module> import <name>`, not `import`")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_FROM or node.level:
                problems.append(f"line {node.lineno}: from {node.module} import not allowed")
            elif node.module == "drsi.policy.api":
                for alias in node.names:
                    if alias.name not in API_NAMES:
                        problems.append(f"line {node.lineno}: drsi.policy.api exports only {sorted(API_NAMES)}")
            else:
                for alias in node.names:
                    if alias.name == "*" or alias.name.startswith("_"):
                        problems.append(f"line {node.lineno}: from {node.module} import {alias.name} not allowed")
                        continue
                    bad = _module_attr_problem(node.module, alias.name)
                    if bad:
                        problems.append(f"line {node.lineno}: {bad}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: use of {node.id} not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            # a local named like a special method becomes one if its namespace ever builds a class
            problems.append(f"line {node.lineno}: names starting with __ ({node.id}) not allowed")
        elif isinstance(node, ast.arg) and node.arg.startswith("__"):
            problems.append(f"line {node.lineno}: parameter {node.arg} not allowed")
        elif isinstance(node, ast.alias) and (node.asname or "").startswith("_"):
            problems.append(f"line {node.lineno}: import alias {node.asname} not allowed")
        elif isinstance(node, ast.ExceptHandler) and (node.type is None or {
                n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)} & CATCH_ALL):
            problems.append(f"line {node.lineno}: catch Exception (or narrower), not everything")
        elif type(node).__name__ == "TryStar":
            problems.append(f"line {node.lineno}: except* not allowed")
        elif isinstance(node, ast.Attribute):
            is_super_init = (node.attr == "__init__" and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Name) and node.value.func.id == "super")
            if is_super_init:
                continue
            if node.attr.startswith("_") or node.attr in BANNED_ATTRS or node.attr.startswith(BANNED_ATTR_PREFIXES):
                problems.append(f"line {node.lineno}: attribute .{node.attr} not allowed (prefix-only)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global/nonlocal not allowed")
        elif isinstance(node, ast.NamedExpr):
            problems.append(f"line {node.lineno}: assignment expressions (:=) not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.decorator_list:
            problems.append(f"line {node.lineno}: decorators not allowed")
        elif isinstance(node, ast.Match):
            problems.append(f"line {node.lineno}: match statements not allowed (patterns read attributes)")
    for node in ast.walk(tree):
        # every attribute write or delete, however expressed (assignment, loop or comprehension target,
        # with-as, augmented assignment, del), must be a plain `self.name`
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                problems.append(f"line {node.lineno}: writing attributes of anything but self is not allowed")
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if [c.name for c in classes] != ["OptimalPolicy"] or classes[0] not in tree.body:
        problems.append("exactly one class, a top-level OptimalPolicy, is allowed (no helper classes)")
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("__") \
                and fn.name != "__init__":
            problems.append(f"line {fn.lineno}: special method {fn.name} not allowed")
        if isinstance(fn, ast.AsyncFunctionDef):
            problems.append(f"line {fn.lineno}: async functions not allowed")
    has_solve = any(
        isinstance(n, ast.ClassDef) and n.name == "OptimalPolicy"
        and any(isinstance(f, ast.FunctionDef) and f.name == "solve" for f in n.body)
        for n in tree.body)
    if not has_solve:
        problems.append("must define class OptimalPolicy with a solve(self, question, budget=None) method")
    return problems


SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
    "isinstance", "iter", "len", "list", "map", "max", "min", "next", "pow", "print", "range", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "super", "tuple", "zip", "True", "False", "None",
    "Exception", "ArithmeticError", "IndexError", "KeyError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError", "__build_class__",
]


def safe_builtins() -> dict:
    """The only builtins a policy runs with: no reflection, I/O, eval or unrestricted import, so a gap
    in the static guard still has nothing to reach for at run time."""
    import builtins
    table = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in ALLOWED_FROM or not fromlist:
            raise ImportError(f"import of {name} is not allowed in a policy")
        return builtins.__import__(name, globals, locals, fromlist, level)
    table["__import__"] = restricted_import
    return table
