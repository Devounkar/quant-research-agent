"""Static checks on LLM-written code before it is executed.

This is a guardrail against accidents (file access, network calls, shelling out), not a
security boundary. For real isolation, run the whole project inside Docker or a VM.
"""
from __future__ import annotations

import ast

ALLOWED_MODULES = {
    "numpy", "pandas", "math", "statistics", "scipy", "typing",
    "functools", "itertools", "collections", "dataclasses",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit", "getattr", "setattr", "delattr", "memoryview",
}
BANNED_ATTRS = {
    "to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_sql", "to_hdf",
    "to_feather", "to_html", "to_clipboard", "to_latex", "to_stata", "to_xml",
    "tofile", "save", "savez", "savetxt", "load", "loadtxt", "fromfile", "genfromtxt", "system",
}


def check_code(code: str, require_strategy: bool = True) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError on line {e.lineno}: {e.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    errors.append(f"import of '{alias.name}' is not allowed (allowed: {sorted(ALLOWED_MODULES)})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_MODULES:
                errors.append(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                errors.append(f"dunder attribute '{node.attr}' is not allowed")
            elif node.attr in BANNED_ATTRS or node.attr.startswith("read_"):
                errors.append(f"I/O method '.{node.attr}' is not allowed")

    if require_strategy:
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_signals"]
        if not fns:
            errors.append("strategy must define a top-level function generate_signals(df, params)")
        elif len(fns[0].args.args) < 2:
            errors.append("generate_signals must take two arguments: (df, params)")
    # de-duplicate, keep order
    return list(dict.fromkeys(errors))
