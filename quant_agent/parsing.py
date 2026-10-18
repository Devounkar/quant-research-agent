"""Parses the agent's plain-text replies into a structured Action.

Expected format:
    THOUGHT: ...
    ACTION: tool_name
    ARGS: {"key": "value"}
    CODE:
    ```python
    ...
    ```
Code is sent in a fenced block (not inside JSON) because LLMs often break JSON escaping
when writing multi-line code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


class ParseError(ValueError):
    pass


@dataclass
class Action:
    tool: str
    args: dict = field(default_factory=dict)
    code: str | None = None
    thought: str = ""


_ACTION_RE = re.compile(r"^\s*\**ACTION\**\s*:\s*\**\s*([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def parse_action(text: str) -> Action:
    m = _ACTION_RE.search(text)
    if not m:
        raise ParseError("no line of the form 'ACTION: <tool_name>' was found")
    tool = m.group(1)

    t = re.search(r"THOUGHT\**\s*:\s*(.*?)(?=^\s*\**ACTION)", text, re.DOTALL | re.MULTILINE)
    thought = t.group(1).strip() if t else ""

    rest = text[m.end():]
    args: dict = {}
    a = re.search(r"ARGS\**\s*:\s*\**\s*", rest)
    if a:
        tail = rest[a.end():].lstrip()
        if tail.startswith("```"):
            tail = re.sub(r"^```(?:json)?\s*", "", tail)
        if tail.startswith("{"):
            try:
                args, _ = json.JSONDecoder().raw_decode(tail)
            except json.JSONDecodeError as e:
                raise ParseError(f"ARGS is not valid JSON ({e.msg} at char {e.pos})") from e
            if not isinstance(args, dict):
                raise ParseError("ARGS must be a JSON object")

    code = args.pop("code", None)
    c = _CODE_RE.search(rest)
    if c:
        code = c.group(1)
    return Action(tool=tool, args=args, code=code.strip("\n") if code else None, thought=thought)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of free text (used for the critic's verdict)."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ParseError("no JSON object found")
