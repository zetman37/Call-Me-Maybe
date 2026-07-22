from __future__ import annotations

import json
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set

from src.models import FunctionDefinition, JsonType

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number_candidates(raw_prompt: str) -> FrozenSet[str]:
    return frozenset(_NUMBER_RE.findall(raw_prompt))


def _initial_json_state(
    functions: List[FunctionDefinition], raw_prompt: str = ""
) -> Dict[str, Any]:
    return {
        "stack": [],
        "mode": "top",
        "in_string_for": None,
        "escape": False,
        "literal_remaining": "",
        "opened": False,
        "functions_by_name": {fn.name: fn for fn in functions},
        "pending_key": None,
        "top_keys_used": set(),
        "selected_function": None,
        "param_keys_used": set(),
        "string_role": None,
        "string_buffer": "",
        "candidate_strings": None,
        "number_candidates": _extract_number_candidates(raw_prompt),
        "number_buffer": "",
    }


def _clone_json_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stack": list(state["stack"]),
        "mode": state["mode"],
        "in_string_for": state["in_string_for"],
        "escape": state["escape"],
        "literal_remaining": state["literal_remaining"],
        "opened": state["opened"],
        "functions_by_name": state["functions_by_name"],
        "pending_key": state["pending_key"],
        "top_keys_used": set(state["top_keys_used"]),
        "selected_function": state["selected_function"],
        "param_keys_used": set(state["param_keys_used"]),
        "string_role": state["string_role"],
        "string_buffer": state["string_buffer"],
        "candidate_strings": state["candidate_strings"],
        "number_candidates": state["number_candidates"],
        "number_buffer": state["number_buffer"],
    }


def _start_string(
    state: Dict[str, Any], entering_as_key: bool
) -> Optional[Dict[str, Any]]:
    depth = len(state["stack"])
    role: Optional[str] = None
    candidates: Optional[FrozenSet[str]] = None

    if entering_as_key:
        if depth == 1:
            remaining = {"name", "parameters"} - state["top_keys_used"]
            if not remaining:
                return None
            role = "top_key"
            candidates = frozenset(remaining)
        elif depth == 2 and state["selected_function"] is not None:
            fn = state["selected_function"]
            remaining = set(fn.parameters.keys()) - state[
                "param_keys_used"
            ]
            if not remaining:
                return None
            role = "param_key"
            candidates = frozenset(remaining)
    else:
        if depth == 1 and state["pending_key"] == "name":
            names = set(state["functions_by_name"].keys())
            if not names:
                return None
            role = "name_value"
            candidates = frozenset(names)

    state["mode"] = "string"
    state["in_string_for"] = "key" if entering_as_key else "val"
    state["string_role"] = role
    state["string_buffer"] = ""
    state["candidate_strings"] = candidates
    return state


def _close_string(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = state["candidate_strings"]
    buffer = state["string_buffer"]
    if candidates is not None and buffer not in candidates:
        return None

    role = state["string_role"]
    if role == "top_key":
        state["top_keys_used"].add(buffer)
        state["pending_key"] = buffer
    elif role == "param_key":
        state["param_keys_used"].add(buffer)
        state["pending_key"] = buffer
    elif role == "name_value":
        state["selected_function"] = state["functions_by_name"].get(
            buffer
        )

    state["mode"] = (
        "after_key" if state["in_string_for"] == "key"
        else "after_value"
    )
    state["in_string_for"] = None
    state["string_role"] = None
    state["string_buffer"] = ""
    state["candidate_strings"] = None
    return state


def _advance_literal(
    state: Dict[str, Any], c: str
) -> Optional[Dict[str, Any]]:
    lit = state["literal_remaining"]
    if lit and lit[0] == c:
        state["literal_remaining"] = lit[1:]
        if not state["literal_remaining"]:
            state["mode"] = "after_value"
        return state
    return None


def _advance_value_start(
    state: Dict[str, Any], c: str
) -> Optional[Dict[str, Any]]:
    if c == "{":
        state["stack"].append("obj")
        state["mode"] = "key_or_close"
    elif c == "[":
        state["stack"].append("arr")
        state["mode"] = "value_or_close"
    elif c == '"':
        return _start_string(state, entering_as_key=False)
    elif c in "-0123456789":
        state["mode"] = "number"
        state["number_buffer"] = c
    elif c == "t":
        state["mode"], state["literal_remaining"] = "literal", "rue"
    elif c == "f":
        state["mode"], state["literal_remaining"] = "literal", "alse"
    elif c == "n":
        state["mode"], state["literal_remaining"] = "literal", "ull"
    else:
        return None
    return state


def _advance_after_value(
    state: Dict[str, Any], c: str
) -> Optional[Dict[str, Any]]:
    if not state["stack"]:
        return None
    top = state["stack"][-1]
    if c == ",":
        state["mode"] = "key" if top == "obj" else "value"
        return state
    if c == "}" and top == "obj":
        state["stack"].pop()
        state["mode"] = "after_value"
        return state
    if c == "]" and top == "arr":
        state["stack"].pop()
        state["mode"] = "after_value"
        return state
    return None


def _advance_json_char(
    state: Dict[str, Any], c: str
) -> Optional[Dict[str, Any]]:
    mode = state["mode"]

    if mode == "string":
        if state["escape"]:
            state["escape"] = False
            state["string_buffer"] += c
            return state
        if c == "\\":
            state["escape"] = True
            return state
        if c == '"':
            return _close_string(state)
        candidates = state["candidate_strings"]
        if candidates is not None:
            new_buffer = state["string_buffer"] + c
            if not any(s.startswith(new_buffer) for s in candidates):
                return None
            state["string_buffer"] = new_buffer
        else:
            state["string_buffer"] += c
        return state

    if mode == "number":
        if c in "0123456789+-.eE":
            state["number_buffer"] += c
            return state
        state["mode"] = "after_value"
        state["number_buffer"] = ""
        return _advance_json_char(state, c)

    if mode == "literal":
        return _advance_literal(state, c)

    if c in " \t\n\r":
        return state

    if mode == "top":
        if c == "{":
            state["stack"].append("obj")
            state["mode"] = "key_or_close"
            state["opened"] = True
            return state
        return None

    if mode == "value":
        return _advance_value_start(state, c)

    if mode == "key_or_close":
        if c == '"':
            return _start_string(state, entering_as_key=True)
        if c == "}":
            if not state["stack"] or state["stack"].pop() != "obj":
                return None
            state["mode"] = "after_value"
            return state
        return None

    if mode == "value_or_close":
        if c == "]":
            if not state["stack"] or state["stack"].pop() != "arr":
                return None
            state["mode"] = "after_value"
            return state
        state["mode"] = "value"
        return _advance_json_char(state, c)

    if mode == "after_key":
        if c == ":":
            state["mode"] = "value"
            return state
        return None

    if mode == "after_value":
        return _advance_after_value(state, c)

    if mode == "key":
        if c == '"':
            return _start_string(state, entering_as_key=True)
        return None

    return None


def _advance_json_state(
    state: Dict[str, Any], text: str
) -> Optional[Dict[str, Any]]:
    s = _clone_json_state(state)
    for c in text:
        result = _advance_json_char(s, c)
        if result is None:
            return None
        s = result
    return s


def _is_complete_json_state(state: Dict[str, Any]) -> bool:
    return (
        state["opened"]
        and not state["stack"]
        and state["mode"] == "after_value"
    )


_LEGAL_START_CHARS: Dict[str, Optional[str]] = {
    "top": "{ \t\n\r",
    "value": '{["-0123456789tfn \t\n\r',
    "value_or_close": '{["-0123456789tfn] \t\n\r',
    "key_or_close": '"} \t\n\r',
    "key": '" \t\n\r',
    "after_key": ": \t\n\r",
    "after_value": ",}] \t\n\r",
}


def _legal_next_chars_for_string(
    buffer: str, candidates: FrozenSet[str]
) -> Set[str]:
    legal: Set[str] = set()
    blen = len(buffer)
    for s in candidates:
        if not s.startswith(buffer):
            continue
        if len(s) > blen:
            legal.add(s[blen])
        else:
            legal.add('"')
    return legal


def _value_matches_type(val: Any, ptype: JsonType) -> bool:
    if ptype == "string":
        return isinstance(val, str)
    if ptype == "number":
        return isinstance(val, (int, float)) and not isinstance(
            val, bool
        )
    if ptype == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if ptype == "boolean":
        return isinstance(val, bool)
    if ptype == "object":
        return isinstance(val, dict)
    if ptype == "array":
        return isinstance(val, list)
    if ptype == "null":
        return val is None
    return False


def _is_complete_valid_call(
    obj: Any, functions: List[FunctionDefinition]
) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"name", "parameters"}:
        return False

    name = obj.get("name")
    params = obj.get("parameters")

    if not isinstance(name, str):
        return False

    fn = next((f for f in functions if f.name == name), None)
    if fn is None:
        return False
    if not isinstance(params, dict):
        return False
    if set(params.keys()) != set(fn.parameters.keys()):
        return False
    for param_name, schema in fn.parameters.items():
        if not _value_matches_type(params[param_name], schema.type):
            return False
    return True


def json_prefix_is_plausible(
    prefix: str, functions: List[FunctionDefinition]
) -> bool:
    state = _advance_json_state(
        _initial_json_state(functions), prefix
    )
    if state is None:
        return False
    if _is_complete_json_state(state):
        try:
            obj = json.loads(prefix.strip())
        except json.JSONDecodeError:
            return True
        if not _is_complete_valid_call(obj, functions):
            return False
    return True
