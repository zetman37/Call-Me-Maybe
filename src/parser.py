from __future__ import annotations

import json
import re
from typing import List
from src.models import FunctionDefinition
from src.validation import is_complete_valid_call


def json_prefix_is_plausible(
    prefix: str, functions: List[FunctionDefinition]
) -> bool:
    s = prefix.strip()
    if not s:
        return True
    if s[0] != "{":
        return False

    if len(re.findall(r'(?<!\\)"', s)) % 2 == 1:
        s += '"'

    stack = []
    in_str = False
    escape = False
    for c in s:
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in ("{", "["):
                stack.append(c)
            elif c == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif c == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    for b in reversed(stack):
        s += "}" if b == "{" else "]"

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return True

    if not isinstance(obj, dict):
        return False

    for k in obj.keys():
        if k not in {"name", "parameters"}:
            return False

    name = obj.get("name")
    if name is not None:
        if not isinstance(name, str):
            return False
        fns_by_name = {fn.name: fn for fn in functions}
        if name not in fns_by_name and not any(
            fn_name.startswith(name) for fn_name in fns_by_name
        ):
            return False

        selected_fn = fns_by_name.get(name)
        params = obj.get("parameters")
        if params is not None:
            if not isinstance(params, dict):
                return False
            if selected_fn is not None:
                for pk in params.keys():
                    if pk not in selected_fn.parameters:
                        return False
                for pk, pval in params.items():
                    p_def = selected_fn.parameters.get(pk)
                    if p_def is not None:
                        ptype = p_def.type
                        if ptype == "string" and not isinstance(pval, str):
                            return False
                        if ptype == "number" and not isinstance(
                            pval, (int, float)
                        ):
                            return False
                        if ptype == "integer" and (
                            not isinstance(pval, int) or isinstance(pval, bool)
                        ):
                            return False
                        if ptype == "boolean" and not isinstance(pval, bool):
                            return False

    try:
        obj_orig = json.loads(prefix.strip())
        return is_complete_valid_call(obj_orig, functions)
    except json.JSONDecodeError:
        pass

    return True
