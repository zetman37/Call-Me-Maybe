from __future__ import annotations

from typing import Any, List
from src.models import FunctionDefinition


def is_complete_valid_call(
    obj: Any, functions: List[FunctionDefinition]
) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"name", "parameters"}:
        return False

    name = obj.get("name")
    params = obj.get("parameters")

    if not isinstance(name, str) or not isinstance(params, dict):
        return False

    fn = next((f for f in functions if f.name == name), None)
    if fn is None:
        return False

    if set(params.keys()) != set(fn.parameters.keys()):
        return False

    for pk, p_def in fn.parameters.items():
        val = params[pk]
        ptype = p_def.type
        if ptype == "string" and not isinstance(val, str):
            return False
        if ptype == "number" and not isinstance(val, (int, float)):
            return False
        if ptype == "integer" and (
            not isinstance(val, int) or isinstance(val, bool)
        ):
            return False
        if ptype == "boolean" and not isinstance(val, bool):
            return False
        if ptype == "null" and val is not None:
            return False

    return True
