from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import BaseModel, model_validator

JsonType = Literal[
    "string", "number", "integer", "boolean", "object", "array", "null"
]


class PropertyDefinition(BaseModel):
    type: JsonType


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, PropertyDefinition]
    returns: PropertyDefinition

    @model_validator(mode="before")
    @classmethod
    def standardize_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        if "fn_name" in values and "name" not in values:
            values["name"] = values["fn_name"]

        if "description" not in values:
            values["description"] = ""

        if "parameters" not in values:
            params = {}
            args_names = values.get("args_names", [])
            args_types = values.get("args_types", {})
            for name in args_names:
                py_type = args_types.get(name, "string")
                json_type = "string"
                if py_type == "float" or py_type == "number":
                    json_type = "number"
                elif py_type == "int" or py_type == "integer":
                    json_type = "integer"
                elif py_type == "bool" or py_type == "boolean":
                    json_type = "boolean"
                elif py_type == "dict" or py_type == "object":
                    json_type = "object"
                elif py_type == "list" or py_type == "array":
                    json_type = "array"

                params[name] = {"type": json_type}
            values["parameters"] = params

        if "returns" not in values:
            ret_type = values.get("return_type", "string")
            json_type = "string"
            if ret_type == "float" or ret_type == "number":
                json_type = "number"
            elif ret_type == "int" or ret_type == "integer":
                json_type = "integer"
            elif ret_type == "bool" or ret_type == "boolean":
                json_type = "boolean"
            elif ret_type == "dict" or ret_type == "object":
                json_type = "object"
            elif ret_type == "list" or ret_type == "array":
                json_type = "array"
            values["returns"] = {"type": json_type}

        return values


class PromptItem(BaseModel):
    prompt: str


class DecodingConfig(BaseModel):
    max_new_tokens: int = 64


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
