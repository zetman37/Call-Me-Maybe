from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import BaseModel, ConfigDict, Field

JsonType = Literal[
    "string", "number", "integer", "boolean", "object", "array", "null"
]


class ParameterSchema(BaseModel):
    """Schema for one function parameter"""
    model_config = ConfigDict(extra="forbid")
    type: JsonType


class ReturnSchema(BaseModel):
    """Schema for function return value"""
    model_config = ConfigDict(extra="forbid")
    type: JsonType


class FunctionDefinition(BaseModel):
    """A callable function definition"""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: ReturnSchema


class PromptItem(BaseModel):
    """One prompt input item"""
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCallResult(BaseModel):
    """One function-calling output item"""
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: Dict[str, Any]


class DecodingConfig(BaseModel):
    """Decoding settings"""
    model_config = ConfigDict(extra="forbid")
    max_new_tokens: int = Field(default=64, ge=1, le=2048)
