from __future__ import annotations
import json
from pathlib import Path
from typing import List, Sequence, Type, TypeVar
from pydantic import ValidationError, BaseModel
from src.models import FunctionCallResult

T = TypeVar("T", bound=BaseModel)


def read_json_file(path: Path) -> object:
    """Read and parse Json File"""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise ValueError(f"Input File not found: {path}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not read {path}: {e}")
    except OSError as e:
        raise ValueError(f"Could not read {path}: {e}")


def parse_list_of_models(
    raw: object,
    model_cls: Type[T],
    file_lable: str
) -> List[T]:
    """Validate JSON array into list of pydantic models"""
    if not isinstance(raw, list):
        raise ValueError(f"{file_lable} must contain a JSON array")
    out: List[T] = []
    for idx, item in enumerate(raw):
        try:
            out.append(
                model_cls.model_validate(item)
            )
        except ValidationError as e:
            raise ValueError(
                f"{file_lable} validation error at index {idx}: {e}"
            )
    return out


def write_results(path: Path, results: Sequence[FunctionCallResult]) -> None:
    """Write final output JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [x.model_dump() for x in results]
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError as e:
        raise ValueError(f"Could not write output file {path}: {e}")
