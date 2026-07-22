from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.json_grammar import (
    _initial_json_state,
    _advance_json_state,
    _is_complete_json_state,
    _is_complete_valid_call,
    _legal_next_chars_for_string,
    _LEGAL_START_CHARS,
)
from src.models import DecodingConfig, FunctionDefinition
from src.vocab import _DEBUG, get_leading_char_buckets, get_vocab_map

NEG_INF = -1e30

_UNRESTRICTED_MODES = {"number", "literal"}


def _argmax(values: List[float]) -> int:
    best_idx = 0
    best_val = float("-inf")
    for i, v in enumerate(values):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def _masked_logits(
    logits: List[float], allowed_ids: List[int]
) -> List[float]:
    masked = [NEG_INF] * len(logits)
    for token_id in allowed_ids:
        if 0 <= token_id < len(logits):
            masked[token_id] = logits[token_id]
    return masked


def _candidates_for_state(
    grammar_state: Dict[str, Any],
    vocab_map: Dict[int, str],
    buckets: Dict[str, List[Tuple[int, str]]],
) -> List[Tuple[int, str]]:
    mode = grammar_state["mode"]

    if mode == "string":
        role = grammar_state.get("string_role")
        if role is None:
            return list(vocab_map.items())
        legal_chars = _legal_next_chars_for_string(
            grammar_state["string_buffer"],
            grammar_state["candidate_strings"],
        )
        result: List[Tuple[int, str]] = []
        for c in legal_chars:
            result.extend(buckets.get(c, ()))
        return result

    if mode in _UNRESTRICTED_MODES:
        return list(vocab_map.items())

    legal = _LEGAL_START_CHARS.get(mode)
    if legal is None:
        return list(vocab_map.items())

    result = []
    for c in legal:
        result.extend(buckets.get(c, ()))
    return result


def _narrow_number_candidates(
    grammar_state: Dict[str, Any],
    buckets: Dict[str, List[Tuple[int, str]]],
) -> Optional[List[Tuple[int, str]]]:
    if grammar_state["mode"] != "number":
        return None

    candidates = grammar_state["number_candidates"]
    buffer = grammar_state["number_buffer"]
    if not candidates:
        return None

    matching = {c for c in candidates if c.startswith(buffer)}
    if buffer in matching:
        return None

    longer = [c for c in matching if len(c) > len(buffer)]
    if len(matching) != 1 or len(longer) != 1:
        return None

    next_char = longer[0][len(buffer)]
    narrow = buckets.get(next_char)
    if not narrow:
        return None
    return list(narrow)


def _find_forced_token(
    grammar_state: Dict[str, Any],
    generated_text: str,
    candidates: List[Tuple[int, str]],
    functions: List[FunctionDefinition],
    vocab_size: int,
) -> Optional[Tuple[int, Dict[str, Any], Any]]:
    survivors: List[Tuple[int, str, Dict[str, Any]]] = []

    for token_id, piece in candidates:
        if token_id >= vocab_size:
            continue
        new_state = _advance_json_state(grammar_state, piece)
        if new_state is None:
            continue
        if _is_complete_json_state(new_state):
            candidate_text = (generated_text + piece).strip()
            try:
                obj = json.loads(candidate_text)
            except json.JSONDecodeError:
                continue
            if not _is_complete_valid_call(obj, functions):
                continue
        survivors.append((token_id, piece, new_state))
        if len(survivors) > 1:
            return None

    if len(survivors) != 1:
        return None

    token_id, piece, new_state = survivors[0]
    completed_obj = None
    if _is_complete_json_state(new_state):
        candidate_text = (generated_text + piece).strip()
        try:
            completed_obj = json.loads(candidate_text)
        except json.JSONDecodeError:
            completed_obj = None
    return token_id, new_state, completed_obj


def constrained_generate_json_text(
    model: Any,
    prompt: str,
    functions: List[FunctionDefinition],
    config: DecodingConfig,
    *,
    raw_prompt: str = "",
) -> str:
    if not functions:
        raise ValueError("No function definitions provided")

    input_ids_tensor = model.encode(prompt)
    raw_ids = input_ids_tensor.tolist()
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("Tokenizer produced empty ids")
    if isinstance(raw_ids[0], list):
        input_ids = [int(x) for x in raw_ids[0]]
    else:
        input_ids = [int(x) for x in raw_ids]

    generated_ids: List[int] = []
    generated_text = ""
    vocab_map: Dict[int, str] | None = None
    buckets: Dict[str, List[Tuple[int, str]]] | None = None
    grammar_state = _initial_json_state(functions, raw_prompt)
    forced_count = 0
    model_call_count = 0

    for step in range(config.max_new_tokens):
        if vocab_map is not None:
            assert buckets is not None

            narrow = _narrow_number_candidates(grammar_state, buckets)
            if narrow:
                forced = _find_forced_token(
                    grammar_state,
                    generated_text,
                    narrow,
                    functions,
                    len(vocab_map),
                )
                if forced is not None:
                    token_id, new_state, completed_obj = forced
                    piece = vocab_map[token_id]
                    generated_ids.append(token_id)
                    generated_text += piece
                    grammar_state = new_state
                    forced_count += 1

                    if _DEBUG:
                        print(
                            f"[decoder] step {step}: FORCED "
                            f"(number, no model call) "
                            f"text={generated_text!r}",
                            file=sys.stderr,
                        )

                    if completed_obj is not None:
                        stripped = generated_text.strip()
                        if _DEBUG:
                            print(
                                f"[decoder] completed at step "
                                f"{step}: {stripped!r}",
                                file=sys.stderr,
                            )
                        return stripped
                    continue

            candidates = _candidates_for_state(
                grammar_state, vocab_map, buckets
            )
            forced = _find_forced_token(
                grammar_state,
                generated_text,
                candidates,
                functions,
                len(vocab_map),
            )
            if forced is not None:
                token_id, new_state, completed_obj = forced
                piece = vocab_map[token_id]
                generated_ids.append(token_id)
                generated_text += piece
                grammar_state = new_state
                forced_count += 1

                if _DEBUG:
                    print(
                        f"[decoder] step {step}: FORCED (no model "
                        f"call) text={generated_text!r}",
                        file=sys.stderr,
                    )

                if completed_obj is not None:
                    stripped = generated_text.strip()
                    if _DEBUG:
                        print(
                            f"[decoder] completed at step {step}: "
                            f"{stripped!r}",
                            file=sys.stderr,
                        )
                    return stripped
                continue

        forward_start = time.monotonic() if _DEBUG else 0.0
        logits = model.get_logits_from_input_ids(
            input_ids + generated_ids
        )
        forward_time = (
            time.monotonic() - forward_start if _DEBUG else 0.0
        )
        model_call_count += 1

        if not isinstance(logits, list) or not logits:
            raise ValueError("Model returned invalid logits")

        if vocab_map is None:
            vocab_map = get_vocab_map(model, len(logits))
            buckets = get_leading_char_buckets(model, vocab_map)
            if _DEBUG:
                print(
                    f"[decoder] vocab_size={len(logits)}",
                    file=sys.stderr,
                )
        assert buckets is not None

        mask_start = time.monotonic() if _DEBUG else 0.0
        allowed_ids: List[int] = []
        candidates = _candidates_for_state(
            grammar_state, vocab_map, buckets
        )
        for token_id, piece in candidates:
            if token_id >= len(logits):
                continue
            new_state = _advance_json_state(grammar_state, piece)
            if new_state is None:
                continue
            if _is_complete_json_state(new_state):
                candidate = (generated_text + piece).strip()
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if not _is_complete_valid_call(obj, functions):
                    continue
            allowed_ids.append(token_id)
        mask_time = time.monotonic() - mask_start if _DEBUG else 0.0

        if not allowed_ids:
            if _DEBUG:
                print(
                    f"[decoder] step {step}: no valid next token. "
                    f"text={generated_text!r}",
                    file=sys.stderr,
                )
            raise ValueError("No valid next token under constraints")

        next_id = _argmax(_masked_logits(logits, allowed_ids))
        piece = vocab_map[next_id]
        generated_ids.append(next_id)
        generated_text += piece
        next_state = _advance_json_state(grammar_state, piece)
        if next_state is None:
            raise ValueError(
                "Internal error: committed token broke grammar state"
            )
        grammar_state = next_state

        if _DEBUG:
            print(
                f"[decoder] step {step}: forward={forward_time:.3f}s "
                f"mask={mask_time:.3f}s allowed={len(allowed_ids)} "
                f"candidates={len(candidates)} "
                f"text={generated_text!r}",
                file=sys.stderr,
            )

        if _is_complete_json_state(grammar_state):
            stripped = generated_text.strip()
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if _is_complete_valid_call(obj, functions):
                if _DEBUG:
                    print(
                        f"[decoder] completed at step {step}: "
                        f"{stripped!r} (forced={forced_count} "
                        f"model_calls={model_call_count})",
                        file=sys.stderr,
                    )
                return stripped

    if _DEBUG:
        print(
            f"[decoder] exhausted {config.max_new_tokens} steps. "
            f"forced={forced_count} model_calls={model_call_count} "
            f"text={generated_text!r}",
            file=sys.stderr,
        )
    raise ValueError(
        "Reached max_new_tokens without valid constrained JSON"
    )


def parse_generated_call(
    generated_json_text: str,
) -> Tuple[str, Dict[str, Any]]:
    try:
        obj = json.loads(generated_json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generated output is not valid JSON: {exc}"
        ) from exc

    if not isinstance(obj, dict):
        raise ValueError("Generated output must be JSON object")
    if set(obj.keys()) != {"name", "parameters"}:
        raise ValueError(
            "Generated output must contain only: name, parameters"
        )

    name = obj.get("name")
    params = obj.get("parameters")

    if not isinstance(name, str):
        raise ValueError("Field 'name' must be string")
    if not isinstance(params, dict):
        raise ValueError("Field 'parameters' must be object")

    return name, params


def select_function_by_name(
    name: str, functions: List[FunctionDefinition]
) -> FunctionDefinition:
    for fn in functions:
        if fn.name == name:
            return fn
    raise ValueError(f"Function '{name}' not found in definitions")


def coerce_parameter_types(
    raw_params: Dict[str, Any], fn: FunctionDefinition
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for param_name, schema in fn.parameters.items():
        if param_name not in raw_params:
            raise ValueError(
                f"Missing required parameter: {param_name}"
            )

        val = raw_params[param_name]
        ptype = schema.type

        if ptype == "string":
            if not isinstance(val, str):
                val = str(val)
        elif ptype == "number":
            try:
                val = float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Parameter '{param_name}' must be number"
                ) from exc
        elif ptype == "integer":
            try:
                val = int(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Parameter '{param_name}' must be integer"
                ) from exc
        elif ptype == "boolean":
            if isinstance(val, bool):
                pass
            elif isinstance(val, str) and val.lower() in {
                "true",
                "false",
            }:
                val = val.lower() == "true"
            else:
                raise ValueError(
                    f"Parameter '{param_name}' must be boolean"
                )
        elif ptype == "object":
            if not isinstance(val, dict):
                raise ValueError(
                    f"Parameter '{param_name}' must be object"
                )
        elif ptype == "array":
            if not isinstance(val, list):
                raise ValueError(
                    f"Parameter '{param_name}' must be array"
                )
        elif ptype == "null":
            val = None
        else:
            raise ValueError(f"Unsupported parameter type: {ptype}")

        out[param_name] = val

    return out
