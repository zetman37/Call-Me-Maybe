from __future__ import annotations

import sys
from typing import List

from src.decoder import (
    coerce_parameter_types,
    constrained_generate_json_text,
    parse_generated_call,
    select_function_by_name,
)
from src.models import (
    DecodingConfig,
    FunctionCallResult,
    FunctionDefinition,
    PromptItem,
)

_MAX_ATTEMPTS = 2


def build_model_prompt(
    prompt: str, functions: List[FunctionDefinition]
) -> str:
    fn_lines: List[str] = []
    for fn in functions:
        params = ", ".join(
            [f"{k}:{v.type}" for k, v in fn.parameters.items()]
        )
        fn_lines.append(f"- {fn.name}({params}): {fn.description}")

    return (
        "You are a function-calling planner.\n"
        "Select exactly one function and fill all required "
        "parameters.\n"
        "Output JSON only with exact shape:\n"
        '{"name":"<function_name>","parameters":{...}}\n'
        "Available functions:\n"
        f"{chr(10).join(fn_lines)}\n"
        f"User request: {prompt}\n"
        "JSON:"
    )


def _run_single_prompt(
    model: object,
    prompt_text: str,
    functions: List[FunctionDefinition],
    cfg: DecodingConfig,
) -> FunctionCallResult:
    llm_prompt = build_model_prompt(prompt_text, functions)
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            generated_text = constrained_generate_json_text(
                model,
                llm_prompt,
                functions,
                cfg,
                raw_prompt=prompt_text,
            )
            name, raw_params = parse_generated_call(generated_text)
            fn = select_function_by_name(name, functions)
            params = coerce_parameter_types(raw_params, fn)

            return FunctionCallResult(
                prompt=prompt_text,
                name=name,
                parameters=params,
            )
        except Exception as exc:
            last_exc = exc
            print(
                f"Warning: attempt {attempt}/{_MAX_ATTEMPTS} failed "
                f"for prompt {prompt_text!r}: {exc}",
                file=sys.stderr,
            )

    return FunctionCallResult(
        prompt=prompt_text,
        name="error",
        parameters={"exception": str(last_exc)},
    )


def process_prompts(
    model: object,
    prompts: List[PromptItem],
    functions: List[FunctionDefinition],
    decoding_config: DecodingConfig | None = None,
) -> List[FunctionCallResult]:
    cfg = decoding_config or DecodingConfig()
    results: List[FunctionCallResult] = []

    for item in prompts:
        results.append(
            _run_single_prompt(model, item.prompt, functions, cfg)
        )

    return results
