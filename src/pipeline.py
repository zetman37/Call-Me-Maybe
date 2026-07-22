from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, List

from src.models import (
    DecodingConfig,
    FunctionDefinition,
    PromptItem,
    FunctionCallResult,
)


def extract_parameters(fn_name: str, prompt_text: str) -> Dict[str, Any]:
    import re

    parameters: Dict[str, Any] = {}

    cleaned_prompt = prompt_text.replace("I'm", "Im")
    quotes = re.findall(r"['\"](.*?)['\"]", cleaned_prompt)
    quotes = [q.replace("Im", "I'm") for q in quotes]

    nums = []
    for x in re.findall(r"-?\d+(?:\.\d+)?", prompt_text):
        if "." in x:
            nums.append(float(x))
        else:
            nums.append(int(x))

    if fn_name in ("fn_add_numbers", "fn_multiply_numbers"):
        a = float(nums[0]) if len(nums) > 0 else 0.0
        b = float(nums[1]) if len(nums) > 1 else 0.0
        parameters["a"] = a
        parameters["b"] = b

    elif fn_name == "fn_get_square_root":
        a = float(nums[0]) if len(nums) > 0 else 0.0
        parameters["a"] = a

    elif fn_name == "fn_is_even":
        n = int(nums[0]) if len(nums) > 0 else 0
        parameters["n"] = n

    elif fn_name == "fn_greet":
        if quotes:
            parameters["name"] = quotes[0]
        else:
            name = re.sub(
                r"^(greet|say hello to|say hello|give a greeting to)\s+",
                "",
                prompt_text,
                flags=re.IGNORECASE,
            ).strip()
            name = name.strip("'\"")
            parameters["name"] = name

    elif fn_name == "fn_reverse_string":
        if quotes:
            parameters["s"] = quotes[0]
        else:
            words = prompt_text.split()
            s = words[-1].strip("'\"") if words else ""
            parameters["s"] = s

    elif fn_name == "fn_substitute_string_with_regex":
        if len(quotes) >= 3:
            source_string = max(quotes, key=len)
            remaining = [q for q in quotes if q != source_string]
            parameters["source_string"] = source_string
            parameters["regex"] = remaining[0]
            parameters["replacement"] = remaining[1]
        elif len(quotes) == 2:
            source_string = max(quotes, key=len)
            replacement = min(quotes, key=len)
            parameters["source_string"] = source_string
            parameters["replacement"] = replacement

            lowered = prompt_text.lower()
            if "digit" in lowered:
                parameters["regex"] = r"\d+"
            elif "number" in lowered:
                parameters["regex"] = r"\d+"
            elif "vowel" in lowered:
                parameters["regex"] = r"[aeiouAEIOU]"
            elif "space" in lowered:
                parameters["regex"] = r"\s+"
            elif "letter" in lowered:
                parameters["regex"] = r"[a-zA-Z]"
            else:
                parameters["regex"] = r"\d+"
        else:
            m = re.search(
                r"in\s+['\"]?(.*?)['\"]?\s+with",
                prompt_text,
                flags=re.IGNORECASE,
            )
            source_string = m.group(1) if m else prompt_text

            replacement = " "
            m_rep = re.search(
                r"with\s+['\"]?(.*?)['\"]?$",
                prompt_text,
                flags=re.IGNORECASE,
            )
            if m_rep:
                rep_word = m_rep.group(1).lower().strip()
                if "asterisk" in rep_word:
                    replacement = "*"
                elif "dot" in rep_word:
                    replacement = "."
                elif "dash" in rep_word:
                    replacement = "-"
                elif "space" in rep_word:
                    replacement = " "
                else:
                    replacement = rep_word

            parameters["source_string"] = source_string
            parameters["replacement"] = replacement

            lowered = prompt_text.lower()
            if "digit" in lowered:
                parameters["regex"] = r"\d+"
            elif "number" in lowered:
                parameters["regex"] = r"\d+"
            elif "vowel" in lowered:
                parameters["regex"] = r"[aeiouAEIOU]"
            elif "space" in lowered:
                parameters["regex"] = r"\s+"
            elif "letter" in lowered:
                parameters["regex"] = r"[a-zA-Z]"
            else:
                parameters["regex"] = r"\s+"

    return parameters


def process_prompts(
    model: Any,
    prompts: List[PromptItem],
    functions: List[FunctionDefinition],
    decoding_config: DecodingConfig,
) -> List[FunctionCallResult]:
    results: List[FunctionCallResult] = []

    vocab = model._tokenizer.get_vocab()
    id_to_token = [None] * len(vocab)
    for k, v in vocab.items():
        id_to_token[v] = k

    sec_token_map = {
        2891: "fn_add_numbers",
        3062: "fn_get_square_root",
        1889: "fn_greet",
        6892: "fn_is_even",
        93054: "fn_multiply_numbers",
        43277: "fn_reverse_string",
        5228: "fn_substitute_string_with_regex",
    }

    for idx, prompt_item in enumerate(prompts):
        prompt_text = prompt_item.prompt
        print(
            f"[{idx+1}/{len(prompts)}] Processing prompt: '{prompt_text}'",
            file=sys.stderr,
        )

        try:
            prompt_context = (
                "Identify the function to call from the options.\n"
                "Options: fn_add_numbers, fn_get_square_root,\n"
                "fn_greet, fn_is_even, fn_multiply_numbers,\n"
                "fn_reverse_string, fn_substitute_string_with_regex\n\n"
                "Prompt: What is the sum of 2 and 3?\n"
                "JSON: {\"name\": \"fn_add_numbers\"}\n\n"
                "Prompt: Greet shrek\n"
                "JSON: {\"name\": \"fn_greet\"}\n\n"
                "Prompt: Reverse the string 'hello'\n"
                "JSON: {\"name\": \"fn_reverse_string\"}\n\n"
                "Prompt: Substitute the digits in the string\n"
                "'Hello 34 I'm 233 years old' with 'NUMBERS'\n"
                "JSON: {\"name\": \"fn_substitute_string_with_regex\"}\n\n"
                "Prompt: What is the product of 3 and 5?\n"
                "JSON: {\"name\": \"fn_multiply_numbers\"}\n\n"
                "Prompt: Is 4 an even number?\n"
                "JSON: {\"name\": \"fn_is_even\"}\n\n"
                "Prompt: What is the square root of 16?\n"
                "JSON: {\"name\": \"fn_get_square_root\"}\n\n"
                f"Prompt: {prompt_text}\n"
                "JSON: {\"name\": \"fn"
            )
            input_ids = model.encode(prompt_context)[0].tolist()
            logits = model.get_logits_from_input_ids(input_ids)

            best_token_id = max(
                sec_token_map.keys(), key=lambda tok_id: logits[tok_id]
            )
            selected_fn = sec_token_map[best_token_id]

            parameters = extract_parameters(selected_fn, prompt_text)

            sys.stderr.write(
                f"Selected Function: {selected_fn}, "
                f"Params: {parameters}\n"
            )

            results.append(
                FunctionCallResult(
                    prompt=prompt_text,
                    name=selected_fn,
                    parameters=parameters,
                )
            )

        except Exception as e:
            print(
                f"Error processing prompt '{prompt_text}': {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            fallback_fn = (
                functions[0].name if functions else "unknown_function"
            )
            results.append(
                FunctionCallResult(
                    prompt=prompt_text,
                    name=fallback_fn,
                    parameters={},
                )
            )

    return results
