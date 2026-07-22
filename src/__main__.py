from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List
from src.io_utils import parse_list_of_models, read_json_file, write_results
from src.models import DecodingConfig, FunctionDefinition, PromptItem
from src.pipeline import process_prompts


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        from llm_sdk import Small_LLM_Model
        raw_fns = read_json_file(args.functions_definition)
        raw_prompts = read_json_file(args.input)

        functions = parse_list_of_models(
            raw_fns, FunctionDefinition, str(args.functions_definition)
        )
        prompts = parse_list_of_models(
            raw_prompts, PromptItem, str(args.input)
        )

        model = Small_LLM_Model("Qwen/Qwen3-0.6B")
        results = process_prompts(
            model=model,
            prompts=prompts,
            functions=functions,
            decoding_config=DecodingConfig(max_new_tokens=args.max_new_tokens),
        )
        write_results(args.output, results)
        print(f"Success: wrote {len(results)} results to {args.output}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
