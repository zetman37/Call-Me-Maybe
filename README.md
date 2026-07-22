*This project has been created as part of the 42 curriculum by asylla.*

## Description

This project implements a function-calling system using a small language model (`Qwen/Qwen3-0.6B` through the provided `llm_sdk`).

Given a natural-language prompt, it returns a strict JSON object containing:
- `name`: selected function name
- `parameters`: typed arguments matching the function schema

The core requirement is reliability: output must be valid JSON and schema-compliant.

## Instructions

### Install

```bash
make install
```

### Run (default paths)

```bash
make run
```

### Run (custom paths)

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Debug

```bash
make debug
```

### Lint + type check

```bash
make lint
```

Optional strict mode:

```bash
make lint-strict
```

### Clean caches

```bash
make clean
```

## Algorithm explanation

The program uses constrained decoding over next-token logits:

1. Build an LLM prompt containing all function definitions and user request.
2. Generate output one token at a time.
3. For each step:
   - inspect all candidate token IDs,
   - decode each candidate token piece,
   - keep only tokens whose appended text is still a plausible prefix of the target JSON shape.
4. Mask all invalid token logits to negative infinity.
5. Select the best valid token.
6. Stop once a fully parseable JSON object with exact keys `{name, parameters}` is produced.
7. Validate function name against available definitions.
8. Coerce parameter values to required types (`number`, `integer`, `boolean`, etc.).

This ensures robust structured outputs compared to prompt-only generation.

## Design decisions

- **Pydantic-first schemas** for strict validation (`extra="forbid"`).
- **Modular architecture**:
  - `models.py`: schema contracts
  - `io_utils.py`: file handling and JSON validation
  - `decoder.py`: constrained decoding + parsing + type coercion
  - `pipeline.py`: orchestration
  - `__main__.py`: CLI + error management
- **Graceful failure behavior**: program does not crash on prompt-level failures.
- **No private SDK usage**: only public methods from provided `Small_LLM_Model`.

## Performance analysis

Expected targets:
- High function selection / argument extraction accuracy (goal: 90%+)
- 100% parseable output JSON
- Runtime under a few minutes for standard datasets

Notes:
- Current token filtering loops over full vocabulary each step (simple and explicit, but expensive).
- Can be optimized later with caching candidate token pieces/prefix states.

## Challenges faced

- Small models can drift from strict JSON formatting.
- Token-level constrained decoding is necessary to force reliable structure.
- Different tokenizer behaviors require careful handling of single-token decoding and prompt IDs shape.

## Testing strategy

- Validate malformed/missing input files.
- Validate malformed function schemas.
- Verify output JSON has exact keys and expected types.
- Test edge prompts (empty text, special characters, ambiguous requests).
- Run linting and static typing checks (`flake8`, `mypy`).

## Example usage

Input:
```json
{"prompt": "What is the sum of 2 and 3?"}
```

Output item:
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources

- Pydantic docs: https://docs.pydantic.dev/
- Python typing: https://docs.python.org/3/library/typing.html
- JSON standard: https://www.json.org/json-en.html
- Constrained decoding (overview): https://arxiv.org/abs/2307.09702

### How AI was used

AI was used for:
- project structure
- initial implementation drafts
- README structuring and wording
