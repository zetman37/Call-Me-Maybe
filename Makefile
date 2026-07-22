.PHONY: install run lint format clean

PYTHON := python
UV := uv

install:
	$(UV) sync

run:
	$(UV) run --active $(PYTHON) -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

lint:
	$(UV) run --active flake8 src
	$(UV) run --active mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find src -type d -name "__pycache__" -exec rm -rf {} +
