.PHONY: install test lint eval demo
install:
	pip install -e ".[dev]"
lint:
	ruff check . && mypy
test:
	pytest -q
eval:
	erp-copilot eval --min-pass 1.0
demo:
	erp-copilot ask "Why is INV-1003 on hold and how do I get it released?"
