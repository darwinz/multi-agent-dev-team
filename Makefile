.PHONY: setup test verify demo web requirements

setup:
	uv sync

test:
	uv run pytest

verify:
	REQUIRE_LIVE=1 uv run pytest -rs

demo:
	uv run coder --demo --reset-memory

web:
	uv run coder-web

requirements:
	uv export --no-hashes --no-emit-project --format requirements-txt > requirements.txt
