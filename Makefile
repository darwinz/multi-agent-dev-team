.PHONY: setup test lint docs verify demo web requirements team-demo team-large team-linked-list team-bst

setup:
	uv sync

test:
	uv run pytest

lint:
	uvx ruff@0.17.4 check .

docs:
	PYTHONHASHSEED=0 uv run --with pdoc==16.0.0 python -m scripts.build_api_docs

verify:
	REQUIRE_LIVE=1 uv run pytest -m live -rs

demo:
	uv run coder --demo --reset-memory

web:
	uv run coder-web

requirements:
	uv export --no-hashes --no-emit-project --format requirements-txt > requirements.txt

team-demo:
	uv run team --demo api --reset-memory

team-large:
	uv run team --demo large --reset-memory

team-linked-list:
	uv run team --demo linked-list --reset-memory

team-bst:
	uv run team --demo bst --reset-memory
