.PHONY: setup test verify demo web requirements team-demo team-large team-linked-list

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

team-demo:
	uv run team --demo api --reset-memory

team-large:
	uv run team --demo large --reset-memory

team-linked-list:
	uv run team --demo linked-list --reset-memory
