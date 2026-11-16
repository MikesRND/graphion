default:
    @just --list

sync:
    uv sync --locked

fix:
    uv run --locked ruff check --fix .
    uv run --locked ruff format .

fmt-check:
    uv run --locked ruff format --check .

lint:
    uv run --locked ruff check .

types:
    uv run --locked mypy

test *args:
    uv run --locked pytest {{args}}

docs:
    uv run --locked python docs/check.py

build:
    uv build

check: fmt-check lint types test docs

ci: check build
