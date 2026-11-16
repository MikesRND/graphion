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
    uv run --locked python -m pyrefly check --config pyproject.toml

[positional-arguments]
test *args:
    uv run --locked pytest "$@"

docs:
    uv run --locked python docs/check.py

build:
    uv build

[positional-arguments]
commits base="origin/main":
    #!/usr/bin/env bash
    set -euo pipefail
    git log --no-merges --format=%s --end-of-options "$1..HEAD" | awk '
        /^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9-]+\))?!?: [^[:space:]].*$/ { next }
        /^Revert ".+"$/ { next }
        { print "Invalid commit subject: " $0 > "/dev/stderr"; invalid = 1 }
        END { exit invalid }
    '

check: fmt-check lint types test docs

ci: check build
