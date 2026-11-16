# List the available recipes.
default:
    @just --list

# Install the locked project and development dependencies.
sync:
    uv sync --locked

# Apply safe Ruff fixes and format Python.
fix:
    uv run --locked ruff check --fix .
    uv run --locked ruff format .

# Check Python formatting without changing files.
fmt-check:
    uv run --locked ruff format --check .

# Run the Ruff linter.
lint:
    uv run --locked ruff check .

# Check types with Pyrefly.
types:
    uv run --locked python -m pyrefly check --config pyproject.toml

# Run pytest, passing any arguments through.
[positional-arguments]
test *args:
    uv run --locked pytest "$@"

# Check documentation links, anchors and Python example syntax.
docs:
    uv run --locked python docs/check.py

# Build the wheel and source distribution.
build:
    uv build

# Check that commit subjects since the base follow Conventional Commits.
[positional-arguments]
commits base="origin/main":
    #!/usr/bin/env bash
    set -euo pipefail
    git log --no-merges --format=%s --end-of-options "$1..HEAD" | awk '
        /^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9_.\/-]+\))?!?: [^[:space:]].*$/ { next }
        /^Revert ".+"$/ { next }
        { print "Invalid commit subject: " $0 > "/dev/stderr"; invalid = 1 }
        END { exit invalid }
    '

# Run every pre-merge check.
check: fmt-check lint types test docs

# Run the pre-merge checks and build the distributions, as CI does.
ci: check build
