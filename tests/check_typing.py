"""Check expected typing failures: uv run python -m tests.check_typing."""

import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    fixture = Path(__file__).with_name("typing_invalid_authoring.py.txt")
    source = fixture.read_text()
    expected = [
        (number, match.group(1))
        for number, line in enumerate(source.splitlines(), 1)
        if (match := re.search(r"# error: ([\w-]+)", line))
    ]
    with TemporaryDirectory() as directory:
        target = Path(directory) / "invalid_authoring.py"
        target.write_text(source)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                # Cached diagnostics can retain a previous temporary path.
                "--no-incremental",
                "--follow-imports=silent",
                "--no-error-summary",
                "--no-pretty",
                "--show-error-codes",
                "--no-color-output",
                str(target),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        diagnostics = [line for line in result.stdout.splitlines() if ": note:" not in line]
        pattern = re.compile(re.escape(str(target)) + r":(\d+): error: .* \[([\w-]+)\]$")
        actual = [(int(match.group(1)), match.group(2)) for line in diagnostics if (match := pattern.fullmatch(line))]
    if result.returncode != 1 or result.stderr or actual != expected or len(diagnostics) != len(expected):
        print(result.stdout + result.stderr, end="")
        print(f"Expected {expected}; received {actual} (exit {result.returncode})")
        return 1
    print(f"PASS: {len(expected)} expected authoring typing errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
