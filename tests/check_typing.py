"""Check that Pyrefly rejects the invalid authoring fixture."""

import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    fixture = Path(__file__).resolve().with_name("typing_invalid_authoring.py.txt")
    project_root = fixture.parents[1]
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
                "pyrefly",
                "check",
                "--config",
                "pyproject.toml",
                "--output-format",
                "json",
                str(target),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        try:
            diagnostics = json.loads(result.stdout)["errors"]
            actual = sorted(
                (error["line"], error["name"])
                for error in diagnostics
                if (project_root / error["path"]).resolve() == target.resolve()
            )
        except (ValueError, TypeError, KeyError):
            print(result.stdout + result.stderr, end="")
            return 1
    if result.returncode != 1 or actual != expected or len(diagnostics) != len(expected):
        print(result.stdout + result.stderr, end="")
        print(f"Expected {expected}, received {actual} (exit {result.returncode})")
        return 1
    print(f"PASS: {len(expected)} expected authoring typing errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
