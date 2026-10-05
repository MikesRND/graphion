"""Check documentation links, anchors and Python example syntax."""

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")


def anchors(text):
    found = set(re.findall(r'<a id="([^"]+)"', text))
    counts = Counter()
    for title in re.findall(r"^#{1,6} (.+)$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        found.add(slug if not counts[slug] else f"{slug}-{counts[slug]}")
        counts[slug] += 1
    return found


def pages():
    archived_plans = DOCS / "plans" / "done"
    found = sorted(page for page in DOCS.rglob("*.md") if not page.is_relative_to(archived_plans))
    return found + sorted((ROOT / "examples").rglob("*.md")) + [ROOT / "README.md"]


def check_pages(pages):
    errors = []
    for page in pages:
        text = page.read_text()
        explicit = re.findall(r'<a id="([^"]+)"', text)
        if len(explicit) != len(set(explicit)):
            errors.append(f"duplicate explicit anchor: {page.relative_to(ROOT)}")
        # Examples may be fragments or depend on other definitions. Syntax
        # checking does not establish runtime support for the documented API.
        for code in re.findall(r"```python\n(.*?)```", text, re.S):
            try:
                ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"invalid Python example in {page}: {exc}")
        for label, target in LINK.findall(text):
            if re.match(r"[a-zA-Z][\w+.-]*:", target):
                continue
            path, _, fragment = unquote(target).partition("#")
            destination = (page.parent / path).resolve() if path else page
            if not destination.is_relative_to(ROOT) or not destination.exists():
                errors.append(f"broken local link in {page.relative_to(ROOT)}: {target}")
                continue
            if fragment and destination.suffix == ".md":
                if fragment not in anchors(destination.read_text()):
                    errors.append(f"missing anchor in {page.relative_to(ROOT)}: {target}")
            elif fragment and re.fullmatch(r"L\d+", fragment):
                if not 1 <= int(fragment[1:]) <= len(destination.read_text().splitlines()):
                    errors.append(f"invalid line reference: {target}")
    return errors


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    checked = pages()
    errors = check_pages(checked)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: links, anchors and Python examples in {len(checked)} pages.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
