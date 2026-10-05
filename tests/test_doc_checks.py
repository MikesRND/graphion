"""The documentation checker reports broken links, anchors and Python examples."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docs import check


class DocumentationCheckTests(unittest.TestCase):
    def check(self, files):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in files.items():
                (root / name).write_text(text)
            with patch.object(check, "ROOT", root):
                return check.check_pages([root / name for name in files if name.endswith(".md")])

    def test_valid_links_anchors_and_examples_pass(self):
        errors = self.check(
            {
                "a.md": '# Title\n\n<a id="x"></a>\n[b](b.md#section) [self](#title) [web](https://example.com)\n'
                "```python\nprint('ok')\n```\n",
                "b.md": "# Section\n[line](c.py#L2)\n",
                "c.py": "first = 1\nsecond = 2\n",
            }
        )
        self.assertEqual(errors, [])

    def test_problems_are_reported(self):
        errors = self.check(
            {
                "a.md": '<a id="x"></a><a id="x"></a>\n[gone](missing.md) [anchor](b.md#nope) [line](c.py#L99)\n'
                "```python\ndef broken(:\n```\n",
                "b.md": "# Section\n",
                "c.py": "only = 1\n",
            }
        )
        self.assertEqual(len(errors), 5, errors)
        for expected in (
            "duplicate explicit anchor",
            "broken local link",
            "missing anchor",
            "invalid line reference",
            "invalid Python example",
        ):
            self.assertTrue(any(error.startswith(expected) for error in errors), (expected, errors))

    def test_links_outside_the_checkout_are_broken(self):
        errors = self.check({"a.md": "[outside](../elsewhere.md)\n"})
        self.assertEqual(len(errors), 1)
        self.assertIn("broken local link", errors[0])

    def test_main_exit_codes(self):
        with patch.object(check, "check_pages", return_value=[]), patch("builtins.print"):
            self.assertEqual(check.main([]), 0)
        with patch.object(check, "check_pages", return_value=["broken"]), patch("builtins.print"):
            self.assertEqual(check.main([]), 1)

    def test_unknown_options_are_rejected(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            check.main(["--run-tests"])
        self.assertEqual(raised.exception.code, 2)

    def test_repository_documentation_passes(self):
        self.assertEqual(check.check_pages(check.pages()), [])


if __name__ == "__main__":
    unittest.main()
