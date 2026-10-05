import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO

from tests.check_typing import main


@unittest.skipUnless(find_spec("mypy") is not None, "requires mypy from the dev dependency group")
class TypingCheckTests(unittest.TestCase):
    def test_repeated_runs_match_current_fixture_diagnostics(self) -> None:
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                output = StringIO()
                with redirect_stdout(output):
                    result = main()
                self.assertEqual(result, 0, output.getvalue())
