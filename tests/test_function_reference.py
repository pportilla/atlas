"""The published session examples stay runnable as command implementations change."""
from contextlib import redirect_stdout
from io import StringIO
import unittest

from docs.site.catalog import ALIASES, ENTRIES
from docs.site.function_reference import FUNCTIONS, reference_for
from engine.app import calculate_resolution
from engine.session import new_session
from engine.session_function_help import function_help


class FunctionReferenceTests(unittest.TestCase):
    def test_catalog_and_descriptions_cover_the_live_interface(self):
        builtins = new_session(None).builtins
        self.assertEqual(set(ENTRIES) | set(ALIASES), {name for name, f in builtins.items() if callable(f)})
        self.assertEqual(FUNCTIONS.keys(), ENTRIES.keys())
        for alias, name in ALIASES.items():
            self.assertIs(builtins[alias], builtins[name])
        for name in ENTRIES:
            with self.subTest(name=name):
                inputs, returns = reference_for(name, builtins[name])
                self.assertTrue(all(description.strip() for _, description in inputs))
                self.assertTrue(returns.strip())

    def test_examples_run_in_fresh_sessions(self):
        for name in (*ENTRIES, *ALIASES):
            with self.subTest(name=name):
                session = new_session(calculate_resolution)
                card = function_help(name, session.builtins)
                example = card.split("\n\nExample\n", 1)[1].split("\n\nDocs  ", 1)[0]
                source = "\n".join(line.removeprefix("  ") for line in example.splitlines())
                with redirect_stdout(StringIO()):
                    list(session.run(source))
