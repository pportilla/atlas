"""The published session examples stay runnable as command implementations change."""
import unittest

from docs.site.catalog import ALIASES, ENTRIES
from docs.site.function_reference import FUNCTIONS, reference_for
from engine.session import new_session
from engine.session_function_help import function_help
from tests.cli_support import run_prompt
from tests.zoo_support import TestZooMixin


class FunctionReferenceTests(TestZooMixin, unittest.TestCase):
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
                card = function_help(name, new_session(None).builtins)
                example = card.split("\n\nExample\n", 1)[1].split("\n\nDocs  ", 1)[0]
                commands = [line.removeprefix("  ") for line in example.splitlines()]
                expected = [line.partition("  # ")[2] for line in commands]
                self.assertTrue(any(expected), "Show a result that the reader can check")
                status, output, error = run_prompt("\n".join([*commands, "quit"]))
                self.assertEqual((status, error), (0, ""))
                self.assertEqual([chunk.rstrip("\n") for chunk in output.split("atlas> ")[1:-1]], expected)
