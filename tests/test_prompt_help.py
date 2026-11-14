import shlex
import tempfile
import unittest
from pathlib import Path

from engine.session import new_session
from tests.cli_support import run_prompt


class PromptHelpTests(unittest.TestCase):
    def ask(self, command):
        status, output, error = run_prompt(command + "\nquit\n")
        self.assertEqual(status, 0)
        return output.split("atlas> ", 1)[1].rsplit("atlas> ", 1)[0].strip(), error

    def test_variables_help_forms_show_how_to_define_and_inspect_variables(self):
        cards = []
        for command in ("variables help", "variables -h", "variables --help", "help variables"):
            with self.subTest(command=command):
                card, error = self.ask(command)
                self.assertEqual(error, "")
                self.assertRegex(card, r"""\w+\s*=\s*coordinates\(["']\w+["']""")
                self.assertIn(":workspace", card)
                cards.append(card)
        self.assertEqual(len(set(cards)), 1)

    def test_registered_functions_each_have_usage_and_an_example(self):
        for name, function in new_session(None).builtins.items():
            if callable(function):
                with self.subTest(name=name):
                    card, error = self.ask(f"{name} -h")
                    self.assertEqual(error, "")
                    self.assertIn(f"{name}(", card)
                    self.assertIn("example", card.lower())
                    self.assertGreaterEqual(len(card.splitlines()), 3)

    def test_resolution_and_script_help_remain_available(self):
        resolve, error = self.ask("help resolve")
        self.assertEqual(error, "")
        self.assertIn("resolve(", resolve)
        self.assertIn("--vars", resolve)
        script, error = self.ask("help script")
        self.assertEqual(error, "")
        self.assertIn("coordinates(", script)
        self.assertIn("./atlas run", script)

    def test_error_suggestions_help_the_user_recover(self):
        for command, expected in (
            ('coordintes("x")', "coordinates"),
            ('polynomal "x"', "polynomial"),
            ("factor()", "factor"),
            ("help variabels", "variables"),
        ):
            with self.subTest(command=command):
                output, error = self.ask(command + "\nprint(17)")
                self.assertIn("17", output)
                self.assertIn(expected, error)
                self.assertIn("help", error.lower())
        for command in ("help zzzyyy", "algebra zzzyyy"):
            with self.subTest(command=command):
                output, error = self.ask(command + "\nprint(17)")
                self.assertIn("17", output)
                self.assertIn("zzzyyy", error)
                self.assertIn("help", error.lower())

    def test_nested_call_and_keyword_errors_point_to_the_failed_call(self):
        for command, topic in (
            ("print(factor())", "factor"),
            ('coordinates("x", coefs=QQ)', "coordinates"),
        ):
            with self.subTest(command=command):
                _, error = self.ask(command)
                self.assertRegex(error, rf"\bhelp\s+{topic}\b")
                if topic == "coordinates":
                    self.assertIn("coeffs", error)

        _, error = self.ask('x = coordinates("x")\nprint(x.multiplicity())')
        self.assertIn("x.multiplicity(factor)", error)
        self.assertNotIn("help multiplicity", error)

    def test_help_preserves_expressions_bindings_and_saved_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "saved.atlas"
            statements = ["variables = 8", "h = 2", "variables - h", 'print("variables -h")']
            output, error = self.ask("\n".join([
                *statements[:2], "variables help", "factor -h", *statements[2:],
                ":save " + shlex.quote(str(saved)),
            ]))
            self.assertEqual(error, "")
            self.assertRegex(output, r"atlas> 6\n")
            self.assertIn("atlas> variables -h\n", output)
            self.assertEqual(saved.read_text().splitlines(), statements)
