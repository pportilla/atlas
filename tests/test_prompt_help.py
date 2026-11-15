import shlex
import tempfile
import unittest
from pathlib import Path

from engine.session import new_session
from tests.cli_support import run_prompt


class PromptHelpTests(unittest.TestCase):
    cusp = 'x, y = coordinates("x", "y")\nK = resolve(y^2-x^3)\n'

    def ask(self, command):
        status, output, error = run_prompt(command + "\nquit\n")
        self.assertEqual(status, 0)
        return output.split("atlas> ", 1)[1].rsplit("atlas> ", 1)[0].strip(), error

    def assert_example_runs(self, setup, card):
        example = card.split("Example\n", 1)[1]
        source = "\n".join(line.removeprefix("  ") for line in example.splitlines() if line.startswith("  "))
        self.assertTrue(source)
        _, error = self.ask(setup + source)
        self.assertEqual(error, "")

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

    def test_recorded_complex_help_explains_its_convention(self):
        for request in ("help K.dual_complex", "K.dual_complex -h"):
            with self.subTest(request=request):
                card, error = self.ask(self.cusp + request)
                self.assertEqual(error, "")
                prose = " ".join(card.lower().split())
                self.assertIn("algebraic closure", prose)
                self.assertIn("strict transform", prose)
                self.assertIn("K.branches", card)

    def test_chart_lookup_and_branch_attachment_at_the_prompt(self):
        commands = [
            "K.dual_complex",
            "K.dual_complex.vertices",
            "K.branches[0].meets",
            "C = K.branches[0].chart",
            "C.id",
            "K.chart(C.id) == C",
            "K.chart(K.charts[0].id) == K.charts[0]",
            "C.strict_transform",
        ]
        output, error = self.ask(self.cusp + "\n".join(commands))
        self.assertEqual(error, "")
        self.assertIn("3 vertices, 2 edges", output)
        self.assertIn("E1 — E3, E2 — E3", output)
        self.assertIn("atlas> [E1, E2, E3]", output)
        self.assertIn("atlas> [E3]", output)
        self.assertIn("B3.u", output)
        self.assertEqual(output.count("atlas> True"), 2)
        self.assertIn("atlas> v-1", output)

    def test_help_examples_use_the_saved_resolution_and_chart(self):
        setup = self.cusp + 'patch = K.chart("B3.u")\n'
        for request, fields in (
            ("help K.charts", ("K.charts[0]", "K.chart(")),
            ("K.chart help", ("K.chart(",)),
            ("help K.branches", ("K.branches[0].chart", "K.branches[0].meets")),
            ("patch -h", ("patch.coordinates", "patch.strict_transform", "patch.map", "patch.divisors")),
        ):
            with self.subTest(request=request):
                card, error = self.ask(setup + request)
                self.assertEqual(error, "")
                for field in fields:
                    self.assertIn(field, card)
                self.assert_example_runs(setup, card)

    def test_help_for_a_smooth_curve_without_recorded_branches(self):
        setup = 'x, y = coordinates("x", "y")\nS = resolve(y)\n'
        for field in ("dual_complex", "dual_complex.vertices", "branches"):
            with self.subTest(field=field):
                card, error = self.ask(setup + "help S." + field)
                self.assertEqual(error, "")
                self.assertNotIn("S.branches[0]", card)
                if field == "branches":
                    self.assertRegex(card.lower(), r"no (?:recorded )?branch|branch.{0,50}(?:unavailable|absent)")
                self.assert_example_runs(setup, card)

    def test_static_topics_and_typo_suggestions(self):
        for topic, word in (("resolution.charts", "chart"), ("chart.strict_transform", "equation")):
            with self.subTest(topic=topic):
                card, error = self.ask("help " + topic)
                self.assertEqual(error, "")
                self.assertIn(word, card.lower())
        _, error = self.ask(self.cusp + "help K.chrts")
        self.assertIn("help K.charts", error)
        _, error = self.ask("K = 3\nhelp K.charts")
        self.assertIn("Unknown help topic", error)

    def test_help_follows_live_algebra_values_and_function_aliases(self):
        setup = ('x, y = coordinates("x", "y")\nf = x^2+y\nI = ideal(f)\nA = quotient_ring(I)\n'
                 'a = A(x)\nphi = A.projection\nalias = factor\nd = dimension(I)\nring = x.parent\n')
        for topic, example_part in (("ring", "ring.coordinates"), ("f", "ring=f.parent"),
                                    ("f.parent", "f.parent.coordinates"), ("I", "I.generators"),
                                    ("I.parent", "ring=I.parent"), ("A", "A.ambient.coordinates"),
                                    ("a", "a.representative"), ("phi", "phi.images"), ("d", "d.value"),
                                    ("phi.codomain", "phi.codomain.gens"), ("alias", "alias(")):
            with self.subTest(topic=topic):
                card, error = self.ask(setup + "help " + topic)
                self.assertEqual(error, "")
                self.assertIn(example_part, card)
                self.assert_example_runs(setup, card)
        from engine.app import make_parser
        from engine.prompt_help import topic_names
        session = new_session(None)
        list(session.run(setup))
        self.assertTrue({"f.parent", "I.ambient", "A", "phi.codomain", "phi.domain", "alias"} <=
                        topic_names(session, make_parser()))

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
