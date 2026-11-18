"""Check the public contract for help on a new session function and its aliases."""
import re
import unittest
from pathlib import Path

from engine.session import new_session
from engine.session_function_help import function_documentation, function_help


ROOT = Path(__file__).resolve().parents[1]


class FunctionHelpTests(unittest.TestCase):
    def test_every_registered_function_points_to_a_real_guide_section(self):
        session = new_session(None)
        for name, function in session.builtins.items():
            if not callable(function):
                continue
            with self.subTest(name=name):
                card = function_help(name, session.builtins)
                header, example = card.split("\n\nExample\n")
                self.assertEqual(len(header.splitlines()), 2)
                self.assertIn(name + "(", example)
                self.assertIn("\n\nDocs  ", card)
                reference = card.split("/blob/main/")[-1]
                path, anchor = reference.split("#")
                source = (ROOT / path).read_text()
                anchors = set(re.findall(r'<a id="([^"]+)"', source))
                for heading in re.findall(r"^#{1,6} (.+)$", source, re.MULTILINE):
                    anchors.add(re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-"))
                self.assertIn(anchor, anchors)

    def test_new_function_docstring_supplies_summary_example_and_reference(self):
        def echo(label):
            """Return the supplied label.

            Extra API notes belong in the guide.

            >>> label = echo("echo(")
            >>> print(label)

            Docs: docs/sessions.md#run-or-save-a-calculation
            """
            return label

        session = new_session(None)
        session.builtins["again"] = echo
        card = function_help("again", session.builtins)
        self.assertEqual(card.split("\n\n")[0], "again(label)\nReturn the supplied label.")
        self.assertIn('  label = again("echo(")', card)
        self.assertTrue(card.endswith("/docs/sessions.md#run-or-save-a-calculation"))
        list(session.run(card.split("Example\n  ")[1].splitlines()[0]))
        self.assertEqual(session.variables["label"], "echo(")

    def test_docs_trailer_is_required_and_must_point_to_a_section(self):
        def new_function(x):
            return x

        summary = "Return the input value.\n\n>>> new_function(1)"
        for trailer in ("", "Docs: docs/sessions.md", "Docs: no such link", "Docs: docs/sessions.md#"):
            with self.subTest(trailer=trailer):
                new_function.__doc__ = summary + "\n\n" + trailer
                with self.assertRaisesRegex(ValueError, "new_function needs a final Docs:"):
                    function_help("new_function", {"new_function": new_function})

    def test_help_for_a_saved_result_uses_the_functions_own_docstring_reference(self):
        functions = new_session(None).builtins
        reference = function_documentation("ae_codimension")
        self.assertEqual(reference, "examples/local-algebra.md#compare-the-cusp-and-its-parametrization")
        self.assertTrue(function_help("ae_codimension", functions).endswith(reference))
