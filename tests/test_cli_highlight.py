import io
import os
import re
import unittest
from unittest.mock import patch

from engine.cli_highlight import (command_pieces, expression_pieces, live_terminal, paint_card, paint_hint,
                                  paint_value, repaint_entry)
from engine.session import new_session
from engine.session_help import function_help
from tests.cli_support import run_prompt

SGR = re.compile(r"\033\[[0-9;]*m")


def roles(pieces):
    return {text: role for text, role in pieces if text.strip()}


class HighlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = set(new_session(None).builtins)

    def test_expression_words_take_the_role_of_what_they_name(self):
        line = 'R = resolve(y^2 - x^3, mode="intrinsic").dual_complex  # cusp'
        found = roles(expression_pieces(line, names={"x", "y"}, functions=self.functions))
        self.assertEqual(found["R"], "name")          # saved by this statement
        self.assertEqual(found["resolve"], "function")
        self.assertEqual(found["x"], "name")
        self.assertEqual(found["2"], "number")
        self.assertEqual(found["mode"], "field")      # keyword argument
        self.assertEqual(found['"intrinsic"'], "string")
        self.assertEqual(found["dual_complex"], "field")
        self.assertEqual(found["# cusp"], "comment")

    def test_unknown_names_stay_plain_so_typos_stand_out(self):
        found = roles(expression_pieces("resolv(f)", names={"f"}, functions=self.functions))
        self.assertIsNone(found["resolv"])
        self.assertEqual(found["f"], "name")

    def test_literals_and_keywords(self):
        found = roles(expression_pieces("for k in [True, False, None]: pass", functions=self.functions))
        self.assertEqual((found["for"], found["True"], found["False"], found["None"]),
                         ("keyword", "ok", "false", "muted"))

    def test_verdicts_are_marked_only_in_displayed_values(self):
        value = "Resolution(-x^3+y^2, mode=embedded, status=resolved, charts=6)"
        self.assertEqual(roles(expression_pieces(value, verdicts=True))["resolved"], "ok")
        self.assertIsNone(roles(expression_pieces(value))["resolved"])
        self.assertEqual(roles(expression_pieces("complete, incomplete", verdicts=True))["incomplete"], "false")

    def test_shell_commands(self):
        found = roles(command_pieces("help milnor_number", commands={"help"}, topics=self.functions))
        self.assertEqual((found["help"], found["milnor_number"]), ("command", "function"))
        found = roles(command_pieces('resolve "y^2-x^3" --vars x,y', commands={"resolve"}))
        self.assertEqual((found["resolve"], found['"y^2-x^3"'], found["--vars"]), ("command", "string", "option"))
        self.assertEqual(roles(command_pieces(":save cusp.atlas", commands={":save"}))[":save"], "command")

    def test_painting_preserves_the_text(self):
        card = function_help("milnor_number", new_session(None).builtins)
        painted = paint_card(card, functions=self.functions, commands={"help"})
        self.assertNotEqual(painted, card)
        self.assertEqual(SGR.sub("", painted), card)
        hint = "Did you mean 'resolve'? Type help resolve."
        self.assertEqual(SGR.sub("", paint_hint(hint, commands={"help", "resolve"})), hint)
        value = "Dual complex (embedded, complete): 3 vertices"
        self.assertEqual(SGR.sub("", paint_value(value)), value)
        self.assertEqual(paint_value(value, color=False), value)
        self.assertEqual(paint_value("x" * 30000), "x" * 30000)

    def test_repaint_redraws_the_echoed_row_in_place(self):
        stream = io.StringIO()
        prompt = (("atlas", "brand"), ("> ", "muted"))
        line = "f = y^2"
        repaint_entry(stream, prompt, line, expression_pieces(line), columns=80)
        written = stream.getvalue()
        self.assertTrue(written.startswith("\0337\033[1A\r") and written.endswith("\033[K\0338"))
        self.assertEqual(SGR.sub("", written[len("\0337\033[1A\r"):-len("\033[K\0338")]), "atlas> " + line)

    def test_repaint_leaves_rows_it_cannot_place(self):
        prompt = (("atlas", "brand"), ("> ", "muted"))
        for line, columns in (("x" * 80, 80), ("f = ü", 80), ("\tf", 80), ("   ", 80)):
            with self.subTest(line=line):
                stream = io.StringIO()
                repaint_entry(stream, prompt, line, expression_pieces(line), columns)
                self.assertEqual(stream.getvalue(), "")

    def test_redirected_prompt_output_stays_plain(self):
        self.assertFalse(live_terminal())
        with patch.dict(os.environ, {"TERM": "xterm-256color"}):
            os.environ.pop("NO_COLOR", None)
            status, out, err = run_prompt("""
                x, y = coordinates("x", "y")
                membership(x*y, ideal(x)).value
                help milnor_number
                foo
            """)
        self.assertEqual(status, 0)
        printed = SGR.sub("", out)
        self.assertIn("True", printed)
        self.assertIn("milnor_number(polynomial, *, point=None", out)
        self.assertNotIn("\0337\033[1A", out)
        self.assertIn("Type ':workspace' to see your variables.", err)


if __name__ == "__main__":
    unittest.main()
