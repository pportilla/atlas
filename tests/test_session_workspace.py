"""The interactive ring and variable summary and its plain-text counterpart."""
import io
import os
import unittest
from fractions import Fraction
from unittest.mock import patch

from engine.session import new_session
from engine.session_panel import SessionPanel, workspace
from engine.session_values import Resolution
from tests.cli_support import run_prompt


class Terminal(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 1


class SessionWorkspaceTests(unittest.TestCase):
    def test_distinct_rings_follow_live_bindings_after_reassignment(self):
        session = new_session(None)
        list(session.run('x,y = coordinates("x","y")\na,b = coordinates("x","y")\nf = x+y'))
        self.assertEqual(workspace(session.variables)[0], ["QQ[x, y] (x)", "QQ[x, y] (a)"])
        list(session.run("x = 17"))
        rings, bindings = workspace(session.variables)
        self.assertEqual(rings, ["QQ[x, y] (y)", "QQ[x, y] (a)"])
        self.assertIn("x = 17", bindings)
        list(session.run("f = 0\ny = 1"))
        self.assertEqual(workspace(session.variables)[0], ["QQ[x, y]"])

    def test_scalar_and_chart_rings_and_safe_value_previews(self):
        session = new_session(None)
        list(session.run('S = ideal(variables=[])\nn = 10^4500'))
        resolution = Resolution({"variables": ["x", "y"], "dual_complex": {"vertices": []},
                                 "charts": [{"id": "C1", "coordinates": ["u", "v"], "ambient_ideal": ["u*v"]}]})
        rings, bindings = workspace(dict(session.variables, C=resolution.chart("C1"), q=Fraction(1, 3)))
        self.assertEqual(rings, ["QQ", "QQ[u, v] (quotient)"])
        self.assertIn("n = number", bindings)
        self.assertIn("q = 1/3", bindings)

        marked = Resolution({"variables": ["x"], "dual_complex": {"vertices": []},
                             "coefficient_field": {"generator": "theta"},
                             "charts": [{"id": "M", "coordinates": ["u"]}]})
        with patch("engine.session_values.chart_space", side_effect=AssertionError("unexpected field computation")):
            marked_rings, _ = workspace({"C": marked.chart("M")})
        self.assertIn("u", marked_rings[0])

    def test_narrow_panel_counts_omitted_bindings_and_clips_wide_text(self):
        stream = Terminal()
        with patch.dict(os.environ, TERM="xterm"), patch("os.get_terminal_size", return_value=os.terminal_size((36, 10))):
            with SessionPanel(stream) as panel:
                panel.refresh({name: name for name in "abcdefgh"})
                row = panel.rows[1]
                visible = [name for name in "abcdefgh" if f"{name} =" in row]
                self.assertIn("h", visible)
                self.assertIn(f"+{8-len(visible)}", row)
                panel.refresh({"message": "界"*24})
                row = panel.rows[1]
                self.assertIn("界", row)
                self.assertTrue(row.endswith("…"))
                self.assertLessEqual(len(row) + row.count("界"), 35)
                self.assertNotIn("h =", row)
                with patch("os.get_terminal_size", return_value=os.terminal_size((16, 10))):
                    panel.refresh({name: name for name in "abcdefgh"})
                    smallest = panel.rows[1]
                    self.assertLessEqual(len(smallest), 15)
                    self.assertIn("+7" if "h" in smallest else "+8", smallest)

    def test_plain_workspace_escapes_control_text_and_preserves_unicode(self):
        value = "界\x1b[2J\n\r\t\x07\u202e"
        source = f"message = {value!r}\n:workspace\nquit\n"
        with patch.dict(os.environ, TERM="dumb"):
            code, output, error = run_prompt(source)
        self.assertEqual((code, error), (0, ""))
        self.assertIn("message = " + repr(value), output)
        for control in ("\x1b", "\r", "\t", "\x07", "\u202e"):
            self.assertNotIn(control, output)

    def test_no_terminal_control_output_on_dumb_or_redirected_streams(self):
        for stream, term in ((io.StringIO(), "xterm"), (Terminal(), "dumb")):
            with self.subTest(term=term), patch.dict(os.environ, TERM=term):
                with SessionPanel(stream) as panel:
                    panel.refresh({"x": 1})
                self.assertEqual(stream.getvalue(), "")
