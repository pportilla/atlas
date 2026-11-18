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
        list(session.run('x,y = coordinates("x","y")\na,b = coordinates("x","y", fresh=True)\nf = x+y'))
        self.assertEqual(workspace(session.variables).header()[0],
                         ["R1 = QQ[x, y] → x, y, f", "R2 = QQ[x, y] → a, b"])

        def unnamed():
            # Without saved ring names, equal presentations fall back to a path.
            return workspace({k: v for k, v in session.variables.items() if k not in session.rings.names}).header()

        self.assertEqual(unnamed()[0], ["x.parent = QQ[x, y] → x, y, f", "a.parent = QQ[x, y] → a, b"])
        list(session.run("x = 17"))
        rings, bindings = workspace(session.variables).header()
        self.assertCountEqual(rings, ["R1 = QQ[x, y] → y, f", "R2 = QQ[x, y] → a, b"])
        self.assertIn("x = 17", bindings)
        self.assertCountEqual(unnamed()[0], ["y.parent = QQ[x, y] → y, f", "a.parent = QQ[x, y] → a, b"])
        # A named ring stays in the workspace until it is deleted.
        list(session.run("f = 0\ny = 1"))
        self.assertEqual(workspace(session.variables).header()[0], ["R1 = QQ[x, y]", "R2 = QQ[x, y] → a, b"])
        self.assertEqual(unnamed()[0], ["QQ[x, y] → a, b"])
        list(session.run("delete_ring(R1)"))
        self.assertEqual(workspace(session.variables).header()[0], ["R2 = QQ[x, y] → a, b"])

    def test_maps_and_mixed_collections_keep_both_endpoint_rings_distinct(self):
        session = new_session(None)
        list(session.run('x = coordinates("x")\na = coordinates("x", fresh=True)\n'
                         'm = hom(x.parent, a.parent, images=(a,))\npair = (x,a)\nf = polynomial("x", ring=x.parent)'))
        inventory = workspace(session.variables)
        first, second = inventory.rings.values()
        self.assertEqual([(entry.role, entry.name) for entry in first.members],
                         [("Coordinates", "x"), ("Polynomials", "f")])
        self.assertEqual([entry.text(expanded=True) for entry in second.members], ["a (generator x)"])
        self.assertEqual([entry.text(expanded=True) for entry in inventory.other],
                         ["m = R1 → R2", "pair = tuple[2]"])
        map_only = workspace({"m": session.variables["m"]})
        self.assertFalse(map_only.rings)
        self.assertEqual(map_only.other[0].text(), "m = QQ[x] (m.domain) → QQ[x] (m.codomain)")

    def test_help_and_workspace_leave_native_gcd_results_unexpanded(self):
        from engine.session_help import algebra_help_values, algebra_value_help

        session = new_session(None)
        list(session.run('x,y = coordinates("x","y")\nf = gcd(x*y, x^2)\nz = gcd(0*x, 0*y)\nalias = f'))
        f, z = (session.variables[name] for name in ("f", "z"))
        topics = algebra_help_values(session)
        card = algebra_value_help("alias", topics["alias"], session.variables)
        self.assertIn("alias is a polynomial in R1 (QQ[x, y]).", card)
        card = algebra_value_help("z.multiplicity", topics["z.multiplicity"], session.variables)
        self.assertIn("z is zero and has no finite factor multiplicity.", card)
        inventory = workspace(session.variables)
        self.assertIs(session.variables["alias"], f)
        self.assertIn("z = 0", [v.text(expanded=True) for v in inventory.rings[f.parent].members])
        for value in (f, z):
            self.assertNotIn("_value", vars(value))
            self.assertNotIn("_record", vars(value))

    def test_saved_empty_coordinate_tuples_belong_to_their_own_ring(self):
        session = new_session(None)
        list(session.run('A = quotient_ring(ideal(ring=QQ))\ng = A.coordinates\np = QQ.coordinates\na = ()'))
        inventory = workspace(session.variables)
        self.assertEqual(inventory.header()[0], ["A = QQ/(0) → g", "QQ → p"])
        self.assertEqual([entry.text(expanded=True) for entry in inventory.other], ["a = tuple[0]"])
        self.assertEqual(inventory.rings[session.variables["A"]].members[0].text(expanded=True), "g = ()")

    def test_scalar_and_chart_rings_and_safe_value_previews(self):
        session = new_session(None)
        list(session.run('S = ideal(variables=[])\nn = 10^4500'))
        resolution = Resolution({"variables": ["x", "y"], "dual_complex": {"vertices": []},
                                 "charts": [{"id": "C1", "coordinates": ["u", "v"], "ambient_ideal": ["u*v"]}]})
        rings, bindings = workspace(dict(session.variables, C=resolution.chart("C1"), q=Fraction(1, 3))).header()
        self.assertEqual(rings, ["QQ → S", "QQ[u, v] (quotient) → C"])
        self.assertIn("n = number", bindings)
        self.assertIn("q = 1/3", bindings)

        marked = Resolution({"variables": ["x"], "dual_complex": {"vertices": []},
                             "coefficient_field": {"generator": "theta"},
                             "charts": [{"id": "M", "coordinates": ["u"]}]})
        with patch("engine.session_values.chart_space", side_effect=AssertionError("unexpected field computation")):
            marked_rings, _ = workspace({"C": marked.chart("M")}).header()
        self.assertIn("u", marked_rings[0])

    def test_quotient_elements_and_polynomial_parent_handles_are_grouped_by_owner(self):
        session = new_session(None)
        list(session.run('x, y = coordinates("x", "y")\nP = x.parent\nA = quotient_ring(ideal(x))\nw = A(y)'))
        with patch("engine.session_quotient._evaluate", side_effect=AssertionError("workspace triggered reduction")):
            rings, bindings = workspace(session.variables).header()
        self.assertEqual(rings, ["P = QQ[x, y] → x, y", "A = P/(x) → w"])
        self.assertEqual(bindings, ["x", "y", "w = [y] in A"])

    def test_quotient_coordinates_keep_their_names_when_their_representatives_coincide(self):
        session = new_session(None)
        list(session.run('x, y = coordinates("x", "y")\nK = quotient_ring(ideal(x,y))\nu,v = K.coordinates\nz = K(x+y)\nA = K'))
        with patch("engine.session_quotient._evaluate", side_effect=AssertionError("workspace triggered reduction")):
            rings, bindings = workspace(session.variables).header()
            self.assertEqual(rings, ["R1 = QQ[x, y] → x, y", "K = A = R1/(x, y) → u, v, z"])
            self.assertEqual(bindings, ["x", "y", "u = [x] in K", "v = [y] in K", "z = [0] in K"])
            del session.variables["K"]
            self.assertIn("u = [x] in A", workspace(session.variables).header()[1])
            del session.variables["A"]
            rings, bindings = workspace(session.variables).header()
            self.assertIn("R1/(x, y) → u, v, z", rings)
            self.assertIn("v = [y]", bindings)

    def test_long_quotient_presentation_follows_rebinding_of_the_defining_ideal(self):
        session = new_session(None)
        list(session.run('a,b,c,d = coordinates("a","b","c","d")\nI = ideal(a,b,c,d)\nK = quotient_ring(I)'))
        original = session.variables["I"]
        with patch.dict(os.environ, TERM="xterm"), patch("os.get_terminal_size", return_value=os.terminal_size((42, 12))):
            with SessionPanel(Terminal()) as panel:
                panel.refresh(session.variables, assigned="I")
                self.assertIn("K = R1/I", panel.rows[0])
                list(session.run("I = ideal(a,b,c)"))
                panel.refresh(session.variables, assigned="I")
                self.assertNotIn("/I", panel.rows[0])
                session.variables["I"] = original
                panel.refresh(session.variables, assigned="I")
                self.assertIn("K = R1/I", panel.rows[0])
                self.assertLessEqual(len(panel.rows[0]), 38)

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

    def test_expanded_narrow_workspace_wraps_each_saved_binding_once(self):
        source = 'alpha = 14\nbeta = 23\ngamma = 32\n:workspace\nquit\n'
        with patch.dict(os.environ, TERM="dumb"), patch("shutil.get_terminal_size", return_value=os.terminal_size((22, 12))):
            code, output, err = run_prompt(source)
        self.assertEqual((code, err), (0, ""))
        section = output.split("Other values\n", 1)[1].split("atlas>", 1)[0]
        self.assertEqual(section.splitlines(), ["  Scalars", "    alpha = 14", "    beta = 23", "    gamma = 32"])

    def test_no_terminal_control_output_on_dumb_or_redirected_streams(self):
        for stream, term in ((io.StringIO(), "xterm"), (Terminal(), "dumb")):
            with self.subTest(term=term), patch.dict(os.environ, TERM=term):
                with SessionPanel(stream) as panel:
                    panel.refresh({"x": 1})
                self.assertEqual(stream.getvalue(), "")
