"""Ring names, the active ring and ring deletion at the prompt and in scripts."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from engine.session import new_session
from engine.session_completion import candidates
from engine.session_expressions import SessionError
from engine.session_panel import SessionPanel, print_workspace, workspace
from engine.session_polynomial import QQ
from engine.session_quotient import QuotientElement
from engine.session_transcript import SessionTranscript
from tests.cli_support import run_prompt, run_script


def session_with(source):
    session = new_session(None)
    list(session.run(dedent(source).lstrip()))
    return session


class Terminal(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 1


class RingNameTests(unittest.TestCase):
    def check_at_both_entry_points(self, source, expected, shown=()):
        """The prompt also displays the values of expression statements, listed first in shown."""
        source = dedent(source).lstrip()
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), [*shown, *expected])

    def test_every_ring_in_use_has_a_name_until_you_choose_one(self):
        session = session_with('''
            x, y = coordinates("x", "y")
            u, v = coordinates("x", "y", fresh=True)
            X, Y = coordinates("x", "y")
        ''')
        names = session.variables
        self.assertIs(names["R1"], names["x"].parent)
        self.assertIs(names["R2"], names["u"].parent)
        self.assertIs(names["X"], names["x"])
        self.assertNotIn("R3", names)

        # Your own name replaces the one Atlas chose, which becomes free again.
        list(session.run("P = R2"))
        self.assertNotIn("R2", names)
        self.assertIs(names["P"], names["u"].parent)
        list(session.run('t = coordinates("t")'))
        self.assertIs(names["R2"], names["t"].parent)

        # A loop variable is a further name and leaves the chosen name in place.
        list(session.run("for r in [R1]:\n    n = 1"))
        self.assertIs(names["R1"], names["r"])

        # A ring that still holds values gets a free name when its last name is rebound.
        list(session.run("P = 1"))
        self.assertIs(names["R3"], names["u"].parent)
        ring = names["t"].parent
        list(session.run("t = 0\nR2 = 5"))
        self.assertFalse(any(value is ring for value in names.values()))
        self.assertEqual(names["R2"], 5)

        # A coordinate declaration may take a name that Atlas chose, but not one of yours.
        list(session.run('R3 = coordinates("R3")'))
        self.assertIs(names["R4"], names["u"].parent)
        self.assertIs(names["R5"], names["R3"].parent)
        with self.assertRaisesRegex(SessionError, "would replace"):
            list(session.run('r = coordinates("r")'))

    def test_ring_returns_the_session_parent_of_the_declared_coordinates(self):
        self.check_at_both_entry_points("""
            R = ring('x','y')
            f = y^2+x^5
            print(f.parent == R, f == polynomial("y^2+x^5", ring=R), active_ring() == None)
            R = ring("x", "y", coeffs=QQ)
            X, Y = coordinates("x", "y")
            print(X == x, x.parent == R, ring("x", "y") == R, ring("y", "x") == R)
            S = ring("x", "y", fresh=True)
            print(S == R, x.parent == S, y.parent == S, f.parent == R, X.parent == R)
            a = "s"
            b = "t"
            make = ring
            T = make(a, b)
            print(T.variables, (s+t).parent == T)
        """, ["True True True", "True True True False", "False True True True True", "('s', 't') True"])
        status, out, err = run_prompt("R = ring('x','y')\nf = y^2+x^5\n:workspace\nquit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertRegex(out, r"R = QQ\[x, y\]\s+Coordinates\s+x, y\s+Polynomials\s+f = x\^5\+y\^2")

    def test_conflicting_ring_declaration_preserves_bindings_and_accepts_coordinate_aliases(self):
        setup = "R = ring('x','y')\nf = y^2+x^5\n"
        conflict = "S = ring('z','x')\n"
        failed = run_script(setup + conflict)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("'x'", failed.stderr)
        recovery = "z, x2 = coordinates('z', 'x')\nS = z.parent\n"
        for command in recovery.splitlines():
            self.assertIn(command, failed.stderr)

        verify = "print(f.parent == R, x.parent == R, x2.parent == S, R == S)\n"
        status, out, err = run_prompt(setup + conflict + recovery + verify + "quit\n")
        self.assertEqual(status, 0)
        self.assertIn("coordinates(", err)
        self.assertIn("True True True False", out)
        replay = run_script(setup + recovery + verify)
        self.assertEqual((replay.returncode, replay.stdout, replay.stderr), (0, "True True True False\n", ""))

    def test_ring_preflights_all_names_before_binding(self):
        session = session_with('R = ring("x","y")\nf = x+y\nn = 1\nuse_ring(R)')
        original = session.variables.copy()
        ring_state = session.rings.state()
        for command in ('R = ring("z","x")', 'S = ring("z","n")', 'S = ring("z","f")',
                        'S = ring("z","R")', 'S = ring("z","S", fresh=True)',
                        'S = ring("z","ideal", fresh=True)', 'S = ring("z","for")',
                        'S = ring("z","help")'):
            with self.subTest(command=command):
                with self.assertRaises(SessionError):
                    list(session.run(command))
                self.assertEqual(session.variables.keys(), original.keys())
                for name, value in original.items():
                    self.assertIs(session.variables[name], value)
                self.assertEqual(session.rings.state(), ring_state)

    def test_ring_names_the_ring_it_creates(self):
        session = session_with('x, y = coordinates("x", "y")\nR = ring("x", "y")\nT = use_ring(ring("s", "t"))')
        names = session.variables
        self.assertIs(names["R"], names["x"].parent)
        self.assertNotIn("R1", names)
        self.assertIs(session.rings.active, names["T"])
        self.assertEqual(workspace(names, session.rings.active).header()[0],
                         ["R = QQ[x, y] → x, y", "T = QQ[s, t] (active)"])
        # Rebinding R to another ring leaves QQ[x, y] with a free name for its coordinates.
        list(session.run('R = ring("u", "z")'))
        self.assertIs(names["R1"], names["x"].parent)
        self.assertIs(names["R"], names["u"].parent)
        for command, message in (("ring()", "coordinate names"), ('ring("x", "x")', "distinct"),
                                 ('ring("x", fresh=1)', "fresh must be"), ('ring = 1', "reserved")):
            with self.subTest(command=command), self.assertRaisesRegex(SessionError, message):
                list(session.run(command))

    def test_generated_ring_name_is_available_as_a_coordinate(self):
        self.check_at_both_entry_points("""
            t = coordinates("t")
            S = ring("R1", "R2")
            print(R1.parent == S, R2.parent == S, R3 == t.parent)
        """, ["True True True"])

    def test_independent_same_named_rings_keep_their_own_names_in_both_entry_points(self):
        self.check_at_both_entry_points("""
            x, y = coordinates("x", "y")
            u, v = coordinates("x", "y", fresh=True)
            f = y^2 - x^3
            print(R1 == f.parent, R2 == u.parent, R1 == R2)
            S = R2
            g = polynomial("x^2+y", ring=S)
            print(g.parent == u.parent, g == u^2 + v)
        """, ["True True False", "True True"])

    def test_ring_name_follows_live_bindings(self):
        self.check_at_both_entry_points("""
            x = coordinates("x")
            u = coordinates("x", fresh=True)
            lookup = ring_name
            print(lookup(x.parent), lookup(u.parent), lookup(QQ))
            A = x.parent
            B = A
            print(lookup(B), lookup(ring("unsaved")))
            A = 10
            print(lookup(x.parent))
            K = ideal(x^2).quotient_ring
            print(lookup(K), lookup(K.ambient))
        """, ["R1 R2 QQ", "A None", "B", "K B"])

        session = session_with('x = coordinates("x")\nA = x.parent')
        original = session.variables["A"]
        lookup = session.builtins["ring_name"]
        self.assertIsNone(lookup(session_with('x = coordinates("x")').variables["R1"]))
        list(session.run("d = delete_ring(A)"))
        self.assertIsNone(lookup(original))
        status, out, err = run_prompt('lookup = ring_name\nhelp lookup\nx = coordinates("x")\n'
                                      'x = lookup(x)\nprint(lookup(x.parent))\nquit\n')
        self.assertEqual(status, 0)
        self.assertIn("lookup(ring)", out)
        self.assertIn("docs/sessions.md#name-select-and-delete-rings", out)
        self.assertIn("Pass a live ring", err)
        self.assertIn("atlas> R1\n", out)

    def test_names_reach_nested_collections_at_both_entry_points(self):
        self.check_at_both_entry_points("""
            values = [coordinates("x"), [1, coordinates("x", fresh=True)]]
            both = (values, values)
            print(R1 == values[0].parent, R2 == values[1][1].parent)
            R1 = "occupied"
            print(R3 == values[0].parent)
            d = delete_ring(R2)
            print(d.names, ring_name(R3))
        """, ["True True", "True", "('values', 'R2', 'both') R3"])

    def test_ring_lookup_and_record_traversal_use_only_existing_metadata(self):
        session = session_with('x = coordinates("x")\nK = ideal(x^2).quotient_ring\na = K(x)')
        quotient = session.variables["K"]
        snapshot = dict(vars(quotient))
        with patch("engine.session_quotient._evaluate", side_effect=AssertionError("unexpected reduction")):
            list(session.run("name = ring_name(a.parent)\ncheck_unit = a.is_unit"))
            self.assertEqual(session.variables["name"], "K")
            self.assertEqual(vars(quotient).keys(), snapshot.keys())
            self.assertTrue(all(vars(quotient)[key] is value for key, value in snapshot.items()))
            # Linked records may share subcollections. Entries retain the same
            # dependencies regardless of which root the session scans first.
            items = []
            record = {"items": items, "class": session.variables["a"]}
            items.append(record)
            session.variables.update(record=record, items=items)
            self.assertEqual(session.builtins["delete_ring"](quotient).names,
                             ("K", "a", "check_unit", "record", "items"))

    def test_detached_records_and_saved_aliases_avoid_revisiting_their_contents(self):
        session = session_with('x = coordinates("x")\nJ = groebner(ideal(x))\nvalues = [1, x]')
        with patch("engine.session_rings._collection_rings", side_effect=AssertionError("rescanned a saved collection")):
            list(session.run("receipt = J.record\nrows = receipt.result.generators\ncopy = values"))
            self.assertIs(session.variables["copy"], session.variables["values"])
            self.assertEqual(session.variables["receipt"], session.variables["J"].record)

        self.check_at_both_entry_points("""
            x = coordinates("x")
            items = [[x]]
            restored = items[len(delete_ring(x.parent).names) - 3]
            print(ring_name(restored[0].parent))
        """, ["R1"])

    def test_names_and_deletion_follow_inclusions_and_normalization_components(self):
        self.check_at_both_entry_points("""
            R = ring("x", "y")
            E = eliminate(ideal(x-y), x)
            C = normalization(ideal(y^2-x^3)).components[0]
            K = E.quotient_ring
            print(ring_name(E.inclusion.domain), ring_name(E.inclusion.codomain))
            print(ring_name(C.ambient), ring_name(C.prime.parent))
            deleted = delete_ring(R)
            print(deleted.names, ring_name(R1), ring_name(R2))
        """, ["R1 R", "R2 R", "('R', 'x', 'y', 'E', 'C', 'K') R1 R2"])

    def test_calculations_name_new_rings_in_statement_order_in_both_entry_points(self):
        # The two ideal sums at the end run as one native group in the script.
        self.check_at_both_entry_points("""
            x, y, z = coordinates("x", "y", "z")
            I = ideal(x - y, y - z)
            A = eliminate(I, x)
            B = ideal_sum(I, I)
            C = eliminate(I, y)
            D = ideal_sum(B, I)
            A = ideal_sum(B, I)
            print(R2.variables, R3.variables)
            print(A.parent == R1, C.parent == R3, D.parent == R1)
        """, ["('y', 'z') ('x', 'z')", "True True True"])

    def test_native_groups_do_not_rebind_ring_names(self):
        session = session_with('''
            x, y = coordinates("x", "y")
            u, v = coordinates("x", "y", fresh=True)
            I = ideal(x, y)
        ''')
        list(session.run("R2 = ideal_sum(I, I)\nR4 = ideal_sum(I, I)"))
        names = session.variables
        self.assertIs(names["R3"], names["u"].parent)
        self.assertIs(names["R2"].parent, names["R1"])

    def test_active_ring_reads_strings_and_holds_ideals_of_numbers(self):
        session = session_with('''
            x, y = coordinates("x", "y")
            u, v = coordinates("x", "y", fresh=True)
        ''')
        for command in ('polynomial("x+y")', "ideal()", "ideal(1)"):
            with self.subTest(command=command), self.assertRaisesRegex(SessionError, r"use_ring\(f\.parent\)"):
                list(session.run(command))
        self.assertIsNone(session.builtins["active_ring"]())

        list(session.run(dedent('''
            use_ring(R2)
            f = polynomial("x^2+y")
            I = ideal()
            a = 1
            J = ideal(a, 2)
            K = ideal(x)
            q = polynomial("1/2", ring=QQ)
            L = ideal(q)
            g = polynomial("x", ring=R1)
        ''')))
        names = session.variables
        for name in ("f", "I", "J"):
            self.assertIs(names[name].parent, names["R2"])
        self.assertEqual(names["f"], names["u"] ** 2 + names["v"])
        self.assertIs(names["K"].parent, names["R1"])
        self.assertIs(names["L"].parent, QQ)
        self.assertIs(names["g"].parent, names["R1"])
        self.assertIs(session.builtins["active_ring"](), names["R2"])

        list(session.run('A = ideal(x^2 - y).quotient_ring\nuse_ring(A)\nc = polynomial("x^2")\nZ = ideal()'))
        self.assertIsInstance(names["c"], QuotientElement)
        self.assertIs(names["c"].parent, names["A"])
        self.assertEqual(names["c"], names["A"](names["y"]))
        self.assertIs(names["Z"].parent, names["A"])

        list(session.run("use_ring(None)"))
        self.assertIsNone(session.rings.active)
        for command, message in (("use_ring(QQ)", "QQ has no coordinates"), ("use_ring(x)", "Pass a ring"),
                                 ("use_ring(5)", "Pass a ring")):
            with self.subTest(command=command), self.assertRaisesRegex(SessionError, message):
                list(session.run(command))
        self.assertIsNone(session.rings.active)

    def test_resolve_and_monodromy_check_read_strings_in_the_active_ring(self):
        self.check_at_both_entry_points("""
            x, y = coordinates("x", "y")
            use_ring(R1)
            R = resolve("y^2-x^3")
            M = monodromy_check("y^2-x^3")
            print(R.polynomial.parent == x.parent, M.resolution.polynomial.parent == R1)
        """, ["True True"], shown=["QQ[x, y]"])
        session = session_with('x, y = coordinates("x", "y")\nuse_ring(ideal(x).quotient_ring)')
        with self.assertRaisesRegex(SessionError, "affine polynomial coordinates"):
            list(session.run('resolve("y^2-x^3")'))

    def test_deleting_a_ring_removes_everything_that_depends_on_it(self):
        self.check_at_both_entry_points("""
            x, y = coordinates("x", "y")
            X, Y, z = coordinates("x", "y", "z")
            f = y^2 - x^3
            include = hom(R1, R2, images=(X, Y))
            g = include(f) + z
            K = ideal(f).quotient_ring
            a = K(x)
            pair = [f, 1]
            R = resolve(f)
            C = R.chart("B3.u")
            s = C.strict_transform
            use_ring(K)
            d = delete_ring(R1)
            print(d.names)
            print(d.was_active, active_ring() == None, g.parent == R2, s.parent == R3)
        """, ["('x', 'y', 'R1', 'f', 'include', 'K', 'a', 'pair', 'R', 'C')", "True True True True"],
           shown=["QQ[x, y]/(x^3-y^2) (grevlex)"])

    def test_deleting_a_quotient_keeps_its_ambient_ring(self):
        session = session_with('''
            x, y = coordinates("x", "y")
            K = ideal(y^2 - x^3).quotient_ring
            u, v = K.coordinates
            J = ideal(v, ring=K)
            L = J.quotient_ring
            w = L(x)
            use_ring(R1)
        ''')
        deletion = session.builtins["delete_ring"](session.variables["K"])
        self.assertEqual(deletion.names, ("K", "u", "v", "J", "L", "w"))
        self.assertFalse(deletion.was_active)
        self.assertEqual(str(deletion), "Deleted QQ[x, y]/(x^3-y^2) (grevlex): K, u, v, J, L, w")
        self.assertEqual(list(session.variables), ["x", "y", "R1"])
        self.assertIs(session.rings.active, session.variables["R1"])
        for command, message in (("delete_ring(QQ)", "built in"), ("delete_ring(x)", "Pass a ring")):
            with self.subTest(command=command), self.assertRaisesRegex(SessionError, message):
                list(session.run(command))
        self.assertEqual(list(session.variables), ["x", "y", "R1"])

    def test_deletion_follows_shared_and_nested_collections(self):
        session = session_with('x = coordinates("x")\nL = [x, 1]\nM = L\nN = [L, 2]\nk = [3]')
        deletion = session.builtins["delete_ring"](session.variables["R1"])
        self.assertEqual(deletion.names, ("x", "R1", "L", "M", "N"))
        self.assertEqual(list(session.variables), ["k"])
        list(session.run("d = delete_ring(coordinates(\"t\").parent)"))
        self.assertIn("d = delete_ring", workspace(session.variables).header()[1])

    def test_a_failed_prompt_command_restores_names_and_the_active_ring(self):
        session, log = new_session(None), SessionTranscript()
        setup = 'x, y = coordinates("x", "y")\nu, v = coordinates("x", "y", fresh=True)\nI = ideal(x)\nuse_ring(R1)'
        list(session.run(setup, transcript=log))
        before = dict(session.variables)
        for command in ("(use_ring(R2), missing)", "(delete_ring(R1), missing)",
                        "for r in [R2]:\n    delete_ring(r)\n    use_ring(None)\n    S = R1\n    missing"):
            with self.subTest(command=command):
                with self.assertRaises(SessionError):
                    list(session.run(command, transcript=log))
                self.assertEqual(list(session.variables), list(before))
                self.assertTrue(all(session.variables[name] is value for name, value in before.items()))
                self.assertIs(session.rings.active, before["R1"])
        self.assertEqual(log.statements, setup.splitlines())

        # The restored names are still known when a later group rebinds one.
        list(session.run("R2 = ideal_sum(I, I)\nR4 = ideal_sum(I, I)", transcript=log))
        self.assertIs(session.variables["R3"], session.variables["u"].parent)
        list(session.run("S = R1", transcript=log))
        self.assertNotIn("R1", session.variables)

    def test_saved_prompt_session_replays_ring_choices_in_a_script(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rings.atlas"
            status, output, error = run_prompt(dedent(f"""
                x, y = coordinates("x", "y")
                u, v = coordinates("x", "y", fresh=True)
                use_ring(R2)
                f = polynomial("x*y")
                (delete_ring(R1), missing)
                d = delete_ring(R1)
                print(f == u*v, d.names, active_ring() == R2)
                :save {path}
                quit
            """).lstrip())
            self.assertEqual(status, 0)
            self.assertIn("Unknown name 'missing'", error)
            replay = run_script(path.read_text())
        self.assertEqual((replay.returncode, replay.stderr), (0, ""))
        self.assertEqual(replay.stdout, "True ('x', 'y', 'R1') True\n")
        self.assertIn("True ('x', 'y', 'R1') True", output)

    def test_workspace_marks_the_active_ring(self):
        session = session_with('x, y = coordinates("x", "y")\nu, v = coordinates("x", "y", fresh=True)\nuse_ring(R2)')
        names, active = session.variables, session.rings.active
        rings, _ = workspace(names, active).header()
        self.assertEqual(rings, ["R1 = QQ[x, y] → x, y", "R2 = QQ[x, y] (active) → u, v"])
        with redirect_stdout(io.StringIO()) as listing, patch.dict(os.environ, NO_COLOR="1"):
            print_workspace(names, active)
        self.assertIn("R2 = QQ[x, y] (active)\n  Coordinates", listing.getvalue())
        self.assertIn("R1 = QQ[x, y]\n  Coordinates", listing.getvalue())
        self.assertIn("R1", candidates("R", session))

        # Selecting another ring redraws the header even when the binding is unchanged.
        with patch.dict(os.environ, TERM="xterm"), patch("os.get_terminal_size", return_value=os.terminal_size((80, 12))):
            with SessionPanel(Terminal()) as panel:
                panel.refresh(names, assigned="u", active=names["R2"])
                self.assertIn("R2 = QQ[x, y] (active)", panel.rows[0])
                panel.refresh(names, assigned="u", active=names["R1"])
                self.assertIn("R1 = QQ[x, y] (active)", panel.rows[0])
                self.assertNotIn("R2 = QQ[x, y] (active)", panel.rows[0])


if __name__ == "__main__":
    unittest.main()
