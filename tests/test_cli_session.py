import io
import os
import subprocess
import sys
import unittest
from textwrap import dedent
from unittest.mock import patch

from engine.session import new_session
from engine.session_polynomial import coordinates
from tests.cli_support import ROOT, run_prompt, run_script


class CliSessionTests(unittest.TestCase):
    @unittest.skipIf(sys.version_info < (3, 14), "Eager help formatting applies to Python 3.14 and newer")
    def test_resolve_does_not_prepare_terminal_help_while_parsing_a_job(self):
        import argparse
        from engine.resolve_cli import _ResolutionParser

        with patch.object(argparse.HelpFormatter, "__init__", side_effect=AssertionError("formatted help")):
            args = _ResolutionParser().parse_args(["x+y", "--vars", "x,y", "--quick", "--json"])
        self.assertEqual((args.polynomial, args.vars, args.search, args.json), ("x+y", "x,y", "quick", True))

    def test_resolve_command_uses_the_same_parser_and_skips_unrelated_imports(self):
        from engine.app import make_parser, run_command
        parser = make_parser()
        for argv in (["resolve", "--help"], ["resolve", "x+y", "--vars", "x,y", "extra"],
                     ["resolve", "x+y", "--vars", "x,y", "--quick", "--exhaustive"]):
            with self.subTest(argv=argv):
                out, err = io.StringIO(), io.StringIO()
                with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err), self.assertRaises(SystemExit) as stop:
                    run_command(parser, argv)
                direct = subprocess.run([str(ROOT / "atlas"), *argv], capture_output=True, text=True, timeout=10)
                self.assertEqual((direct.returncode, direct.stdout, direct.stderr),
                                 (stop.exception.code, out.getvalue(), err.getvalue()))

        probe = dedent("""
            import runpy, sys
            class Guard:
                def find_spec(self, name, path=None, target=None):
                    if name in ("engine.algebra_cli", "http.server"):
                        raise AssertionError(name)
            sys.meta_path.insert(0, Guard())
            sys.argv = ["atlas", "resolve", "--quick", "--vars", "x,y", "--json", "-x+y"]
            runpy.run_path("atlas", run_name="__main__")
        """)
        direct = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(direct.returncode, 0, direct.stderr)

    def test_standalone_script_roundtrips_large_decimals(self):
        env = os.environ.copy()
        env.pop("PYTHONINTMAXSTRDIGITS", None)
        integer = "9" * 4301
        source = f'x = coordinates("x")\nn = {integer}\nprint(n)\nprint(normal_form(x/n, ideal(x^2), compute="cpu"))\n'
        result = subprocess.run([str(ROOT / "atlas"), "run", "-"], input=source,
                                capture_output=True, text=True, env=env, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [integer, f"1/{integer}*x"])
        if hasattr(sys, "set_int_max_str_digits"):
            env["PYTHONINTMAXSTRDIGITS"] = "640"
            limited = subprocess.run([str(ROOT / "atlas"), "run", "-"], input=source,
                                     capture_output=True, text=True, env=env, timeout=15)
            self.assertEqual(limited.returncode, 2)
            self.assertIn("640 digits", limited.stderr)

    def test_coordinate_redeclaration_reuses_the_parent_at_prompt_and_in_script(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            F = y^2 + x
            x, y = coordinates("x", "y")
            print(F.parent == x.parent, F+x == y^2+2*x)
            x, y = coordinates("x", "y", fresh=True)
            g = polynomial("x^2+y", ring=F.parent)
            print(g.parent == F.parent, g == x^2+y)
            u, v = coordinates("x", "y", coeffs=QQ)
            x, y = F.parent.coordinates
            print(g == x^2+y, u.parent == F.parent)
        """)
        expected = ["True True", "True False", "True True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").strip().splitlines(), expected)

    def test_overlapping_coordinate_declarations_preserve_live_bindings(self):
        setup = 'x,y = coordinates("x","y")\nf = y^2+x^3\n'
        conflict = "x,z = coordinates('x','z')\n"
        hint = "x2, z = coordinates('x', 'z')"
        stopped = run_script(setup + conflict + 'print("continued")\n')
        self.assertEqual(stopped.returncode, 2)
        self.assertEqual(stopped.stdout, "")
        self.assertIn("'x' in QQ[x, y]", stopped.stderr)
        self.assertIn(hint, stopped.stderr)
        status, out, err = run_prompt(setup + conflict + ":workspace\n"
                                      "print(f.parent == x.parent, f-x^3 == y^2)\nquit\n")
        self.assertEqual(status, 0)
        self.assertIn(hint, err)
        self.assertIn("QQ[x, y]", out)
        self.assertRegex(out, r"Coordinates\s+x, y")
        self.assertRegex(out, r"Polynomials\s+f = ")
        self.assertNotIn("QQ[x, z]", out)
        self.assertIn("True True", out)

        source = setup + hint + '\nprint(x.parent == f.parent, x2.parent == z.parent, x.parent == x2.parent)'
        script = run_script(source)
        self.assertEqual((script.returncode, script.stdout.strip(), script.stderr), (0, "True True False", ""))

    def test_coordinate_alias_checks_all_names_before_changing_any(self):
        session = new_session(None)
        list(session.run('a = coordinates("x")\na2 = 2\nmake = coordinates'))
        with self.assertRaisesRegex(ValueError, r"z, a3 = make\('z', 'x'\)"):
            list(session.run('z, a = make("z","x")'))
        self.assertNotIn("z", session.variables)
        self.assertEqual(session.variables["a"].variables, ("x",))
        list(session.run('flag = True\nz, a = make("z", "x", fresh=flag)'))
        self.assertEqual(session.variables["a"].variables, ("z", "x"))

    def test_indexed_declarations_protect_every_kind_of_saved_binding(self):
        session = new_session(None)
        list(session.run('x, y = coordinates("x", "y")\nf = x+y\nn = 7\nA = quotient_ring(ideal(x))\n'
                         'a = A(x)\nG = coordinates("x", "z")\nmake = coordinates'))
        originals = session.variables.copy()
        for name in ("x", "f", "n", "A", "a", "G", "make"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, rf"{name}2 = make\('x', 'z'\)\[0\]"):
                list(session.run(f'{name} = (make("x","z"))[0]'))
            self.assertIs(session.variables[name], originals[name])
        # An ordinary assignment to an existing name is deliberate.
        list(session.run('x, z = G\nf = make("x","z",fresh=True)[0]'))
        self.assertIs(session.variables["x"].parent, originals["G"][0].parent)
        self.assertIsNot(session.variables["f"].parent, session.variables["x"].parent)

    def test_saved_coordinate_tuple_requires_generators_in_the_same_parent(self):
        session = new_session(None)
        list(session.run('G = coordinates("x","y")\nG = coordinates("x","y")'))
        original = session.variables["G"]
        with self.assertRaisesRegex(ValueError, r"G2 = coordinates\('x', 'z'\)"):
            list(session.run('G = coordinates("x","z")'))
        self.assertIs(session.variables["G"], original)
        list(session.run('G = coordinates("x","y",fresh=True)'))
        with self.assertRaisesRegex(ValueError, r"G2 = coordinates\('x', 'y'\)"):
            list(session.run('G = coordinates("x","y")'))

    def test_coordinate_declaration_protects_computed_values_equal_to_generators(self):
        setup = 'x,y = coordinates("x","y")\nP = x.parent\noriginal = x\nx = x+0\n'
        conflict = 'x, other = coordinates("x","y")\n'
        script = run_script(setup + conflict)
        self.assertEqual(script.returncode, 2)
        self.assertIn("x2, other = coordinates('x', 'y')", script.stderr)
        status, out, err = run_prompt(setup + conflict + 'print(x == original, x.parent == P)\nquit\n')
        self.assertEqual(status, 0)
        self.assertIn("True True", out)
        self.assertIn("x2, other = coordinates('x', 'y')", err)
        session = new_session(None)
        list(session.run(setup))
        saved = session.variables["x"]
        with self.assertRaisesRegex(ValueError, "would replace"):
            list(session.run(conflict))
        self.assertIs(session.variables["x"], saved)
        self.assertNotIn("other", session.variables)

    def test_explicit_assignment_selects_coordinate_objects_from_an_existing_ring(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            P = x.parent
            I = ideal(x-y)
            K = I.quotient_ring
            x, y = K.coordinates
            print(x.parent == K, x == y)
            x = P.coordinates[0]
            print(x.parent == P, y.parent == K)
            x, y = K.ambient.coordinates
            pair = (x, y)
            pair = coordinates("x", "y")
            print(pair == P.coordinates, y.parent == P)
        """).lstrip()
        expected = ["True True", "True True", "True True"]
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").strip().splitlines(), expected)

    def test_chart_coordinates_keep_their_positions_even_if_their_values_coincide(self):
        from engine.session_algebra import polynomial
        from engine.session_polynomial import QQ
        from engine.session_values import Resolution

        self.assertEqual(QQ.coordinates, ())
        self.assertIs(QQ.coordinates.parent, QQ)
        graph = Resolution({"variables": ["t"], "dual_complex": {"vertices": []}, "charts": [
            {"id": "G", "coordinates": ["a", "b"], "ambient_ideal": ["a", "b"]}
        ]}).chart("G")
        a, b = graph.coordinates
        self.assertEqual(a, 0)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)
        self.assertIs(graph.parent.coordinates[0], a)
        self.assertIs(graph.parent.coordinates[1], b)
        self.assertEqual(polynomial("b", variables=(a, b)), 0)
        with self.assertRaisesRegex(ValueError, "order"):
            polynomial("b", variables=(b, a))

    def test_concurrent_first_declarations_return_one_sessions_coordinates(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from engine.session_polynomial import _CoordinateSpace

        first_started, second_started = Event(), Event()
        make_parent = _CoordinateSpace.named

        def overlap(*args, **kwargs):
            if first_started.is_set():
                second_started.set()
            else:
                first_started.set()
                second_started.wait(timeout=1)
            return make_parent(*args, **kwargs)

        with patch.object(_CoordinateSpace, "named", side_effect=overlap), ThreadPoolExecutor(2) as pool:
            declare = new_session(None).builtins["coordinates"]
            a = pool.submit(declare, "x", "y")
            self.assertTrue(first_started.wait(timeout=2))
            b = pool.submit(declare, "x", "y")
            self.assertIs(a.result(), b.result())
            self.assertIs(a.result().parent, declare("x", "y").parent)

    def test_quoted_polynomials_reuse_chart_and_elimination_coordinates_at_both_entry_points(self):
        source = """
            x, y = coordinates("x", "y")
            R = resolve(y^2 - x^3)
            C = R.chart("B3.u")
            s = polynomial("v-1", variables=C.coordinates)
            zero = polynomial("u-u", ring=s.parent)
            print(s == C.strict_transform, zero+s == s)
            other = polynomial("v-1", variables=R.chart("B3.v").coordinates)
            print(s == other)
            E = eliminate(ideal(y-x^2, x-2), x, compute="cpu")
            q = polynomial("y^2/2-8", variables=E.coordinates)
            copy = polynomial("y^2/2-8", ring=q.parent)
            print(membership(q, E, compute="cpu").value, q == y^2/2-8, copy == q)
            u, v = coordinates("u", "v")
            t = coordinates("y")
            print(s.parent == u.parent, q.parent == t.parent)
        """
        expected = ["True True", "False", "True False True", "False False"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, out, err = run_prompt(dedent(source) + "quit\n")
        self.assertEqual((status, err), (0, ""))
        for line in expected:
            self.assertIn(f"atlas> {line}\n", out)

    def test_script_follows_cusp_from_branch_to_charts(self):
        result = run_script("""
            x, y = coordinates("x", "y")
            f = y^2 - x^3
            R = resolve(f)
            R
            branch = R.branches[0]
            E = branch.meets[0]
            print(R.status, len(R.divisors), len(R.charts))
            print(R.polynomial == f, R.polynomial + x^3 == y^2, R.record.polynomial == "-x^3+y^2")
            print(E.id, E.self_intersection, E.total_multiplicity)
            u, v = branch.chart.coordinates
            print(branch.chart.id, branch.chart.strict_transform == v-1)
            for chart in E.charts:
                total = chart.pullback(f)
                e = E.equation_in(chart)
                print(chart.id, e, total.multiplicity(e), total.multiplicity(chart.strict_transform))
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "resolved 3 6", "True True True", "E3 -1 6", "B3.u True", "B3.u u 6 1", "B3.v v 6 1",
        ])

    def test_prompt_computes_cusp_multiplicity_from_the_pulled_back_equation(self):
        status, out, err = run_prompt("""
            x, y = coordinates("x", "y")
            f = y^2 - x^3
            R = resolve(f)
            E = R.divisor("E3")
            C = R.branches[0].chart
            u, v = C.coordinates
            total = C.pullback(R.polynomial)
            total == u^6 * v^3 * C.strict_transform
            total == C.ambient_total_transform
            total.multiplicity(E.equation_in(C))
            C.pullback(f^2).multiplicity(u)
            C.pullback((1/2)*x+y) == (1/2)*u^2*v+u^3*v^2
            C.map["y"] == C.pullback(y)
            quit
        """)
        self.assertEqual(status, 0)
        self.assertEqual(err, "")
        self.assertEqual(out.count("atlas> True\n"), 4)
        self.assertIn("atlas> 6\n", out)
        self.assertIn("atlas> 12\n", out)

    def test_optional_field_and_exact_coefficients(self):
        result = run_script("""
            x, y = coordinates("x", "y", coeffs=QQ)
            f = (1/2)*y^2 - x^3
            print(f.coeffs, 2*f, -x^2)
            print((x+y)^2 == x^2+2*x*y+y^2, x-x == 0, x^0 == True)
            if x-x:
                print("unexpected nonzero polynomial")
            a, b = coordinates("x", "y")
            print(x == a)
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["QQ -2*x^3+y^2 -x^2", "True True False", "True"])

    def test_factor_multiplicity_counts_exact_division_and_rejects_undefined_orders(self):
        u, v = coordinates("u", "v")
        factor = 2*u+3*v
        product = 5*factor**3*(u+1)
        self.assertEqual(product.multiplicity(factor), 3)
        self.assertEqual(product.multiplicity(u+v), 0)
        with self.assertRaisesRegex(ValueError, "zero polynomial"):
            (u-u).multiplicity(factor)
        for constant in (u-u, u**0):
            with self.subTest(constant=constant), self.assertRaisesRegex(ValueError, "nonconstant"):
                product.multiplicity(constant)

    def test_file_stops_at_coordinate_space_error(self):
        result = run_script("""
            x, y = coordinates("x", "y")
            a, b = coordinates("x", "y", fresh=True)
            print("before")
            f = x + a
            print("after")
        """)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["before"])
        self.assertIn("curve.atlas:4:", result.stderr)
        self.assertIn("coordinate spaces", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cross_resolution_charts_are_rejected(self):
        result = run_script("""
            x, y = coordinates("x", "y")
            f = y^2 - x^3
            A = resolve(f)
            B = resolve(f)
            print(A.divisor("E3").equation_in(B.chart("B3.u")))
        """)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("curve.atlas:5:", result.stderr)
        self.assertIn("different resolutions", result.stderr)

    def test_equations_and_pullbacks_preserve_coordinate_spaces(self):
        for invalid in ("C.pullback(a)", 'C.coordinates[0] + R.chart("B3.v").coordinates[0]'):
            with self.subTest(invalid=invalid):
                result = run_script(f"""
                    x, y = coordinates("x", "y")
                    a, b = coordinates("x", "y", fresh=True)
                    R = resolve(y^2-x^3)
                    C = R.chart("B3.u")
                    {invalid}
                """)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("curve.atlas:5:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_string_input_provides_source_coordinates_for_pullback(self):
        result = run_script("""
            R = resolve("v^2-u^3", variables=["u","v"])
            a, b = R.coordinates
            C = R.chart("B1.u")
            u, v = C.coordinates
            print(a == u, R.polynomial == b^2-a^3, C.pullback(R.polynomial) == u^2*(v^2-u))
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["False True True"])

    def test_archived_source_equation_retains_the_whole_ring_and_large_exponents(self):
        from engine.session_values import Resolution

        degree = 2**80
        coefficient = 2**140 + 17
        record = {"variables": ["x", "y"], "polynomial": f"{coefficient}*x^{degree}/3",
                  "charts": [], "dual_complex": {"vertices": [], "cells": [], "status": "complete"},
                  "mode": "embedded"}
        restored = Resolution(record)
        x, y = restored.coordinates
        self.assertEqual(restored.polynomial + y, coefficient * x**degree / 3 + y)
        foreign_x, _ = coordinates("x", "y")
        self.assertNotEqual(restored.polynomial, coefficient * foreign_x**degree / 3)
        self.assertIs(restored.polynomial, restored.polynomial)
        restored.record["variables"].clear()
        self.assertEqual(restored.record["variables"], ["x", "y"])

    def test_algebraic_center_pullback_reduces_coefficients_in_their_number_field(self):
        result = run_script("""
            x, y = coordinates("x", "y")
            f = (y^2-2*x^2)^2-x^5
            R = resolve(f)
            C = R.chart("B5.v")
            u, v = C.coordinates
            total = C.pullback(f)
            print(total == u^4*v^10*C.strict_transform)
            print(total.multiplicity(R.divisor("E5").equation_in(C)))
            print(total.multiplicity(C.strict_transform))
            recorded = polynomial("8+a6*v+v^2-u", variables=C.coordinates)
            print(recorded == C.strict_transform)
            theta = polynomial("theta", ring=recorded.parent)
            print(4*theta == polynomial("a6", variables=C.coordinates), theta^2 == 2)
            print(polynomial("1/a6", variables=C.coordinates) * (4*theta) == 1,
                  polynomial("1/a6^2", variables=C.coordinates) == 1/32)
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["True", "10", "1", "True", "True True", "True True"])

    def test_printed_chart_polynomial_roundtrips_over_a_composite_field(self):
        from engine.session_algebra import polynomial
        from engine.session_values import Resolution

        # The recorded theta is sqrt(2). A generator of the composite field
        # also has to express sqrt(3), so its printed name must be distinct.
        R = Resolution({
            "variables": ["x"], "dual_complex": {"vertices": []},
            "coefficient_field": {"generator": "theta", "exact_root": "sqrt(2)"},
            "algebraic_numbers": {"b": {"exact": "sqrt(3)"}},
            "charts": [{"id": "C", "coordinates": ["u"], "map": {"x": "u"},
                        "strict_transform": "theta*u+b"}],
        })
        C = R.chart("C")
        self.assertEqual(polynomial("theta^2+b^2", variables=C.coordinates), 5)
        self.assertEqual(polynomial(str(C.strict_transform), variables=C.coordinates), C.strict_transform)

    def test_quoted_and_typed_coefficient_division_at_both_entry_points(self):
        source = dedent("""
            x,y,z = coordinates("x","y","z")
            C = resolve(x^4*y^2-2*z^6,mode="intrinsic",compute="cpu").chart("B4")
            u,v,w = C.coordinates
            theta = polynomial("theta", ring=C.parent)
            print(polynomial("(u1+u2)/(theta+1)", ring=C.parent) == (u+v)/(theta+1))
            print(polynomial("1/(theta+1)^3", ring=C.parent) == 1/(theta+1)^3)
            print(polynomial("1/(2/(theta+1)-theta)", ring=C.parent) == 1/(2/(theta+1)-theta))
            print(polynomial("u1^1180591620717411303424/(theta+1)", ring=C.parent)*(theta+1)==u^1180591620717411303424)
        """).lstrip()
        expected = ["True"] * 4
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

        from engine.session_algebra import polynomial
        from engine.session_values import Resolution

        field_chart = Resolution({
            "variables": ["x"], "dual_complex": {"vertices": []},
            "coefficient_field": {"generator": "theta", "exact_root": "sqrt(2)"},
            "charts": [{"id": "C", "coordinates": ["u"], "map": {"x": "u"}, "strict_transform": "u+theta"}],
        }).chart("C")
        for denominator, error in (("theta^2-2", "nonzero"), ("u+theta", "coefficient field"),
                                   ("unlisted", "Unknown chart variable")):
            with self.subTest(denominator=denominator), self.assertRaisesRegex(ValueError, error):
                polynomial("1/(" + denominator + ")", ring=field_chart.parent)

    def test_indexed_roots_define_distinct_polynomial_ideals_in_a_binomial_chart(self):
        result = run_script("""
            x, y, z = coordinates("x", "y", "z")
            R = resolve(x^4*y^2-2*z^6, mode="intrinsic")
            C = R.chart("B4")
            u, v, w = C.coordinates
            e1, g1 = R.divisor("E7.1").ideal_in(C)
            e2, g2 = R.divisor("E7.2").ideal_in(C)
            print(e1 == v, e2 == v, g1 != g2)
            print(g1*g2 == C.strict_transform, g1*g2 == 1-2*u^2)
            S = resolve(x^4*y^2-z^6, mode="intrinsic")
            D = S.chart("B4")
            p, q, t = D.coordinates
            print(S.divisor("E7.1").ideal_in(D)[1] == 1+p)
            print(S.divisor("E7.2").ideal_in(D)[1] == 1-p)
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["True True True", "True True", "True", "True"])

    def test_general_blowup_does_not_compute_multiplicity_in_the_wrong_ring(self):
        from engine.general_resolution import search
        from engine.polynomial import parse

        # The graph chart has y = r1_2*x. Divisibility in its free
        # polynomial representatives can give a different answer.
        def general_search(expression, names, *, mode, **options):
            record = search(parse(expression, names), names, mode, seconds=5)
            return 200, dict(record, variables=names, polynomial=expression, mode=mode)

        session = new_session(general_search)
        list(session.run(dedent("""
            x, y, z = coordinates("x", "y", "z")
            R = resolve(x^2-y^2*z)
            G = R.chart("G1")
            a, b = R.divisor("E1").ideal_in(G)
            X, Y, Z, t = G.coordinates
            X0, Y0, Z0, t0 = G.ambient.coordinates
        """)))
        self.assertEqual(list(session.run("G.ambient_ideal[0] == t0*X0-Y0")),
                         [(True, True)])
        self.assertEqual(list(session.run("t*X == Y")),
                         [(True, True)])
        self.assertEqual(list(session.run("G.strict_transform[0].variables == a.variables")),
                         [(True, True)])
        with self.assertRaisesRegex(ValueError, "quotient ring"):
            list(session.run("G.pullback(y).multiplicity(a)"))

    def test_intrinsic_point_and_identity_chart(self):
        result = run_script("""
            x, y = coordinates("x", "y")
            R = resolve(y^2-x^3, mode="intrinsic")
            B = R.branches[0]
            D = B.meets[0]
            a, b = D.ideal_in(B.chart)
            u, v = B.chart.coordinates
            print(D.id, a == u, b == v)
            smooth = resolve(y-x)
            C = smooth.charts[0]
            print(C.id, len(smooth.divisors), C.pullback(y-x) == C.strict_transform)
            u, v = C.coordinates
            zero = C.pullback(0)
            half = C.pullback(1/2)
            print(zero.variables == u.variables, zero == 0, half.variables == u.variables, half == 1/2)
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["P1 True True", "U0 0 True", "True True True True"])

    def test_prompt_keeps_variable_after_failure_and_accepts_legacy_commands(self):
        status, out, err = run_prompt("""
            x, y = coordinates("x", "y")
            R = resolve(y^2-x^3)
            R = resolve(x+1)
            R.divisor("E3").equation_in(R.chart("B3.v"))
            resolve 'y^2 - x^3' --vars x,y
            quit
        """)
        self.assertEqual(status, 0)
        self.assertIn("origin", err)
        self.assertIn("atlas> v\n", out)
        self.assertIn("RESOLVED", out)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
