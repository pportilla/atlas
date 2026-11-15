"""Check parent identities across public operations at the prompt and in files."""

import unittest
from textwrap import dedent

from engine.session import new_session
from engine.session_values import Resolution
from tests.cli_support import run_prompt, run_script


class RingOwnershipTests(unittest.TestCase):
    def check_at_both_entry_points(self, source, expected):
        source = dedent(source).lstrip()
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_extension_of_coordinates_and_projection_preserve_their_endpoints(self):
        self.check_at_both_entry_points("""
            x, y = coordinates("x", "y")
            f = y^2+x^3
            X, Y, z = coordinates("x","y","z")
            include = hom(f.parent, z.parent, images=(X, Y))
            g = include(f) + z
            print(g == Y^2+X^3+z, f.parent == x.parent, g.parent == z.parent)
            A = quotient_ring(ideal(f))
            p = A.projection
            print(p.domain == f.parent, p.codomain == A, p(f) == 0, p(x) == A(x))
        """, ["True True True", "True True True True"])

    def test_resolution_preserves_the_specified_parent_for_text_and_polynomials(self):
        self.check_at_both_entry_points("""
            R = resolve("y^2-x^3", variables=["x","y"])
            x, y = coordinates("x","y")
            X, Y = coordinates("x","y", fresh=True)
            S = resolve("y^2-x^3", ring=X.parent)
            T = resolve(Y^2-X^3)
            M = monodromy_check("y^2-x^3", variables=(X,Y))
            print(R.polynomial.parent == x.parent, S.polynomial.parent == X.parent)
            print(S.polynomial == T.polynomial, M.resolution.polynomial == S.polynomial)
            C = S.chart("B3.u")
            phi = C.pullback_map
            print(phi.domain == S.polynomial.parent, phi.codomain == C.parent)
            print(phi.images == (C.map.x, C.map.y), phi(S.polynomial) == C.pullback(T.polynomial))
            print(C.pullback(S.polynomial).multiplicity(S.divisor("E3").equation_in(C)))
        """, ["True True", "True True", "True True", "True True", "6"])

    def test_normalization_map_uses_the_component_and_normalized_quotients(self):
        self.check_at_both_entry_points("""
            x, y = coordinates("x","y")
            C = normalization(ideal(y^2-x^3)).components[0]
            phi = C.map
            t, = C.coordinates
            print(phi.domain.ambient == x.parent, phi.codomain.ambient == t.parent)
            print(quotient_ring(C.prime, compute="cpu") == phi.domain, C.relations.quotient_ring == phi.codomain)
            print(phi(phi.domain(x)) == phi.codomain(t^2), phi(phi.domain(y)) == phi.codomain(t^3))
            print(phi.domain(y^2-x^3) == 0, C.map == phi)
        """, ["True True", "True True", "True True", "True True"])

    def test_partial_archived_chart_only_transports_recorded_coordinates(self):
        record = {"variables": ["x", "y"], "dual_complex": {"vertices": []},
                  "charts": [{"id": "G", "coordinates": ["u"], "map": {"x": "u"}},
                             {"id": "missing", "coordinates": ["u"]}]}
        resolution = Resolution(record)
        x, y = resolution.coordinates
        chart = resolution.chart("G")
        u, = chart.coordinates
        self.assertEqual(chart.pullback(3*x+1), 3*u+1)
        self.assertIs(chart.pullback(1).parent, chart.parent)
        with self.assertRaisesRegex(ValueError, "no recorded image for y"):
            chart.pullback(y)
        for chart in resolution.charts:
            with self.assertRaisesRegex(ValueError, "every original coordinate"):
                _ = chart.pullback_map

    def test_same_spelling_does_not_authorize_mixed_operations(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            u, v = coordinates("x", "y", fresh=True)
            I = ideal(x^2, y)
            J = ideal(u^2, v)
            A = quotient_ring(I)
            B = quotient_ring(J)
        """)))
        cases = (
            "x + u", "x * u", "x / (u^0)", "x.multiplicity(u)",
            "ideal(x, u)", 'polynomial("x", variables=(x,v))',
            "intersection(I,J)", "quotient(I,J)", "localize(I,J)",
            "normal_form(u,I)", "local_normal_form(v,I)", "membership(u,I)",
            "eliminate(I,u)", "resultant(x,y,u)", "gcd(x,u)",
            "A(u)", "A(B(v))", "A.projection(u)", "A(x) + B(u)",
            "hom(x.parent, u.parent, images=(x,v))",
        )
        for command in cases:
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "space|ring"):
                list(session.run(command))
