"""Jung's method for surface singularities, checked independently of its construction.

Laufer's formula mu = 12 p_g + K^2 + chi(E) - 1 holds for hypersurface surface
singularities. The Milnor number comes from Atlas's local algebra, a separate
calculation; the graph shapes are compared with the known resolution graphs.
"""
import json
import subprocess
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

from engine.app import calculate_resolution
from engine.jung import JungDeclined, JungResolution, jung_record
from engine.session import new_session
from tests.cli_support import run_prompt, run_script

ROOT = Path(__file__).resolve().parents[1]
KEY = "z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)"


def build_available():
    return (ROOT / "native" / "build" / "jung").is_file()


def neighbours(J):
    adjacent = {c["id"]: [] for c in J.curves}
    for a, b, points in J.edges:
        adjacent[a] += [b] * points
        adjacent[b] += [a] * points
    return adjacent


def arms(J, centre):
    """Lengths of the chains hanging from a vertex of a tree."""
    adjacent, lengths = neighbours(J), []
    for start in adjacent[centre]:
        previous, current, length = centre, start, 1
        while len(adjacent[current]) == 2:
            previous, current = current, next(v for v in adjacent[current] if v != previous)
            length += 1
        lengths.append(length)
    return sorted(lengths)


@unittest.skipUnless(build_available(), "Build the native Jung resolver with make")
class JungMethodTests(unittest.TestCase):
    def setUp(self):
        self.session = new_session(calculate_resolution)
        self.run_lines('x, y, z = coordinates("x", "y", "z")')

    def run_lines(self, source):
        for _ in self.session.run(source):
            pass

    def jung(self, f):
        self.run_lines(f"J = jung_resolution({f})\nmu = milnor_number({f}).value")
        return self.session.variables["J"], self.session.variables["mu"]

    def assert_laufer(self, J, mu):
        self.assertEqual(Fraction(J.milnor_number_from_laufer), mu)
        self.assertEqual(12 * J.geometric_genus + J.K2 + J.euler_characteristic - 1, mu)
        invariants = J.record["minimal_good_resolution"]["invariants"]
        self.assertTrue(invariants["negative_definite"])
        self.assertTrue(invariants["gorenstein"])

    def test_rational_double_points_have_their_dynkin_graphs(self):
        cases = {"x^2+y^3+z^5": (8, [1, 2, 4]), "x^2+(y+z)^3+y^4": (6, [1, 2, 2]),
                 "(x+2*y)^2+(y+z)^3+z^4": (6, [1, 2, 2]), "(x+y)^2+z^3+x^3": (4, [1, 1, 1])}
        for f, (rank, lengths) in cases.items():
            with self.subTest(f=f):
                J, mu = self.jung(f)
                self.assert_laufer(J, mu)
                self.assertEqual((len(J.curves), J.geometric_genus, J.K2), (rank, 0, 0))
                self.assertEqual({(c["genus"], c["self_intersection"]) for c in J.curves}, {(0, -2)})
                centre = next(c for c, around in neighbours(J).items() if len(around) == 3)
                self.assertEqual(arms(J, centre), lengths)

    def test_a20_needs_a_sheared_projection_and_gives_a_chain(self):
        J, mu = self.jung("x*y-z^21")
        self.assert_laufer(J, mu)
        self.assertEqual(J.projection["degree_in_fibre"], 2)
        self.assertEqual(len(J.curves), 20)
        self.assertEqual(sorted(len(v) for v in neighbours(J).values()), [1, 1] + [2] * 18)

    def test_key_surface_matches_the_hand_resolution(self):
        J, mu = self.jung(KEY)
        self.assertEqual(mu, 39)
        self.assert_laufer(J, mu)
        self.assertEqual((J.geometric_genus, J.K2, J.euler_characteristic, len(J.curves)), (3, -8, 12, 11))
        adjacent = neighbours(J)
        weights = {c["id"]: c["self_intersection"] for c in J.curves}
        centre = next(c for c, w in weights.items() if w == -4)
        self.assertEqual(Counter(weights.values()), Counter({-2: 10, -4: 1}))
        forks = [v for v in adjacent[centre] if len(adjacent[v]) == 3]
        leaves = [v for v in adjacent[centre] if len(adjacent[v]) == 1]
        self.assertEqual((len(adjacent[centre]), len(forks), len(leaves)), (4, 3, 1))
        for fork in forks:
            self.assertEqual(sorted(len(adjacent[v]) for v in adjacent[fork] if v != centre), [1, 1])

    def test_cusp_needs_the_global_monodromy_test(self):
        J, mu = self.jung("x*y*z+x^4+y^4+z^4")
        self.assert_laufer(J, mu)
        self.assertGreater(J.record["global_monodromy_tests"], 0)
        self.assertEqual([c["self_intersection"] for c in J.curves], [-3, -3, -3])
        self.assertEqual(sorted(len(v) for v in neighbours(J).values()), [2, 2, 2])
        self.assertEqual(set(J.canonical_cycle.values()), {-1})

    def test_higher_order_surfaces_satisfy_laufers_formula(self):
        expected = {"x^3+y^3+z^4": 1, "x^2*y+y^4+z^3": 1, "(x*y-z^2)^2+x^5+y^5": 4}
        for f, genus in expected.items():
            with self.subTest(f=f):
                J, mu = self.jung(f)
                self.assert_laufer(J, mu)
                self.assertEqual(J.geometric_genus, genus)

    def test_non_isolated_surface_is_declined(self):
        with self.assertRaisesRegex(JungDeclined, "not isolated"):
            jung_record("(x+y+z)^2-z*(x-y)^2", ("x", "y", "z"))
        with self.assertRaisesRegex(JungDeclined, "nonreduced"):
            jung_record("(x^2+y^3+z^5)^2", ("x", "y", "z"))
        record = jung_record("x+y^2", ("x", "y", "z"))
        self.assertEqual((record["status"], record["minimal_good_resolution"]["vertices"]), ("smooth", []))

    def test_prompt_script_and_cli_return_the_same_resolution(self):
        source = 'x, y, z = coordinates("x", "y", "z")\nJ = jung_resolution(x^3+y^3+z^4)\nprint_json(curves=J.curves, edges=J.edges, pg=J.geometric_genus)\n'
        done = run_script(source)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        script = done.stdout
        status, prompt, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        record = json.loads(subprocess.check_output([str(ROOT / "atlas"), "jung", "x^3+y^3+z^4", "--json"],
                                                    text=True, cwd=ROOT, timeout=60))
        from_cli = JungResolution(record)
        expected = json.loads(script)
        self.assertIn(script.strip(), prompt)
        self.assertEqual(expected["curves"], from_cli.curves)
        self.assertEqual(expected["pg"], from_cli.geometric_genus)
        declined = subprocess.run([str(ROOT / "atlas"), "jung", "(x+y+z)^2-z*(x-y)^2"],
                                  capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(declined.returncode, 2)
        self.assertIn("not isolated", declined.stderr)

    def test_help_card_reads_the_saved_record_and_owner(self):
        status, output, error = run_prompt("""
            x, y, z = coordinates("x", "y", "z")
            J = jung_resolution(x^2+y^3+z^5)
            help J
            help J.geometric_genus
            help jung_resolution
            J.K2
            quit
        """)
        self.assertEqual((status, error), (0, ""))
        for text in ("J is a Jung resolution of x^2 + y^3 + z^5 in R1 (QQ[x, y, z]).",
                     "J.geometric_genus is p_g", "docs/jung-method.md#fields-of-a-jung-resolution",
                     "docs/jung-method.md#resolve-a-surface-with-jungs-method"):
            self.assertIn(text, output)

    def test_completion_lists_the_fields_without_computing(self):
        from engine.session_completion import candidates
        self.run_lines("J = jung_resolution(x^2+y^3+z^5)")
        self.assertEqual(candidates("J.geo", self.session), ["J.geometric_genus"])
        self.assertIn("J.curves", candidates("J.c", self.session))

    def test_saved_resolution_keeps_its_ring_named(self):
        from engine.session_rings import owner
        self.run_lines("J = jung_resolution(x^2+y^3+z^5)")
        J = self.session.variables["J"]
        self.assertIs(owner(J), self.session.variables["x"].parent)


if __name__ == "__main__":
    unittest.main()
