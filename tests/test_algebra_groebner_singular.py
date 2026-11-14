"""Compare the reusable QQ handlers with a live Singular standard basis."""
import shutil
import subprocess
import unittest

from engine import native_polynomial as native
from engine.algebra_groebner import ideal_membership, normal_form, reduced_basis
from engine.polynomial import parse_expression


def singular_reference(generators, queries, coordinates, order):
    names = [x.name for x in coordinates]
    singular_order = {"lex": "lp", "grevlex": "dp"}[order]
    lines = [
        "option(redSB);",
        f"ring R=0,({','.join(names)}),{singular_order};",
        "short=0;",  # Print explicit products and powers.
        f"ideal I={','.join(map(str, generators)) or '0'};",
        "ideal G=std(I);",
        "print(size(G));",
        "for (int j=1; j<=size(G); j++) { print(G[j]/leadcoef(G[j])); }",
        *(f"print(reduce({f},G));" for f in queries),
        "quit;",
    ]
    result = subprocess.run(["Singular", "-q"], input="\n".join(lines), text=True,
                            capture_output=True, check=True, timeout=30)
    size, *polynomials = result.stdout.splitlines()
    # Parse the reference answer independently of the native polynomial parser.
    values = [native.from_terms(coordinates, parse_expression(p, names, _reference=True))
              for p in polynomials]
    return values[:int(size)], values[int(size):]


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class SingularComparisonTests(unittest.TestCase):
    def test_sparse_normal_form_across_the_packed_degree_boundary(self):
        x, y = coordinates = native.symbols("x y")
        query = x + y**40000
        _, expected = singular_reference((x,), (query,), coordinates, "lex")
        basis = reduced_basis((x,), coordinates, order="lex")
        self.assertEqual(normal_form(query, basis), expected[0])
        # This larger literal is outside Singular 4.4.1's ordinary poly^int
        # syntax. Setting x=0 gives an independent exact expected remainder.
        power = 2**64
        self.assertEqual(normal_form(x + y**power, basis), y**power)

    def test_reduced_basis_remainders_and_membership(self):
        x, y = coordinates = native.symbols("x y")
        queries = (x**2 + y/2, y**2 - 1, y, x, native.coerce(0))
        for order in ("lex", "grevlex"):
            for generators in ((x**2 - 1, y - x), (x + y**2, y**3), (), (x, x + 1)):
                with self.subTest(generators=generators, order=order):
                    expected_basis, remainders = singular_reference(
                        generators, queries, coordinates, order)
                    basis = reduced_basis(generators, coordinates, order=order)
                    self.assertCountEqual(basis.exprs, expected_basis)
                    self.assertEqual(len(remainders), len(queries))
                    for polynomial, remainder in zip(queries, remainders):
                        self.assertEqual(normal_form(polynomial, basis), remainder)
                        self.assertIs(ideal_membership(polynomial, basis), remainder == 0)
