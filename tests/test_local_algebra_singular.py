"""Compare rational quotient algebras and hypersurface invariants with Singular."""
import shutil
import subprocess
import unittest

from engine.algebra import dispatch


def singular(order, commands, variables=("x", "y")):
    script = [
        'LIB "sing.lib";',
        f"ring R=0,({','.join(variables)}),{order};",
        "option(redSB);",
        *commands,
        "quit;",
    ]
    run = subprocess.run(["Singular", "-q", "--no-rc"], input="\n".join(script),
                         text=True, capture_output=True, check=True, timeout=30)
    answers = []
    for line in run.stdout.splitlines():
        if line.startswith("ANSWER "):
            answers.append(int(line.removeprefix("ANSWER ")))
        elif line.strip():
            raise AssertionError(f"Unexpected Singular output: {run.stdout}\n{run.stderr}")
    if run.stderr:
        raise AssertionError(run.stderr)
    return answers


def calculate(operation, *, variables=("x", "y"), order="grevlex", **operands):
    ring = {"field": "QQ", "variables": list(variables), "order": order}
    response = dispatch({"operation": operation, "ring": ring, "operands": operands, "compute": "cpu"})
    return response["result"]


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class LocalAlgebraSingularTests(unittest.TestCase):
    def check_basis(self, result, scope, generators):
        self.assertEqual(result["scope"], scope)
        order = "dp" if scope == "global" else "ds"
        commands = [f"ideal G=std({generators});", 'print("ANSWER "+string(vdim(G)));']
        if result["basis"]:
            # Columns are the coordinates of the Atlas classes in Singular's
            # standard monomials. A nonzero determinant proves they are a basis.
            commands += [
                "ideal K=kbase(G);",
                f"ideal A={','.join(result['basis'])};",
                "matrix C=coeffs(reduce(A,G,2),K);",
                'print("ANSWER "+string(nrows(C)));',
                'print("ANSWER "+string(ncols(C)));',
                'print("ANSWER "+string(det(C)!=0));',
            ]
        dimension, *certificate = singular(order, commands, result["ring"]["variables"])
        self.assertIs(result["finite"], dimension >= 0)
        self.assertEqual(result["dimension"], dimension if dimension >= 0 else None)
        if dimension < 0:
            self.assertIsNone(result["basis"])
        elif dimension == 0:
            self.assertEqual(result["basis"], [])
        else:
            self.assertEqual(len(result["basis"]), dimension)
            self.assertEqual(certificate, [dimension, dimension, 1])

    def test_global_and_origin_quotients(self):
        for ideal in (
            [],
            ["0", "0"],
            ["1"],
            ["x-1", "y"],
            ["x-y-y^2", "y^3"],
            ["0", "3/7*x^2", "y-x", "3/7*x^2"],
            ["x^2*(x-1)", "y*(x-1)"],  # The line x=1 misses the origin.
        ):
            for operation, scope in (("quotient-basis", "global"), ("local-quotient-basis", "origin")):
                with self.subTest(operation=operation, ideal=ideal):
                    result = calculate(operation, ideal=ideal)
                    self.assertEqual(result["type"], "quotient-basis")
                    self.check_basis(result, scope, f"ideal({','.join(ideal) or '0'})")

    def test_local_order_ties_with_a_component_away_from_the_origin(self):
        # At the origin the factor 1-x is a unit and xz=y². The plane x=1
        # makes the global quotient infinite-dimensional.
        variables = ("x", "y", "z")
        ideal = [f"(1-x)*({f})" for f in ("x^2+y^3", "y^3", "z^2", "x*z-y^2+z^3")]
        generators = f"ideal({','.join(ideal)})"
        for order, kept, excluded in (("lex", "y^2", "x*z"), ("grevlex", "x*z", "y^2")):
            with self.subTest(order=order):
                local = calculate("local-quotient-basis", variables=variables, order=order, ideal=ideal)
                self.assertEqual(local["dimension"], 7)
                self.assertIn(kept, local["basis"])
                self.assertNotIn(excluded, local["basis"])
                self.check_basis(local, "origin", generators)
                global_result = calculate("quotient-basis", variables=variables, order=order, ideal=ideal)
                self.assertFalse(global_result["finite"])
                self.check_basis(global_result, "global", generators)

    def test_jacobian_algebra_and_hypersurface_numbers(self):
        for polynomial in (
            "x+y^2",                 # Smooth at the origin.
            "1/3*x^3+3/5*y^2",      # Cusp with rational coefficients.
            "y^2+x^3+x^4",          # Another critical point at x=-3/4.
            "x^2*y",                # Nonisolated critical locus.
            "x^5+y^5+x^3*y^3",      # Milnor number 16, Tjurina number 15.
        ):
            with self.subTest(polynomial=polynomial):
                result = calculate("local-jacobian-algebra", polynomial=polynomial)
                self.assertEqual(result["type"], "local-jacobian-algebra")
                self.check_basis(result, "origin", f"jacob(poly({polynomial}))")
                fx, fy = result["generators"]
                correct_fx, correct_fy, mu, tau = singular("ds", [
                    f"poly f={polynomial};",
                    f'print("ANSWER "+string(({fx})==diff(f,x)));',
                    f'print("ANSWER "+string(({fy})==diff(f,y)));',
                    'print("ANSWER "+string(milnor(f)));',
                    'print("ANSWER "+string(tjurina(f)));',
                ])
                self.assertEqual((correct_fx, correct_fy), (1, 1))
                for operation, expected in (("milnor-number", mu), ("tjurina-number", tau)):
                    with self.subTest(operation=operation):
                        result = calculate(operation, polynomial=polynomial)
                        self.assertEqual(result["type"], "integer")
                        self.assertEqual(result["scope"], "origin")
                        self.assertIs(result["finite"], expected >= 0)
                        self.assertEqual(result["value"], expected if expected >= 0 else None)
