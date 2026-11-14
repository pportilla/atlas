"""Check local quotient and hypersurface invariants at rational points in Singular."""
from fractions import Fraction
import shutil
import unittest

from engine.polynomial import InputError
from tests.test_local_algebra_singular import calculate, singular
from tests.test_local_standard_singular import answer, translated_ring


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class PointInvariantsSingularTests(unittest.TestCase):
    def at_point(self, operation, variables, order, point, **operands):
        result = calculate(operation, variables=variables, order=order, point=point, **operands)
        self.assertEqual(result["scope"], "point")
        self.assertEqual(result["point"], [str(Fraction(q)) for q in point])
        self.assertEqual(result["ring"], {"field": "QQ", "variables": list(variables), "order": order})
        return result

    def check_basis(self, result, original, expected_dimension):
        variables, order = result["ring"]["variables"], result["ring"]["order"]
        commands = [f"ideal original={original};", f"ideal proposed={','.join(result['basis'] or []) or '0'};",
                    *translated_ring(order, result["point"]), answer("vdim(G)")]
        if expected_dimension > 0:
            # In a local algebra of length d the maximal ideal satisfies m^d=0.
            # Reduce in the global jet quotient to obtain exact QQ coordinates
            # for each translated candidate, without local unit multipliers.
            centered = ("u", "v", "w")[:len(variables)]
            commands.extend([
                "ideal candidates=shift(proposed);",
                f"ring JetRing=0,({','.join(centered)}),dp;",
                f"ideal Q=std(imap(T,J)+maxideal({expected_dimension}));",
                "ideal A=imap(T,candidates);",
                "matrix C=coeffs(reduce(A,Q),kbase(Q));",
                answer("vdim(Q)"), answer("nrows(C)"), answer("ncols(C)"), answer("det(C)!=0"),
            ])
        dimension, *rank = singular("dp", commands, variables)
        self.assertEqual(dimension, expected_dimension)
        self.assertEqual(result["dimension"], dimension if dimension >= 0 else None)
        self.assertIs(result["finite"], dimension >= 0)
        if dimension < 0:
            self.assertIsNone(result["basis"])
        elif dimension == 0:
            self.assertEqual(result["basis"], [])
        else:
            self.assertEqual(len(result["basis"]), dimension)
            self.assertEqual(rank, [dimension, dimension, dimension, 1])

    def test_quotient_on_and_off_components(self):
        # The point (1/2,-2/3) has length two and the component x=2 is a line.
        ideal = ["(x-2)*(x-1/2)^2", "(x-2)*(y+2/3)"]
        for point, dimension in ((["2/4", "-4/6"], 2), ([2, 0], -1), ([0, 0], 0)):
            with self.subTest(point=point):
                result = self.at_point("local-quotient-basis", ("x", "y"), "grevlex", point, ideal=ideal)
                self.assertEqual(result["type"], "quotient-basis")
                self.check_basis(result, ",".join(ideal), dimension)
        for ideal, expected in (([], -1), (["0"], -1), (["2/7"], 0)):
            with self.subTest(ideal=ideal):
                result = self.at_point("local-quotient-basis", ("x", "y"), "lex", [2, 3], ideal=ideal)
                self.check_basis(result, ",".join(ideal) or "0", expected)

    def test_three_variable_order_tie_at_a_translated_point(self):
        variables, point = ("x", "y", "z"), ["1/2", "-2/3", 2]
        u, v, w = "(x-1/2)", "(y+2/3)", "(z-2)"
        ideal = [f"(1-{u})*({g})" for g in (f"{u}^2+{v}^3", f"{v}^3", f"{w}^2",
                                              f"{u}*{w}-{v}^2+{w}^3")]
        for order, kept, excluded in (("lex", "v^2", "u*w"), ("grevlex", "u*w", "v^2")):
            with self.subTest(order=order):
                result = self.at_point("local-quotient-basis", variables, order, point, ideal=ideal)
                self.check_basis(result, ",".join(ideal), 7)
                # Compare polynomials exactly after moving the returned basis
                # into centered coordinates. A full-rank basis alone would
                # also pass if the implementation ignored the order.
                answers = singular("dp", [
                    f"ideal original={','.join(ideal)};", f"ideal proposed={','.join(result['basis'])};",
                    *translated_ring(order, result["point"]), "ideal B=shift(proposed);",
                    "int hasKept,hasExcluded; int i;",
                    f"for(i=1;i<=size(B);i++){{if(B[i]=={kept}){{hasKept=1;}}"
                    f"if(B[i]=={excluded}){{hasExcluded=1;}}}}",
                    answer("hasKept"), answer("hasExcluded"),
                ], variables)
                self.assertEqual(answers, [1, 0])

    def test_jacobian_and_hypersurface_numbers(self):
        variables = ("x", "y")
        curve = "y^2-x^3*(x-1)^2"
        u, v = "(x-1/2)", "(y+2/3)"
        for f, point, expected_mu, expected_tau in (
            (curve, [0, 0], 2, 2), (curve, [1, 0], 1, 1), (curve, [4, 24], 0, 0),
            (f"{u}^5+{v}^5+{u}^3*{v}^3", ["2/4", "-4/6"], 16, 15),
            (f"{u}^2*{v}", ["1/2", 1], -1, -1), ("0", [2, 3], -1, -1),
        ):
            for order in ("lex", "grevlex"):
                with self.subTest(f=f, point=point, order=order):
                    algebra = self.at_point("local-jacobian-algebra", variables, order, point, polynomial=f)
                    self.assertEqual(algebra["type"], "local-jacobian-algebra")
                    self.check_basis(algebra, f"jacob(poly({f}))", expected_mu)
                    fx, fy = algebra["generators"]
                    dx, dy, mu, tau = singular("dp", [
                        f"poly f={f};", "ideal original=jacob(f);",
                        answer(f"({fx})==diff(f,x)"), answer(f"({fy})==diff(f,y)"),
                        *translated_ring(order, algebra["point"]), "poly F=shift(f);",
                        answer("milnor(F)"), answer("tjurina(F)"),
                    ], variables)
                    self.assertEqual((dx, dy, mu, tau), (1, 1, expected_mu, expected_tau))
                    for operation, dimension in (("milnor-number", mu), ("tjurina-number", tau)):
                        number = self.at_point(operation, variables, order, point, polynomial=f)
                        self.assertEqual(number["type"], "integer")
                        self.assertIs(number["finite"], dimension >= 0)
                        self.assertEqual(number["value"], dimension if dimension >= 0 else None)

    def test_critical_point_off_the_hypersurface(self):
        # The origin lies on f=0; the specified critical point does not.
        f = "(x-1/2)^2+(y+2/3)^2-25/36"
        checks = singular("dp", [f"poly f={f};", "ideal original=jacob(f);",
                                 *translated_ring("grevlex", ["1/2", "-2/3"]), "poly F=shift(f);",
                                 answer("reduce(F,std(maxideal(1)))==-25/36"), answer("vdim(G)")])
        self.assertEqual(checks, [1, 1])
        for operation in ("local-jacobian-algebra", "milnor-number", "tjurina-number"):
            with self.subTest(operation=operation), self.assertRaises(InputError):
                calculate(operation, point=["1/2", "-2/3"], polynomial=f)

    def test_explicit_origin_and_legacy_result(self):
        for operation in ("local-quotient-basis", "local-jacobian-algebra", "milnor-number", "tjurina-number"):
            with self.subTest(operation=operation):
                args = {"ideal": ["x^2", "y"]} if operation == "local-quotient-basis" else {"polynomial": "y^2-x^3"}
                origin = calculate(operation, **args)
                explicit = self.at_point(operation, ("x", "y"), "grevlex", ["0/4", 0], **args)
                self.assertEqual(origin["scope"], "origin")
                self.assertNotIn("point", origin)
                self.assertEqual(explicit, {**origin, "scope": "point", "point": ["0", "0"]})
