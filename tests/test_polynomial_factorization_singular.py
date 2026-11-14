"""Compare units, monic irreducible factors and multiplicities with Singular."""
from fractions import Fraction
import shutil
import subprocess
import unittest

from engine.algebra import dispatch
from engine.polynomial import parse_expression


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class FactorizationSingularTests(unittest.TestCase):
    def test_rational_factorization_in_both_orders(self):
        names = ["x", "y", "z"]
        f = "-3/5*(2*x*z+3*y^2)^3*(x^2+1)*(y-1)^2"
        for order, singular_order in (("lex", "lp"), ("grevlex", "dp")):
            with self.subTest(order=order):
                script = f"""
                    ring R=0,(x,y,z),{singular_order};
                    short=0;
                    list L=factorize({f});
                    number c=1;
                    poly g;
                    for (int j=1; j<=size(L[1]); j++) {{
                        g=L[1][j];
                        c=c*leadcoef(g)^L[2][j];
                        if (deg(g)>0) {{ print(g/leadcoef(g)); print(L[2][j]); }}
                    }}
                    print(c);
                    quit;
                """
                oracle = subprocess.run(["Singular", "-q"], input=script, text=True,
                                        capture_output=True, check=True, timeout=30)
                *rows, unit = oracle.stdout.splitlines()
                self.assertEqual(len(rows), 6, oracle.stdout)
                expected = [(parse_expression(g, names, _reference=True), int(e))
                            for g, e in zip(rows[::2], rows[1::2])]
                result = dispatch({"operation": "factor", "ring": {"variables": names, "order": order},
                                   "operands": {"polynomial": f}})["result"]
                self.assertEqual(Fraction(result["unit"]), Fraction(unit))
                self.assertCountEqual(
                    [(parse_expression(g["polynomial"], names, _reference=True), g["multiplicity"])
                     for g in result["factors"]], expected)
