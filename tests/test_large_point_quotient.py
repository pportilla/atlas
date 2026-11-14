"""Nonisolated point germs and GMP-sized isolated quotient lengths."""
import unittest

from engine.algebra import dispatch


def calculate(op, variables, point, *, order="grevlex", **operands):
    return dispatch({"operation": op, "ring": {"variables": variables, "order": order}, "compute": "cpu",
                     "operands": dict(operands, point=point)})["result"]


class LargePointQuotientTests(unittest.TestCase):
    def test_symbolic_pivot_detects_dependent_coordinate_powers(self):
        n = 2**64 + 7
        # At (2,4), 2*4^((N-1)/2) = 2^N. The two quadratic rows then
        # coincide modulo x-2,w-4. Changing the coefficient to three
        # makes them independent and kills both squares.
        for coefficient, expected in ((2, ["1", "y", "z", "y*z", "z^2", "y*z^2"]),
                                      (3, ["1", "y", "z", "y*z"])):
            with self.subTest(coefficient=coefficient):
                rows = ["x-2", "w-4", f"x^{n}*y^2+z^2", f"{coefficient}*w^{(n-1)//2}*y^2+z^2", "z^3"]
                result = calculate("local-quotient-basis", ["x", "w", "y", "z"], [2, 4, 0, 0], ideal=rows)
                self.assertEqual(result["basis"], expected)

    def test_shifted_coefficients_beyond_memory_with_coincident_leading_terms(self):
        n = 2**64 + 7
        names, point = ["x", "y", "z"], [2, 0, 0]
        for order in ("lex", "grevlex"):
            with self.subTest(order=order):
                # Addition and subtraction recover y² and z² at this point.
                rows = ["x-2", f"x^{n}*y^2+z^2", f"x^{n}*y^2-z^2"]
                result = calculate("local-quotient-basis", names, point, order=order, ideal=rows)
                self.assertEqual(result["basis"], ["1", "y", "z", "y*z"])

                # The gradient starts with 2h, 3*2^N*y²+z², 2yz (h=x-2).
                # Its length-four graded quotient has standard monomials
                # 1,y,z,z². All cubics vanish and f adds no relation.
                f = f"(x-2)^2+x^{n}*y^3+y*z^2+z^4"
                jacobian = calculate("local-jacobian-algebra", names, point, order=order, polynomial=f)
                self.assertEqual(jacobian["basis"], ["1", "y", "z", "z^2"])
                quotient = calculate("local-quotient-basis", names, point, order=order, ideal=jacobian["generators"])
                self.assertEqual(quotient["basis"], jacobian["basis"])
                for op in ("milnor-number", "tjurina-number"):
                    self.assertEqual(calculate(op, names, point, order=order, polynomial=f)["value"], 4)

    def test_torus_coefficient_beyond_memory_has_a_small_staircase(self):
        n = 2**64 + 7
        # In the x=2 slice, y² is a nonzero scalar multiple of z².
        # The leading ideal is (X,y²,z³) independently of that scalar.
        result = calculate("local-quotient-basis", ["x","y","z"], [2,0,0],
                           ideal=["x-2", f"x^{n}*y^2+z^2", "z^3"])
        self.assertEqual(result["dimension"], 6)
        self.assertEqual(result["basis"], ["1","y","z","y*z","z^2","y*z^2"])

    def test_milnor_and_tjurina_need_only_the_support_of_the_initial_forms(self):
        n = 2**64 + 7
        # Completing the square gives X²+(z+y²)²+y³(x^N-y). Since
        # x^N-y is a unit, this is formally an A2 hypersurface over C.
        # The local Jacobian heads are X,y²,z. In particular m²⊂J
        # and the whole hypersurface equation is already in J.
        f = f"(x-2)^2+z^2+2*y^2*z+x^{n}*y^3"
        for op in ("milnor-number", "tjurina-number"):
            result = calculate(op, ["x","y","z"], [2,0,0], polynomial=f)
            self.assertEqual((result["finite"],result["value"]), (True,2))

    def test_wide_coefficient_respects_the_local_tie_order(self):
        n = 2**64 + 7
        # Modulo the cube of the maximal ideal every degree-two
        # monomial survives except the head of 2^N*a*c+b².
        axes = ["a","b","c"]
        cubics = [f"{axes[i]}*{axes[j]}*{axes[k]}" for i in range(3)
                  for j in range(i,3) for k in range(j,3)]
        ordered = ["1","a","b","c","a^2","a*b","a*c","b^2","b*c","c^2"]
        for order, head in (("lex","a*c"), ("grevlex","b^2")):
            result = calculate("local-quotient-basis", ["x",*axes], [2,0,0,0], order=order,
                               ideal=["x-2",f"x^{n}*a*c+b^2",*cubics])
            self.assertEqual(result["dimension"], 9)
            self.assertEqual(result["basis"], [m for m in ordered if m != head])

    def test_repeated_coupled_hypersurface_remains_nonisolated(self):
        n = 2**64 + 7
        g = f"(x^{n}*y^{n}+x+y-3)"
        f = f"-7/3*{g}^2"
        for op in ("milnor-number", "tjurina-number", "local-jacobian-algebra"):
            result = calculate(op, ["x", "y"], [1, 1], polynomial=f)
            self.assertFalse(result["finite"])
            if op == "local-jacobian-algebra":
                self.assertIsNone(result["basis"])
                self.assertEqual(len(result["generators"]), 2)
                explicit = calculate("local-quotient-basis", ["x","y"], [1,1],
                                     ideal=result["generators"])
                self.assertEqual((explicit["finite"],explicit["dimension"],explicit["basis"]),
                                 (False,None,None))
                off_factor = calculate("local-quotient-basis", ["x","y"], [1,-1],
                                       ideal=result["generators"])
                self.assertEqual((off_factor["finite"],off_factor["dimension"],off_factor["basis"]),
                                 (True,0,[]))
            else:
                self.assertIsNone(result["value"])
        # The Jacobian of g and x-y at this point is invertible.
        # Adding the cube of the second parameter gives an isolated A2
        # singularity, a check against accepting a near-square.
        near = f"{f}+(x-y)^3"
        for op in ("milnor-number", "tjurina-number"):
            self.assertEqual(calculate(op, ["x", "y"], [1, 1], polynomial=near)["value"], 2)

    def test_product_staircase_counts_more_than_128_bits(self):
        m, n = 2**67+3, 2**73+5
        # At (1,0,0), x^N+x-2 is a parameter, so its square contributes
        # length one. The local quotient then has y^m=z^n=0.
        degree = 2**64 + 7
        f = f"(x^{degree}+x-2)^2+y^{m+1}+z^{n+1}"
        for op in ("milnor-number", "tjurina-number"):
            result = calculate(op, ["x", "y", "z"], [1, 0, 0], polynomial=f)
            self.assertTrue(result["finite"])
            self.assertEqual(result["value"], m*n)
            self.assertNotIn("basis", result)

    def test_sparse_support_cancellation_changes_the_staircase(self):
        n = 2**64 + 7
        # Restricting x to 1 makes yz zero. Then y²+z³ gives z⁴=0.
        # This requires a new leading monomial and leaves precisely five.
        result = calculate("local-quotient-basis", ["x","y","z"], [1,0,0],
                           ideal=["x-1", "y^2+z^3", f"y*z+(x^{n}-1)*z^2"])
        self.assertEqual(result["dimension"], 5)
        self.assertEqual(result["basis"], ["1", "y", "z", "z^2", "z^3"])
