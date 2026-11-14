"""Native local dimension and regularity for nonpure and nonreduced point germs."""
from itertools import combinations
from random import Random
import unittest

from engine import local_dimension
from engine import native_polynomial as np
from engine.algebra import dispatch
from engine.local_dimension_input import cpu_activity_buffer, cpu_invariants, measured_cpu_invariants
from engine.polynomial import InputError, parse_expression


def solve(ideal, *, names=("x","y"), point=None, order="grevlex"):
    names = list(names)
    variables = tuple(map(np.Symbol, names))
    rows = [np.from_terms(variables, parse_expression(f, names)) for f in ideal]
    operands = {"ideal": rows}
    if point is not None:
        operands["point"] = point
    ring = {"field": "QQ", "variables": names, "order": order}
    return tuple(local_dimension.handle(op, ring, operands) for op in ("local-dimension", "local-smoothness"))


class LocalDimensionKernelTests(unittest.TestCase):
    def test_input_activity_is_optional_only_for_nonmodular_shapes(self):
        variables = tuple(map(np.Symbol, ("x", "y", "z")))
        x, y, z = variables
        ideal = (x*y, x*z)
        # A caller can reuse a previously populated native counters buffer.
        counts = cpu_activity_buffer(*([7]*8))
        result = cpu_invariants(variables, ideal, ("0","0","0"), False, True, counts)
        self.assertEqual(result, (2, 3, 0, False))
        self.assertEqual(list(counts), [0]*8)
        # Change points, order and operand order between calls. Only input
        # handles may be reused, the support and tangent tests run each time.
        for rows, point, expected in ((ideal, ("1","0","0"), (1,1,2,True)),
                                      (ideal[::-1], ("0","0","0"), (2,3,0,False))):
            result, counts = measured_cpu_invariants(variables, rows, point, True, True)
            self.assertEqual(result, expected)
            self.assertIsNone(counts)
        _, counts = measured_cpu_invariants(variables, (x+y, x*z), ("0","0","0"), False, True)
        self.assertIsInstance(counts, cpu_activity_buffer)

    def check_local(self, ideal, expected_dimension, embedding, **kwargs):
        dimension, smooth = solve(ideal, **kwargs)
        self.assertEqual(dimension["type"], "integer")
        self.assertEqual((dimension["value"], dimension["local_dimension"]), (expected_dimension, expected_dimension))
        self.assertEqual(smooth["type"], "boolean")
        self.assertEqual((smooth["local_dimension"], smooth["embedding_dimension"]), (expected_dimension, embedding))
        self.assertEqual(smooth["jacobian_rank"], len(smooth["ring"]["variables"])-embedding)
        self.assertIs(smooth["value"], expected_dimension == embedding)
        return dimension, smooth

    def test_germs_on_distinct_components_and_their_intersection(self):
        ideal = ["x*y", "x*z"]
        for point, dimension, embedding in ((None,2,3), (["2/3",0,0],1,1), ([0,0,"-1/4"],2,2)):
            for order in ("lex","grevlex"):
                with self.subTest(point=point, order=order):
                    self.check_local(ideal, dimension, embedding, names=("x","y","z"), point=point, order=order)

    def test_nilpotent_embedded_at_origin(self):
        self.check_local(["y^2", "x*y"], 1, 2)
        self.check_local(["y^2", "x*y"], 1, 1, point=["-1/3",0])
        self.check_local(["x^5", "y^2"], 0, 2)

    def test_empty_ring_and_outside_support(self):
        scalar, _ = self.check_local([], 0, 0, names=())
        self.assertEqual(scalar["scope"], "origin")
        self.assertNotIn("point", scalar)
        for ideal, names, point in ((["1"], (), []), (["x^2*y","x"], ("x","y"), [1,0])):
            with self.subTest(ideal=ideal), self.assertRaisesRegex(InputError, "point must lie on"):
                solve(ideal, names=names, point=point)

    def test_nonhomogeneous_cancellation_and_unit_factors_at_fractional_point(self):
        x, y, z = "(x-1/2)", "(y+2/3)", "(z-1/5)"
        f = f"({x}*{y}+{z}^2)"
        rows = [f"{f}*({x}+1)", f"{f}+{x}^2"]
        for order in ("lex", "grevlex"):
            record, _ = self.check_local(rows, 1, 3, names=("x","y","z"), order=order,
                                         point=["2/4","-2/3","1/5"])
            self.assertEqual(record["scope"], "point")
            self.assertEqual(record["point"], ["1/2","-2/3","1/5"])

    def test_wide_hypersurface_at_nonzero_point(self):
        n = 2**64+7
        self.check_local([f"(x^{n}-1+y)^2"], 1, 2, point=[1,0])
        self.check_local([f"x^{n}-1+y"], 1, 1, point=[1,0])
        self.check_local([f"x^{2*n}-y^{n}"], 1, 1, point=[2,4])
        self.check_local([f"x^{n}-1", "y^2"], 0, 1, point=[1,0])
        self.check_local([f"x^{n}-1+y", "y^2"], 0, 1, point=[1,0])
        self.check_local([f"(x^{n}-1+y)*y", "y^2"], 1, 2, point=[1,0])
        self.check_local([f"(x^{n}-1+y)*(1+y)", "y^2*(1+y)"], 0, 1, point=[1,0])
        self.check_local([f"(x^{n}-1+y)^2", "(y+x-1)^2"], 0, 2, point=[1,0])
        self.check_local([f"(x^{n}-1+y)*(1+y)", f"(x^{n}-1+y)*(1+y^2)"], 1, 1, point=[1,0])
        self.check_local([f"x^{n}-1+y", "y^2", "z^2"], 0, 2, names=("x","y","z"), point=[1,0,0])
        self.check_local([f"(x^{n}-1+y)*z", "y^2*(1+y)", "z^2*(1+x)"], 1, 3,
                         names=("x","y","z"), point=[1,0,0])

    def test_principal_jets_on_fixed_and_moving_coordinates(self):
        center = ["-3/7", 0]
        u = "(7*x+3)"
        self.check_local([f"{u}^4 + {u}*y + y^3"], 1, 2, point=center)
        self.check_local([f"{u}^4 + (x^2+1)*y + y^3"], 1, 1, point=center)
        self.check_local([f"{u}^2 + {u} + y^3"], 1, 1, point=center)
        with self.assertRaisesRegex(InputError, "point must lie on"):
            solve([f"2 + {u}^4 + {u}*y"], point=center)

    def test_large_power_coordinates_at_fractional_and_negative_points(self):
        # Unit factors vary by generator. The nilpotent z direction affects
        # the original tangent space even at a smooth point of the support.
        a, b = "(x^1001+1)", f"({2**1001}*y^1001-1)"
        rows = [f"x^7*{a}*{b}", f"x^5*y^9*{a}*z^2", f"y^3*{b}*z^2"]
        for order in ("lex", "grevlex"):
            for x, embedding in ((-1, 3), (1, 2)):
                with self.subTest(order=order, x=x):
                    self.check_local(rows, 1, embedding, names=("x","y","z"),
                                     point=[x,"1/2",0], order=order)

    def test_large_coupled_generators_with_hidden_equations(self):
        n = 2**64+7
        u, v = f"(x^{n}*y^{n}+x+y-3)", "(x-y)"
        a = f"{u}^2"
        b = f"{a}+{v}^3"
        self.check_local([a, b, f"{b}+z^4"], 0, 3, names=("x","y","z"), point=[1,1,0])
        self.check_local([f"(x^{n}+x-2)*y", f"(x^{n}+x-2)*z", "y*z"], 1, 3,
                         names=("x","y","z"), point=[1,0,0])

    def test_large_graph_cancellation_with_positive_local_dimension(self):
        # With x=1 the remaining germ is (y²+z³,yz), times a free w axis.
        # Its initial generators by themselves leave a spurious z direction.
        # An S-polynomial gives z⁴. The original x^1001 w^1001 term
        # would require 1,004,004 terms when translated at x=w=1.
        ideal = ["x-1", "y^2+z^3", "y*z+z^2*((x^1001-1)*w^1001+(x-1)*(w+1))"]
        operands = {"ideal": ideal, "point": [1, 0, 0, 1]}
        for order in ("lex", "grevlex"):
            for op in ("local-dimension", "local-smoothness"):
                with self.subTest(order=order, operation=op):
                    result = dispatch({"ring": {"variables": list("xyzw"), "order": order},
                                       "operands": operands, "operation": op, "compute": "cpu"})["result"]
                    self.assertEqual(result["local_dimension"], 1)
                    if op == "local-smoothness":
                        self.assertEqual((result["value"], result["embedding_dimension"],
                                          result["jacobian_rank"]), (False, 3, 1))

    def test_monomial_supports_against_coordinate_subspaces(self):
        rng = Random(21017)
        names = list("abcdef")
        for _ in range(28):
            rows = [rng.sample(range(6), rng.randrange(1,5)) for _ in range(rng.randrange(3,12))]
            ideal = ["*".join(names[i]+"^2" for i in row) for row in rows]
            # The largest coordinate plane not containing all axes in any
            # generator's support is a component of maximal dimension.
            dimension = max(len(S) for k in range(7) for S in combinations(range(6),k)
                            if not any(set(row).issubset(S) for row in rows))
            self.check_local(ideal, dimension, 6, names=names)
