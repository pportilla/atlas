import unittest

from engine.kernel_json import kernel_json
from engine.toric_graph import _request


def candidates(variables, polynomial, faces):
    return kernel_json("atlas_toric_graph", "candidates", variables=variables,
                       polynomial=polynomial, faces=faces)


class RepeatedFaceGraphCandidates(unittest.TestCase):
    def test_repeated_edge_for_the_quintic_cusp(self):
        names = ["x", "y", "z", "t"]
        face = "t^2*x^4-2*t^7*x^2+t^12"
        remainder = "x^5+y^5+z^5+t^2*x*y*z+t^13"
        equation = remainder + "+" + face
        graph, = candidates(names, equation, [face, face])
        expected = dict(auxiliary="w", equation="w-x^2+t^5+y^5+2*z^5",
                        divisor=remainder + "+t^2*(w+y^5+2*z^5)^2"
                        "+w^2*(w-x^2+t^5+y^5+2*z^5)")
        # Native support also independently substitutes the graph into its
        # divisor and checks the original polynomial exactly.
        self.assertEqual(_request("support", names, equation, graph),
                         _request("support", names, equation, expected))

    def test_wide_rational_binomial_and_auxiliary_name_collision(self):
        names = ["x", "t", "w"]
        a = 2**78 + 1
        c = 2**151 + 3
        face = f"7*(x^{a}-({c}/3)*t^{a + 1})^2"
        remainder = f"x^{2*a + 1}+t^{2*a + 3}+w^3"
        equation = remainder + "+" + face
        graph, = candidates(names, equation, [face])
        self.assertEqual(graph["auxiliary"], "w1")
        support = _request("support", names, equation, graph)
        # The convenient auxiliary intercepts are one on the graph and
        # three on its divisor, so the support of their product contains w1^4.
        self.assertIn(["0", "0", "0", "4"], support)

    def test_non_powers_and_faces_with_different_coefficients_are_rejected(self):
        names = ["x", "y"]
        remainder = "x^5+y^5"
        for face in ["x^2+x*y+y^2", "x^2+x*y+y^3", "x^2+x*y+x^2*y"]:
            with self.subTest(face=face):
                self.assertEqual(candidates(names, remainder + "+" + face, [face]), [])
        self.assertEqual(candidates(names, remainder + "+(x+y)^2", ["(x-y)^2"]), [])

    def test_higher_power_with_monomial_multiplier(self):
        names = ["x", "t"]
        face = "-3*t*(2*x+t^2)^3"
        equation = "x^4+t^9+" + face
        graph, = candidates(names, equation, [face])
        self.assertTrue(_request("support", names, equation, graph))
