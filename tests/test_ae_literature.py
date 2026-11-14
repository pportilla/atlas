"""Published extended right-left codimensions at the origin.

All coefficients are rational. Extending QQ to the authors' real or complex
field preserves the ranks in the finite jets of these polynomial germs.
"""
import unittest

from tests.test_local_algebra_singular import calculate


# Sharland, equations (6)–(7) and Table 1, in the original coordinates.
# https://arxiv.org/pdf/1202.4992#page=6
SHARLAND = [
    ("Ahat2", ["x", "y^2+x*z+x^2*y", "y*z", "z^2+y^3"], 18, 2, 10),
    ("Bhat3", ["x", "y^2+x*z", "z^2-x*y", "y^3+y^2*z+y*z^2-z^3"], 33, 3, 7),
    ("Ahat3", ["x", "y^3+x*z+x^4*y", "y*z", "z^2+y^5"], 186, 3, 32),
    # Equation (7) allows either sign of alpha for ell=2.
    *[(f"Bhat5 ({sign})", ["x", "y^2+x*z", f"z^2{sign}x*y", "y^5+y^4*z+y*z^4-z^5"], 252, 3, 17)
      for sign in ("+", "-")],
]


class PublishedAeCodimensions(unittest.TestCase):
    def check_germs(self, variables, cases):
        for name, components, expected in cases:
            with self.subTest(germ=name):
                result = calculate("ae-codimension", variables=variables, components=components)
                self.assertIs(result["finite"], expected is not None, result)
                self.assertEqual(result["value"], expected, result)

    def test_plane_branches(self):
        # Ishikawa and Janeczko, Theorems 8.7 and 9.2. The three W18 curves
        # have delta = 9 and symplectic defects 0, 2, 1. Ae-codim = delta - defect.
        # https://pages.mini.pw.edu.pl/~janeczkos/Ishi_Jan.pdf#page=20
        self.check_germs(("t",), [
            ("W18, monomial", ["t^4", "t^7"], 9),
            ("W18, ninth-order term", ["t^4", "t^7+t^9"], 7),
            ("W18, thirteenth-order term", ["t^4", "t^7+t^13"], 8),
        ])

    def test_space_curve(self):
        # Hernandes, Rodrigues Hernandes and Ruas give a normal-space basis in Example 4.1.
        # https://web.icmc.usp.br/SCATUSU/Cadernos_matematica/2006/v7_n1/v7_n1_A%20e%20-codimension%20of%20germs%20of%20analytic%20monomial%20curves.pdf
        self.check_germs(("t",), [("monomial curve in four-space", ["t^6", "t^9", "t^17", "t^25"], 49)])

    def test_plane_to_plane(self):
        # Yoshida, Kabata and Ohmoto, sections 1.1–1.3 and Table 2.
        # https://www.math.sci.hokudai.ac.jp/~ohmoto/Papers/yko.pdf#page=5
        self.check_germs(("x", "y"), [
            ("sharksfin", ["x^2+y^3", "y^2+x^3"], 2),
            ("odd-shaped sharksfin", ["x^2+y^5", "y^2+x^3"], 3),
            ("deltoid", ["x^2-y^2+x^3", "x*y"], 2),
            ("gulls", ["x", "x*y^2+y^4+y^5"], 2),
        ])

    def test_corank_two_surfaces(self):
        # Marar and Nuño-Ballesteros, Theorem 2.7 and Example 3.6 (double fold),
        # and Theorem 3.4 (unequal weights 1,2 preclude finite determinacy).
        # Section 4.1 gives 51 explicitly. In Example 4.4 the forms of weighted
        # degrees 10 and 13 are A-finite, so Proposition 4.3 gives 246 and 433.
        # https://www.uv.es/nuno/Preprints/corank2.pdf#page=7
        self.check_germs(("x", "y"), [
            ("homogeneous double fold", ["x^2", "y^2", "x^3+x*y^2+x^2*y-y^3"], 12),
            ("unequal-weight double fold", ["x^2", "y^2", "x^3+x*y"], None),
            ("double cusp, degree 5", ["x^2", "x*y+y^3", "x^2*y+y^5"], 51),
            ("double cusp, degree 10", ["x^2", "x*y+y^3", "x^5+2*x^4*y^2+y^10"], 246),
            ("double cusp, degree 13", ["x^2", "x*y+y^3", "x^6*y+y^13"], 433),
        ])

    def test_simple_surface_germs(self):
        # Costa, Martins and Nuño-Ballesteros, Table 1 (Mond's simple germs).
        # https://www.journalofsing.org/volume22/costa-martins-nuno_ballesteros.pdf#page=2
        self.check_germs(("x", "y"), [
            ("F4", ["x", "y^2", "x^3*y+y^5"], 4),
            ("H3", ["x", "x*y+y^8", "y^3"], 3),
        ])

    def test_projections_from_three_space(self):
        # Nabarro, Theorem 2.3, Table 2. Use orbit codimensions in the Ae column.
        # For N1 and N2, a=1 satisfies a != 0 and 4a^3+27 != 0. For N12,
        # a=b=c=1 satisfies 4b-1 != 0 and 4b-1+6ac != 0.
        # https://web.icmc.usp.br/SCATUSU/Cadernos_matematica/2001/v2_n2/v2_n2_Projections%20of%20hypersurfaces%20in%20R4to%20planes.pdf#page=5
        self.check_germs(("x", "y", "z"), [
            ("N1", ["x", "x*y+y^3+y^2*z+z^3+z^5"], 3),
            ("N2", ["x", "x*y+y^3+y^2*z+z^3"], 4),
            ("N12", ["x", "x*y*z+y^2*z+z^3+x^2*y+x^2*z+y*z^3+z^4"], 7),
        ])

    def test_map_into_four_space(self):
        # Houston and Wik Atique, Theorem 4.8(iii) and Remark 4.9.
        # https://arxiv.org/pdf/1302.1104#page=11
        self.check_germs(("x", "y", "z"), [("P2", ["x", "y", "z^3+x*z", "z^5+y*z"], 2)])

    def test_sharland_original_maps(self):
        for name, components, expected, k, last_weight in SHARLAND:
            with self.subTest(germ=name):
                result = calculate("ae-codimension", variables=("x", "y", "z"), components=components)
                self.assertEqual((result["finite"], result["value"]), (True, expected))
                self.assertEqual(result["method"], "weighted_tangent_ranks")
                self.assertEqual((result["contact_order"], result["last_weight"]), (k, last_weight))

    def test_sharland_with_nonlinear_target_change(self):
        # Adding the square of the first target coordinate to the fourth
        # preserves the normal space and breaks homogeneity in these coordinates.
        for name, components, expected, _, _ in SHARLAND[:2]:
            with self.subTest(germ=name):
                sheared = [*components[:-1], components[-1] + "+x^2"]
                result = calculate("ae-codimension", variables=("x", "y", "z"), components=sheared)
                self.assertEqual((result["finite"], result["value"]), (True, expected))
                self.assertEqual(result["method"], "extended_tangent_jets")

    def test_four_variable_augmentations(self):
        # Breva Ribes and Oset Sinha, Table 1. For 3_P take P of type A2
        # with Milnor number 2. The family 5_k has Ae-codimension k-1.
        # https://arxiv.org/pdf/2203.09223#page=17
        self.check_germs(("x", "y", "z", "t"), [
            ("3_P, P of type A2", ["x", "y", "z", "t^3+(x^3+y^2+z^2)*t"], 2),
            ("5_k, k=2", ["x", "y", "z", "t^5+x*t+y*t^2+z^2*t^3"], 1),
        ])
