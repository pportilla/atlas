import unittest
from fractions import Fraction
from engine.newton_faces import certify_face

class NewtonFaceTests(unittest.TestCase):
    def test_primitive_input_budget_and_signed_laurent_translation(self):
        face = {(-9,4):Fraction(-1,5), (-8,3):Fraction(-2,5), (-7,2):Fraction(-1,5)}
        cert = certify_face(face)
        self.assertEqual(cert["edge"]["direction"], [1,-1])
        self.assertEqual(cert["status"], "degenerate")
        scaled = {e:c*(2**8300) for e,c in face.items()}
        self.assertEqual(certify_face(scaled)["gcd_degree"], 1)
        bad = {**face,(-8,3):Fraction(1,2**8192)}
        self.assertIn("denominators", certify_face(bad)["reason"])
        bad = {**face,(-8,3):2**8192}
        self.assertIn("coefficient or exponent", certify_face(bad)["reason"])
    def test_pivots_with_constant_and_dependent_coordinates(self):
        offset = 2**99
        face = {(offset,1,3,4):1, (offset,1,3,6):1, (offset,3,7,4):1}
        self.assertEqual(certify_face(face)["projection_axes"], [1,3])
        self.assertEqual(certify_face({(offset,1,3,4):1})["projection_axes"], [])

    def test_projected_nonsimplex_faces(self):
        # After extracting a monomial, the first is (1+x)(1+y);
        # changing the last coefficient removes its torus singularity.
        face = {(0,0,2,0):1, (1,0,1,0):1, (0,1,1,0):1, (1,1,0,0):1}
        self.assertEqual(certify_face(face)["status"], "degenerate")
        face[(1,1,0,0)] = 2
        certified = certify_face(face)
        self.assertTrue(certified["unit_ideal"])
        self.assertEqual(certified["projection_axes"], [0,1])

    def test_primitive_character_with_gaps(self):
        # An ambient primitive direction records four geometric roots, even
        # though the exponents only generate the index-two sublattice.
        face = {(6,0,0,0):1, (4,2,0,0):2, (2,4,0,0):1}
        result = certify_face(face)
        self.assertEqual(result["edge"]["direction"], [1,-1,0,0])
        self.assertEqual(result["edge"]["degree"], 4)
        self.assertEqual(result["gcd_degree"], 2)
        face[(4,2,0,0)] = 1
        self.assertEqual(certify_face(face)["status"], "nondegenerate")

    def test_origin_singularities_excluded_from_torus(self):
        # x^2+y^2 is singular only at the origin; a unit monomial shift
        # and additional constant torus factors preserve smoothness.
        cert = certify_face({(3,0,7,2):1,(1,2,7,2):1})
        self.assertEqual(cert["status"], "nondegenerate")
        self.assertEqual(cert["face_dimension"], 1)
