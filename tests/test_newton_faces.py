import unittest
from engine.newton_faces import certify_face

class NewtonFaceTests(unittest.TestCase):
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
