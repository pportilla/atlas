import os
import unittest
from fractions import Fraction
from engine.newton_faces import certify_face, edge_polynomial, face_search

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

    def test_exhaustive_edge_keeps_wide_labels_and_root_multiplicities(self):
        # (1+T^N)^2 has N distinct double roots on the torus. Both the
        # character exponent and the common scale exceed decimal digit limits.
        degree = 2**16000
        scale = Fraction(7, 3**12000)
        face = {(0,): scale, (degree,): 2*scale, (2*degree,): scale}
        self.assertEqual(certify_face(face)["status"], "undetermined")
        proof = certify_face(face, exhaustive=True)
        self.assertEqual((proof["status"], proof["gcd_degree"]), ("degenerate", degree))
        self.assertEqual(edge_polynomial(proof["edge"]), face)

    def test_exhaustive_face_after_native_basis_limit(self):
        # Each input coefficient fits the bounded protocol. The intermediate
        # Buchberger coefficients of this biquadratic exceed its 16384-bit cap.
        face = {(i,j): 2**(900-3*i-j) + 3**(450-4*j-i) + 1 for i in range(3) for j in range(3)}
        self.assertIn("basis exceeded", certify_face(face)["reason"])
        proof = certify_face(face, exhaustive=True)
        self.assertEqual((proof["status"], proof["basis_size"], proof["unit_ideal"]), ("nondegenerate", 1, True))

    def test_exhaustive_distinguishes_proper_and_unit_jacobians(self):
        # The first polynomial is (1+x^N)(1+y). Changing just the xy
        # corner removes every singular point from the torus.
        degree = 2**65
        face = {(0,0): 1, (degree,0): 1, (0,1): 1, (degree,1): 1}
        self.assertEqual(certify_face(face)["status"], "undetermined")
        self.assertFalse(certify_face(face, exhaustive=True)["unit_ideal"])
        face[(degree,1)] = 2**17000
        self.assertTrue(certify_face(face, exhaustive=True)["unit_ideal"])

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "The sparse wide kernel handles multi-word exponents")
    def test_exhaustive_uncompressed_exponent_beyond_a_machine_word(self):
        # df/dy=1+x^(N+1); a common zero with f would also require
        # 1+x^N=0. Consecutive powers would then give x=1, a contradiction.
        degree = 2**70
        face = {(0,0): 1, (degree,0): 1, (0,1): 1, (degree+1,1): 1}
        self.assertTrue(certify_face(face, exhaustive=True)["unit_ideal"])

    def test_quick_then_exhaustive_resolution_and_orbit_recheck(self):
        from engine.atlas_torus import _orbit
        from engine.newton_backend import resolve_newton
        face = {(4,0): 1, (2,2): 2**8300, (0,4): 1}
        rays = ((1,0), (0,1))
        self.assertIsNone(resolve_newton(face, ["x","y"], "embedded", defer_outputs=True))
        self.assertIsNone(_orbit(rays, tuple(face.items()), False))
        with face_search(True):
            result = resolve_newton(face, ["x","y"], "embedded", defer_outputs=True)
        self.assertEqual(result["status"], "resolved")
        proofs = result["certificate"]["compact_faces"]
        self.assertEqual([row["gcd_degree"] for row in proofs if row["face_dimension"] == 1], [0])
        self.assertEqual(_orbit(rays, tuple(face.items()), True)[0], 4)
        self.assertEqual(certify_face(face)["status"], "undetermined")

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "Exercises the native modular replay hooks")
    def test_exhaustive_uses_requested_modular_replay(self):
        from engine.native_library import replay_scope
        face = {(0,0): 1, (8192,0): 1, (0,1): 1, (8192,1): 2}
        with replay_scope("gpu", backend="reference", index=0) as scope:
            self.assertTrue(certify_face(face, exhaustive=True)["unit_ideal"])
        self.assertGreater(scope.report()["backends"]["reference"]["primes"], 0)

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
