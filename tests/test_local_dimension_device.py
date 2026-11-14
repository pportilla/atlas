"""Certified CPU and device paths for local dimension and smoothness."""
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch
from tests.large_coupled_ideal import COUPLED_RANK_ONE, POINT, VARIABLES


OPERATIONS = ("local-dimension", "local-smoothness")


def solve(operation, compute):
    x, y, z = "(x-1/2)", "(y+2/3)", "(z-1/3)"
    u, v = f"({x}+2*{y})", f"(3*{x}+{y})"
    # Here 5v-4 is a unit. The local ring is QQ[u,v]/(u^3,v^3),
    # with one independent Jacobian row from the third coordinate.
    return dispatch({"operation": operation, "compute": compute,
                     "ring": {"variables": ["x", "y", "z"]},
                     "operands": {"ideal": [f"{u}^3", f"{v}^3*(5*{v}-4)", z],
                                  "point": ["1/2", "-2/3", "1/3"]}})


class LocalDimensionDeviceTests(unittest.TestCase):
    def test_unmaterializable_jacobian_entries_have_an_exact_rank(self):
        n = 2**64+7
        f = f"(x-2+x^{n}*y)"
        for rows, dimension, rank in (([f, "y"], 0, 2), ([f, "y^2"], 0, 1),
                                      ([f, f"(1+x^{n})*{f}"], 1, 1)):
            for operation in OPERATIONS:
                for policy in ("cpu", "auto", "gpu"):
                    with self.subTest(rows=rows, operation=operation, policy=policy):
                        record = dispatch({"operation": operation, "compute": policy,
                                           "ring": {"variables": ["x", "y"]},
                                           "operands": {"ideal": rows, "point": ["2", "0"]}})
                        result = record["result"]
                        self.assertEqual((result["point"], result["local_dimension"]), (["2","0"], dimension))
                        if operation == "local-smoothness":
                            self.assertEqual(result["jacobian_rank"], rank)
                            self.assertEqual(result["embedding_dimension"], 2-rank)
                            self.assertIs(result["value"], dimension == 2-rank)
                            if os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1" and policy == "gpu":
                                stages = record["execution"]["stages"]
                                witness = next(s for s in stages if s["operation"] == "jacobian_row_selection")
                                self.assertIn(witness["backend"], ("triton","opencl"))
                                self.assertTrue(witness["device"])
                                self.assertEqual(witness["candidate_rank"], rank)
                                certificate = next(s for s in stages if s["operation"] == "certify_jacobian_row_space")
                                self.assertEqual((certificate["backend"], certificate["candidate_rank"]), ("cpu", rank))

    def test_bad_modular_prime_for_wide_row_recomputes_exact_rank(self):
        # The second row is independent over QQ but reduces to zero at the
        # device prime. Checking the reported rows must reject rank one.
        request = {"operation": "local-smoothness", "compute": "gpu", "ring": {"variables": ["x", "y"]},
                   "operands": {"ideal": [f"x-2+x^{2**64+7}*y", "1000000007*y"], "point": ["2", "0"]}}
        record = dispatch(request)
        self.assertEqual(record["result"]["jacobian_rank"], 2)
        self.assertIs(record["result"]["value"], True)
        if os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1":
            witness = next(s for s in record["execution"]["stages"] if s["operation"] == "jacobian_row_selection")
            self.assertEqual(witness["candidate_rank"], 1)
            self.assertTrue(any("did not span" in reason for reason in record["execution"]["fallback_reasons"]))

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_wide_coupled_initial_forms_on_device(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                request = {"operation": operation, "ring": {"variables": VARIABLES, "order": "grevlex"},
                           "operands": {"ideal": COUPLED_RANK_ONE, "point": POINT}}
                cpu = dispatch({**request, "compute": "cpu"})
                gpu = dispatch({**request, "compute": "gpu"})
                result = gpu["result"]
                self.assertEqual(result, cpu["result"])
                self.assertEqual((result["point"], result["local_dimension"]), (POINT, 0))
                stages = gpu["execution"]["stages"]
                replay = next(stage for stage in stages if stage["operation"] == "groebner_replay"
                              and stage["backend"] in ("opencl", "triton"))
                self.assertTrue(replay["device"])
                self.assertGreater(replay["primes"], 0)
                self.assertTrue(any(stage["operation"] == "groebner_certification" and
                                    stage["backend"] == "cpu" and stage["bases"] > 0
                                    for stage in stages[stages.index(replay) + 1:]))
                if operation == "local-smoothness":
                    self.assertIs(result["value"], False)
                    self.assertEqual((result["embedding_dimension"], result["jacobian_rank"]), (2, 1))
                    witness = next(stage for stage in stages if stage["operation"] == "jacobian_row_selection")
                    self.assertIn(witness["backend"], ("opencl", "triton"))
                    self.assertTrue(witness["device"])
                    self.assertEqual(witness["candidate_rank"], 1)
                    self.assertTrue(any(stage["operation"] == "certify_jacobian_row_space" and
                                        stage["backend"] == "cpu" and stage["candidate_rank"] == 1
                                        for stage in stages[stages.index(witness) + 1:]))

    def test_unavailable_device_uses_exact_cpu(self):
        with patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl", RESOLUTION_GPU_INDEX=str(2**31-1)):
            for operation in OPERATIONS:
                with self.subTest(operation=operation):
                    cpu = solve(operation, "cpu")["result"]
                    self.assertEqual((cpu["scope"], cpu["point"], cpu["local_dimension"]),
                                     ("point", ["1/2", "-2/3", "1/3"], 0))
                    if operation == "local-dimension":
                        self.assertEqual(cpu["value"], 0)
                    else:
                        self.assertIs(cpu["value"], False)
                        self.assertEqual((cpu["embedding_dimension"], cpu["jacobian_rank"]), (2, 1))
                    for policy in ("auto", "gpu"):
                        report = solve(operation, policy)
                        self.assertEqual(report["result"], cpu)
                        self.assertEqual(report["execution"]["backend"], "cpu")
                        if policy == "gpu":
                            self.assertTrue(report["execution"]["note"])

    @unittest.skipUnless(os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1",
                         "Set ATLAS_TEST_ALGEBRA_GPU=1 to require a physical GPU")
    def test_device_replay_and_exact_certification(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                cpu = solve(operation, "cpu")["result"]
                self.assertEqual(solve(operation, "auto")["result"], cpu)
                gpu = solve(operation, "gpu")
                self.assertEqual(gpu["result"], cpu)
                stages = gpu["execution"]["stages"]
                replay = next(s for s in stages if s["operation"] == "groebner_replay")
                certificate = next(s for s in stages if s["operation"] == "groebner_certification")
                self.assertEqual(replay["label"], "local_dimension")
                self.assertIn(replay["backend"], ("opencl", "triton"))
                self.assertTrue(replay["device"])
                self.assertGreater(replay["primes"], 0)
                self.assertEqual(certificate["backend"], "cpu")
                self.assertGreater(certificate["bases"], 0)
                self.assertLess(stages.index(replay), stages.index(certificate))
                if operation == "local-smoothness":
                    selection = next(s for s in stages if s["operation"] == "jacobian_row_selection")
                    check = next(s for s in stages if s["operation"] == "certify_jacobian_row_space")
                    self.assertIn(selection["backend"], ("opencl", "triton"))
                    self.assertTrue(selection["device"])
                    self.assertEqual((selection["candidate_rank"], check["candidate_rank"]), (1, 1))
                    self.assertEqual(check["backend"], "cpu")
                    self.assertLess(stages.index(selection), stages.index(check))
