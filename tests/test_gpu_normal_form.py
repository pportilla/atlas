import os
import unittest
from unittest.mock import patch

from engine.gpu_devices import available_devices
from engine.gpu_normal_form import _certify, _opencl_witnesses, normal_form
from engine.native_library import _kernel, replay_scope


class MonomialNormalFormTests(unittest.TestCase):
    # One full and one partial Triton block, large admissible exponents, a
    # rational surviving coefficient, and the same term divisible by two heads.
    basis = [(7, 11, 0), (6, 0, 21), (32767, 0, 0)]
    terms = [(i % 17, i % 19, i % 23) for i in range(121)] + [(32767, 0, 0)]

    @staticmethod
    def expected(terms, basis):
        return [next((j for j, head in enumerate(basis)
                      if all(h <= e for h, e in zip(head, term))), -1) for term in terms]

    def test_host_rejects_bad_divisibility(self):
        witnesses = self.expected(self.terms, self.basis)
        _certify(self.terms, self.basis, witnesses)
        for i, bad in ((0, 0), (121, -1), (0, len(self.basis)), (0, -2)):
            with self.subTest(i=i, bad=bad), self.assertRaises(ArithmeticError):
                changed = witnesses.copy()
                changed[i] = bad
                _certify(self.terms, self.basis, changed)
        with self.assertRaises(ArithmeticError):
            _certify(self.terms, self.basis, [])

    def test_selection_and_host_verified_exact_coefficients(self):
        from engine.algebra_execution import algebra_execution
        basis = ["1 7 11 0", "1 6 0 21", "1 32767 0 0"]
        chunks = [f"-1234567890987654321/17 {a} {b} {c}" for a,b,c in self.terms]
        row = ";".join(chunks)
        reference = self.expected(self.terms, self.basis)
        answer = ";".join(s for s,j in zip(chunks, reference) if j == -1)
        with patch("engine.gpu_normal_form._opencl_witnesses", return_value=reference) as launch:
            with algebra_execution("gpu", backend="opencl") as run:
                self.assertEqual(normal_form(("a", "b", "c"), basis, row), answer)
                stage = run.report()["stages"][-1]
                self.assertEqual(stage["operation"], "monomial_normal_form")
                self.assertEqual(stage["terms"], len(chunks))
                self.assertEqual(stage["removed"], sum(j != -1 for j in reference))
            for compute in ("cpu", "auto"):
                with replay_scope(compute, backend="opencl"):
                    self.assertIsNone(normal_form(("a", "b", "c"), basis, row))
            with replay_scope("gpu", backend="opencl"):
                self.assertIsNone(normal_form(("a", "b", "c"), basis, chunks[1]))
                self.assertIsNone(normal_form(("a", "b", "c"), [";".join(basis)], row))
                self.assertIsNone(normal_form(("a", "b", "c"), basis, row+"*"+chunks[0]))
            launch.assert_called_once()

    def test_device_bounds_reject_unrepresentable_inputs_before_launch(self):
        from engine.gpu_normal_form import monomial_witnesses
        with replay_scope("gpu", backend="opencl"), patch("engine.gpu_normal_form._opencl_witnesses") as launch:
            for powers in ((32767, 1), (-1, 0), (2**70, 0)):
                self.assertIsNone(monomial_witnesses(2, [powers] * 64, [(1, 0)]))
            launch.assert_not_called()

    @unittest.skipIf(_kernel() is None, "Requires the compiled Atlas reducer")
    def test_rejected_result_retries_exact_cpu(self):
        from engine.algebra import dispatch
        request = {"operation": "normal-form", "compute": "gpu",
                   "ring": {"variables": ["x"], "field": "QQ", "order": "lex"},
                   "operands": {"ideal": ["x"], "polynomial": "(1+x)^63/7"}}
        with patch("engine.gpu_normal_form._opencl_witnesses", return_value=[-1]*64), \
                patch.dict(os.environ, RESOLUTION_GPU_BACKEND="opencl"):
            result = dispatch(request)
        self.assertEqual(result["result"]["polynomial"], "1/7")
        report = result["execution"]
        self.assertEqual(report["device"], "CPU")
        self.assertIn("Host rejected", " ".join(report["fallback_reasons"]))

    def test_listed_physical_devices(self):
        devices = available_devices()
        if not devices:
            self.skipTest("Requires a physical GPU")
        expected = self.expected(self.terms, self.basis)
        for device in devices:
            with self.subTest(device=device):
                if device["backend"] == "opencl":
                    result = _opencl_witnesses(self.terms, self.basis, device["index"])
                else:
                    from engine.triton_monomial import witnesses
                    result = witnesses(self.terms, self.basis, device["torch_device"])
                self.assertEqual(list(result), expected)

    @unittest.skipUnless(os.getenv("TRITON_INTERPRET") == "1", "Requires Triton interpreter")
    def test_triton_interpreter(self):
        from engine.triton_monomial import witnesses
        self.assertEqual(witnesses(self.terms, self.basis, "cpu"), self.expected(self.terms, self.basis))
