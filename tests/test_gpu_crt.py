"""Exact pair lifting, damaged-device recovery, and public CPU/device agreement."""
import ctypes
import os
import unittest
from unittest.mock import patch

from engine.algebra import dispatch
from engine.gpu_crt import _CRT, merge_pairs
from engine.gpu_devices import triton_device
from engine.native_library import _kernel


def groebner_request(compute):
    scale = 2**301 + 123
    return dispatch({"operation": "groebner", "compute": compute,
                     "ring": {"field": "QQ", "variables": ["x", "y"], "order": "lex"},
                     "operands": {"ideal": [f"x^2+{scale}*y-1", "x*y-2"]}})


@unittest.skipUnless(hasattr(_kernel(), "atlas_groebner_swap_crt_pairs"), "build the paired lift")
class PairedCRT(unittest.TestCase):
    def test_native_lift_and_recovery_after_a_damaged_pair(self):
        baseline = groebner_request("cpu")["result"]
        for failure in ("none", "bad_residue", "failed_readback"):
            calls = []

            @_CRT
            def reference(images, moduli, pairs, width, out):
                calls.append(pairs)
                for pair in range(pairs):
                    p, q, inverse = moduli[3*pair:3*pair+3]
                    for j in range(width):
                        left, right = images[2*pair*width+j], images[(2*pair+1)*width+j]
                        out[pair*width+j] = left + p*((right-left)*inverse % q)
                if failure == "bad_residue":
                    out[pairs*width-1] ^= 1
                return int(failure == "failed_readback")

            with self.subTest(failure=failure), patch("engine.gpu_crt.dispatch_crt", reference), \
                 patch.dict(os.environ, {"RESOLUTION_GPU_BACKEND": "triton"}), \
                 patch("engine.gpu_devices.triton_device", return_value=None):
                self.assertEqual(groebner_request("gpu")["result"], baseline)
                self.assertGreater(max(calls), 1)

    def test_device_pair_boundaries_and_complete_basis(self):
        device = triton_device()
        if device is None:
            self.skipTest("physical Triton accelerator required")
        primes = [(2147483647, 2147483629), (3, 5), (2147483629, 3)]
        width = 257
        images, moduli, expected = [], [], []
        for p, q in primes:
            inverse = pow(p, -1, q)
            left = [(p-1-k*k) % p for k in range(width)]
            right = [(7*k-q//2) % q for k in range(width)]
            images.extend(left + right)
            moduli.extend((p, q, inverse))
            expected.extend(a+p*((b-a)*inverse % q) for a,b in zip(left,right))
        images = (ctypes.c_uint32*len(images))(*images)
        moduli = (ctypes.c_uint32*len(moduli))(*moduli)
        output = (ctypes.c_uint64*len(expected))()
        merge_pairs(ctypes.cast(images, ctypes.POINTER(ctypes.c_uint32)),
                    ctypes.cast(moduli, ctypes.POINTER(ctypes.c_uint32)), len(primes),
                    width, output, device["torch_device"])
        self.assertEqual(output[:], expected)
        with patch.dict(os.environ, {"RESOLUTION_GPU_BACKEND": "triton"}):
            gpu = groebner_request("gpu")
            self.assertEqual(gpu["result"], groebner_request("cpu")["result"])
        self.assertTrue(any(s["operation"] == "crt_pair_lift" and s["backend"] == "triton"
                            for s in gpu["execution"]["stages"]))
