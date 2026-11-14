"""Compare batched traces and device failure paths with known scalar bases."""
import ctypes
import hashlib
import unittest

from benchmarks.groebner_hard import cyclic, katsura, payload
from engine.native_library import _kernel


_U32 = ctypes.c_uint32
_PTR = ctypes.POINTER(_U32)
_SCREEN = ctypes.CFUNCTYPE(ctypes.c_int, _U32, _U32, _U32, *([_PTR] * 6))


@unittest.skipUnless(hasattr(_kernel(), "atlas_groebner_swap_trace_screen"), "rebuild the modular kernel")
class TraceConstruction(unittest.TestCase):
    def test_ordered_layers_and_declined_device(self):
        lib = _kernel()
        swap = lib.atlas_groebner_swap_trace_screen
        swap.argtypes, swap.restype = [ctypes.POINTER(ctypes.c_void_p)], None
        native_rows = lib.atlas_groebner_trace_rows_cpu
        native_rows.argtypes, native_rows.restype = _SCREEN._argtypes_, ctypes.c_int
        modular = lib.atlas_groebner_modular
        modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_long),
                            ctypes.POINTER(ctypes.c_double)]
        modular.restype = ctypes.c_void_p

        def solve(source, screen):
            calls = []

            @_SCREEN
            def callback(prime, jobs, width, initial, offsets, pivots, columns, coefficients, result):
                calls.append(jobs)
                if screen == "declined":
                    return 1
                if screen == "invalid":
                    result[0] = prime
                    return 0
                return native_rows(prime, jobs, width, initial, offsets, pivots, columns, coefficients, result)

            pointer = ctypes.cast(callback, ctypes.c_void_p)
            swap(ctypes.byref(pointer))
            try:
                counts = (ctypes.c_long * 7)()
                handle = modular(source, len(source), counts, None)
                self.assertTrue(handle, lib.atlas_groebner_error())
                try:
                    self.assertTrue(calls)
                    self.assertGreater(counts[3], 0)
                    return lib.atlas_groebner_text(handle), counts[4]
                finally:
                    lib.atlas_groebner_free(handle)
            finally:
                swap(ctypes.byref(pointer))

        # Hashes of the complete basis text from the original scalar trace.
        cases = ((katsura(6), "grevlex", "3295c09d1f6e805d58123f40cb675c947a990453024a53cc4324d22901602660"),
                 (cyclic(5), "lex", "c6385bf71b1624c8d6004860defab899e23f18eb6dab037861e938db0eb1f805"))
        for problem, order, digest in cases:
            names, equations = problem
            source = payload(equations, len(names), order)
            reference = solve(source, "declined")
            self.assertEqual(hashlib.sha256(reference[0]).hexdigest(), digest)
            for mode in ("batch", "invalid"):
                with self.subTest(names=names, order=order, mode=mode):
                    self.assertEqual(solve(source, mode), reference)
