import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier
import unittest
from unittest.mock import patch

from engine import native_library as native


@unittest.skipUnless(native._kernel() is not None, "native kernel not built")
class ReplayScopeTests(unittest.TestCase):
    problem = b"2 0 0\n1 2 0;2147483629 0 1;-1 0 0\n1 1 1;-2 0 0\n"

    def basis(self):
        lib = native._kernel()
        lib.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                               ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double)]
        lib.atlas_groebner_modular.restype = ctypes.c_void_p
        handle = lib.atlas_groebner_modular(self.problem, len(self.problem), None, None)
        self.assertTrue(handle)
        try:
            return lib.atlas_groebner_text(handle)
        finally:
            lib.atlas_groebner_free(handle)

    def test_concurrent_cpu_and_reference_scopes(self):
        ready = Barrier(2)

        def run(compute):
            with native.replay_scope(compute, backend="reference") as replay:
                ready.wait(timeout=10)
                basis = self.basis()
                result = replay.report()
                self.assertEqual(replay.report(), result)
            self.assertIsNone(native.current_replay_scope())
            return basis, result

        with ThreadPoolExecutor(2) as pool:
            cpu, reference = pool.map(run, ["cpu", "gpu"])
        self.assertEqual(cpu[0], reference[0])
        self.assertEqual(cpu[1]["backend"], "cpu")
        self.assertEqual(reference[1]["backend"], "reference")
        self.assertEqual(cpu[1]["primes"], reference[1]["primes"])
        self.assertGreater(cpu[1]["primes"], 0)

    def test_failed_launch_preserves_basis_and_valid_cpu_count(self):
        with native.replay_scope("cpu") as cpu:
            expected = self.basis()
            baseline = cpu.report()
        with patch("engine.gpu_devices.triton_device", return_value={"torch_device": "cuda:1"}), \
             patch("engine.gpu_replay.replay_triton", side_effect=RuntimeError("allocator failed")), \
             native.replay_scope("gpu", backend="triton", index=1) as gpu:
            self.assertEqual(self.basis(), expected)
            report = gpu.report()
            self.assertEqual(report["primes"], baseline["primes"])
            self.assertEqual(report["backend"], "cpu")
            self.assertIn("allocator failed", report["fallback_reasons"][0])
        with native.replay_scope("cpu") as other:
            self.assertIsNone(other.report())
            other.merge(report)
            self.assertEqual(other.report(), report)

    def test_copied_context_cannot_rebase_parent_thread_counters(self):
        with native.replay_scope("cpu") as parent:
            self.basis()
            baseline = parent.report()
            self.basis()

            def worker():
                self.assertIsNone(native.current_replay_scope())
                with native.replay_scope("gpu", backend="reference") as child:
                    self.basis()
                    return child.report()

            with ThreadPoolExecutor(1) as pool:
                child = pool.submit(copy_context().run, worker).result()
            self.assertEqual(child["backend"], "reference")
            report = parent.report()
            self.assertEqual(report["backends"], {"cpu": {"batches": 2 * baseline["batches"],
                                                         "primes": 2 * baseline["primes"]}})
            if "certification" in report:
                self.assertEqual(report["certification"]["bases"], 2)
                self.assertEqual(child["certification"]["bases"], 1)

    @unittest.skipUnless(hasattr(native._kernel(), "atlas_groebner_certification_counts"), "rebuild the modular kernel")
    def test_partial_failed_readback_restores_inputs_and_counts_cpu_verification(self):
        with native.replay_scope("cpu"):
            exact = self.basis()

        def damaged_readback(code, length, pool, work, primes, count, pools, status, index):
            # A failed transfer might already have overwritten generator slots.
            ctypes.memset(pools, 0, count * pool * ctypes.sizeof(ctypes.c_uint32))
            return 1

        with patch.object(native, "_opencl_replay", return_value=damaged_readback), \
             native.replay_scope("gpu", backend="opencl") as scope:
            self.assertEqual(self.basis(), exact)
            report = scope.report()
        self.assertEqual(report["backend"], "cpu")
        self.assertNotIn("devices", report)
        self.assertIn("OpenCL replay failed", report["fallback_reasons"][0])
        cert = report["certification"]
        self.assertEqual(cert["bases"], 1)
        self.assertGreaterEqual(cert["attempts"], cert["bases"])
        self.assertGreater(cert["verify_ms"], 0)
        self.assertGreater(cert["lift_ms"], 0)

    def test_lazy_selection_restores_outer_context_on_error(self):
        with native.replay_scope("gpu", backend="triton", index=1) as gpu, \
             patch("engine.gpu_devices.triton_device", side_effect=AssertionError("unneeded probe")):
            self.assertEqual(gpu.settings(), {"compute": "gpu", "backend": "triton", "index": 1})
            with self.assertRaisesRegex(ValueError, "bad candidate"):
                with native.replay_scope("cpu") as cpu:
                    self.assertIs(native.current_replay_scope(), cpu)
                    raise ValueError("bad candidate")
            self.assertIs(native.current_replay_scope(), gpu)
            self.assertIsNone(gpu.report())
        self.assertIsNone(native.current_replay_scope())

    def test_nested_counts_and_auto_worker_settings(self):
        with patch.dict("os.environ", {"RESOLUTION_GPU_BACKEND": "triton", "ATLAS_GROEBNER_GPU": ""}):
            with native.replay_scope("auto") as parent:
                self.assertEqual(parent.compute, "auto")
                with native.replay_scope(**parent.settings()) as child:
                    self.basis()
                    report = child.report()
                    self.assertEqual(report["backend"], "cpu")
                self.assertIsNone(parent.report())
                parent.merge(report)
                self.assertEqual(parent.report(), report)

    def test_listed_opencl_device_and_overlapping_cpu_run_the_certified_replay(self):
        from engine.native_backend import gpu_devices
        devices = gpu_devices()
        if not devices or native._opencl_replay() is None:
            self.skipTest("requires an OpenCL GPU and the indexed replay plugin")
        ready = Barrier(2)
        def solve(compute):
            with native.replay_scope(compute, backend="opencl", index=devices[0]["index"]) as scope:
                ready.wait(timeout=10)
                basis = self.basis()
                ready.wait(timeout=30)
                return basis, scope.report()

        with ThreadPoolExecutor(2) as pool:
            (exact, cpu), (candidate, gpu) = pool.map(solve, ("cpu", "gpu"))
        self.assertEqual(candidate, exact)
        self.assertEqual(cpu["backend"], "cpu")
        self.assertNotIn("devices", cpu)
        self.assertEqual(gpu["backend"], "opencl", gpu)
        self.assertGreater(gpu["primes"], 0)
        self.assertEqual(gpu["primes"], cpu["primes"])
        self.assertEqual(gpu["devices"]["opencl"]["name"], devices[0]["name"])
        if "certification" in gpu:
            self.assertEqual(cpu["certification"]["bases"], 1)
            self.assertEqual(gpu["certification"]["bases"], 1)

    @unittest.skipUnless(hasattr(native._kernel(), "atlas_groebner_cpu_trace_counts"), "rebuild the modular kernel")
    def test_nested_trace_scope_and_batched_cpu_fallback(self):
        from benchmarks.groebner_hard import katsura, payload
        names, polys = katsura(6)
        source = payload(polys, len(names), "grevlex")
        lib = native._kernel()

        def solve():
            handle = lib.atlas_groebner(source, len(source))
            self.assertTrue(handle, lib.atlas_groebner_error())
            try:
                return lib.atlas_groebner_text(handle)
            finally:
                lib.atlas_groebner_free(handle)

        with patch.object(native, "_opencl_trace", return_value=None), \
             patch.object(native, "_opencl_replay", return_value=None), \
             native.replay_scope("gpu", backend="opencl") as outer:
            with native.replay_scope("cpu") as cpu:
                expected = solve()
                counts = cpu.report()["trace"]["backends"]["cpu"]
                self.assertGreater(counts["pairs"], 0)
                self.assertFalse(cpu.trace_unavailable)
            self.assertIsNone(outer.report())
            self.assertEqual(solve(), expected)
            report = outer.report()
            self.assertEqual(report["trace"]["backends"], {"cpu": counts})
            self.assertEqual(outer.trace_unavailable, {"opencl"})
            self.assertEqual(report["backend"], "cpu")
