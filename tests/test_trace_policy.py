import ctypes
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engine import native_library as native
from engine import trace_policy


class TracePolicyTests(unittest.TestCase):
    device = {"backend": "opencl", "index": 0, "name": "test GPU", "driver": "1.0"}
    shape = (2147483629, 26, 171, 2558)

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "profile.json"
        for mock in (patch.dict(os.environ, ATLAS_TRACE_PROFILE=str(self.path)),
                     patch.object(trace_policy, "_host", return_value={"kernels": "test-kernel-fingerprint"}),
                     patch.object(native, "_warm_trace", set())):
            mock.start()
            self.addCleanup(mock.stop)

    def record(self, **changes):
        row = dict(zip(("prime", "jobs", "width", "edges"), self.shape))
        row.update(samples=7, cpu_ms=10.0, gpu_ms=4.0, cpu_low_ms=9.0, gpu_high_ms=5.0)
        return dict(row, **changes)

    def profiles(self, *rows, cold=2.0):
        trace_policy.write_profile(self.path, self.device, rows, cold)
        return trace_policy.load_profiles("opencl", 0)

    def test_noise_and_disagreement_keep_cpu(self):
        for row in (self.record(samples=2), self.record(gpu_ms=9.1), self.record(gpu_high_ms=9.5)):
            with self.subTest(row=row):
                self.assertFalse(self.profiles(self.record(), row))

    def test_cold_start_new_shapes_and_device_mismatch(self):
        profiles = self.profiles(self.record(), cold=5.0)
        warm = {("opencl", 0, None)}
        with patch.object(trace_policy, "_selected_device", return_value=self.device) as probe:
            self.assertIsNone(trace_policy.fastest_backend(profiles, self.shape, set(), set()))
            self.assertIsNone(trace_policy.fastest_backend(profiles, (101, *self.shape[1:]), warm, set()))
            probe.assert_not_called()
            self.assertEqual(trace_policy.fastest_backend(profiles, self.shape, warm, set()), "opencl")
        with patch.object(trace_policy, "_selected_device", return_value=dict(self.device, driver="2.0")):
            skipped = set()
            self.assertIsNone(trace_policy.fastest_backend(profiles, self.shape, warm, skipped))
            self.assertEqual(skipped, {"opencl"})

    def test_missing_stale_and_invalid_profiles(self):
        self.assertFalse(trace_policy.load_profiles("auto", 0))
        self.profiles(self.record())
        self.assertFalse(trace_policy.load_profiles("opencl", 1))
        data = json.loads(self.path.read_text())
        data["host"]["kernels"] = "old"
        self.path.write_text(json.dumps(data))
        self.assertFalse(trace_policy.load_profiles("auto", 0))
        self.path.write_text("{")
        self.assertFalse(trace_policy.load_profiles("auto", 0))

    @unittest.skipUnless(hasattr(native._kernel(), "atlas_groebner_cpu_trace_counts"), "requires native build")
    def test_auto_sends_only_measured_winner_and_keeps_cpu_exact(self):
        from benchmarks.groebner_hard import katsura, payload
        names, polynomials = katsura(6)
        data = payload(polynomials, len(names), "grevlex")
        lib = native._kernel()
        lib.atlas_groebner_trace_rows_cpu.argtypes = native._TRACE._argtypes_
        lib.atlas_groebner_trace_rows_cpu.restype = ctypes.c_int

        def solve(mode):
            with native.replay_scope(mode, backend="opencl") as scope:
                h = lib.atlas_groebner(data, len(data))
                self.assertTrue(h, lib.atlas_groebner_error())
                try:
                    return lib.atlas_groebner_text(h), scope.report()
                finally:
                    lib.atlas_groebner_free(h)

        self.profiles(self.record())
        # Model a fast device by executing its ABI with the native reference.
        def screen(*args):
            return lib.atlas_groebner_trace_rows_cpu(*args[:-1])

        with patch.object(trace_policy, "_selected_device", return_value=self.device) as probe, \
             patch.object(native, "_opencl_trace", return_value=screen):
            basis, cpu = solve("cpu")
            probe.assert_not_called()
            other, auto = solve("auto")
            self.assertEqual(basis, other)
            rows = auto["trace"]["backends"]
            self.assertEqual(rows["opencl"], {"batches": 1, "pairs": self.shape[1]})
            self.assertEqual(rows["cpu"]["pairs"] + self.shape[1], cpu["trace"]["backends"]["cpu"]["pairs"])
            self.assertEqual(auto["backend"], "cpu")
