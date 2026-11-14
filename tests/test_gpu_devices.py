import io
import os
import sys
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from engine import app, gpu_devices, native_backend


class GPUCommand(unittest.TestCase):
    def test_interpreter_cannot_count_as_a_gpu(self):
        with patch.dict(os.environ, TRITON_INTERPRET="1"), patch.object(gpu_devices, "_probe_triton") as probe:
            self.assertIsNone(gpu_devices.triton_device(0))
            probe.assert_not_called()

    def test_alias_and_negative_polynomial_in_both_orders(self):
        parser = app.make_parser()
        for flags, polynomial in ((["--gpu", "-x^2+y"], "-x^2+y"),
                                  (["-x^2+y", "--gpu"], "-x^2+y"),
                                  (["--compute=gpu", "--x^2+y"], "--x^2+y")):
            with self.subTest(flags=flags), redirect_stdout(io.StringIO()):
                with patch.object(app, "run_job", return_value=(200, {"status": "resolved"})) as job:
                    self.assertEqual(app.run_command(parser, ["resolve", *flags, "--vars", "x,y", "--json"]), 0)
                self.assertEqual(job.call_args.args[0]["compute"], "gpu")
                self.assertEqual(job.call_args.args[0]["polynomial"], polynomial)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parser.parse_args(["resolve", "x+y", "--vars", "x,y", "--gpu", "--compute", "cpu"])

    def test_devices_report_backend_local_indices(self):
        cl = {"index": 0, "name": "OpenCL card"}
        triton = {"index": 1, "name": "Triton card", "torch_device": "cuda:1", "backend": "triton"}
        with patch.object(native_backend, "gpu_devices", return_value=[cl]), \
             patch.object(gpu_devices, "_triton_candidates", return_value=("cuda:0", "cuda:1")), \
             patch.object(gpu_devices, "triton_device", side_effect=[None, triton]):
            self.assertEqual(gpu_devices.available_devices(), [dict(cl, backend="opencl"), triton])
            self.assertNotIn("backend", cl)


class TritonDiscovery(unittest.TestCase):
    def setUp(self):
        # Exercise the discovery/dispatch logic without requiring all three vendors in CI.
        self.torch = MagicMock()
        self.torch.cuda.device_count.return_value = 2
        self.torch.xpu.device_count.return_value = 1
        self.torch.xpu.get_device_name.return_value = "Intel test GPU"
        self.torch.xpu.device.return_value = nullcontext()
        self.torch.cuda.device.return_value = nullcontext()
        self.tensor = self.torch.full.return_value
        self.tensor.item.return_value = 169
        self.kernel = MagicMock()
        self.target = SimpleNamespace(backend="xpu", arch={"arch": "pvc"})
        driver = SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: self.target))
        modules = {"torch": self.torch, "triton": MagicMock(), "triton.runtime": SimpleNamespace(driver=driver),
                   "engine.triton_probe": SimpleNamespace(square_modulo=self.kernel)}
        for context in (patch.dict(sys.modules, modules), patch.dict(os.environ, TRITON_INTERPRET="0")):
            context.start()
            self.addCleanup(context.stop)
        for function in (gpu_devices._triton_candidates, gpu_devices._probe_triton):
            function.cache_clear()
            self.addCleanup(function.cache_clear)

    def test_xpu_probed_with_backend_local_ordinal(self):
        self.assertEqual(gpu_devices._triton_candidates(), ("cuda:0", "cuda:1", "xpu:0"))
        self.assertIsNone(gpu_devices.triton_device(0))  # XPU driver cannot launch on a CUDA tensor.
        self.torch.full.assert_not_called()
        self.assertEqual(gpu_devices.triton_device(2), {"index": 2, "name": "Intel test GPU", "backend": "triton",
                                                       "runtime": "xpu", "architecture": "pvc", "torch_device": "xpu:0"})
        self.torch.xpu.device.assert_called_once_with(0)
        self.torch.xpu.get_device_name.assert_called_once_with(0)
        self.assertEqual(self.torch.full.call_args.kwargs["device"], "xpu:0")
        self.kernel.__getitem__.return_value.assert_called_once_with(self.tensor)

    def test_xpu_bad_integer_result_is_unavailable(self):
        self.tensor.item.return_value = 0
        self.assertIsNone(gpu_devices.triton_device(2))
        self.kernel.__getitem__.return_value.assert_called_once_with(self.tensor)

    def test_cuda_and_hip_keep_indices_and_device_scope(self):
        self.torch.cuda.get_device_name.return_value = "test GPU"
        for index, backend in enumerate(("cuda", "hip")):
            self.target.backend, self.target.arch = backend, "arch"
            with self.subTest(backend=backend):
                result = gpu_devices.triton_device(index)
                self.assertEqual((result["index"], result["runtime"], result["torch_device"]),
                                 (index, backend, f"cuda:{index}"))
                self.torch.cuda.device.assert_called_with(index)


@unittest.skipUnless(native_backend._LIBRARY.is_file(), "ordinary kernel is not built")
class GPURankDispatch(unittest.TestCase):
    def test_native_rank_verifies_triton_witness(self):
        variables = list("xyz")
        terms = [((1, 1, 0), "1"), ((0, 0, 2), "1")]
        reference = native_backend.resolve_native(terms, variables, "intrinsic", "cpu")["certificate"]
        device = dict(index=0, backend="triton", name="test GPU")
        for pivots, expected in ((reference["minor_rows"], "triton"), ([0]*reference["rank"], "cpu_after_triton")):
            with self.subTest(pivots=pivots), patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"), \
                 patch.object(gpu_devices, "triton_device", return_value=device), \
                 patch.dict(sys.modules, {"engine.gpu_rank": SimpleNamespace(full_rank_rows=lambda *args: pivots)}):
                proof = native_backend.resolve_native(terms, variables, "intrinsic", "gpu")["certificate"]
                self.assertEqual(proof["rank_backend"], expected)
                self.assertEqual(proof["minor_rows"], reference["minor_rows"])
                if expected == "triton":
                    self.assertEqual(proof["device"], device)
                    self.assertIn("CPU independently verified", proof["verification"])

    def test_cpu_and_small_jobs_do_not_probe_triton(self):
        with patch.object(gpu_devices, "triton_device", side_effect=AssertionError("unexpected probe")) as probe, \
             patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton", RESOLUTION_GPU_MIN_COLS="512"):
            for policy in ("cpu", "auto"):
                proof = native_backend.resolve_native([((1, 1, 0), "1"), ((0, 0, 2), "1")],
                                                      list("xyz"), "intrinsic", policy)["certificate"]
                self.assertEqual(proof["rank_backend"], "cpu")
            probe.assert_not_called()

    def test_missing_triton_recovers_cpu_rank(self):
        with patch.object(gpu_devices, "triton_device", return_value=None), \
             patch.dict(os.environ, RESOLUTION_GPU_BACKEND="triton"):
            proof = native_backend.resolve_native([((1, 1, 0), "1"), ((0, 0, 2), "1")],
                                                  list("xyz"), "intrinsic", "gpu")["certificate"]
            self.assertEqual(proof["rank_backend"], "cpu_after_triton")
            self.assertIn("unavailable", proof["rank_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
