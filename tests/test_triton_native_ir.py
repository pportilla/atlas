"""Compile the native kernel definitions even on hosts without a GPU driver."""
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.native_library import _kernel
from engine.triton_ir import native_ir


@unittest.skipUnless(find_spec("triton") and _kernel(), "Requires Triton and the native kernel")
class NativeTritonIR(unittest.TestCase):
    def test_device_modules_compile(self):
        from triton.backends import backends
        from triton.backends.compiler import GPUTarget
        from triton.compiler import compile

        targets = [target for backend, target in (("nvidia", GPUTarget("cuda", 80, 32)),
                                                   ("amd", GPUTarget("hip", "gfx942", 64)))
                   if backend in backends]
        if not targets:
            self.skipTest("Requires an offline NVIDIA or AMD compiler")
        modules = [("trace_probe", ()), ("trace_levels", (2147483629,)),
                   ("rank_small", (1000000009, 32, 5, 1)), ("crt_pair", (256,)), ("determinant_small", (6,)),
                   ("rank_pivot", (1000000007, 64, 32, 0)), ("rank_eliminate", (1000000007, 8, 32, 1)),
                   ("strict_charts", (3, 256)), ("ray_pairs", (3, 256)), ("gather_charts", (3, 256)),
                   ("monomial_witness", (6, 64)), ("replay_axpy", (256, 2)), ("replay_axpy", (256, 3))]
        modules.extend((name, (128,)) for name in ("poly_add_words", "poly_outer_product",
                       "poly_point_product", "poly_outer_words", "poly_point_words", "poly_dense_convolution"))
        modules.extend([("poly_root_screen", (128,)), ("poly_binomial_steps", (2,128))])
        modules.extend(("residue_batch", (action, 128)) for action in range(9))
        modules.extend(("finite_field_batch", (action,2,32)) for action in range(12))
        modules.append(("replay_clear", (256,)))
        for operation in ("pivot", "axpy", "check", "inverse", "scale"):
            modules.append(("replay_" + operation, (256,)))
        with TemporaryDirectory() as tmp:
            for name, parameters in modules:
                path = Path(tmp, name + ".ttir")
                path.write_text(native_ir(name, parameters), encoding="ascii")
                for target in targets:
                    with self.subTest(kernel=name, parameters=parameters, target=target):
                        options = {"num_warps": 2 if name == "trace_levels" else 4}
                        compiled = compile(str(path), target=target, options=options)
                        self.assertTrue(compiled.kernel)
