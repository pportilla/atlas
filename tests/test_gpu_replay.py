import ctypes
import os
import random
import unittest

import numpy as np
import sympy as sp

from engine import gpu_replay
from engine.groebner import _kernel, _line


def scalar_replay(code, pool_size, work_size, prime, pool):
    """One prime, instruction by instruction, as native/opencl_groebner.cpp does it. Returns the status."""
    square, work, at = prime*prime, [0]*work_size, 0
    while at < len(code):
        op = code[at]
        if op == 1:
            for t in range(code[at + 1]):
                work[t] = 0
            at += 2
        elif op in (2, 3):
            src, count = code[at + 1], code[at + 2]
            for t in range(count):
                i = code[at + 3 + t]
                w = work[i] + (pool[src + t] if op == 2 else prime - pool[src + t])
                work[i] = w - square if w >= square else w
            at += 3 + count
        elif op == 4:
            column, src, count = code[at + 1], code[at + 2], code[at + 3]
            f = work[column] % prime
            work[column] = 0
            if f:
                for t in range(count):
                    i = code[at + 4 + t]
                    w = work[i] + (prime - f)*pool[src + t]
                    work[i] = w - square if w >= square else w
            at += 4 + count
        elif op == 5:
            count = code[at + 1]
            if any(work[code[at + 2 + t]] % prime for t in range(count)):
                return 0
            at += 2 + count
        else:
            target, count = code[at + 1], code[at + 2]
            base = work[code[at + 3]] % prime
            if not base:
                return 0
            inverse = pow(base, prime - 2, prime)
            pool[target] = 1
            for t in range(1, count):
                pool[target + t] = work[code[at + 3 + t]] % prime * inverse % prime
            at += 3 + count
    return 1


def random_program(source, length, rows):
    """Reductions of random rows: each new row is a combination of earlier ones, made monic and stored."""
    code, filled = [], rows*length
    for _ in range(6):
        columns = source.sample(range(length), length)
        code += [1, length]
        first = source.randrange(filled//length)*length
        code += [2, first, length, *columns]
        other = source.randrange(filled//length)*length
        code += [3, other, length, *source.sample(range(length), length)]
        pivot = source.randrange(filled//length)*length
        code += [4, columns[0], pivot, length, *source.sample(range(length), length)]
        if source.random() < 0.3:
            code += [5, 2, *source.sample(range(length), 2)]
        code += [6, filled, length, *source.sample(range(length), length)]
        filled += length
    return code, filled


class ReplayParts(unittest.TestCase):
    def test_reference_follows_the_instructions(self):
        source = random.Random(20261114)
        primes = [2147483629, 2147483587, 1000000007, 65537, 101, 7]
        for trial in range(20):
            length, rows = source.randrange(3, 9), source.randrange(2, 5)
            code, pool_size = random_program(source, length, rows)
            pools = np.zeros((len(primes), pool_size), dtype=np.uint32)
            for i, p in enumerate(primes):
                pools[i, :rows*length] = [source.randrange(p) for _ in range(rows*length)]
            expected, wanted = [], []
            for i, p in enumerate(primes):
                pool = [int(v) for v in pools[i]]
                wanted.append(scalar_replay(code, pool_size, length, p, pool))
                expected.append(pool)
            status = np.zeros(len(primes), dtype=np.uint32)
            gpu_replay.replay_reference(np.array(code, dtype=np.uint32), pool_size, length,
                                        np.array(primes, dtype=np.uint32), pools, status)
            with self.subTest(trial=trial):
                self.assertEqual(list(status), wanted)
                for i in range(len(primes)):
                    if wanted[i]:
                        self.assertEqual([int(v) for v in pools[i]], expected[i])


@unittest.skipUnless(_kernel() and os.getenv("ATLAS_GROEBNER") != "sympy", "the native kernel is not in use")
class ReplayThroughTheKernel(unittest.TestCase):
    """The kernel traces a Groebner computation, replays it for many primes on
    its own threads and through the backend, and counts the primes on which
    the two differ."""

    def compare(self, backend, count=24):
        library = _kernel()
        library.atlas_groebner_replay_compare.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_long)]
        library.atlas_groebner_replay_compare.restype = ctypes.c_long
        x, y, z, w = sp.symbols("x y z w")
        problems = [
            ((x, y, z), [x**2*y - z**3 + 1, x*y**2 + z**2 - x, x**3 - y*z + 2]),
            ((x, y, z, w), [x + y + z + w, x*y + y*z + z*w + w*x, x*y*z + y*z*w + z*w*x + w*x*y, x*y*z*w - 1]),
            ((x, y, z), [3*x**3 - 7*y*z + 5, x*y - 11*z**2 + 2*y, 13*x*z**2 - y**3 + x]),
        ]
        self.assertTrue(gpu_replay.use(library, backend))
        try:
            for gens, ideal in problems:
                payload = (f"{len(gens)} 0 0\n" + "".join(_line(gens, f) + "\n" for f in ideal)).encode("ascii")
                seconds, sizes = (ctypes.c_double*3)(), (ctypes.c_long*3)()
                with self.subTest(ideal=ideal):
                    self.assertEqual(library.atlas_groebner_replay_compare(payload, len(payload), count, seconds, sizes), 0)
                    self.assertGreater(sizes[0], 0)
        finally:
            library.atlas_groebner_use_batch(None)

    def test_reference_backend(self):
        self.compare("reference")

    @unittest.skipUnless(gpu_replay.triton_available(), "Triton needs PyTorch with CUDA and the triton package")
    def test_triton_backend(self):
        self.compare("triton")


if __name__ == "__main__":
    unittest.main()
