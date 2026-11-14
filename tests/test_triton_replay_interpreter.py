import os
import unittest


@unittest.skipUnless(os.getenv("TRITON_INTERPRET") == "1", "requires the Triton interpreter")
class InterpretedReplayTests(unittest.TestCase):
    def test_parallel_checks_and_scaling(self):
        import numpy as np
        from engine.gpu_replay import replay_reference, replay_triton
        # Two term blocks with nonzero residuals for the rejected prime, and
        # a zero leading coefficient which rejects another replay.
        terms, block = 35, 16
        indices = list(range(terms))
        check = [5, terms-1, *indices[1:]]
        code = np.array([1, terms, 2, 0, terms, *indices, *check,
                         6, terms, terms, *indices], dtype=np.uint32)
        primes = np.array([2147483629, 2147483587, 101, 7], dtype=np.uint32)
        base = np.zeros((len(primes), terms * 2), dtype=np.uint32)
        base[:, 0] = [1, 2, 0, 1]
        base[3, (1, 17, 34)] = 6
        expected, result = base.copy(), base.copy()
        wanted, status = np.empty(4, dtype=np.uint32), np.empty(4, dtype=np.uint32)
        replay_reference(code, 2 * terms, terms, primes, expected, wanted)
        replay_triton(code, 2 * terms, terms, primes, result, status, block=block, device="cpu")
        np.testing.assert_array_equal(status, [1, 1, 0, 0])
        np.testing.assert_array_equal(status, wanted)
        np.testing.assert_array_equal(result[status != 0], expected[status != 0])
