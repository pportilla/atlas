"""Reading a saved atlas decodes each chart's expressions in one native call."""
import copy
import unittest
from unittest.mock import patch

from engine import affine_kernel, atlas
from engine.native_library import _kernel
from engine.polynomial import InputError
from engine.resolve import resolve


@unittest.skipUnless(_kernel() is not None, "Build the native kernel with make.")
class RecordPrefetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = resolve(dict(polynomial="(x+y)^2+z^3+x^3", variables=list("xyz"), mode="embedded"))
        cls.saved = cls.record["atlas"]

    def read(self, data, *, prefetch):
        affine_kernel._prefetched.clear()
        atlas._read_polynomial.cache_clear()
        replacement = affine_kernel.prefetch if prefetch else (lambda variables, texts: None)
        with patch.object(affine_kernel, "prefetch", replacement):
            try:
                return atlas.from_json(copy.deepcopy(data), native=True)
            except InputError as error:
                return "InputError", str(error)

    def test_same_charts_with_one_decode_per_chart(self):
        calls = []
        original = affine_kernel._request

        def spy(lines):
            calls.append(lines[0])
            return original(lines)

        with patch.object(affine_kernel, "_request", spy):
            fast = self.read(self.saved, prefetch=True)
        self.assertEqual(fast, self.read(self.saved, prefetch=False))
        self.assertLessEqual(calls.count("decode_lines"), len(self.saved))
        self.assertGreater(calls.count("decode_lines"), 0)
        self.assertEqual(calls.count("decode"), 0)
        self.assertEqual(affine_kernel._prefetched, {})

    def test_invalid_expressions_report_the_single_decode_error(self):
        broken = copy.deepcopy(self.saved)
        chart = next(c for c in broken if c["region"] or c["fibre"])
        key = "region" if chart["region"] else "fibre"
        chart[key] = [*chart[key], "u1**", "u1+"]
        self.assertEqual(self.read(broken, prefetch=True), self.read(broken, prefetch=False))
        self.assertEqual(self.read(broken, prefetch=True)[0], "InputError")
        malformed = copy.deepcopy(self.saved)
        malformed[0]["components"] = [None]
        with self.assertRaises(Exception) as fast:
            self.read(malformed, prefetch=True)
        with self.assertRaises(Exception) as slow:
            self.read(malformed, prefetch=False)
        self.assertEqual((type(fast.exception), str(fast.exception)), (type(slow.exception), str(slow.exception)))


if __name__ == "__main__":
    unittest.main()
