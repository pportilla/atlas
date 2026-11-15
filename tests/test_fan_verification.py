"""The one-call native fan verifier agrees with the per-chart verifier."""
import copy
import unittest
from unittest.mock import patch

from engine import monomial_atlas
from engine.native_library import _kernel
from engine.polynomial import parse
from engine.resolve import resolve
from engine.strata import InvalidChart


def outcome(record, polynomial, *, native, **options):
    """The verifier's answer, or the type and message of what it raised."""
    options.setdefault("integrals", False)
    replacement = monomial_atlas._verify_fan_natively if native else (lambda *a, **k: monomial_atlas._PER_CHART)
    with patch.object(monomial_atlas, "_verify_fan_natively", replacement):
        try:
            return monomial_atlas.fan_outputs(record, polynomial, **options)
        except (InvalidChart, ArithmeticError) as error:
            return type(error).__name__, str(error)


@unittest.skipUnless(_kernel() is not None and hasattr(_kernel(), "atlas_fan_verification_abi"),
                     "Build the native fan verifier with make.")
class NativeFanVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fans = {}
        for poly, names in (("x^5+z*y^3+z^15+t^4*x+t^8", "xyzt"), ("x^2+y^3+z^5", "xyz"),
                            ("x*y*z+x^4+y^4+z^4", "xyz"), ("x^2*y+y^4+z^3", "xyz")):
            record = resolve(dict(polynomial=poly, variables=list(names), mode="embedded"))
            record.pop("atlas", None)
            cls.fans[poly] = record, parse(poly, list(names))

    def assertSameVerification(self, record, polynomial, **options):
        calls = []
        original = monomial_atlas._verify_fan_natively

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with patch.object(monomial_atlas, "_verify_fan_natively", spy):
            fast = outcome(record, polynomial, native=True, **options)
        self.assertEqual(calls, [1])
        self.assertEqual(fast, outcome(record, polynomial, native=False, **options))
        return fast

    def test_fresh_fans_give_the_same_atlas_and_complex(self):
        for poly, (record, polynomial) in self.fans.items():
            with self.subTest(poly=poly):
                atlas, found = self.assertSameVerification(record, polynomial)
                self.assertEqual(len(atlas), len(record["charts"]))
                self.assertEqual(found["dual_complex"]["status"], "complete")
                self.assertSameVerification(record, polynomial, complex=False)

    def test_failures_are_reported_at_the_same_chart_with_the_same_message(self):
        record, polynomial = self.fans["x^2*y+y^4+z^3"]
        charts = record["charts"]
        exceptional = next(i for i, c in enumerate(charts) if max(c["ray_indices"]) >= 3)

        wrong_map = copy.deepcopy(record)
        chart = wrong_map["charts"][exceptional]
        chart["map"]["x"] = chart["map"]["x"] + "+" + chart["coordinates"][0]
        self.assertEqual(self.assertSameVerification(wrong_map, polynomial)[0], "InvalidChart")

        # A later chart names the same exceptional ray through a duplicate index.
        relabelled = copy.deepcopy(record)
        later = [i for i, c in enumerate(charts) if max(c["ray_indices"]) >= 3][-1]
        indices = relabelled["charts"][later]["ray_indices"]
        position = indices.index(max(indices))
        relabelled["fan"]["rays"].append(list(relabelled["fan"]["rays"][indices[position]]))
        indices[position] = len(relabelled["fan"]["rays"]) - 1
        self.assertEqual(self.assertSameVerification(relabelled, polynomial),
                         ("InvalidChart", "One exceptional ray must have a single global component label."))

        # A different equation keeps the fan but loses normal crossings.
        cusp = parse("x^2*y+y^4+z^3+x*z^2", list("xyz"))
        self.assertSameVerification(record, cusp)

    def test_saved_split_intrinsic_and_integral_requests_use_the_per_chart_verifier(self):
        record, polynomial = self.fans["x*y*z+x^4+y^4+z^4"]
        for options in (dict(intrinsic=True), dict(integrals=True)):
            with self.subTest(options=options), patch.object(monomial_atlas, "_verify_fan_natively",
                                                             side_effect=AssertionError):
                monomial_atlas.fan_outputs(record, polynomial, **options)


if __name__ == "__main__":
    unittest.main()
