import ctypes
import json
import os
from pathlib import Path
import sys
import unittest

import sympy as sp
from sympy import QQ
from sympy.polys.polytools import GroebnerBasis

from engine.groebner import _kernel, _line, colon, groebner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


@unittest.skipUnless(_kernel() and os.getenv("ATLAS_GROEBNER") != "sympy", "the native Groebner kernel is not in use")
class GroebnerKernelTests(unittest.TestCase):
    def test_graded_lex_and_its_ordered_quotients(self):
        from unittest.mock import patch
        from engine import groebner as gb
        x,y,z = sp.symbols("x y z")
        for field, coefficient in ((sp.QQ, 2), (sp.QQ.algebraic_field(sp.sqrt(2)), sp.sqrt(2))):
            rows = (x*z-y**2, z**2-coefficient*x, x*y-3)
            ref = sp.groebner(rows, x,y,z, order="grlex", domain=field)
            f = x**2*z + y*z**3 + x*y
            expected = ref.reduce(f)
            with patch.object(gb, "_MODE", "native"), patch.object(sp, "groebner", side_effect=AssertionError):
                actual = gb.groebner(rows, (x,y,z), order="grlex", domain=field)
                self.assertEqual(actual.exprs, ref.exprs)
                self.assertEqual(actual.reduce(f), expected)

    def test_packed_basis_with_unpacked_remainder(self):
        from unittest.mock import patch
        from engine import groebner as gb
        if not hasattr(_kernel(), "atlas_wide_abi"):
            self.skipTest("the sparse QQ arithmetic ABI is unavailable")
        x,t = sp.symbols("x t")
        # A pre-existing small parameter relation receives a sparse
        # polynomial of high degree after a chart pulls it back.
        with patch.object(gb, "_MODE", "native"), patch.object(gb, "CHECK", False):
            basis = groebner((t**2-2,), (x,t))
            large = x**(10**10)
            self.assertEqual(basis.reduce((t**2+t)*large)[1], t*large+2*large)
            self.assertTrue(basis.contains(large*(t**2-2)))

    def test_compound_qq_codec_with_colliding_names_and_invalid_input(self):
        from engine.blowup_ideal import _expressions
        from engine.groebner import _poly
        from sympy.polys.polyerrors import BasePolynomialError
        a,b = sp.Dummy("v"), sp.Dummy("v")
        v = (a,b)
        f = (a+2*b)**6 + a*(a-b)**3/2
        expected = sp.Poly(f, *v, domain=QQ)
        self.assertEqual(_poly(v, _line(v, f))[0], expected)
        self.assertEqual(_expressions(v, (f,0), QQ), (expected.as_expr(),))
        self.assertIn("*", _line(v, (a+b)*(a-b)))
        # The QQ contract rejects a non-polynomial fraction even when its
        # numerator and denominator cancel in a rational function field.
        with self.assertRaises(BasePolynomialError):
            _expressions(v, (f+(a**2-b**2)/(a-b),), QQ)
        imaginary = sp.Symbol("I")
        with self.assertRaises(BasePolynomialError):
            _line((imaginary,a), (imaginary+a)**3 + sp.I)

    def assert_same(self, ideal, gens, order):
        native = groebner(ideal, gens, order=order)
        self.assertNotIsInstance(native, GroebnerBasis)  # the kernel answered
        reference = sp.groebner(ideal, *gens, order=order, domain=QQ)
        self.assertEqual(list(native.exprs), list(reference.exprs))
        self.assertEqual(native.polys, list(reference.polys))
        return native, reference

    def test_reduced_bases_agree_with_sympy_in_both_orders(self):
        x, y, z, t = sp.symbols("x y z t")
        half = sp.Rational(1, 2)
        ideals = [((x**2 + y, x*y - 1), (x, y)),
                  ((x**2 + y**2 + z**2 - 1, x*y - sp.Rational(2, 3)*z, z**3 - x), (x, y, z)),
                  (((x**2 + half*y)*(y*z - 3)*(x + y + z), (x - y)*(x + y) + z, x*y*z - 1), (x, y, z)),
                  ((x*y, 1 - t*x), (t, x, y)),
                  ((x**3 - y**2, x**2*z - y**3, 7*z**2 - x*y + half), (x, y, z)),
                  ((x - 1, sp.Integer(3)), (x, y)), ((), (x, y))]
        for ideal, gens in ideals:
            for order in ("grevlex", "lex"):
                with self.subTest(ideal=ideal, order=order):
                    self.assert_same(ideal, gens, order)

    def test_remainders_and_membership_agree_with_sympy(self):
        x, y, z = sp.symbols("x y z")
        ideal = (x**2 + y**2 + z**2 - 1, x*y - sp.Rational(2, 3)*z, z**3 - x)
        native, reference = self.assert_same(ideal, (x, y, z), "grevlex")
        for f in (sp.Rational(5, 7)*x**4*y + z**5 - 3, (x + sp.Rational(2, 3)*y)*(z**3 - x)*(y - 1) + 5,
                  x*ideal[1] + ideal[2], sp.Integer(0), x):
            with self.subTest(f=f):
                self.assertEqual(native.reduce(f)[1], reference.reduce(f)[1])
                self.assertEqual(native.contains(f), reference.contains(f))

    def test_a_declared_groebner_prefix_gives_the_same_basis(self):
        x, y, z = sp.symbols("x y z")
        known = tuple(groebner((x**3 - y**2, x*y - z**2), (x, y, z)).exprs)
        extra = (x*z - y + 1, y**2*z - 2)
        self.assertEqual(list(groebner((*known, *extra), (x, y, z), known=len(known)).exprs),
                         list(sp.groebner((*known, *extra), x, y, z, order="grevlex", domain=QQ).exprs))

    def test_ideal_quotients_agree_with_sympy(self):
        x, y, z = sp.symbols("x y z")
        cases = [((x*y,), (x,), (x, y)), ((x*y, y**2), (x, y), (x, y)),
                 ((x**2*y - z, x*z**2 - y**2), (x*y - z, z**2), (x, y, z)),
                 ((x*y*z, x**2*(y - 1)), (x*(y - 1)*(z + 2), y**2*z), (x, y, z)),
                 ((x**3 - y**2, sp.Rational(1, 2)*x*y - z), (y*z, x + 3*z), (x, y, z)), ((x,), (x,), (x, y))]
        for ideal, by, gens in cases:
            with self.subTest(ideal=ideal, by=by):
                ring = QQ.old_poly_ring(*gens, order="grevlex")
                reference = [ring.to_sympy(f) for f in ring.ideal(*ideal).quotient(ring.ideal(*by)).gens]
                self.assertEqual(list(colon(ideal, by, gens)),
                                 list(sp.groebner(reference, *gens, order="grevlex", domain=QQ).exprs))

    def test_native_limits_and_explicit_symbolic_fallback(self):
        from unittest.mock import patch
        from engine import groebner as gb
        x, y = sp.symbols("x y")
        with patch.object(gb,"_MODE","native"), patch.object(sp,"groebner",side_effect=AssertionError):
            # The unpacked FLINT ring accepts exponents above the packed limit.
            self.assertNotIsInstance(groebner((x**40000-y,), (x,y)), GroebnerBasis)
            with self.assertRaisesRegex(ArithmeticError,"ATLAS_GROEBNER=sympy"):
                groebner((x+y,), (x,y), domain=sp.GF(3))
            with self.assertRaisesRegex(ArithmeticError,"ATLAS_GROEBNER=sympy"):
                groebner((x+y,), (x,y), order=sp.polys.orderings.lex)
            with patch.object(gb,"_kernel",return_value=None):
                with self.assertRaisesRegex(ArithmeticError,"make groebner"):
                    groebner((x+y,), (x,y))
        for mode in ("sympy","check"):
            with patch.object(gb,"_MODE",mode), patch.object(gb,"_kernel",return_value=None):
                self.assertEqual(groebner((x+y,), (x,y), domain=sp.GF(3)).exprs,[x+y])

    def test_recorded_verifier_bases(self):
        # Every basis the ideal-blowup verifier asked for on the benchmark germs.
        from groebner import kernel_times
        corpus = json.loads((ROOT / "benchmarks" / "groebner-corpus.json").read_text())
        self.assertEqual(kernel_times(corpus, 1)[1], 0)


def katsura(n):
    """The Katsura ideal in n + 1 variables, as kernel lines."""
    rows = []
    for m in range(n):
        terms = {}
        for i in range(-n, n + 1):
            if abs(m - i) <= n:
                e = [0] * (n + 1)
                e[abs(i)] += 1
                e[abs(m - i)] += 1
                terms[tuple(e)] = terms.get(tuple(e), 0) + 1
        e = [0] * (n + 1)
        e[m] = 1
        terms[tuple(e)] = terms.get(tuple(e), 0) - 1
        rows.append(terms)
    terms = {}
    for i in range(-n, n + 1):
        e = [0] * (n + 1)
        e[abs(i)] = 1
        terms[tuple(e)] = terms.get(tuple(e), 0) + 1
    terms[tuple([0] * (n + 1))] = -1
    rows.append(terms)
    return [";".join(f"{c} {' '.join(map(str, e))}" for e, c in row.items() if c) for row in rows]


@unittest.skipUnless(_kernel(), "the native Groebner kernel is not built")
class ModularRouteTests(unittest.TestCase):
    """The route through modular images must return the same reduced basis as
    the exact one: it is certified over Q (native/KERNEL.md, section 7)."""

    def setUp(self):
        self.library = _kernel()
        self.library.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                                        ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double)]
        self.library.atlas_groebner_modular.restype = ctypes.c_void_p
        self.library.atlas_groebner_route.argtypes = [ctypes.c_void_p]
        self.library.atlas_groebner_hilbert.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.library.atlas_groebner_hilbert.restype = ctypes.c_void_p
        self.library.atlas_hilbert_numerator.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.library.atlas_hilbert_numerator.restype = ctypes.c_void_p

    def bases(self, variables, order, lines):
        payload = (f"{variables} {order} 0\n" + "".join(line + "\n" for line in lines)).encode()
        counts = (ctypes.c_long * 7)()
        modular = self.library.atlas_groebner_modular(payload, len(payload), counts, None)
        self.assertTrue(modular)
        exact = self.library.atlas_groebner(payload, len(payload))
        texts = [self.library.atlas_groebner_text(handle).decode() for handle in (modular, exact)]
        used = self.library.atlas_groebner_route(exact)
        for handle in (modular, exact):
            self.library.atlas_groebner_free(handle)
        return texts[0], texts[1], counts, used

    def test_same_basis_as_the_exact_route(self):
        x, y, z = sp.symbols("x y z")
        gens = (x, y, z)
        ideals = [(x**2 + y**2 + z**2 - 1, x*y - sp.Rational(2, 3)*z, z**3 - x), (x**3 - y**2, x**2*z - y**3, 7*z**2 - x*y + 1),
                  (x*y - z, x*z - y**2 + sp.Rational(1, 5), y*z**2 - x - 3), (x - 1, sp.Integer(3)), (x*y, x*z, y*z),
                  (x**2*y - z**3, x*y**2 - z**3 + z)]
        problems = [(3, order, [_line(gens, f) for f in ideal]) for ideal in ideals for order in (0, 1)]
        problems += [(4, 0, katsura(3)), (4, 1, katsura(3)), (5, 0, katsura(4)), (5, 1, katsura(4)), (2, 0, [])]
        for variables, order, lines in problems:
            with self.subTest(lines=lines, order=order):
                modular, exact, _, _ = self.bases(variables, order, lines)
                self.assertEqual(modular, exact)

    def test_primes_that_must_not_be_trusted(self):
        # 2147483629 is the first prime the route tries. It divides a leading
        # coefficient, then another coefficient, of these generators.
        p = 2147483629
        for lines in ([f"{p} 2 0;1 0 1;-1 0 0", "1 1 1;-2 0 0"], [f"1 2 0;{p} 0 1;-1 0 0", "1 1 1;-2 0 0"]):
            with self.subTest(lines=lines):
                modular, exact, _, _ = self.bases(2, 0, lines)
                self.assertEqual(modular, exact)
        # (x, z) meet (1 - p*y, z - 1), with generators none of whose
        # coefficients p divides. Modulo p the ideal becomes (x, z): the set
        # {x, z} is a Groebner basis, contains the ideal and has the leading
        # monomials seen modulo p, yet is wrong. Homogenizing excludes it.
        lines = ["1 0 0 2;-1 0 0 1", "1 1 0 1;-1 1 0 0", f"1 0 0 1;1 0 1 1;{-(p + 1)} 0 1 2", f"1 1 0 0;1 1 1 0;{-(p + 1)} 1 1 1"]
        for order in (0, 1):
            with self.subTest(order=order):
                modular, exact, _, _ = self.bases(3, order, lines)
                self.assertEqual(modular, exact)
                self.assertNotEqual(sorted(modular.split("\n")[:-1]), ["1 0 0 1", "1 1 0 0"])

    def test_routes_taken_when_coefficients_swell(self):
        # Degrees compared first: modular images.
        modular, exact, counts, used = self.bases(7, 0, katsura(6))
        self.assertEqual(used, 1)
        self.assertEqual(modular, exact)
        # A lexicographic order: the Hilbert function of the homogenized ideal.
        modular, exact, counts, used = self.bases(6, 1, katsura(5))
        self.assertEqual(used, 2)
        self.assertEqual(modular, exact)
        self.assertEqual(len(exact.split("\n")) - 1, 6)
        self.assertGreater(counts[0], 50)  # primes: the coefficients have thousands of bits

    def test_the_hilbert_route_gives_the_same_basis(self):
        x, y, z = sp.symbols("x y z")
        gens = (x, y, z)
        ideals = [(x**2 + y**2 + z**2 - 1, x*y - sp.Rational(2, 3)*z, z**3 - x), (x*y - z, x*z - y**2 + sp.Rational(1, 5), y*z**2 - x - 3),
                  (x - 1, sp.Integer(3)), (x*y, x*z, y*z), (x**2*y - z**3, x*y**2 - z**3 + z), (x*y*z - 1, x + y)]
        problems = [(3, [_line(gens, f) for f in ideal]) for ideal in ideals] + [(4, katsura(3)), (5, katsura(4)), (2, [])]
        for variables, lines in problems:
            with self.subTest(lines=lines):
                payload = (f"{variables} 1 0\n" + "".join(line + "\n" for line in lines)).encode()
                handles = [self.library.atlas_groebner_hilbert(payload, len(payload)), self.library.atlas_groebner(payload, len(payload))]
                self.assertTrue(all(handles))
                texts = [self.library.atlas_groebner_text(handle) for handle in handles]
                for handle in handles:
                    self.library.atlas_groebner_free(handle)
                self.assertEqual(texts[0], texts[1])
        # The route is for orders that do not compare degrees first.
        payload = b"2 0 0\n1 1 0\n"
        self.assertFalse(self.library.atlas_groebner_hilbert(payload, len(payload)))

    def test_hilbert_numerators_count_standard_monomials(self):
        from itertools import product
        from math import comb
        import random
        random.seed(7)
        for _ in range(60):
            m = random.randint(1, 4)
            generators = [tuple(random.randint(0, 3) for _ in range(m)) for _ in range(random.randint(0, 6))]
            text = (f"{m}\n" + "".join(" ".join(map(str, g)) + "\n" for g in generators)).encode()
            answer = self.library.atlas_hilbert_numerator(text, len(text))
            self.assertTrue(answer)
            numerator = list(map(int, ctypes.string_at(answer).decode().split()))
            self.library.atlas_groebner_release(answer)
            for d in range(8):
                standard = sum(1 for e in product(range(d + 1), repeat=m) if sum(e) == d
                               and not any(all(a >= b for a, b in zip(e, g)) for g in generators))
                value = sum(c*comb(m - 1 + d - j, m - 1) for j, c in enumerate(numerator) if j <= d)
                self.assertEqual(value, standard, (generators, d))

    def test_opencl_replay_agrees_with_the_host(self):
        plugin = ROOT / "native" / "build" / ("libgroebner_opencl.dylib" if sys.platform == "darwin" else "libgroebner_opencl.so")
        if not plugin.is_file():
            self.skipTest("the OpenCL plugin is not built (make gpu)")
        device = ctypes.CDLL(str(plugin))
        if not device.atlas_replay_opencl_available():
            self.skipTest("no OpenCL GPU")
        library = ctypes.CDLL(str(ROOT / "native" / "build" / plugin.name.replace("_opencl", "")))
        library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
        library.atlas_groebner_use_batch(ctypes.cast(device.atlas_replay_opencl, ctypes.c_void_p))
        library.atlas_groebner_replay_compare.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
                                                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_long)]
        library.atlas_groebner_replay_compare.restype = ctypes.c_long
        payload = ("5 0 0\n" + "".join(line + "\n" for line in katsura(4))).encode()
        seconds, sizes = (ctypes.c_double * 3)(), (ctypes.c_long * 3)()
        try:
            self.assertEqual(library.atlas_groebner_replay_compare(payload, len(payload), 64, seconds, sizes), 0)
        finally:
            library.atlas_groebner_use_batch(None)


if __name__ == "__main__":
    unittest.main()
