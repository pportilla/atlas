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
    def test_rational_monomial_rows_in_chart_order(self):
        from unittest.mock import patch
        from engine import groebner as gb
        a, b = sp.Dummy("row"), sp.Dummy("row")
        expr = -sp.Rational(3, 5) * a**2 * b
        with patch.object(gb, "_expanded_qq_line", side_effect=AssertionError):
            self.assertEqual(_line((b, a), expr), "-3/5 1 2")
            self.assertEqual(_line((a, b), expr), "-3/5 2 1")
            self.assertEqual(_line((a, b), sp.Mul(0, a, evaluate=False)), "")
        self.assertIsNone(gb._monomial_row((a, b), a**-1))
        self.assertIsNone(gb._monomial_row((a, b), sp.Dummy("row")))

    def test_graded_lex_and_its_ordered_quotients(self):
        from unittest.mock import patch
        from engine import groebner as gb
        x,y,z = sp.symbols("x y z")
        for field, coefficient in ((sp.QQ, 2), (sp.QQ.algebraic_field(sp.sqrt(2)), sp.sqrt(2))):
            rows = (x*z-y**2, z**2-coefficient*x, x*y-3)
            ref = sp.groebner(rows, x,y,z, order="grlex", domain=field)
            f = x**2*z + y*z**3 + x*y
            expected = ref.reduce(f)
            with patch.object(sp, "groebner", side_effect=AssertionError):
                actual = gb.groebner(rows, (x,y,z), order="grlex", domain=field)
                self.assertEqual(actual.exprs, ref.exprs)
                self.assertEqual(actual.reduce(f), expected)

    def test_packed_basis_with_unpacked_remainder(self):
        if not hasattr(_kernel(), "atlas_wide_abi"):
            self.skipTest("the sparse QQ arithmetic ABI is unavailable")
        x,t = sp.symbols("x t")
        # A pre-existing small parameter relation receives a sparse
        # polynomial of high degree after a chart pulls it back.
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

    def test_monomial_deletions_between_fraction_free_reduction_steps(self):
        x, y, z = sp.symbols("x y z")
        # x^20 is settled before the fifteen y -> 1/2 steps. After z*y^16
        # disappears the settled and pending coefficients share a factor of 3,
        # which the kernel extracts at its eighth scaling.
        kept = x**20/11 + y**15/7 - sp.Rational(5, 13)
        f = sp.expand(z*y**16 + 3*(kept + z*sum(y**i for i in range(16))))
        remainder = sp.expand(3*kept.subs(y, sp.Rational(1, 2)))
        for order in ("lex", "grevlex"):
            for ideal, expected in (((z, 2*y - 1), remainder), ((), f), ((1,), sp.Integer(0))):
                with self.subTest(order=order, ideal=ideal):
                    native, reference = self.assert_same(ideal, (x, y, z), order)
                    self.assertEqual(reference.reduce(f)[1], expected)
                    self.assertEqual(native.reduce(f)[1], expected)
                    self.assertTrue(native.contains(f - expected))

    def test_membership_with_settled_terms_and_cancellation(self):
        from unittest.mock import patch

        x, y, z = sp.symbols("x y z")
        relation = 2*y - z - 1
        # y^2 first creates lower terms that later cancel. An x or a z term
        # can settle the answer before that complete reduction finishes.
        member = 4*y**2 - (z + 1)**2
        for ideal, cases in (
            ((relation,), ((x + member, x), (2*y, z + 1), (member, 0))),
            ((), ((sp.Integer(0), 0), (x, x))),
            ((1,), ((x + member, 0),)),
        ):
            basis, reference = self.assert_same(ideal, (x, y, z), "lex")
            for f, expected in cases:
                with self.subTest(ideal=ideal, f=f):
                    self.assertEqual(reference.reduce(f)[1], expected)
                    self.assertEqual(basis.reduce(f)[1], expected)
                    self.assertIs(basis.contains(f), expected == 0)

        # Retain the packed basis across polynomial requests beyond its range.
        basis = groebner((relation,), (x, y, z), order="lex")
        for f, expected in ((x**40000*relation, 0), (x**40000*(relation + 1), x**40000)):
            self.assertIs(basis.contains(f), expected == 0)
            self.assertEqual(basis.reduce(f)[1], expected)

        with patch.object(_kernel(), "atlas_groebner_contains", None, create=True):
            self.assertTrue(basis.contains(member))
            self.assertFalse(basis.contains(member + x))

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

    def test_native_limits(self):
        from unittest.mock import patch
        from engine import groebner as gb
        x, y = sp.symbols("x y")
        with patch.object(sp,"groebner",side_effect=AssertionError):
            # The unpacked FLINT ring accepts exponents above the packed limit.
            self.assertNotIsInstance(groebner((x**40000-y,), (x,y)), GroebnerBasis)
            with self.assertRaisesRegex(ArithmeticError,"specified ring or order"):
                groebner((x+y,), (x,y), domain=sp.GF(3))
            with self.assertRaisesRegex(ArithmeticError,"specified ring or order"):
                groebner((x+y,), (x,y), order=sp.polys.orderings.lex)
            with patch.object(gb,"_kernel",return_value=None):
                with self.assertRaisesRegex(ArithmeticError,"make groebner"):
                    groebner((x+y,), (x,y))

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

    def test_removable_content_keeps_the_exact_strategy(self):
        # The centered dense quartics are the eager-degree case. In lex order,
        # reducing y²+z² by B*y-z gives the single term (B²+1)*z² in the
        # lazy head loop. Its primitive coefficient is one.
        b = (1 << 100) + 7
        cases = [
            (3, 0, ["1 4 0 0;3 2 1 1;2 0 3 1;1 1 1 2;2 0 0 4",
                    "1 0 4 0;-2 3 1 0;1 2 0 2;3 0 1 3",
                    "1 0 0 4;2 1 3 0;1 0 2 2;3 3 0 1"]),
            (2, 1, [f"{b} 1 0;-1 0 1", "1 2 0;1 0 2"]),
        ]
        for variables, order, rows in cases:
            with self.subTest(order=order):
                modular, exact, _, route = self.bases(variables, order, rows)
                self.assertEqual(exact, modular)
                self.assertEqual(route, 0)

    def test_monomial_deletions_stay_within_the_direct_work_budget(self):
        # 820 terms disappear under x. Only the 41 powers of y remain.
        dense = ";".join(f"1 {i} {j}" for i in range(41) for j in range(41 - i))
        payload = f"2 0 0\n1 1 0\n{dense}\n".encode()
        handle = self.library.atlas_groebner(payload, len(payload))
        self.assertTrue(handle)
        try:
            expected = ";".join(f"1 0 {j}" for j in reversed(range(41)))
            self.assertEqual(self.library.atlas_groebner_text(handle).decode().splitlines(),
                             [expected, "1 1 0"])
            self.assertEqual(self.library.atlas_groebner_route(handle), 0)
        finally:
            self.library.atlas_groebner_free(handle)

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


@unittest.skipUnless(hasattr(_kernel(), "atlas_groebner_use_f4"), "rebuild the modular kernel")
class F4TraceRouteTests(unittest.TestCase):
    """The modular route traces a graded reverse lexicographic problem with F4
    matrices: of the homogenized generators (native/KERNEL.md, section 7.5),
    or of the generators themselves when the homogenized ideal would need a
    larger basis (7.6). The certified basis must be the one the Buchberger
    trace and the exact engine give."""

    def setUp(self):
        self.library = _kernel()
        self.library.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                                        ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double)]
        self.library.atlas_groebner_modular.restype = ctypes.c_void_p
        self.library.atlas_groebner_text.restype = ctypes.c_char_p
        self.library.atlas_groebner_use_f4.argtypes, self.library.atlas_groebner_use_f4.restype = [ctypes.c_int], None
        self.addCleanup(self.library.atlas_groebner_use_f4, 1)

    def modular(self, variables, order, lines, f4=True):
        payload = (f"{variables} {order} 0\n" + "".join(line + "\n" for line in lines)).encode()
        counts = (ctypes.c_long * 7)()
        self.library.atlas_groebner_use_f4(int(f4))
        handle = self.library.atlas_groebner_modular(payload, len(payload), counts, None)
        self.assertTrue(handle, self.library.atlas_groebner_error())
        try:
            return self.library.atlas_groebner_text(handle).decode(), self.library.atlas_groebner_modular_route(), counts
        finally:
            self.library.atlas_groebner_free(handle)

    def exact(self, variables, order, lines):
        payload = (f"{variables} {order} 0\n" + "".join(line + "\n" for line in lines)).encode()
        handle = self.library.atlas_groebner(payload, len(payload))
        self.assertTrue(handle, self.library.atlas_groebner_error())
        try:
            return self.library.atlas_groebner_text(handle).decode()
        finally:
            self.library.atlas_groebner_free(handle)

    def test_same_basis_as_the_buchberger_trace_and_the_exact_engine(self):
        cyclic5 = ["1 1 0 0 0 0;1 0 1 0 0 0;1 0 0 1 0 0;1 0 0 0 1 0;1 0 0 0 0 1",
                   "1 1 1 0 0 0;1 0 1 1 0 0;1 0 0 1 1 0;1 0 0 0 1 1;1 1 0 0 0 1",
                   "1 1 1 1 0 0;1 0 1 1 1 0;1 0 0 1 1 1;1 1 0 0 1 1;1 1 1 0 0 1",
                   "1 1 1 1 1 0;1 0 1 1 1 1;1 1 0 1 1 1;1 1 1 0 1 1;1 1 1 1 0 1", "1 1 1 1 1 1;-1 0 0 0 0 0"]
        cyclic6 = ["1 1 0 0 0 0 0;1 0 1 0 0 0 0;1 0 0 1 0 0 0;1 0 0 0 1 0 0;1 0 0 0 0 1 0;1 0 0 0 0 0 1",
                   "1 1 1 0 0 0 0;1 0 1 1 0 0 0;1 0 0 1 1 0 0;1 0 0 0 1 1 0;1 0 0 0 0 1 1;1 1 0 0 0 0 1",
                   "1 1 1 1 0 0 0;1 0 1 1 1 0 0;1 0 0 1 1 1 0;1 0 0 0 1 1 1;1 1 0 0 0 1 1;1 1 1 0 0 0 1",
                   "1 1 1 1 1 0 0;1 0 1 1 1 1 0;1 0 0 1 1 1 1;1 1 0 0 1 1 1;1 1 1 0 0 1 1;1 1 1 1 0 0 1",
                   "1 1 1 1 1 1 0;1 0 1 1 1 1 1;1 1 0 1 1 1 1;1 1 1 0 1 1 1;1 1 1 1 0 1 1;1 1 1 1 1 0 1",
                   "1 1 1 1 1 1 1;-1 0 0 0 0 0 0"]
        reimer4 = [";".join(f"{2 * (-1)**i} " + " ".join(str(k if j == i else 0) for j in range(4)) for i in range(4))
                   + ";-1 0 0 0 0" for k in range(2, 6)]
        # The same systems with one variable scaled by 2^70+1, and in seventh powers of the variables.
        scaled = [";".join(f"{2**70 + 1} {term[2:]}" if term[2] == "1" else term for term in row.split(";")) for row in cyclic6]
        powers = [";".join(" ".join([term.split()[0]] + [str(7 * int(e)) for e in term.split()[1:]]) for term in row.split(";"))
                  for row in reimer4]
        # The trace expected: 2 of the homogenized generators, 3 of the generators themselves.
        problems = [
            ("Katsura-4", 5, katsura(4), 2), ("Katsura-6", 7, katsura(6), 2),
            # Every degree of the homogenized ideal meets the Hilbert bound.
            ("cyclic-5", 5, cyclic5, 2),
            ("principal", 2, ["3 2 1;-1/7 0 0"], 2),
            ("wide coefficients", 2, [f"1 2 0;{2**100 + 1} 0 1;-1 0 0", f"1 1 1;{-(2**128) - 1} 0 0"], 2),
            ("rational coefficients", 3, ["1 2 0 0;2/3 0 1 1;-5/7 0 0 0", "1 1 1 0;-11/13 0 0 2", "1 0 0 3;-1 1 0 0;1/2 0 1 0"], 2),
            ("wide exponents", 2, ["1 40 0;-1 0 37;2 3 3", "1 5 30;-3 0 0"], 2),
            # Homogeneous generators are traced as they are, with or without the bound.
            ("monomials", 3, ["1 1 1 0", "1 1 0 1", "1 0 1 1"], 2),
            ("homogeneous, bound missed", 3, ["1 2 0 0;-1 1 1 0", "1 1 1 0;-1 0 2 0", "1 1 0 1;-1 0 1 1"], 2),
            # A degree misses the bound, or there is none, but the homogenized
            # basis is not larger than the basis of the ideal.
            ("twisted cubic", 3, ["1 1 0 1;-1 0 2 0", "1 0 1 0;-1 2 0 0", "1 0 0 1;-1 1 1 0"], 2),
            ("repeated generator", 2, ["1 2 0;-3 0 1", "2 2 0;-6 0 1", "1 1 2;7 0 0"], 2),
            ("overdetermined", 2, ["1 3 0;-2 1 1;5 0 0", "1 2 1;-1 0 2;1 1 0", "1 0 3;-1 1 0", "1 2 0;1 0 2;-4 0 0"], 2),
            # The homogenized ideal needs a larger basis: the generators are traced as they are.
            ("cyclic-6", 6, cyclic6, 3),
            ("Reimer-4", 4, reimer4, 3),
            ("cyclic-6, wide coefficients", 6, scaled, 3),
            ("Reimer-4, wide exponents", 4, powers, 3),
            ("the unit ideal", 2, ["1 2 0;1 0 1;1 0 0", "1 1 1;-1 0 0", "1 0 2;-1 1 0", "1 3 0;-2 0 0"], 3),
        ]
        for name, variables, lines, expected in problems:
            with self.subTest(name=name):
                basis, route, counts = self.modular(variables, 0, lines)
                self.assertEqual(route, expected)
                self.assertGreater(counts[3], 0)  # certified over Q
                old, old_route, _ = self.modular(variables, 0, lines, f4=False)
                self.assertEqual(old_route, 1)
                self.assertEqual(basis, old)
                self.assertEqual(basis, self.exact(variables, 0, lines))

    def test_trace_primes_that_must_not_be_trusted(self):
        # 2147483629 is the first prime tried. Modulo it the two generators
        # coincide and the trace has rank one; every replay shows rank two,
        # so the trace is discarded and the next prime gives (x^2, y^2).
        p = 2147483629
        lines = ["1 2 0;1 0 2", f"1 2 0;{p + 1} 0 2"]
        basis, route, counts = self.modular(2, 0, lines)
        self.assertEqual(route, 2)
        self.assertEqual(basis, "1 2 0\n1 0 2\n")
        self.assertEqual(basis, self.exact(2, 0, lines))
        self.assertGreaterEqual(counts[2], 2)  # traces
        self.assertGreater(counts[1], 0)       # rejected primes
        # The same prime divides a leading coefficient, then another coefficient.
        for lines in ([f"{p} 2 0;1 0 1;-1 0 0", "1 1 1;-2 0 0"], [f"1 2 0;{p} 0 1;-1 0 0", "1 1 1;-2 0 0"]):
            with self.subTest(lines=lines):
                basis, route, _ = self.modular(2, 0, lines)
                self.assertEqual(route, 2)
                self.assertEqual(basis, self.exact(2, 0, lines))
        # The same for a direct trace: modulo p the last generator of this
        # Reimer system is the first one, over Q their difference gives x1.
        reimer4 = [";".join(f"{2 * (-1)**i} " + " ".join(str(k if j == i else 0) for j in range(4)) for i in range(4))
                   + ";-1 0 0 0 0" for k in range(2, 6)]
        lines = reimer4 + [reimer4[0] + f";{p} 1 0 0 0"]
        basis, route, counts = self.modular(4, 0, lines)
        self.assertEqual(route, 3)
        self.assertEqual(basis, self.exact(4, 0, lines))
        self.assertGreaterEqual(counts[2], 2)
        self.assertGreater(counts[1], 0)
        # A prime dividing the leading coefficient of a generator.
        lines = [f"{p}" + reimer4[0][1:]] + reimer4[1:]
        basis, route, _ = self.modular(4, 0, lines)
        self.assertIn(route, (2, 3))
        self.assertEqual(basis, self.exact(4, 0, lines))

    def test_which_problems_keep_the_buchberger_trace(self):
        from engine import native_library as native
        lines = katsura(4)
        # Orders that do not compare degrees first.
        for order in (1, 2):
            with self.subTest(order=order):
                basis, route, _ = self.modular(5, order, lines)
                self.assertEqual(route, 1)
        # A unit among the generators, and no generator at all.
        self.assertEqual(self.modular(2, 0, ["1 1 0;-1 0 0", "3 0 0"])[:2], ("1 0 0\n", 1))
        self.assertEqual(self.modular(2, 0, [])[:2], ("", 1))
        # A solve that asks for a device keeps the trace its replay program comes from.
        reference, _, _ = self.modular(5, 0, lines)
        with native.replay_scope("gpu", backend="reference") as scope:
            basis, route, _ = self.modular(5, 0, lines)
            self.assertEqual((basis, route), (reference, 1))
            self.assertGreater(scope.report()["backends"]["reference"]["primes"], 0)
        with native.replay_scope("cpu") as scope:
            basis, route, _ = self.modular(5, 0, lines)
            self.assertEqual((basis, route), (reference, 2))
            report = scope.report()
            self.assertEqual(report["backend"], "cpu")
            self.assertGreater(report["primes"], 0)
            self.assertGreater(report["trace"]["backends"]["cpu"]["pairs"], 0)

    def test_profile_reports_the_trace_and_its_replays(self):
        profile = self.library.atlas_groebner_f4_profile
        profile.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_long),
                            ctypes.POINTER(ctypes.c_void_p)]
        lines = katsura(6)
        payload = ("7 0 0\n" + "".join(line + "\n" for line in lines)).encode()
        seconds, counts, text = (ctypes.c_double * 2)(), (ctypes.c_long * 9)(), ctypes.c_void_p()
        self.assertEqual(profile(payload, len(payload), 5, 2, seconds, counts, ctypes.byref(text)), 0)
        # 41 elements; a complete intersection: no combination is dependent
        # and every degree meets the Hilbert bound; all replays agree.
        self.assertEqual((counts[0], counts[4], counts[6], counts[7]), (41, 0, 1, 5))
        # The basis modulo the trace prime is the certified basis modulo that prime.
        prime = counts[8]
        modular = ctypes.string_at(text).decode()
        self.library.atlas_groebner_release(text)

        def residues(basis, convert):
            return sorted(tuple(sorted((tuple(map(int, term.split()[1:])), convert(term.split()[0]))
                                       for term in row.split(";"))) for row in basis.splitlines())

        def image(coefficient):
            numerator, _, denominator = coefficient.partition("/")
            return int(numerator) * pow(int(denominator or 1), -1, prime) % prime

        self.assertEqual(residues(modular, int), residues(self.exact(7, 0, lines), image))
        # Generators that miss the bound are traced as they are: the basis
        # modulo the prime is again the reduced basis itself.
        lines = ["1 1 0 1;-1 0 2 0", "1 0 1 0;-1 2 0 0", "1 0 0 1;-1 1 1 0", "1 3 0 0;-2 0 0 1;1 0 1 0"]
        payload = ("3 0 0\n" + "".join(line + "\n" for line in lines)).encode()
        self.assertEqual(profile(payload, len(payload), 3, 2, seconds, counts, ctypes.byref(text)), 0)
        self.assertEqual((counts[6], counts[7]), (0, 3))
        prime = counts[8]
        modular = ctypes.string_at(text).decode()
        self.library.atlas_groebner_release(text)
        self.assertEqual(residues(modular, int), residues(self.exact(3, 0, lines), image))
        self.assertEqual(profile(b"2 1 0\n1 1 0\n", 12, 0, 1, seconds, counts, None), -1)


if __name__ == "__main__":
    unittest.main()
