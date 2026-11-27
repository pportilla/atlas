"""Newton modifications in coordinates adapted to the singular locus.

A fan of the input coordinates can have a boundary ray over a coordinate
subspace of the hypersurface that is not singular, or a face with a repeated
factor that is a coordinate after a substitution. The germs are fourfolds and
fivefolds on which the quick search used to end with a curve of failed
crossings on an orbit. The expected zeta functions are computed here from
the theorem of Sebastiani and Thom and the formula of Milnor and Orlik.
"""
from fractions import Fraction
from math import gcd, lcm
import unittest

import sympy

from engine.kernel_json import kernel_json
from engine.newton_faces import face_search
from engine.newton_frames import METHOD, adapted_frame, adapted_polynomial, compose_map
from engine.polynomial import encode_sparse_row, format_poly, parse, sparse_row
from engine.resolution_zetas import replay_zetas
from engine.resolve import resolve
from engine.strata import InvalidChart, InvalidZetaCertificate

FOURFOLD = ["x", "y", "z", "t", "u"]
FIVEFOLD = ["x", "y", "z", "t", "u", "v"]


def quick(polynomial, names, mode="embedded"):
    request = dict(polynomial=polynomial, variables=names, mode=mode, search="quick", compute="cpu")
    if mode == "embedded":
        request["zetas"] = ["monodromy", "topological"]
    return resolve(request)


def frame(polynomial, names):
    with face_search(False):
        found = adapted_frame(parse(polynomial, names), names)
    if found is None:
        return None
    steps, adapted, fan = found
    return [(step["coordinate"], step["offset"]) for step in steps], adapted, fan


def steps_of(record):
    translation = record["certificate"]["coordinate_translation"]
    return [(step["coordinate"], step["offset"]) for step in translation["steps"]]


def cyclic_product(left, right):
    """Tensor product of sums of regular representations of cyclic groups."""
    result = {}
    for a, m in left.items():
        for b, k in right.items():
            order = lcm(a, b)
            result[order] = result.get(order, 0) + m * k * gcd(a, b)
    return {order: m for order, m in result.items() if m}


def brieskorn_pham(exponents):
    """Milnor and Orlik: the monodromy on the vanishing cohomology of sum x_i^(a_i)."""
    result = {1: 1}
    for a in exponents:
        result = cyclic_product(result, {a: 1, 1: -1})
    return result


def join_zeta(exponents):
    """Zeta factors of g + sum y_i^(a_i), for a monomial g with at least two variables.

    The Milnor fibre of g is a union of tori permuted by the monodromy, with
    alternating character -1 on its reduced cohomology. By Sebastiani and Thom
    the reduced cohomology of the join is that of the Brieskorn-Pham germ,
    shifted by the dimension of the first fibre plus one.
    """
    sign = -1 if len(exponents) % 2 == 0 else 1
    character = {order: sign * m for order, m in brieskorn_pham(exponents).items()}
    character[1] = character.get(1, 0) + 1
    return {order: -m for order, m in character.items() if m}


def isolated_zeta(exponents):
    sign = 1 if len(exponents) % 2 else -1
    character = {order: sign * m for order, m in brieskorn_pham(exponents).items()}
    character[1] = character.get(1, 0) + 1
    return {order: -m for order, m in character.items() if m}


def factors(record):
    return {row["multiplicity"]: row["exponent"] for row in record["monodromy_zeta"]["factors"]}


def weighted_milnor_number(weights):
    value = Fraction(1)
    for weight in weights:
        value *= 1 / Fraction(weight) - 1
    return value


class NativeFrameSteps(unittest.TestCase):
    def test_boundary_image_in_the_smooth_locus(self):
        # The x-axis lies on x^2*y+z^2+t^3+u^4 and is not singular: the ray
        # (0,12,6,4,3) of the fan would blow it up. The y-axis is singular.
        steps, adapted, fan = frame("x^2*y+z^2+t^3+u^4", FOURFOLD)
        self.assertEqual(steps, [("y", "x")])
        self.assertEqual(adapted, parse("x^3+x^2*y+z^2+t^3+u^4", FOURFOLD))
        self.assertFalse(fan["isomorphism_off_origin"])
        self.assertTrue(all(fan["rays"][i][1] == 0 for i in fan["boundary_rays"]))

    def test_power_keeps_the_newton_polyhedron(self):
        # x = X + y would turn x^3 into (X+y)^3. The normal (7,2) of the
        # edge from x^3 to x*y^7 needs k*2 > 7.
        p = parse("x^3+x*y^7+z^3+t^3+u^3", FOURFOLD)
        normals = [[7, 2, 7, 7, 7], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        choices = kernel_json("atlas_resolution_candidate", "boundary_translations", width=5,
                              row=encode_sparse_row(p), subspaces=[[1]],
                              normals=[[str(v) for v in ray] for ray in normals])
        self.assertEqual([(c["axis"], format_poly(sparse_row(c["shift"]), FOURFOLD)) for c in choices], [(0, "y^4")])
        adapted = sparse_row(choices[0]["adapted"])
        self.assertEqual(adapted, parse("(x+y^4)^3+(x+y^4)*y^7+z^3+t^3+u^3", FOURFOLD))

    def test_subspaces_in_order_of_dimension(self):
        # The planes {x=t=u=0}, ... lie on x*y*z+t^2+u^3; only the axes are singular.
        steps, adapted, fan = frame("x*y*z+t^2+u^3", FOURFOLD)
        # One step for each of the planes, by a power of the product of its free coordinates.
        self.assertEqual([coordinate for coordinate, _ in steps], ["z", "y", "x"])
        for (_, offset), free in zip(steps, (("x", "y"), ("x", "z"), ("y", "z"))):
            (exponent, coefficient), = parse(offset, FOURFOLD).items()
            self.assertEqual(coefficient, 1)
            self.assertEqual({FOURFOLD[i] for i, k in enumerate(exponent) if k}, set(free))
            self.assertEqual(len({k for k in exponent if k}), 1)
        self.assertEqual(adapted, adapted_polynomial(parse("x*y*z+t^2+u^3", FOURFOLD), FOURFOLD, [
            {"coordinate": coordinate, "offset": offset} for coordinate, offset in steps]))
        # The boundary rays that are left lie over the three singular axes.
        for index in fan["boundary_rays"]:
            self.assertEqual(sum(value == 0 for value in fan["rays"][index]), 1)

    def test_least_subspaces_outside_the_singular_locus(self):
        # x*y+z^2*t+u^3*v is singular along the (t,v)-plane. A ray over the
        # (y,v)-plane leaves the singular locus because of the y-axis alone.
        from engine.fan_centres import smooth_ray_images
        equation = "x*y+z^2*t+u^3*v"
        self.assertEqual(smooth_ray_images(equation, FIVEFOLD, [(1, 0, 1, 1, 1, 0)]), [[1]])
        self.assertEqual(smooth_ray_images(equation, FIVEFOLD, [(1, 1, 1, 0, 1, 0)]), [])
        self.assertEqual(smooth_ray_images(equation, FIVEFOLD, [(0, 0, 1, 1, 1, 1)]), [[0], [1]])

    def test_repeated_face_factor(self):
        steps, adapted, _ = frame("t^4+u^4+y^3+z^3+x^2+2*x*y+y^2", FOURFOLD)
        self.assertEqual(steps, [("x", "-y")])
        self.assertEqual(adapted, parse("x^2+y^3+z^3+t^4+u^4", FOURFOLD))

    def test_two_factors_of_a_degenerate_face(self):
        # (x+2*z^2)*(x+y+z^2) is singular in its torus where both factors vanish.
        p = parse("(x+2*z^2)*(x+y+z^2)+t^5+u^7", FOURFOLD)
        face = parse("(x+2*z^2)*(x+y+z^2)", FOURFOLD)
        choices = kernel_json("atlas_resolution_candidate", "face_translations", width=5,
                              row=encode_sparse_row(p), face=encode_sparse_row(face))
        shifts = {(c["axis"], format_poly(sparse_row(c["shift"]), FOURFOLD)) for c in choices}
        self.assertIn((0, "-2*z^2"), shifts)
        self.assertIn((1, "-z^2-x"), shifts)
        first = next(c for c in choices if c["axis"] == 0 and format_poly(sparse_row(c["shift"]), FOURFOLD) == "-2*z^2")
        self.assertEqual(sparse_row(first["adapted"]), parse("x*(x+y-z^2)+t^5+u^7", FOURFOLD))
        # A single factor without multiplicity is left alone.
        smooth = parse("x+y+z^2", FOURFOLD)
        self.assertEqual(kernel_json("atlas_resolution_candidate", "face_translations", width=5,
                                     row=encode_sparse_row(p), face=encode_sparse_row(smooth)), [])

    def test_quadratic_discriminant_certifies_irreducible_sections(self):
        from engine.atlas_torus import _absolute_torus_section
        names = ["a", "b", "c", "d"]
        def irreducible(text):
            return _absolute_torus_section(tuple(sorted(parse(text, names).items())))
        # d^2 + c*A*(c*A+b) with A = b*c+a: the discriminant is square-free.
        self.assertTrue(irreducible("b^2*c^4+2*a*b*c^3+a^2*c^2+b^2*c^2+a*b*c+d^2"))
        self.assertTrue(irreducible("a^2+b^2*c^2+b^2*d^4"))
        # a^2+b^2*c^2 splits over Q(i): its discriminant is a square.
        self.assertFalse(irreducible("a^2+b^2*c^2"))

    def test_frame_without_obstruction_is_left_alone(self):
        self.assertIsNone(frame("x^2+y^3+z^3+t^4+u^4", FOURFOLD))
        self.assertIsNone(frame("x^2*y^3+z^2*t^3+u^5", FOURFOLD))

    def test_chain_rejects_a_step_that_is_not_triangular(self):
        p = parse("x^2*y+z^2+t^3+u^4", FOURFOLD)
        self.assertEqual(adapted_polynomial(p, FOURFOLD, [{"coordinate": "y", "offset": "x"}]),
                         parse("x^3+x^2*y+z^2+t^3+u^4", FOURFOLD))
        for offset in ("y*x", "1+x", "0"):
            with self.assertRaises(InvalidChart):
                adapted_polynomial(p, FOURFOLD, [{"coordinate": "y", "offset": offset}])

    def test_composed_maps_invert(self):
        steps = [{"coordinate": "z", "offset": "x*y"}, {"coordinate": "y", "offset": "x*z"}]
        chart = {"coordinates": ["a", "b", "c", "d", "e"],
                 "map": dict(zip(FOURFOLD, ["a*b", "b*c", "c", "d*a", "e"]))}
        image = compose_map(chart, FOURFOLD, steps)
        symbols = sympy.symbols("a b c d e")
        a, b, c, d, e = symbols
        y = b*c + (a*b)*c
        expected = dict(x=a*b, y=y, z=c + (a*b)*y, t=d*a, u=e)
        for name in FOURFOLD:
            self.assertEqual(sympy.expand(sympy.sympify(image[name].replace("^", "**")) - expected[name]), 0)
        back = compose_map(dict(chart, map=image), FOURFOLD, steps, inverse=True)
        self.assertEqual({k: parse(v, chart["coordinates"]) for k, v in back.items()},
                         {k: parse(v, chart["coordinates"]) for k, v in chart["map"].items()})


class AdaptedNewtonResolutions(unittest.TestCase):
    def assert_charts_pull_back(self, record, names, count=12):
        """Each chart map pulls the input back to its exceptional monomial times the strict transform."""
        source = sympy.symbols(" ".join(names))
        f = sympy.sympify(record["polynomial"].replace("^", "**"), dict(zip(names, source)))
        charts = record["charts"]
        stride = max(1, len(charts) // count)
        for chart in charts[::stride]:
            local = dict(zip(chart["coordinates"], sympy.symbols(" ".join(chart["coordinates"]))))
            def read(text):
                return sympy.sympify(text.replace("^", "**"), local)
            images = {name: read(chart["map"][name]) for name in names}
            pullback = f.subs(dict(zip(source, (images[name] for name in names))), simultaneous=True)
            product = read(chart["pullback_factor"]) * read(chart["strict_transform"])
            self.assertEqual(sympy.expand(pullback - product), 0, chart["id"])

    def test_join_with_a_nonconvenient_monomial(self):
        record = quick("x^2*y+z^2+t^3+u^4", FOURFOLD)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(steps_of(record), [("y", "x")])
        self.assertEqual(record["certificate"]["coordinate_translation"]["method"], METHOD)
        self.assertEqual(record["certificate"]["boundary_image_ideals"], [["x", "z", "t", "u"]])
        self.assertEqual(factors(record), join_zeta([2, 3, 4]))
        self.assertEqual(record["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 7)
        self.assertEqual(record["topological_zeta"]["status"], "computed")
        self.assert_charts_pull_back(record, FOURFOLD)

    def test_join_with_three_singular_axes(self):
        record = quick("x*y*z+t^2+u^3", FOURFOLD)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual([coordinate for coordinate, _ in steps_of(record)], ["z", "y", "x"])
        self.assertEqual(factors(record), join_zeta([2, 3]))
        self.assertEqual(record["topological_zeta"]["status"], "computed")
        self.assert_charts_pull_back(record, FOURFOLD, count=6)

    def test_fivefold_join(self):
        record = quick("x^2*y+z^2+t^2+u^2+v^5", FIVEFOLD)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual(steps_of(record), [("y", "x")])
        self.assertEqual(factors(record), join_zeta([2, 2, 2, 5]))
        self.assertEqual(record["topological_zeta"]["status"], "computed")

    def test_hidden_brieskorn_pham_coordinates(self):
        hidden = quick("t^4+u^4+y^3+z^3+x^2+2*x*y+y^2", FOURFOLD)
        direct = quick("x^2+y^3+z^3+t^4+u^4", FOURFOLD)
        self.assertEqual((hidden["status"], hidden["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual(steps_of(hidden), [("x", "-y")])
        self.assertNotIn("coordinate_translation", direct["certificate"])
        self.assertEqual(factors(hidden), isolated_zeta([2, 3, 3, 4, 4]))
        for name in ("monodromy_zeta", "topological_zeta"):
            self.assertEqual(hidden[name]["expression"], direct[name]["expression"])
        self.assert_charts_pull_back(hidden, FOURFOLD)

    def test_isolated_nonconvenient_germ(self):
        # x^3+x*y^7+z^4+z*t^5+u^2 is weighted homogeneous with weights
        # (1/3, 2/21, 1/4, 3/20, 1/2). It has no power of y or of t.
        record = quick("x^3+x*y^7+z^4+z*t^5+u^2", FOURFOLD)
        self.assertEqual((record["status"], record["engine"]), ("resolved", "newton-polyhedron"))
        self.assertEqual(steps_of(record), [("x", "y^4"), ("z", "t^2")])
        self.assertTrue(record["fan"]["isomorphism_off_origin"])
        mu = weighted_milnor_number(["1/3", "2/21", "1/4", "3/20", "1/2"])
        self.assertEqual(mu, 323)
        self.assertEqual(record["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 1 + mu)
        self.assertEqual(record["topological_zeta"]["status"], "computed")

    def test_intrinsic_resolution_in_adapted_coordinates(self):
        record = quick("x*y^3+z^2+t^2+u^2", FOURFOLD, mode="intrinsic")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(steps_of(record), [("x", "y")])

    def test_intrinsic_sections_over_the_singular_axes(self):
        # The sections over the three singular axes are quadratic in t.
        record = quick("x*y*z+t^2+u^3", FOURFOLD, mode="intrinsic")
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["dual_complex"]["status"], "complete")
        self.assertEqual(len(steps_of(record)), 3)

    def test_replay_checks_the_steps(self):
        record = quick("x^2*y+z^2+t^3+u^4", FOURFOLD)
        saved = {key: value for key, value in record.items()
                 if key in ("status", "engine", "mode", "variables", "polynomial", "atlas", "charts", "fan",
                            "certificate", "dual_complex", "schema_version", "origin", "base_field",
                            "geometric_field", "steps")}
        replay = replay_zetas(dict(record=saved, zetas=["monodromy", "topological"], compute="cpu"))
        for name in ("monodromy_zeta", "topological_zeta"):
            self.assertEqual(replay[name]["expression"], record[name]["expression"])
        # Another substitution gives another adapted polynomial, which the saved charts contradict.
        forged = dict(saved, certificate=dict(saved["certificate"], coordinate_translation={
            "method": METHOD, "steps": [{"coordinate": "y", "offset": "x^2"}]}))
        forged["atlas"] = [dict(chart, torus=dict(chart["torus"], coordinate_translation=forged["certificate"][
            "coordinate_translation"])) for chart in saved["atlas"]]
        with self.assertRaises(InvalidZetaCertificate):
            replay_zetas(dict(record=forged, zetas=["monodromy"], compute="cpu"))


class NewtonHandOver(unittest.TestCase):
    """A construction that declines leaves the others to run, and the search never ends a request."""

    def search(self, polynomial, names, mode="embedded"):
        from engine.newton_backend import resolve_newton
        diagnostics, verified = {}, []
        with face_search(False):
            record = resolve_newton(parse(polynomial, names), names, mode, seconds=20, defer_outputs=True,
                                    verified=verified, integrals=True, diagnostics=diagnostics)
        return record, diagnostics, verified

    def test_rejected_construction_leaves_the_next_one_to_run(self):
        from unittest.mock import patch
        with patch("engine.newton_trials.hyperplane_fan", side_effect=InvalidChart("The fan was rejected.")), \
                patch("engine.newton_trials.resolve_trials", side_effect=ArithmeticError("No packed form")):
            record, diagnostics, verified = self.search("x^2*y+z^2+t^3+u^4", FOURFOLD)
        self.assertEqual(record["engine"], "newton-polyhedron")
        self.assertEqual(steps_of(record), [("y", "x")])
        self.assertEqual(diagnostics["hyperplane_fan"], "The fan was rejected.")
        self.assertEqual(diagnostics["monomial_trials"], "No packed form.")
        self.assertEqual(verified, [])

    def test_declined_search_returns_nothing_with_its_reasons(self):
        from unittest.mock import patch
        from engine.centres import GiveUp
        with patch("engine.newton_frames.resolve_adapted", side_effect=GiveUp("No frame")):
            record, diagnostics, verified = self.search("x^2*y+z^2+t^3+u^4", FOURFOLD)
        self.assertIsNone(record)
        self.assertEqual(diagnostics["adapted_frame"], "No frame.")
        self.assertEqual(diagnostics["point_repair"],
                         "The strict transform fails normal crossings on a positive-dimensional locus "
                         "in chart N1. Its dimension on this orbit is 3.")
        self.assertEqual(verified, [])

    def test_no_adapted_frame_is_a_recorded_decline(self):
        # (x^3-y^4)^2 is a repeated factor of a face that no substitution makes a coordinate.
        from engine.newton_frames import resolve_adapted
        diagnostics = {}
        with face_search(False):
            record = resolve_adapted(parse("(x^3-y^4)^2+z^2+t^3+u^3", FOURFOLD), FOURFOLD, "embedded",
                                     diagnostics=diagnostics)
        self.assertIsNone(record)
        self.assertIn("adapted_frame", diagnostics)

    def test_surfaces_keep_the_origin_blowups_before_the_adapted_frames(self):
        # In three variables the default search has the linear frames only.
        # The dispatcher asks for the adapted ones after the blowups from the
        # origin, and both methods give the same zeta functions.
        from unittest.mock import patch
        from engine.newton_backend import resolve_newton
        names = ["x", "y", "z"]
        for source, steps in (("(x+y)^2+z^3+x^3", 1), ("(x+y)^2-(x-y)^2*z", 2),
                              ("(z+2*x)^4+y*(3*z^4-x^4)", 2), ("(y+2*z)^3/7-x*(y^3+z^3)", 2)):
            with self.subTest(source=source):
                polynomial = parse(source, names)
                with face_search(False):
                    self.assertIsNone(resolve_newton(polynomial, names, "embedded", defer_outputs=True))
                    adapted = resolve_newton(polynomial, names, "embedded", defer_outputs=True, frames="adapted")
                self.assertEqual(adapted["engine"], "newton-polyhedron")
                self.assertEqual(len(steps_of(adapted)), steps)
                with patch("engine.reduced_centre.smooth_reduced_graph", return_value=None):
                    blowups = quick(source, names)
                    with patch("engine.origin_backend.resolve_origin", return_value=None):
                        fan = quick(source, names)
                self.assertEqual((blowups["status"], blowups["engine"]), ("resolved", "canonical-blowups"))
                self.assertEqual((fan["status"], fan["engine"]), ("resolved", "newton-polyhedron"))
                for name in ("monodromy_zeta", "topological_zeta"):
                    self.assertEqual(fan[name]["status"], "computed")
                    self.assertEqual(fan[name]["expression"], blowups[name]["expression"])

    def test_nonreduced_input_is_not_a_decline(self):
        from unittest.mock import patch
        from engine.polynomial import NonReducedError
        with patch("engine.newton_frames.resolve_adapted", side_effect=NonReducedError("The germ is nonreduced.")):
            with self.assertRaises(NonReducedError):
                self.search("x^2*y+z^2+t^3+u^4", FOURFOLD)

    def test_dispatcher_hands_over_after_a_defect(self):
        # An internal error of the Newton code is recorded and the other methods still run.
        from unittest.mock import patch
        from engine.resolve import _resolve_polynomial
        calls = []

        def origin(*args, **kwargs):
            calls.append(kwargs.get("tangent_only", False))
            return None

        request = dict(polynomial="x^2*y+z^2+t^3+u^4", variables=FOURFOLD, mode="embedded", search="quick",
                       compute="cpu")
        from time import perf_counter
        with patch("engine.newton_backend.resolve_newton", side_effect=KeyError("ray")), \
                patch("engine.origin_backend.resolve_origin", side_effect=origin), \
                patch("engine.general_resolution.search", return_value=None), face_search(False):
            result = _resolve_polynomial(parse(request["polynomial"], FOURFOLD), request, perf_counter())
        self.assertEqual(calls, [True, False])
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("KeyError", result["diagnostic"]["newton_declines"]["defect"])
        self.assertIn("The Newton search returned no certified atlas.", result["diagnostic"]["description"])

    def test_rejected_proposal_is_replaced_and_leaves_no_atlas(self):
        # The verifier rejects the Newton charts, once by a failed check and
        # once by an internal error. The smooth-centre rule is asked next,
        # and the result carries nothing of the rejected proposal. A bounded
        # attempt keeps the budgets of each method and may end without an atlas.
        from unittest.mock import patch
        request = dict(polynomial="x^2*y+z^2+t^3+u^4", variables=FOURFOLD, mode="embedded", search="quick",
                       compute="cpu", zetas=["monodromy", "topological"])
        for error, text in ((InvalidChart("Normal crossings fail on the owned origin fibre."), "Normal crossings fail"),
                            (KeyError("ray"), "internal error (KeyError")):
            asked = []

            def canonical(*args, **kwargs):
                asked.append(kwargs.get("safe", False))
                return None

            with self.subTest(error=type(error).__name__), \
                    patch("engine.strata.proposal_chart_outputs", side_effect=error), \
                    patch("engine.resolve.canonical_record", side_effect=canonical):
                result = resolve(dict(request, bounded=True))
            self.assertEqual(asked, [False])
            self.assertEqual((result["status"], result["charts"]), ("inconclusive", []))
            self.assertNotIn("atlas", result)
            self.assertNotIn("fan", result)
            self.assertIn(text, result["diagnostic"]["description"])
            self.assertIn("After that proposal was rejected", result["diagnostic"]["description"])

    def test_rejected_proposal_lands_in_the_safe_fallback(self):
        # A quick request does not end on a rejected Newton proposal: the
        # smooth-centre rules take over without a deadline of their own.
        from unittest.mock import patch

        class Reached(Exception):
            pass

        calls = []

        def canonical(*args, **kwargs):
            calls.append((kwargs.get("safe"), kwargs.get("seconds")))
            raise Reached

        with patch("engine.strata.proposal_chart_outputs", side_effect=KeyError("ray")), \
                patch("engine.resolve.canonical_record", side_effect=canonical):
            with self.assertRaises(Reached):
                quick("x^2*y+z^2+t^3+u^4", FOURFOLD)
        self.assertEqual(calls, [(True, None)])

    def test_declined_search_lands_in_the_safe_smooth_centre_rules(self):
        # When no construction applies, a quick request goes on with the
        # smooth-centre rules as its safe method, after the repeated-tangent rule.
        from unittest.mock import patch
        from time import perf_counter
        from engine.resolve import _resolve_polynomial

        class Reached(Exception):
            pass

        calls = []

        def origin(*args, **kwargs):
            calls.append((kwargs.get("tangent_only", False), kwargs.get("safe", False), kwargs.get("seconds")))
            if kwargs.get("safe"):
                raise Reached
            return None

        request = dict(polynomial="(x^3-y^4)^2+z^2+t^3+u^3", variables=FOURFOLD, mode="embedded", search="quick",
                       compute="cpu")
        with patch("engine.newton_backend.resolve_newton", return_value=None), \
                patch("engine.origin_backend.resolve_origin", side_effect=origin), face_search(False):
            with self.assertRaises(Reached):
                _resolve_polynomial(parse(request["polynomial"], FOURFOLD), request, perf_counter())
        # The repeated-tangent rule has its share of the time budget; the safe method has no deadline.
        from engine.quick_budget import QUICK_SECONDS, TANGENT_SHARE
        self.assertEqual(calls, [(True, False, TANGENT_SHARE * QUICK_SECONDS), (False, True, None)])


if __name__ == "__main__":
    unittest.main()
