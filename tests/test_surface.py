"""Normal crossings of a smooth surface with exceptional divisors: the procedure of engine/surface.py."""
import unittest

import sympy as sp

import engine.surface as surface
from engine.local_blowups import acampo_over
from engine.monodromy import monodromy_zeta
from engine.newton_backend import resolve_newton
from engine.polynomial import format_poly, parse
from engine.resolve import resolve
from engine.topological import topological_zeta

x, y, z, s = sp.symbols("x y z s")
CASES = {
    # name: (components, fibre, kinds of the centres: p for points, c for a curve)
    "tangent along a line": ([("E1", 2, 3, (z,)), ("strict", 1, 1, (z - x**2,))], (x, y, z), "cc"),
    "node of the trace": ([("E1", 2, 3, (z,)), ("strict", 1, 1, (z - x*y,))], (x, y, z), "pc"),
    "cusp of the trace": ([("E1", 2, 3, (z,)), ("strict", 1, 1, (z - y**2 + x**3,))], (x, y, z), "pppcccc"),
    "two divisors share a line, a third crosses it":
        ([("E1", 2, 3, (x,)), ("E2", 3, 4, (y,)), ("E3", 5, 2, (z,)), ("strict", 1, 1, (x - y,))], (x, y, z), "c"),
    "tacnode of the trace": ([("E1", 2, 3, (z,)), ("strict", 1, 1, (z - y**2 + x**4,))], (x, y, z), "ppccc"),
    "three lines of the trace": ([("E1", 2, 3, (z,)), ("strict", 1, 1, (z - x*y*(x - y),))], (x, y, z), "pcc"),
    "double conic": ([("E1", 2, 3, (z,)), ("strict", 1, 1, (z - (x**2 + y**2 - 1)**2,))], (x - 1, y, z), "cc"),
}


def zeta(data, strata):
    value = 0
    for key, chi in strata.items():
        term = sp.Integer(chi)
        for label in key:
            term /= data[label][0]*s + data[label][1]
        value += term
    return sp.factor(sp.cancel(value))


def no_plane(*arguments, **options):
    raise surface.NoModel("forced")
    yield


class SurfaceProcedure(unittest.TestCase):
    def test_plane_and_implicit_forms_agree(self):
        models = surface._models
        for name, (components, fibre, kinds) in CASES.items():
            answers = []
            for implicit in (False, True):
                # With no plane model the surface is used as it is.
                surface._models = (lambda *a, **k: iter([(a[0], (), a[2], list(a[3]), tuple(a[1]), 0)])) if implicit else models
                try:
                    data, strata, charts, centres = acampo_over((x, y, z), components, fibre, rules=("complete",))
                finally:
                    surface._models = models
                self.assertEqual("".join(centre["kind"][0] for centre in centres), kinds, name)
                self.assertTrue(all(centre["rule"] == "surface" for centre in centres), name)
                answers.append(zeta(data, strata))
            self.assertEqual(answers[0], answers[1], name)

    def test_same_zeta_function_as_the_short_rule(self):
        for name in ("node of the trace", "cusp of the trace", "three lines of the trace"):
            components, fibre, _ = CASES[name]
            both = [zeta(*acampo_over((x, y, z), components, fibre, rules=(rule,))[:2]) for rule in ("complete", "direct")]
            self.assertEqual(both[0], both[1], name)

    def test_where_the_short_rule_fails(self):
        from engine.local_blowups import GiveUp
        components, fibre, _ = CASES["two divisors share a line, a third crosses it"]
        with self.assertRaises(GiveUp):
            acampo_over((x, y, z), components, fibre, rules=("direct",))
        data, _, _, centres = acampo_over((x, y, z), components, fibre)
        self.assertEqual(data["B1"], (6, 7))   # N = 1 + 2 + 3, nu = 2 + (3 - 1) + (4 - 1)

    def test_newton_engine_on_a_nonisolated_germ(self):
        # The binomial x^2*y - z^3 has its own engine; through the Newton
        # modification of a sheared copy it needs the invariant (the strict
        # transform is singular) and then the surface procedure.
        names = list("xyz")
        p = parse("x^2*y-z^3", names)
        record = resolve_newton(p, names, "embedded")
        record.update(variables=names, polynomial=format_poly(p, names), mode="embedded", origin=[0, 0, 0])
        rules = {centre["rule"] for step in record["steps"] for centre in step.get("centers", [])}
        self.assertEqual(rules, {"invariant", "surface"})
        mine = monodromy_zeta(record)
        other = resolve({"polynomial": "x^2*y-z^3", "variables": names, "mode": "embedded", "search": "quick",
                         "monodromy": True})
        self.assertEqual(other["engine"], "binomial-coordinate-blowups")
        self.assertEqual(mine["expression"], other["monodromy_zeta"]["expression"])
        self.assertEqual(topological_zeta(record, monodromy=mine)["expression"], other["topological_zeta"]["expression"])

    def test_tricuspidal_quartic(self):
        result = resolve({"polynomial": "x^2*y^2+y^2*z^2+z^2*x^2-2*x*y*z*(x+y+z)+x^5+y^5+z^5", "variables": list("xyz"),
                          "mode": "embedded", "search": "quick", "monodromy": True})
        self.assertEqual(result["monodromy_zeta"]["milnor_fiber_euler_characteristic"], 34)   # mu = 33
        self.assertEqual(result["topological_zeta"]["expression"],
                         "(54*s^2 + 125*s + 69)/((s + 1)*(4*s + 3)*(30*s + 23))")


if __name__ == "__main__":
    unittest.main()
