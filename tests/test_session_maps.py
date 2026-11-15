import json
import unittest

from engine.session import new_session
from engine.session_values import Resolution
from tests.cli_support import run_prompt, run_script


class RingMapTests(unittest.TestCase):
    def test_simultaneous_substitution_and_wide_inputs_at_both_entry_points(self):
        source = """x, y = coordinates("x", "y")
u, z = coordinates("x", "z")
phi = hom(x.parent, u.parent, images=(z+1, u))
print(phi.domain == x.parent, phi.codomain == u.parent, phi.images == (z+1, u))
print(phi(x*y + y/3) == (z+1)*u + u/3, phi(0).parent == u.parent)
swap = hom(x.parent, x.parent, images=(y,x))
print(swap(x^2*y) == y^2*x)
both = hom(x.parent, u.parent, images=(z,z))
n = 2^65+3
print(both((2^150+1)*x^n*y^2) == (2^150+1)*z^(n+2))
at = hom(x.parent, QQ, images=(1/2, 3))
print(at(x*y) == 3/2, at(x*y).parent == QQ)
constants = hom(QQ, u.parent, images=())
print(constants(at(y/2)).parent == u.parent)
"""
        expected = ["True True True", "True True", "True", "True", "True True", "True"]
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_foreign_bindings_and_source_relations(self):
        session = new_session(None)
        list(session.run("""x = coordinates("x")
z = coordinates("z")
other = coordinates("z", fresh=True)
A = quotient_ring(ideal(x^2))
B = quotient_ring(ideal(z^3))
a, = A.gens
b, = B.gens
phi = hom(A, B, images=(b^2,))
check = phi(a) == b^2
end = hom(A, x.parent, images=(0,))
zero = end(a)
"""))
        self.assertTrue(session.variables["check"])
        self.assertEqual(session.variables["zero"], 0)
        self.assertIs(session.variables["zero"].parent, session.variables["x"].parent)
        for source, message in (
            ("hom(A, B, images=(b,))", "relation"),
            ("hom(A, z.parent, images=(z,))", "relation"),
            ("hom(x.parent, z.parent, images=(other,))", "coordinate spaces"),
            ("hom(x.parent, z.parent, images=())", "one image"),
            ("hom(x.parent, B, images=(z,))", "residue class"),
            ("phi(x)", "residue class"),
            ("phi(b)", "residue class"),
        ):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, message):
                list(session.run(source))

    def test_ground_and_zero_ring_maps_preserve_the_unit(self):
        session = new_session(None)
        list(session.run("""Z = quotient_ring(ideal(1, ring=QQ))
S = quotient_ring(ideal(ring=QQ))
p = hom(QQ, Z, images=())
q = hom(Z, Z, images=())
lift = hom(S, QQ, images=())
"""))
        z, s = session.variables["Z"], session.variables["S"]
        self.assertEqual(session.variables["p"](1), z(0))
        self.assertEqual(session.variables["q"](z(0)), z(0))
        self.assertEqual(session.variables["lift"](s(2)), 2)
        with self.assertRaisesRegex(ValueError, "relation 1"):
            list(session.run("hom(Z, QQ, images=())"))

    def test_maps_from_graph_charts_check_unreduced_relations(self):
        session = new_session(None)
        graph = Resolution({"variables": ["x"], "dual_complex": {"vertices": []},
                            "charts": [{"id": "G", "coordinates": ["a", "b"],
                                        "ambient_ideal": ["a-b^2"]}]})
        session.variables["G"] = graph.chart("G")
        list(session.run("""a,b = G.coordinates
z = coordinates("z")
identity = hom(G.parent, G.parent, images=(a,b))
param = hom(G.parent, z.parent, images=(z^2,z))
project = hom(G.ambient, G.parent, images=(a,b))
"""))
        a, b = session.variables["a"], session.variables["b"]
        self.assertEqual(str(a), str(b**2))
        self.assertEqual(session.variables["identity"](a+b), a+b)
        z = session.variables["z"]
        self.assertEqual(session.variables["param"](a+b), z**2+z)
        relation, = graph.chart("G").session_attribute("ambient_ideal")
        self.assertEqual(session.variables["project"](relation), 0)
        with self.assertRaisesRegex(ValueError, "domain relation"):
            list(session.run("hom(G.parent, z.parent, images=(0,z))"))

        marked = Resolution({"variables": ["x"], "dual_complex": {"vertices": []},
                             "coefficient_field": {"generator": "theta", "exact_root": "sqrt(2)"},
                             "charts": [{"id": "C", "coordinates": ["u"], "map": {"x": "theta*u"}}]})
        session.variables["C"] = marked.chart("C")
        list(session.run("u = C.coordinates[0]\nh = hom(z.parent, C.parent, images=(u,))"))
        self.assertEqual(session.variables["h"](z), session.variables["u"])
        with self.assertRaisesRegex(ValueError, "coefficient-field embedding"):
            list(session.run("hom(C.parent, z.parent, images=(z,))"))

    def test_projected_results_retain_the_inclusion_and_replay_source(self):
        session = new_session(None)
        list(session.run("""x,y,z = coordinates("x","y","z")
I = ideal(x-z, y-z)
J = eliminate(I, x)
t,s = J.coordinates
r = resultant(x-y,x-z,x)
one = resultant(x-y,1,x)
first = J.inclusion(t+s) == y+z
second = r.inclusion(r) == y-z
third = r.inclusion.domain == J.inclusion.domain
fourth = one.inclusion(one).parent == x.parent
A,B,C = coordinates("x","y","z", fresh=True)
K = eliminate(ideal(A-C,B-C),A)
ground = eliminate(I,x,y,z)
fifth = ground.parent == QQ
sixth = ground.inclusion(1).parent == I.parent
"""))
        for key in ("first", "second", "third", "fourth", "fifth", "sixth"):
            self.assertTrue(session.variables[key], key)
        original = session.variables["J"].inclusion
        fresh = session.variables["K"].inclusion
        self.assertIsNot(original.domain, fresh.domain)
        self.assertIsNot(original.codomain, fresh.codomain)
        with self.assertRaisesRegex(ValueError, "coordinate spaces"):
            fresh(original.domain.coordinates[0])
        request = {"operation": "eliminate", "ring": {"variables": ["x", "y"]},
                   "operands": {"ideal": ["x-y", "y^2"], "variables": ["x"]}}
        list(session.run(f"saved = algebra_request({json.dumps(request)!r})"))
        included = session.variables["saved"].inclusion
        self.assertEqual(included.domain.variables, ("y",))
        self.assertEqual(included.codomain.variables, ("x", "y"))
        self.assertEqual(included(included.domain.coordinates[0]), included.codomain.coordinates[1])
