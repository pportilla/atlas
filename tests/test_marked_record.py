import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MarkedRecordTests(unittest.TestCase):
    def test_binomial_field_coordinates_preserve_each_requested_root_index(self):
        from fractions import Fraction
        from engine.marked_field import flint_literal, program
        from engine.native_library import field_request
        from engine.plane_backend import coefficient_field
        from engine.polynomial import parse_expression

        sextic = (1,0,0,0,0,0,-2)
        def family(p):
            return tuple((p,i) for i in reversed(range(1,len(p))))
        for requests, degree in (
            (family(sextic), 12),
            (family((1,0,0,0,2**181)), 8),
            (family((3,0,0,0,2)), 8),
            (family((1,0,0,0,0,-1)), 4),
            (family((2,0,0,0,0,0,-128)), 2),
            (((sextic,2),(sextic,1)), 6),
            (((sextic,1),(sextic,1)), 6),
            (((sextic,1),((1,0,-3),2)), 12),
        ):
            with self.subTest(requests=requests):
                roots = tuple(flint_literal(p,i) for p,i in requests)
                value = coefficient_field(roots, exhaustive=True, root_images=True)
                self.assertEqual(value["degree"], degree)
                self.assertEqual(coefficient_field(roots, root_images=True), value)
                self.assertEqual(len(value["values"]), len(roots))
                for (p,i),image in zip(requests, value["values"]):
                    # Independently embed the original selected root with the
                    # generic QQbar solver in exactly the returned mark.
                    source = f"flintroot {len(p)-1} " + " ".join(map(str,reversed(p))) + f" {i-1}"
                    row = field_request("atlas_field_coefficient", (program(value["exact"]),source))
                    expected = {(j,):Fraction(c) for j,c in enumerate(row.split()) if Fraction(c)}
                    self.assertEqual(parse_expression(image,["T"]), expected)

    def test_earlier_root_coordinates_follow_successive_primitive_elements(self):
        from unittest.mock import patch
        import sympy as sp
        from engine.marked_field import flint_literal
        from engine.plane_backend import coefficient_field

        # The two conjugates of sqrt(2) must retain their separate marks
        # when sqrt(3) and sqrt(5) enlarge the field. Include a rational input.
        roots = (flint_literal((2,-1), 1), flint_literal((1,0,-2), 1),
                 flint_literal((1,0,-2), 2), flint_literal((1,0,-3), 1),
                 flint_literal((1,0,-5), 1))
        with patch("engine.native_process.run", side_effect=AssertionError("unexpected worker")):
            field = coefficient_field(roots, exhaustive=True, root_images=True)
        self.assertEqual(field["degree"], 8)
        t = sp.Symbol("T")
        def parse(text):
            return sp.Poly(sp.sympify(text, locals={"T":t}), t, domain=sp.QQ)
        modulus = parse(field["minimal_polynomial"])
        images = list(map(parse, field["values"]))
        self.assertEqual(images[0], parse("1/2"))
        self.assertEqual(images[1], -images[2])
        for image, square in zip(images[1:], (2,2,3,5)):
            self.assertTrue((image**2-square).rem(modulus).is_zero)
        # FLINT's first root in each of these real quadratic pairs is positive.
        # The marked generator is their sum with 1/2. The positive square
        # roots choose one embedding of this irreducible degree-eight field.
        theta = sp.Rational(1,2) + sp.sqrt(2) + sp.sqrt(3) + sp.sqrt(5)
        self.assertEqual(sp.expand(modulus.eval(theta)), 0)
        for image, value in zip(images[1:], (sp.sqrt(2), -sp.sqrt(2), sp.sqrt(3), sp.sqrt(5))):
            self.assertEqual(sp.expand(image.eval(theta) - value), 0)
        self.assertEqual(coefficient_field(roots, root_images=True), field)

    def test_marked_angles_and_scaled_polynomials(self):
        from engine.marked_field import description, record_field
        from engine.polynomial import InputError
        for literal, equivalent in (("exp(pi/I/5)", "exp(-I*pi/5)"),
                                    ("sin(I*I*pi/7)", "sin(-pi/7)"),
                                    ("tan(pi/7 + pi/7)", "tan(2*pi/7)")):
            with self.subTest(literal=literal):
                self.assertEqual(description(literal), description(equivalent))
        for literal in ("sin(I*pi)", "cos(pi+I)", "csc(0)", "exp(pi)"):
            with self.subTest(literal=literal), self.assertRaises(InputError):
                description(literal)
        field = {"generator": "a", "exact_root": "sqrt(2)"}
        for half in ("0.5", "٥.٠e-1", "1_0 / 20"):
            with self.subTest(half=half):
                field["minimal_polynomial_coefficients"] = [half, 0, -1]
                self.assertIsNotNone(record_field(field, {}, ["a"], "t")[0])
        field["minimal_polynomial_coefficients"] = [1, 0, -3]
        with self.assertRaisesRegex(InputError, "does not define"):
            record_field(field, {}, ["a"], "t")

    def test_native_indices_on_mixed_factors(self):
        import sympy as sp
        from engine import native_library
        from engine.field_coefficient import marked_root_display, root_position
        from engine.marked_field import description
        if not hasattr(native_library._kernel(), "atlas_field_description"):
            self.skipTest("Requires native marked scalar descriptors")
        x = sp.Symbol("x")
        for expression in ((x*x+1)*(x*x+2), (x*x-x+1)*(2*x*x+3)*(x**3-x+1)**2,
                           (x*x+4)*(x**4-12*x*x+20*x+40), x*x*(x*x+2*x+16)**2*(x*x+25)):
            for i in range(sp.degree(expression,x)):
                with self.subTest(expression=expression,index=i):
                    coefficients, index = description(f"CRootOf({expression}, {i})")
                    root = sp.CRootOf(expression, i)
                    poly = sp.Poly(sp.minpoly(root,x),x)
                    self.assertEqual(tuple(poly.all_coeffs()),coefficients)
                    if poly.degree()==1:
                        self.assertEqual(index,1)
                    else:
                        self.assertEqual(marked_root_display(tuple(reversed(coefficients)))[2][index-1],
                                         root_position(poly,root))

    def test_native_weighted_local_charts_can_be_read_without_sympy(self):
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_field_description"):
            self.skipTest("Requires the native chart arithmetic")
        self.guard("""
from engine.origin_backend import _resolve
from engine.newton_backend import resolve_newton
from engine.weighted_record import nonreduced_resolution
from engine.monodromy import monodromy_zeta
from engine.topological import topological_zeta
from engine.polynomial import parse
names=list('xyz')
for f,op in (('(x+y)^2+z^3+x^3', lambda p:_resolve(p,names,'embedded',60)),
             ('z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2', lambda p:resolve_newton(p,names,'embedded'))):
    def support(poly):
        r=op(poly)
        r.update(variables=names,polynomial=f,mode='embedded')
        return r
    doubled=nonreduced_resolution(parse('('+f+')^2',names),names,'embedded',support)
    doubled.update(variables=names,polynomial='('+f+')^2',mode='embedded')
    single=support(parse(f,names))
    old=monodromy_zeta(single)['factors']
    assert monodromy_zeta(doubled)['factors']==[dict(row,multiplicity=row['multiplicity']*2) for row in old]
    assert topological_zeta(doubled)['status']=='computed'
""")

    def test_guarded_marked_literals_and_composita(self):
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_field_description"):
            self.skipTest("Requires native marked scalar descriptors")
        source = """
from engine.marked_field import description, common_fields
assert description('sqrt(2) + sqrt(3)') == ((1,0,-10,0,1),1)
assert description('tan(pi/8)') == ((1,2,-1),1)
assert description('exp(I*pi/5)')[0] == (1,-1,1,-1,1)
assert description('CRootOf((t^2+1)*(t^2+2),-1)') == ((1,0,2),1)
eq,relation,degree,images = common_fields(('sqrt(2)','sqrt(3)'), 't')
assert degree == 4 and len(images) == 2
"""
        self.guard(source)

    def test_weighted_graph_with_several_named_roots(self):
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_field_description"):
            self.skipTest("Requires native marked scalar descriptors")
        self.guard("""
from engine.graph_record import lift_graph
from engine.marked_field import description
r={"algebraic_numbers":{"a":{"exact":"sqrt(2)"},"b":{"exact":"sqrt(3)"}}}
c=dict(coordinates=["x","y","r"], map={"x":"x","y":"y"}, ambient_ideal=["y-x*(r+a+b)"],
       pivot_generator="x", exceptional=[dict(id="E",ideal=["x"])],strict_transform=["y"])
orders=lift_graph(r,c,[("y",2),("x",3)],"1",("x","y"))
assert orders=={"E":5}
assert description(c["weight_field"]["exact"])==description("sqrt(2)+sqrt(3)")
""")

    def test_wide_graph_worker_and_archive_use_the_same_divisor_orders(self):
        self.guard("""
import json
from engine import native_polynomial as np
from engine.general_resolution import _attempt
from engine.resolution_search import _candidate
from engine.graph_record import graph_outputs
from engine.monodromy import monodromy_zeta
from engine.topological import topological_zeta
from engine.strata import InvalidChart
coords = np.symbols('v0:35')
x,y,z = coords[:3]
f = x*x-y*y*z
poly = np.terms(coords,f)
for mode,dimension in [('embedded',34),('intrinsic',33)]:
    state, r = _attempt(coords,f,_candidate(coords,(x,y),'wide Whitney cylinder'),mode,1,30)
    assert state == 'certified'
    r.update(variables=list(map(str,coords)), polynomial=str(f), mode=mode)
    r['atlas'], output = graph_outputs(r,poly,intrinsic=mode=='intrinsic')
    dual = output['dual_complex']
    assert dual['status']=='complete' and len(dual['vertices'])==1
    assert dual['vertices'][0]['dimension']==dimension and not dual['cells']
    r = json.loads(json.dumps(r))
    r.pop('dual_complex')
    assert graph_outputs(r,poly,intrinsic=mode=='intrinsic')[1] == output
    assert monodromy_zeta(r)['expression'] == '1/(1-T^2)'
    assert topological_zeta(r)['expression'] == '(s + 2)/(2*(s + 1)^2)'
    r['atlas'][0]['components'][0]['nu'] += 1
    try:
        graph_outputs(r,poly)
    except InvalidChart:
        pass
    else:
        raise AssertionError('An altered relative Jacobian order passed validation')
""")

    def test_intrinsic_components_of_a_marked_tacnode_product(self):
        self.guard("""
from engine import atlas, native_polynomial as np
from engine.candidate_field import indexed_field
from engine.general_resolution import _attempt
from engine.graph_record import graph_outputs
from engine.resolution_search import ResolutionCandidate
x, y, *free = coords = np.symbols('v0:35')
u, v, w = np.symbols('U V W')
field, theta = indexed_field((1,0,-2),1)
candidate = ResolutionCandidate((x**3,x*y,y**2), (u,v,w), ((x,y,u),(x,y,w)), field,theta,'tacnode product')
f = y**2-x**4
state, record = _attempt(coords,f,candidate,'intrinsic',1,30)
assert state == 'certified'
record.update(variables=list(map(str,coords)), polynomial=str(f), mode='intrinsic')
saved, output = graph_outputs(record,np.terms(coords,f),intrinsic=True,integrals=False)
import json
replay = atlas.boundary_complex(atlas.from_json(json.loads(json.dumps(saved))),len(coords),True)
assert replay == output['dual_complex'] and replay['status'] == 'complete'
assert replay['cells'] == [] and [v['dimension'] for v in replay['vertices']] == [33,33]
""")

    def test_archived_graph_computes_both_integrals_without_symbolic_import(self):
        import sympy as sp
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_field_description"):
            self.skipTest("Requires native marked scalar descriptors")
        from engine.general_resolution import verify_candidate
        from engine.resolution_search import ResolutionCandidate
        x,y,z,u,v = sp.symbols("x y z u v")
        a = sp.sqrt(2)
        candidate = ResolutionCandidate((x,y),(u,v),((x+a*y,y),),sp.QQ.algebraic_field(a),a,"test")
        record = verify_candidate((x,y,z),x*x-y*y*z,candidate,"embedded",3)
        record.update(variables=list("xyz"), polynomial="x^2-y^2*z", mode="embedded")
        del record["coefficient_field"]["minimal_polynomial_coefficients"]
        self.guard("import json\nr=json.loads("+repr(json.dumps(record))+")\n"+"""
from engine.monodromy import monodromy_zeta
from engine.topological import topological_zeta
from engine.graph_record import graph_record_charts
from engine.polynomial import parse
f=parse(r["polynomial"],r["variables"])
for mark in ("sqrt(2)", "CRootOf((t-1)*(t^2-2),2)", "2*cos(pi/4)"):
    r["coefficient_field"]["exact_root"]=mark
    r.pop("atlas",None)
    old=monodromy_zeta(r)
    assert old["expression"]=="1/(1-T^2)"
    topo=topological_zeta(r)["expression"]
    r["atlas"]=graph_record_charts(r,f)
    assert monodromy_zeta(r)["expression"]==old["expression"]
    assert topological_zeta(r)["expression"]==topo
    r["atlas"][0]["components"][0]["multiplicity"]+=1
    with assertion.assertRaises(InvalidChart):
        monodromy_zeta(r)
r.pop("atlas")
r["charts"].pop()
with assertion.assertRaises(InvalidChart):
    monodromy_zeta(r)
""")

    def guard(self, source):
        script = """import os, sys, unittest
os.environ['ATLAS_GROEBNER']='native'
class Guard:
    def find_spec(self,name,*args):
        if name == 'sympy' or name.startswith('sympy.'): raise AssertionError(name)
sys.meta_path.insert(0,Guard())
from engine.strata import InvalidChart
assertion = unittest.TestCase()
""" + source
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
                       capture_output=True,text=True)

    def test_saved_and_fresh_weighted_planes(self):
        from engine import native_library
        if not hasattr(native_library._kernel(), "atlas_field_description"):
            self.skipTest("Requires native marked scalar descriptors")
        self.guard("""
from engine.resolve import resolve
from engine.monodromy import monodromy_zeta
from engine.topological import topological_zeta
r = resolve(dict(polynomial='((x^2+y^2)^2+x^7)^2*x^3', variables=['x','y'],
                 mode='embedded', monodromy=True, search='quick', compute='cpu'))
assert r['monodromy_zeta']['status']=='computed'
assert monodromy_zeta(r)['expression'] == r['monodromy_zeta']['expression']
assert topological_zeta(r)['expression'] == r['topological_zeta']['expression']
r['atlas'][0]['components'][0]['multiplicity'] += 1
with assertion.assertRaises(InvalidChart):
    monodromy_zeta(r)
""")
