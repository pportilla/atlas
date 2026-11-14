import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path

from engine import native_library
from engine.plane_atlas import plane_outputs
from engine.polynomial import parse
from engine.resolve import resolve
from engine.strata import InvalidChart, rational_chart_outputs


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(native_library._MODE == "sympy", "The sparse plane chart verifier uses the native kernel.")
class RationalPlaneAtlas(unittest.TestCase):
    def test_cusp_import_guard_fresh_saved_and_companion(self):
        code = """
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sympy' or fullname.startswith('sympy.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, sys.argv[1])
from engine.resolve import resolve
from engine.polynomial import parse
from engine.strata import rational_chart_outputs
from engine.monodromy import monodromy_zeta
from engine.topological import topological_zeta
request = dict(variables=['x','y'], polynomial='(y-x)^2-x^3', monodromy=True)
for mode, counts in [('embedded', (3,2)), ('intrinsic', (1,0))]:
    record = resolve({**request, 'mode': mode})
    zetas = tuple(record[k]['expression'] for k in ('monodromy_zeta','topological_zeta'))
    assert zetas == ('(1-T^6)/((1-T^2)*(1-T^3))', '(4*s + 5)/((s + 1)*(6*s + 5))'), zetas
    archived = dict(variables=request['variables'], polynomial=request['polynomial'], atlas=record['atlas'])
    p = parse(request['polynomial'], request['variables'])
    _, result = rational_chart_outputs(archived, p, intrinsic=mode=='intrinsic')
    boundary = result['dual_complex']
    assert (len(boundary['vertices']),len(boundary['cells'])) == counts
    if mode == 'embedded':
        archive = dict(archived, status='resolved', mode=mode, engine='saved', charts=[{}])
        assert tuple(f(archive)['expression'] for f in (monodromy_zeta, topological_zeta)) == zetas
    else:
        assert 'strata' not in result
        assert record['monodromy_zeta']['source'] == 'companion embedded resolution'
for sign in ("-", "+"):
    f = f"(y^2{sign}2*x^4)^2-x^11"
    p = parse(f, ['x','y'])
    for mode, counts in [('embedded', (8,7)), ('intrinsic', (2,0))]:
        record = resolve(dict(variables=['x','y'], polynomial=f, mode=mode, monodromy=True))
        zetas = [record[k]['expression'] for k in ('monodromy_zeta','topological_zeta')]
        assert zetas == ['(1-T^8)*(1-T^22)^2/((1-T^4)*(1-T^11)^2)',
                         '(2*s + 3)*(19*s + 9)/((s + 1)*(8*s + 3)*(22*s + 9))']
        archived = dict(variables=['x','y'], polynomial=f, atlas=record['atlas'])
        _, result = rational_chart_outputs(archived, p, intrinsic=mode=='intrinsic')
        boundary = result['dual_complex']
        assert (len(boundary['vertices']), len(boundary['cells'])) == counts
        if mode == 'embedded':
            archive = dict(archived, status='resolved', mode=mode, engine='saved', charts=[{}])
            assert [g(archive)['expression'] for g in (monodromy_zeta, topological_zeta)] == zetas
assert 'sympy' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True,
                                text=True, env={**os.environ, "ATLAS_GROEBNER": "native"})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_multiple_centres_and_deleted_reciprocal_directions(self):
        names, text = ["x","y"], "((y-x)^2-x^5)*((y+x)^2-x^5)"
        record = resolve(dict(variables=names, polynomial=text, mode="embedded", monodromy=True))
        charts, p = record["atlas"], parse(text,names)
        from engine.atlas import from_json, to_json
        self.assertEqual(to_json(from_json(charts)), charts)
        self.assertEqual(len(charts[1]["removed"]), 2)
        _, result = plane_outputs(dict(variables=names, atlas=charts), p)
        self.assertEqual(result["strata"], record["topological_zeta"]["strata"])
        self.assertEqual(result["dual_complex"]["cells"], record["dual_complex"]["cells"])
        for change in ("reciprocal", "child", "order", "strict", "projection", "pivot", "region", "fibre"):
            with self.subTest(change=change):
                altered = copy.deepcopy(charts)
                if change == "reciprocal":
                    altered[1]["removed"].pop()
                elif change == "child":
                    altered.pop()
                elif change == "order":
                    altered[-1]["components"][0]["nu"] += 1
                elif change == "strict":
                    altered[-1]["components"][-1]["ideal"] = ["1"]
                elif change == "projection":
                    altered[2]["map"]["x"] += "+v"
                elif change == "pivot":
                    altered[2]["point_blowup"]["parent"] = 1
                elif change == "region":
                    altered[1]["region"] = []
                else:
                    altered[1]["fibre"] = []
                with self.assertRaises(InvalidChart):
                    rational_chart_outputs(dict(variables=names, atlas=altered), p)

    def test_conjugate_sibling_transitions_are_verified_in_one_field(self):
        names, f = ["x", "y"], "(y^2-2*x^4)^2-x^11"
        p = parse(f, names)
        charts = resolve(dict(variables=names, polynomial=f))["atlas"]
        theta = charts[0]["coefficient_field"]["generator"]
        self.assertEqual(charts[4]["point_blowup"]["center"], ["0", theta])
        self.assertEqual(charts[10]["point_blowup"]["center"], ["0", "-"+theta])
        for change in ("different marked root", "wrong relation", "one sibling twice", "wrong reciprocal", "wrong sheet map"):
            with self.subTest(change=change):
                altered = copy.deepcopy(charts)
                if change == "different marked root":
                    altered[5]["coefficient_field"] = {**altered[5]["coefficient_field"],
                                                       "embedding": "PolynomialRootIndexed(List(-2, 0, 1), 1)"}
                elif change == "wrong relation":
                    for c in altered:
                        c["coefficient_field"] = {**c["coefficient_field"], "minimal_polynomial": theta+"^2+2"}
                elif change == "one sibling twice":
                    for c in altered[10:12]:
                        c["point_blowup"]["center"] = ["0", theta]
                elif change == "wrong reciprocal":
                    altered[3]["removed"][0] = altered[3]["removed"][1]
                else:
                    altered[4]["map"]["y"] = altered[10]["map"]["y"]
                with self.assertRaises(InvalidChart):
                    plane_outputs(dict(variables=names, atlas=altered), p)
        # Coordinates can differ by a multiple of the field relation.
        charts[4]["map"]["y"] += f"+({theta}^2-2)*u"
        self.assertIsNotNone(plane_outputs(dict(variables=names, atlas=charts), p))
