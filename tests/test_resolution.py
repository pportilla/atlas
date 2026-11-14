import json
import os
from pathlib import Path
import re
import subprocess
import unittest
from engine.polynomial import parse, InputError, LimitError
from engine.resolve import resolve

ROOT=Path(__file__).resolve().parents[1]


def plane(source,mode='embedded',variables=('x','y')):
    return resolve(dict(polynomial=source,variables=list(variables),mode=mode))


class ParserTests(unittest.TestCase):
    def test_rational_ring_and_precedence(self):
        eq=lambda a:parse(a,['x','y'])
        self.assertEqual(eq('-(x-y)^2/2+x^2'),eq('x^2/2+x*y-y^2/2'))
        self.assertEqual(eq('x**3-2^3*y'),eq('x^3-8*y'))

    def test_rejections(self):
        for text in ['0','x-x','1+x','x/y','x^-2','x^0.5','x^2^3','x^129','q+x',
                     'x*y;exit','x(1)','x+','(x+y','x / (2-2)','2x']:
            with self.subTest(text=text),self.assertRaises((InputError,LimitError)):
                parse(text,['x','y'])
        for names in [[],['x','x'],['x','2y'],None]:
            with self.assertRaises(InputError):parse('x',names)


class CurveTests(unittest.TestCase):
    def test_cusp_and_maps(self):
        for mode in ['embedded','intrinsic']:
            f='beta^2-alpha^3';result=plane(f,mode,['alpha','beta'])
            for chart in result['charts']:
                pullback=re.sub(r'alpha|beta',lambda m:'('+chart['map'][m[0]]+')',f)
                self.assertEqual(parse(pullback,['u','v']),parse(chart['ambient_total_transform'],['u','v']))
            d=result['dual_complex']
            if mode=='embedded':
                self.assertEqual([(v['self_intersection'],v['total_multiplicity']) for v in d['vertices']],
                                 [(-3,2),(-2,3),(-1,6)])
                self.assertEqual({frozenset(c['vertices']) for c in d['cells']},
                                 {frozenset(('E1','E3')),frozenset(('E2','E3'))})
                self.assertEqual(result['branches'][0]['ambient_components'],['E3'])
            else:
                self.assertEqual(len(result['steps']),1)
                self.assertEqual([v['id'] for v in d['vertices']],['P1'])
                self.assertEqual(result['charts'][1]['exceptional'],[])

    def test_multiple_projective_directions_with_infinity(self):
        r=plane('x*y*(x^2+y^2)','intrinsic');d=r['dual_complex']
        self.assertEqual(len(r['steps']),1);self.assertEqual(len(d['vertices']),4)
        a,b=[{e['id'] for e in c['exceptional']} for c in r['charts']]
        self.assertEqual(len(a|b),4);self.assertEqual(len(a&b),2)
        self.assertEqual(d['cells'],[])
        self.assertTrue(all('PolynomialRootIndexed' in x['exact'] for x in r['algebraic_numbers'].values()))

    def test_deep_algebraic_centers(self):
        r=plane('(y^2-2*x^4)^2-x^11')
        self.assertEqual(len(r['steps']),8);self.assertEqual(len(r['branches']),2)
        self.assertEqual(len(r['dual_complex']['cells']),7)
        self.assertTrue(any(c['exclude_points'] for c in r['charts']))
        self.assertTrue(any(isinstance(c['base_domain'],dict) and c['base_domain']['exclude_sibling_points'] for c in r['charts']))

    def test_nonreduced_germ_but_allow_unit_factors(self):
        with self.assertRaisesRegex(InputError,'nonreduced'):plane('(y^2-x^3)^2','intrinsic')
        divisor=plane('(y^2-x^3)^2')
        self.assertEqual([v['total_multiplicity'] for v in divisor['dual_complex']['vertices']],[4,6,12])
        q=plane('(1+x)^2*(y^2-x^3)')
        self.assertEqual(len(q['steps']),3)

    def test_chart_holes_when_finished_branch_shares_parent(self):
        r=plane('(y-x)*((y-2*x)^2-x^5)','intrinsic')
        self.assertEqual(len(r['branches']),2)
        for c in r['charts'][2:]:
            if c['parent_chart']=='B1.u':
                self.assertIn(['0','1'],c['base_domain']['exclude_sibling_points'])


class DispatchTests(unittest.TestCase):
    def test_flags_and_modes(self):
        r=plane('x+y^2','intrinsic')
        self.assertEqual(r['steps'],[])
        q=resolve({'polynomial':'a^4+b^4+c^4+d^4','variables':list('abcd')})
        self.assertEqual(len(q['charts']),4)
        with self.assertRaises(InputError):resolve({'compute':'tpu'})

    def test_unresolved_is_not_certified(self):
        r=resolve({'polynomial':'x^2*y^2+z^4+w^5','variables':list('xyzw'),'search':'quick'})
        self.assertEqual(r['status'],'inconclusive')
        self.assertTrue(r['can_search'])
        self.assertEqual(r['dual_complex']['status'],'unavailable')
        self.assertEqual(r['charts'],[])

    def test_search_offer_after_parser_budget(self):
        data={'variables':['x','y'],'polynomial':'x+y^160'}
        limited=resolve(dict(data,search='quick'))
        self.assertEqual(limited['status'],'inconclusive')
        self.assertTrue(limited['can_search'])
        self.assertEqual(resolve(data)['status'],'resolved')

    def test_cli_exit_codes(self):
        for src,vs,code in [('y^2-x^3','x,y',0),('x^2*y^2+z^4+w^5','x,y,z,w',3),('1+x','x,y',2)]:
            flags=['--quick'] if code==3 else []
            result=subprocess.run([str(ROOT/'atlas'),'resolve',src,'--vars',vs,'--json',*flags],capture_output=True,text=True)
            self.assertEqual(result.returncode,code,result.stderr)
            self.assertIsInstance(json.loads(result.stdout),dict)

if __name__=='__main__':unittest.main()
