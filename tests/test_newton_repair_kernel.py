"""Independent checks for orbit faces, subface eligibility, finite repairs and their replay."""
from fractions import Fraction
from random import Random
import unittest

from engine import native_polynomial as np
from engine.centre_weights import centre_rows
from engine.kernel import exprs
from engine.newton_repair import repair_operation, supported_rows


class NewtonRepairKernel(unittest.TestCase):
    def test_native_orbit_intersections_and_complementary_directions(self):
        rng = Random(92)
        for _ in range(45):
            masks = [[sorted(rng.sample(range(7), rng.randint(1,7))) for _ in range(6)] for part in range(3)]
            cones = [[5,2,3], [0,4,5]]
            chosen = [dict(chart=j, rays=rng.sample(cone,rng.randrange(1,4))) for j,cone in enumerate(cones)]
            result = repair_operation("orbit_faces", width=3, minima=masks, orbits=chosen, cones=cones)
            all_nonempty = True
            for cone, source, item in zip(cones, chosen, result["orbits"]):
                faces = [sorted(set.intersection(*(set(part[r]) for r in source["rays"]))) for part in masks]
                self.assertEqual(result["faces"][item["face_id"]], faces[0])
                self.assertEqual(item["faces"], faces[1:])
                self.assertEqual(item["free"], [i for i,r in enumerate(cone) if r not in source["rays"]])
                self.assertEqual(item["fixed"], [i for i,r in enumerate(cone) if r in source["rays"]])
                self.assertEqual(item["exceptional"], [r for r in source["rays"] if r>=3])
                self.assertEqual(item["exceptional_axes"], [i for i,r in enumerate(cone) if r>=3])
                all_nonempty &= bool(faces[0])
            self.assertEqual(result["refines"], all_nonempty)

    def test_degenerate_subfaces_and_strict_branch_meeting(self):
        faces = [[], [0,1], [0,1,2,3], [3,4,5]]
        rows = [dict(face_id=i, faces=parts, free=list(range(dim)))
                for i,parts,dim in [(0,[[],[0]],1), (0,[[],[0,1]],1), (0,[[],[]],0),
                                     (1,[[0,1]],0),(2,[[0,1,2],[0,1]],2),(3,[[3,4,5],[4]],1)]]
        expected = ["contained","inadmissible","inadmissible","point","exact","volume"]
        plan = repair_operation("orbit_plan", orbits=rows, faces=faces, degenerate=[1],
                                count_strata=True, all_subfaces=True)
        self.assertEqual([item["action"] for item in plan], expected)
        self.assertEqual(plan[0]["chi"], 0)
        self.assertEqual(plan[-1]["volume_parts"], [0])
        self.assertEqual((plan[-1]["strict_sign"],plan[-1]["off_sign"]), (1,-1))
        skipped = repair_operation("orbit_plan", orbits=rows, faces=faces, degenerate=[1],
                                   count_strata=False, all_subfaces=False)
        self.assertEqual([i["action"] for i in skipped], ["skip"]*3+["point","skip","skip"])

    def test_saturated_orbit_excludes_axes_and_respects_scheme_structure(self):
        x,y = variables = np.symbols("x y")
        a = Fraction(2**97+5, 17)
        for f, kind, ideal in [(x*y*(x+y), "empty", (1,)), ((x-1)**2+(y-a)**2, "points", (x-1, y-a)),
                               (x**(2**69)*(y-a)**3, "positive_dimension", ((y-a)**2,)),
                               (np.coerce(0), "positive_dimension", ())]:
            result = repair_operation("singular_orbit", width=2, equation=centre_rows(variables, (f,))[0])
            self.assertEqual(result["kind"], kind)
            self.assertEqual(exprs(variables,result["ideal"]), ideal)

    def test_separating_graph_elimination_and_wide_euler_rows(self):
        x,y,t = variables = np.symbols("x y t")
        a = Fraction(2**101+1, 13)
        rows = [x-t**(2**69), y-a, t*t+1]
        self.assertEqual(repair_operation("rur_graph", axes=2, rows=centre_rows(variables,rows)), [2,0,1])
        self.assertIsNone(repair_operation("rur_graph", axes=2, rows=centre_rows(variables,[x*y-t, y-a, t*t+1])))
        self.assertEqual(supported_rows(variables, [1, *rows], [2], exact=True), (t*t+1,))
        self.assertEqual(supported_rows(variables, [1, *rows], [2]), (1,t*t+1))
        value = str(2**117+5)
        self.assertEqual(repair_operation("branch_euler", cut=value, branches=["0",value], deleted=["2",None]),
                         [[-1,value,-1],[0,"2",-1],[1,value,1]])

    def test_toric_parent_verifies_wide_map_after_marked_reduction(self):
        from engine.repair_gluing import _repair_geometry
        x,y,t = variables = np.symbols("x y t")
        power = 2**79+3
        rays = [["1",str(power)],["0","1"]]
        linear = [["1/9","0"],["0","-2"]]
        relation = centre_rows(variables,[t*t-2])[0]
        def parent(rays,images):
            return _repair_geometry("parent", width=3,rays=rays,linear=linear,modulus=relation,
                                    names=["X","Y"],images=centre_rows(variables,images))
        field_noise = (t*t-2)*x**(2**68)
        source = parent(rays,[x/9+field_noise,-2*x**power*y])
        target = parent([["1","0"],["0","1"]],[x/9,-2*y])
        self.assertIsNotNone(source)
        forward = _repair_geometry("transition", source=source, target=target)
        backward = _repair_geometry("transition", source=target, target=source)
        self.assertEqual(exprs((x,y),forward),(x,1,x**power*y,1))
        self.assertEqual(exprs((x,y),backward),(x,1,y,x**power))
        self.assertIsNone(parent(rays,[x/9+1,-2*x**power*y]))
        self.assertIsNone(parent([["1","0"],["0","2"]],[x/9,-2*y*y]))
        self.assertIsNone(parent([["1","0"],["-1","1"]],[x/9,-2*y]))
        self.assertIsNone(_repair_geometry("transition",source=source,target=dict(target,names=["Y","X"])))

    def test_deleting_sheet_requires_one_irreducible_nonlinear_parameter(self):
        from engine.repair_gluing import _repair_geometry
        x,y,t = variables = np.symbols("x y t")
        base = dict(width=2, marked=False, ambient=[], removed=True, parent=None, mapping=None)
        def relations(eqs, **fields):
            child = dict(width=3,marked=False,removed=False,parent=0,mapping=2,
                         ambient=centre_rows(variables,eqs))
            return _repair_geometry("relations",charts=[base,{**child,**fields}])
        self.assertEqual(relations([3*t*t-2]),[dict(axis=2, coefficients=["3","0","-2"],relation=0,sheet=1)])
        self.assertEqual(relations([t*t-1]),[])
        for equations in ([(t*t-2)**2],[(t+1)*(t*t-2)],[x*x-3,t*t-2]):
            self.assertIsNone(relations(equations))
        self.assertIsNone(relations([t*t-2],parent=-1))
        self.assertIsNone(relations([t*t-2],mapping=3))
        self.assertIsNone(relations([t*t-2],marked=True))
