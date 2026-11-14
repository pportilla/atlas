"""Geometric and chart-level regression checks for the regular Newton fan."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from math import prod
import re
import unittest

from engine.polynomial import parse
from engine.resolve import resolve
from engine.toric_backend import resolve_toric


BASE = ["x", "y", "z"]
E8 = "x^2+y^3+z^5"
SPLIT = "2*x^2+3*y^4-z^7+x*y*z^4"


def run(equation, mode):
    return resolve({"polynomial": equation, "variables": BASE, "mode": mode})


def determinant(a, b, c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
            + a[2]*(b[0]*c[1]-b[1]*c[0]))


def polynomial(expr, coords):
    # Parse affine polynomials even when the constant term is nonzero.
    marked = parse(f"marker*({expr})", [*coords, "marker"])
    return {powers[:-1]: coeff for powers, coeff in marked.items()}


def neighbors(dual):
    adj = {v["id"]: set() for v in dual["vertices"]}
    for edge in dual["cells"]:
        a, b = edge["vertices"]
        adj[a].add(b)
        adj[b].add(a)
    return adj


class ToricAtlas(unittest.TestCase):
    def test_e8_graph(self):
        result = run(E8, "intrinsic")
        self.assertEqual(result["engine"], "native-toric")
        dual = result["dual_complex"]
        self.assertEqual(dual["status"], "complete")
        self.assertEqual((len(dual["vertices"]), len(dual["cells"])), (8, 7))
        adj = neighbors(dual)
        (center,) = [v for v, n in adj.items() if len(n) == 3]
        # The E8 diagram is distinguished from D8 by its three arm lengths.
        lengths = []
        visited = {center}
        for first in adj[center]:
            last, node, length = center, first, 0
            while True:
                self.assertNotIn(node, visited)
                visited.add(node)
                length += 1
                tail = adj[node] - {last}
                if not tail:
                    break
                self.assertEqual(len(tail), 1)
                last, node = node, next(iter(tail))
            lengths.append(length)
        self.assertEqual(sorted(lengths), [1, 2, 4])
        self.assertEqual(len(visited), 8)

    def test_split_arm_labels_and_equations(self):
        result = run(SPLIT, "intrinsic")
        dual = result["dual_complex"]
        self.assertEqual(result["root_families"]["alpha12"]["degree"], 2)
        self.assertEqual(Fraction(result["root_families"]["alpha12"]["power_value"]), Fraction(-3, 2))
        by_id = {v["id"]: v for v in dual["vertices"]}
        central = next(v["id"] for v in by_id.values() if v.get("label") == "Central curve")
        adj = neighbors(dual)
        for root in ("alpha12_1", "alpha12_2"):
            arm = {v["id"] for v in by_id.values() if v.get("root") == root}
            self.assertEqual(len(arm), 3)
            ends = arm & adj[central]
            self.assertEqual(len(ends), 1)
            other_ends = [v for v in arm if len(adj[v]) == 1]
            self.assertEqual(len(other_ends), 1)
            for v in arm:
                self.assertTrue(adj[v] <= arm | {central})
        for chart in result["charts"]:
            coords = chart["coordinates"]
            strict = polynomial(chart["strict_transform"], coords)
            axes = defaultdict(set)
            for divisor in chart["exceptional"]:
                root = by_id[divisor["id"]].get("root")
                if root not in {"alpha12_1", "alpha12_2"}:
                    continue
                axis, eq = divisor["ideal"]
                axes[axis].add(root)
                # Eliminate the two conjugate roots and recover exactly the
                # restriction of the strict transform to this axis divisor.
                pos, neg = eq.split(f"-{root}*")
                product = polynomial(f"2*({pos})^2+3*({neg})^2", coords)
                restriction = {e: c for e, c in strict.items() if e[coords.index(axis)] == 0}
                self.assertEqual(product, restriction)
            for roots in axes.values():
                self.assertEqual(roots, {"alpha12_1", "alpha12_2"})

    def test_fan_coverage_and_ambient_dual(self):
        for equation in [E8, SPLIT, "x^3+y^5+z^7"]:
            with self.subTest(equation=equation):
                result = run(equation, "embedded")
                fan = result["fan"]
                rays, cones = fan["rays"], fan["cones"]
                self.assertEqual(rays[:3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                self.assertTrue(all(min(r) > 0 for r in rays[3:]))
                edges = defaultdict(list)
                total_area = Fraction(0)
                expected = set()
                for cone in cones:
                    rr = [rays[i] for i in cone]
                    det = determinant(*rr)
                    self.assertEqual(det, 1)
                    # Project the cones to x+y+z=1: this is the ratio of the
                    # triangle area to the coordinate simplex area.
                    total_area += Fraction(det, prod(sum(r) for r in rr))
                    for i, j in zip(cone, cone[1:] + cone[:1]):
                        edges[tuple(sorted((i, j)))].append((i, j))
                    exc = [f"E{i-2}" for i in cone if i >= 3]
                    for k in range(2, len(exc)+1):
                        expected.update(frozenset(face) for face in combinations(exc, k))
                self.assertEqual(total_area, 1)
                for edge, oriented in edges.items():
                    if max(edge) < 3:
                        self.assertEqual(len(oriented), 1)
                    else:
                        self.assertEqual(len(oriented), 2)
                        self.assertEqual(oriented[0], oriented[1][::-1])
                cells = [frozenset(c["vertices"]) for c in result["dual_complex"]["cells"]]
                self.assertEqual(Counter(cells), Counter(expected))
                self.assertEqual(len(result["dual_complex"]["vertices"]), len(rays)-3)

    def test_chart_pullbacks(self):
        for equation in (E8, SPLIT):
            for chart in run(equation, "embedded")["charts"]:
                mapped = re.sub(r'[xyz]', lambda m: "("+chart["map"][m[0]]+")", equation)
                full = f"({chart['pullback_factor']})*({chart['strict_transform']})"
                self.assertEqual(polynomial(mapped, chart["coordinates"]),
                                 polynomial(full, chart["coordinates"]), chart["id"])

    def test_weighted_face_and_local_scope(self):
        for eq in ("x^2+y^4+z^6+x*y^2", "x^2+y^4+z^6+x*y", "x^2+y^4"):
            self.assertIsNone(resolve_toric(parse(eq, BASE), BASE, "embedded"))
        # The repeated factor is a unit at 0. The off-origin hypersurface
        # remains singular along x=1 because the fan is unchanged there.
        local = run(f"({E8})*(1-x)^2", "intrinsic")
        homogeneous = run(E8, "intrinsic")
        self.assertEqual(local["certificate"]["weights"], [15, 10, 6])
        self.assertEqual(local["dual_complex"], homogeneous["dual_complex"])
        self.assertEqual(local["fan"], homogeneous["fan"])


if __name__ == "__main__":
    unittest.main()
