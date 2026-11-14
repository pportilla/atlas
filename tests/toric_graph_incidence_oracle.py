"""Independent Singular absolute-factor oracle for exceptional intersections on graph fans."""
from collections import Counter
from itertools import combinations
import re
import subprocess

from engine.polynomial import format_poly, parse_expression


def singular_jacobians(record):
    graph = record["graph"]
    names = [*record["variables"], graph["auxiliary"]]
    fan = record["fan"]
    polynomials = []
    minima = []
    for equation in ("equation", "divisor"):
        rows = sorted(parse_expression(graph[equation], names, _reference=True).items())
        polynomials.append(rows)
        masks = []
        for ray in fan["rays"]:
            weights = [sum(int(a)*b for a, b in zip(ray, exponent)) for exponent, _ in rows]
            minimum = min(weights)
            masks.append(sum(1 << i for i, weight in enumerate(weights) if weight == minimum))
        minima.append(masks)
    all_terms = tuple((1 << len(rows))-1 for rows in polynomials)
    pairs = set()
    graph_tests = set()
    for cone in fan["cones"]:
        # Expand all subsets by adjoining one ray. Repeated orbits are harmless
        # and each distinct initial ideal is sent to Singular only once.
        subsets = [(all_terms[0], all_terms[1], False)]
        for ray in cone:
            gm, fm = minima[0][ray], minima[1][ray]
            new_exceptional = ray >= len(names)
            subsets += [(g & gm, f & fm, exc or new_exceptional) for g, f, exc in subsets]
        for g, f, exceptional in subsets:
            if not g or not f:
                raise ValueError("Cone crosses a Newton wall")
            if exceptional and g.bit_count() > 1:
                graph_tests.add(g)
                if f.bit_count() > 1:
                    pairs.add((g, f))

    def face(row, mask):
        return format_poly(dict(entry for i, entry in enumerate(polynomials[row]) if mask >> i & 1), names)
    inverse = "torus_inverse"
    while inverse in names:
        inverse += "_"
    body = [f'ring R=0,({",".join([*names, inverse])}),dp;poly inv={inverse}*{"*".join(names)}-1;']
    for g in sorted(graph_tests):
        body.append(f'poly g={face(0,g)};ideal J=std(ideal(inv,g)+jacob(g));'
                    'print("CHECK "+string(size(J))+" "+string(J[1]));kill g,J;')
    minors = [f'diff(g,{a})*diff(f,{b})-diff(g,{b})*diff(f,{a})'
              for j, a in enumerate(names) for b in names[:j]]
    for g, f in sorted(pairs):
        body.append(f'poly g={face(0,g)};poly f={face(1,f)};ideal J=inv,g,f,{",".join(minors)};'
                    'J=std(J);print("CHECK "+string(size(J))+" "+string(J[1]));kill f,g,J;')
    body.append("quit;")
    run = subprocess.run(["Singular", "-q", "--no-rc", "--random=16"], text=True,
                         input="\n".join(body), capture_output=True, check=True)
    expected = len(graph_tests) + len(pairs)
    lines = [line for line in run.stdout.splitlines() if line.startswith("CHECK ")]
    if lines != ["CHECK 1 1"] * expected or "   ?" in run.stdout + run.stderr:
        raise ValueError("Singular failed an orbit Jacobian check: " + run.stdout[:500])
    return expected


def singular_incidence(record):
    graph = record["graph"]
    names = [*record["variables"], graph["auxiliary"]]
    poly = parse_expression(graph["equation"], names, _reference=True)
    terms = sorted(poly.items())
    rays = record["fan"]["rays"]
    initial = []
    for r in rays:
        weights = [sum(int(a) * b for a, b in zip(r, exponent)) for exponent, _ in terms]
        minimum = min(weights)
        initial.append(frozenset(i for i, value in enumerate(weights) if value == minimum))
    face_for = {}
    for maximal in record["fan"]["cones"]:
        exceptional = sorted(r for r in maximal if r >= len(names))
        for k in range(1, len(exceptional) + 1):
            for cone in combinations(exceptional, k):
                if cone not in face_for:
                    face_for[cone] = frozenset.intersection(*(initial[r] for r in cone))
    if any(not f for f in face_for.values()):
        raise ValueError("A cone crosses a wall of the graph's normal fan")
    faces = sorted({tuple(sorted(f)) for f in face_for.values() if len(f) > 1})
    body = ['LIB "absfact.lib";', f'ring R=0,({",".join(names)}),dp;',
            'proc nAbs(poly p){def Rabs=absFactorize(p);setring Rabs;return(absolute_factors[4]);}']
    for i, face in enumerate(faces):
        rows = [terms[j] for j in face]
        common = tuple(min(p[h] for p, _ in rows) for h in range(len(names)))
        # Monomials are invertible on the orbit torus. The ambient-character
        # lattice of this orbit is saturated, so a complementary torus factor
        # does not affect its geometric components.
        reduced = {tuple(a-b for a, b in zip(p, common)): c for p, c in rows}
        source = format_poly(reduced, names)
        body.append(f'print("FACTORS {i} "+string(nAbs({source})));')
    body.append("quit;")
    run = subprocess.run(["Singular", "-q", "--no-rc", "--random=16"], text=True,
                         input="\n".join(body), capture_output=True, check=True)
    output = run.stdout + run.stderr
    if any(line.lstrip().startswith("?") for line in output.splitlines()):
        raise ValueError(output)
    counts = {int(i): int(n) for i, n in re.findall(r"^FACTORS (\d+) (\d+)$", output, re.M)}
    if sorted(counts) != list(range(len(faces))) or min(counts.values(), default=1) < 1:
        raise ValueError(f"Missing absolute factorization result: {output[:1000]}")
    multiplicities = {frozenset(f): counts[i] for i, f in enumerate(faces)}
    incidence = {}
    for cone, f in face_for.items():
        degree = multiplicities.get(f, 0)
        for sheet in range(1, degree + 1):
            facets = []
            if len(cone) > 1:
                for smaller in combinations(cone, len(cone)-1):
                    outer = face_for[smaller]
                    # If the containing face has several absolute components,
                    # its common character/root identifies the same component
                    # on the child orbit. Equal face polynomials ensure this
                    # transport uses the same absolute factor.
                    if multiplicities[outer] > 1 and f != outer:
                        raise ValueError("A split containing face needs explicit factor restriction")
                    facets.append((smaller, sheet if multiplicities[outer] > 1 else 1))
            incidence[cone, sheet] = tuple(sorted(facets))
    return incidence, len(faces)


def atlas_signature(data):
    if data.get("status") != "complete":
        raise ValueError(f"Incomplete dual complex: {data.get('reason')}")
    labels = {}
    for vertex in data["vertices"]:
        labels[vertex["id"]] = ((vertex["ray_index"],), int(vertex.get("root_index", 1)))
    if len(set(labels.values())) != len(data["vertices"]):
        raise ValueError("Duplicate geometric vertex")
    incidence = {value: () for value in labels.values()}
    vertex_by_id = dict(labels)
    for row in sorted(data["cells"], key=lambda cell: cell["dimension"]):
        rays = tuple(sorted(vertex_by_id[v][0][0] for v in row["vertices"]))
        if len(set(rays)) != len(rays) or len(rays) != row["dimension"] + 1:
            raise ValueError(f"Invalid set of incident rays for {row['id']}")
        key = (rays, int(row.get("root_index", 1)))
        if key in incidence or row["id"] in labels:
            raise ValueError(f"Duplicate cell {row['id']}")
        incidence[key] = tuple(sorted(labels[f] for f in row["faces"]))
        labels[row["id"]] = key
    if len(labels) != len(data["vertices"]) + len(data["cells"]):
        raise ValueError("Duplicate vertex or cell IDs")
    return incidence


def compare(actual, expected):
    if actual != expected:
        for key in sorted(actual.keys() | expected.keys()):
            if actual.get(key) != expected.get(key):
                raise AssertionError(f"Cell {key}: Atlas {actual.get(key)}, expected {expected.get(key)}")
    counts = Counter(len(key[0])-1 for key in actual)
    return [counts[i] for i in range(max(counts, default=-1)+1)]
