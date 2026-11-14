"""Exact Euler characteristics of open total-transform components over zero."""
import ctypes
from itertools import combinations
from operator import index

from .polynomial import parse


def _volume(points, dimension):
    """Normalized Newton polytope volume in the full orbit lattice."""
    from .groebner import CHECK, _MODE, _kernel
    points = tuple(dict.fromkeys(tuple(map(index, p)) for p in points))
    if any(len(p) != dimension for p in points):
        raise ValueError("Each lattice point must have the stated dimension.")
    if len(points) <= dimension:
        return 0
    library = _kernel()
    if _MODE == "sympy" or dimension == 0 or not hasattr(library, "atlas_lattice_volume"):
        return _volume_reference(points, dimension)
    operation = library.atlas_lattice_volume
    operation.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    operation.restype = ctypes.c_void_p
    payload = ("\n".join([str(dimension), *(" ".join(map(str, p)) for p in points)]) + "\n").encode()
    result = operation(payload, len(payload))
    if not result:
        raise ArithmeticError("The kernel could not compute the lattice volume.")
    try:
        value = int(ctypes.string_at(result))
    finally:
        library.atlas_groebner_release(result)
    if CHECK and value != _volume_reference(points, dimension):
        raise AssertionError("The normalized lattice volumes differ.")
    return value


def _volume_reference(points, dimension):
    """Normalized volume in the full orbit character lattice Z^dimension.

    The incremental triangulation cones an outside point over every visible
    boundary simplex. Integer supporting forms evaluate to signed normalized
    simplex volumes, including the lattice index of each boundary simplex.
    """
    import sympy as sp
    points = tuple(dict.fromkeys(points))
    if len(points) <= dimension:
        return 0
    if dimension == 1:
        return max(p[0] for p in points) - min(p[0] for p in points)
    origin = points[0]
    differences = [[a-b for a,b in zip(p, origin)] for p in points[1:]]
    if len(points) == dimension + 1:
        return abs(int(sp.Matrix(differences).det()))
    # At most dimension pivot columns are needed from this integer matrix.
    pivots = sp.Matrix(differences).T.rref()[1]
    if len(pivots) < dimension:
        return 0
    simplex = (0, *(1+i for i in pivots))
    total = abs(int(sp.Matrix([differences[i] for i in pivots]).det()))
    centroid_sum = tuple(map(sum,zip(*(points[i] for i in simplex))))

    def support(facet):
        base = points[facet[0]]
        rows = sp.Matrix([[a-b for a,b in zip(points[i],base)] for i in facet[1:]])
        # Removing one column from the (d-1) by d edge matrix produces
        # the maximal minors of the oriented facet.
        normal = []
        for j in range(dimension):
            minor = rows[:, [k for k in range(dimension) if k != j]]
            normal.append((-1)**j * int(minor.det()))
        offset = sum(a*b for a,b in zip(normal, base))
        if sum(a*b for a,b in zip(normal,centroid_sum)) < (dimension+1)*offset:
            normal = [-a for a in normal]
            offset = -offset
        return normal, offset

    facets = {facet:support(facet) for facet in combinations(simplex,dimension)}
    for index, point in enumerate(points):
        if index in simplex:
            continue
        horizon = set()
        for facet,(normal,offset) in list(facets.items()):
            height = sum(a*b for a,b in zip(normal,point)) - offset
            if height >= 0:
                continue
            total -= height
            del facets[facet]
            for ridge in combinations(facet,dimension-1):
                if ridge in horizon:
                    horizon.remove(ridge)
                else:
                    horizon.add(ridge)
        for ridge in horizon:
            facet = tuple(sorted((*ridge,index)))
            facets[facet] = support(facet)
    return total


def toric_contributions(result, poly, variables):
    """Return A'Campo rows for a certified toric atlas, or None to defer.

    Works in either resolution mode: the fan gives the ambient modification.
    A nonreduced embedded atlas supplies its reduced support for cutting out
    the strict transform and its verified original orders for the weights.
    """
    engine = result.get("engine")
    if result.get("status") != "resolved" or engine not in {
            "newton-polyhedron", "native-toric", "binomial-coordinate-blowups"}:
        return None
    fan = result["fan"]
    rays, cones = fan["rays"], fan["cones"]
    n = len(variables)
    source = result.get("input_divisor", {}).get("reduced_equation")
    if engine == "newton-polyhedron":
        source = result["certificate"]["adapted_polynomial"]
    support = parse(source, list(variables), bounded=False) if source else poly
    exponents = list(support)
    if len(rays) == n:
        # Smooth reduced support in the identity atlas. Even with a local
        # unit and a power, its unique component through zero has order
        # equal to the least degree of the original, unadapted input.
        if min(map(sum,exponents)) == 1:
            return [{"component":"smooth strict component", "multiplicity":min(map(sum,poly)),
                     "euler_characteristic":1}]
        return None
    from .binomial_kernel import pairings
    valuations = pairings(rays, exponents)
    orders = [min(row) for row in valuations]
    if any(orders[:n]) or min(orders[n:]) < 1:
        return None
    minimizers = [frozenset(j for j,q in enumerate(row) if q == m)
                  for row,m in zip(valuations,orders)]
    if result.get("input_divisor"):
        weights = {v["id"]:v.get("total_multiplicity") for v in result["dual_complex"]["vertices"]}
        for i in range(n,len(rays)):
            weight = weights.get(f"E{i-n+1}")
            if type(weight) is not int or weight < 1:
                return None
            orders[i] = weight

    # The relative interior of a cone maps to the origin precisely when its
    # rays together have positive support in every original coordinate.
    # Each orbit in E_i° has exactly one exceptional ray. A fiber orbit
    # with only original rays would need the entire unsubdivided orthant.
    ray_support = [sum(1 << j for j,r in enumerate(ray) if r) for ray in rays]
    full_support = (1 << n)-1
    seen = set()
    volumes = {}
    euler = [0]*len(rays)
    for cone in cones:
        original = [i for i in cone if i < n]
        exceptional = [i for i in cone if i >= n]
        if not exceptional:
            return None
        for k in range(len(original)+1):
            for original_face in combinations(sorted(original), k):
                base_support = sum(1 << i for i in original_face)
                for i in exceptional:
                    if base_support | ray_support[i] != full_support:
                        continue
                    orbit = (*original_face,i)
                    if orbit in seen:
                        continue
                    seen.add(orbit)
                    face = frozenset.intersection(*(minimizers[r] for r in orbit))
                    if not face:
                        continue  # both binomial terms vanished on this orbit
                    d = n-len(orbit)
                    if not d:
                        euler[i] += 1  # constant nonzero residual on a point
                    elif len(face) > d:
                        # In the unimodular chart the remaining coordinates
                        # form an integral basis for this orbit's characters.
                        # Full-rank faces fill their saturated affine lattice;
                        # normalized volume therefore depends only on face,d.
                        key = (face,d)
                        if key not in volumes:
                            remaining = [r for r in cone if r not in orbit]
                            points = [tuple(valuations[r][j] for r in remaining) for j in sorted(face)]
                            volumes[key] = _volume(points,d)
                        # chi((C*)^d)=0. For the nondegenerate hypersurface
                        # chi(H)=(-1)^(d-1) Vol_Z(P) (Danilov–Khovanskii).
                        euler[i] += (-1)**d * volumes[key]
    return [{"component":f"E{i-n+1}", "multiplicity":orders[i], "euler_characteristic":euler[i],
             "ray_index":i} for i in range(n,len(rays))]
