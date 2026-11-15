"""Independent triangulation and toric-orbit monodromy checks."""
from itertools import combinations
from engine.monodromy_toric import _volume
from engine.polynomial import parse


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
    from engine.binomial_kernel import pairings
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
