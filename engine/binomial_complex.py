"""Geometric dual complex on a smooth terminal fan for a binomial.

In a chart of a unimodular fan, the strict equation has disjoint monomials
determined by the signs of the pairings with delta. Intersections are taken
with the exceptional coordinate hyperplanes, then glued across the toric
chart cover. All claims are over the algebraic closure of Q.
"""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from math import gcd

from .polynomial import LimitError


def _monomial(coordinates, powers):
    return "*".join(name if power == 1 else f"{name}^{power}"
                    for name, power in zip(coordinates,powers) if power) or "1"


def _equation(coordinates, values, coefficient):
    left = _monomial(coordinates, (max(0,q) for q in values))
    right = _monomial(coordinates, (max(0,-q) for q in values))
    c = abs(coefficient)
    term = right if c == 1 else str(c) if right == "1" else f"{c}*{right}"
    return left + ("-" if coefficient > 0 else "+") + term


def _root_equation(coordinates, values, root):
    left = _monomial(coordinates, (max(0,q) for q in values))
    right = _monomial(coordinates, (max(0,-q) for q in values))
    return f"{left}-{root}" if right=="1" else f"{left}-{root}*{right}"


def binomial_branch_equations(rays, cone, delta, coefficient):
    """Geometric strict branch equations for a nonprimitive terminal chart.

    The coefficient occurs in the defining equation of the common root family.
    Using the same oriented primitive character in every chart fixes the branch
    labels across overlaps, including when the unit monomial switches sides.
    """
    d = gcd(*delta)
    if not Fraction(coefficient) or d == 0:
        raise ValueError("Expected a nonzero binomial coefficient and exponent difference.")
    if d == 1:
        return []
    direction = tuple(e//d for e in delta)
    values = tuple(sum(a*b for a,b in zip(rays[i],direction)) for i in cone)
    if min(values)<0<max(values):
        raise ValueError("A split terminal binomial chart must have a unit monomial.")
    coordinates = [f"u{i+1}" for i in range(len(delta))]
    return [{"root":f"alpha1_{j}","root_index":j,
             "equation":_root_equation(coordinates,values,f"alpha1_{j}")}
            for j in range(1,d+1)]


def binomial_complex(rays, cones, delta, coefficient, mode, *, exceptional_rays=None, orders=None):
    """Return the dual complex and the list of exceptional ideals per affine chart."""
    if len(delta)<2 or any(not isinstance(q,int) for q in delta):
        raise ValueError("The exponent difference must be an integer vector.")
    degree = gcd(*delta)
    if degree<=1:
        return _primitive_complex(rays,cones,delta,coefficient,mode,
                                  exceptional_rays=exceptional_rays,orders=orders)
    coefficient = Fraction(coefficient)
    if not coefficient:
        raise ValueError("A torus binomial must have two nonzero coefficients.")
    direction = tuple(e//degree for e in delta)
    exceptional = frozenset(range(len(delta),len(rays)) if exceptional_rays is None else exceptional_rays)
    cones = tuple(tuple(c) for c in cones)
    pairs = sum(2**len(exceptional.intersection(c))-1 for c in cones)
    if pairs*(degree if mode=="intrinsic" else 1)>1_000_000:
        raise LimitError("The split binomial exceeds the geometric intersection-stratum budget.")
    signs = [sum(a*b for a,b in zip(ray,direction)) for ray in rays]
    for cone in cones:
        values = [signs[i] for i in cone]
        if min(values)<0<max(values):
            raise ValueError("Nonterminal split binomial: both monomials are nonconstant in a chart.")

    # On a sign-separated cone the strict equation is a d-th power of
    # a primitive character equal to c. Every variable in that character is
    # invertible on the strict transform, so its d distinct roots give
    # disjoint geometric branches everywhere, not just on the dense torus.
    dual, chart_divisors = _primitive_complex(rays,cones,direction,1,mode,
                                              exceptional_rays=exceptional,orders=orders)
    dual["root_families"]={"alpha1":{
        "degree":degree,"power_value":str(coefficient),"polynomial":f"T^{degree}-({coefficient})",
        "direction":list(direction),
        "definition":f"alpha1_j is root j of T^{degree} = {coefficient}, ordered by real part then imaginary part, starting at 1.",
    }}
    if mode=="embedded":
        return dual,chart_divisors

    def lift(record,j):
        r={**record,"id":f"{record['id']}.{j}","root":f"alpha1_{j}","root_index":j}
        for field in ("vertices","faces"):
            if field in r:
                r[field]=[f"{key}.{j}" for key in r[field]]
        return r

    for field in ("vertices","cells"):
        dual[field]=[lift(item,j) for item in dual[field] for j in range(1,degree+1)]
    for ci,divisors in enumerate(chart_divisors):
        if not divisors:
            continue
        branches=binomial_branch_equations(rays,cones[ci],delta,coefficient)
        expanded=[]
        for component in divisors:
            for branch in branches:
                j=branch["root_index"]
                ideal=[*component["ideal"],branch["equation"]]
                expanded.append({**lift(component,j),"root_equation":branch["equation"],
                                 "ideal":ideal,"ambient_ideal":list(ideal),"equation":", ".join(ideal)})
        chart_divisors[ci]=expanded
    return dual,chart_divisors


def _primitive_complex(rays, cones, delta, coefficient, mode, *, exceptional_rays=None, orders=None):
    """Return (dual, divisors_by_chart) using coordinate names u1,...,un.

    The caller certifies that the cones form a regular fan and that its
    modification resolves the binomial. delta is left exponent minus right
    exponent; the strict torus equation is X^delta = coefficient. Require
    primitive delta, so each surviving torus hypersurface is geometrically
    integral. By default ray indices >=n are the exceptional rays.

    Within a chart for C, the local piece on an exceptional subset S has a
    unique generic set Z of zero coordinates. Either Z=S, or a surviving
    linear monomial forces one additional coordinate to vanish. This also
    decides overlap gluing: a chart overlap inverts precisely the rays outside
    the common face. Every chart carrying the same (S,Z) contains that face
    and contains the same dense subvariety of its orbit. Pieces with different
    Z have different generic orbits. Thus repeated cells with the same list
    of exceptional ray indices remain distinct when necessary.
    """
    n = len(delta)
    if mode not in ("embedded","intrinsic"):
        raise ValueError("Resolution mode must be embedded or intrinsic.")
    if n < 2 or any(not isinstance(q,int) for q in delta) or gcd(*delta) != 1:
        raise ValueError("The exponent difference must be a primitive integer vector.")
    coefficient = Fraction(coefficient)
    if not coefficient:
        raise ValueError("A torus binomial must have two nonzero coefficients.")
    rays = tuple(tuple(r) for r in rays)
    cones = tuple(tuple(c) for c in cones)
    if any(len(r)!=n or min(r)<0 for r in rays):
        raise ValueError("The affine base requires rays of the given dimension in the positive orthant.")
    if any(len(c)!=n or len(set(c))!=n or min(c)<0 or max(c)>=len(rays) for c in cones):
        raise ValueError("Expected maximal cones with n distinct ray indices.")
    exceptional = frozenset(range(n,len(rays)) if exceptional_rays is None else exceptional_rays)
    if not exceptional.issubset(range(len(rays))):
        raise ValueError("Exceptional ray index is outside the fan.")
    if orders is not None and len(orders)!=len(rays):
        raise ValueError("Divisor orders must be indexed by ray.")
    if sum(2**len(exceptional.intersection(c))-1 for c in cones) > 1_000_000:
        raise LimitError("The binomial fan exceeds the intersection-stratum budget.")

    coords = [f"u{i+1}" for i in range(n)]
    pairings = tuple(sum(a*b for a,b in zip(r,delta)) for r in rays)
    ray_support = [sum(1<<i for i,v in enumerate(r) if v) for r in rays]
    full_support = (1<<n)-1
    # On a chart with one unit monomial, precisely the zero-pairing rays
    # may vanish on the strict transform. With two nonconstant terms the
    # toric chart origin lies on it. In either case the monomial map sends
    # a point to the base origin iff its zero rays cover every base coordinate.
    strata = defaultdict(dict)  # subset -> generic zero-ray set -> hits origin
    chart_pieces = []
    chart_equations = []
    witnesses = {}
    for ci, cone in enumerate(cones):
        positives = tuple(i for i in cone if pairings[i]>0)
        negatives = tuple(i for i in cone if pairings[i]<0)
        allowed_zero = cone if mode=="embedded" or (positives and negatives) else [
            i for i in cone if pairings[i]==0]
        support = 0
        for ray in allowed_zero:
            support |= ray_support[ray]
        meets_origin = support == full_support
        values = tuple(pairings[i] for i in cone)
        chart_equations.append(_equation(coords,values,coefficient))
        indices = sorted(exceptional.intersection(cone))
        pieces = {}
        for size in range(1,len(indices)+1):
            for subset in combinations(indices,size):
                zeros = subset
                if mode=="intrinsic":
                    kill_plus = any(pairings[i]>0 for i in subset)
                    kill_minus = any(pairings[i]<0 for i in subset)
                    if kill_plus and kill_minus:
                        raise ValueError("Nonterminal chart: both binomial terms vanish on an exceptional intersection.")
                    if kill_plus or kill_minus:
                        surviving = negatives if kill_plus else positives
                        if not surviving:
                            continue  # the remaining equation on this coordinate flat is a unit
                        if len(surviving)!=1 or abs(pairings[surviving[0]])!=1:
                            raise ValueError("Nonterminal chart: the surviving monomial is not a single linear coordinate.")
                        zeros = tuple(sorted((*subset,surviving[0])))
                pieces[subset] = zeros
                strata[subset][zeros] = strata[subset].get(zeros,False) or meets_origin
                witnesses.setdefault((subset,zeros),ci)
        chart_pieces.append(pieces)

    ids = {}
    vertices, cells = [], []
    ray_number = {r:i+1 for i,r in enumerate(sorted(exceptional))}
    for subset in sorted(strata,key=lambda t:(len(t),t)):
        zero_sets = sorted(z for z,hits in strata[subset].items() if hits)
        for local_no, zeros in enumerate(zero_sets,1):
            if len(subset)==1:
                ray, = subset
                ident = f"E{ray_number[ray]}" + (f".{local_no}" if len(zero_sets)>1 else "")
                vertex = {"id":ident, "ray_index":ray, "ray":list(rays[ray]),
                          "dimension":n-1-int(mode=="intrinsic"), "generic_zero_rays":list(zeros)}
                if orders is not None:
                    vertex["ambient_order"] = orders[ray]
                vertices.append(vertex)
            else:
                ident = f"C{len(cells)+1}"
                local = chart_pieces[witnesses[(subset,zeros)]]

                def parent(face):
                    return ids[face,local[face]]

                cells.append({"id":ident, "dimension":len(subset)-1,
                              "vertices":[parent((i,)) for i in subset],
                              "faces":[parent(face) for face in combinations(subset,len(subset)-1)],
                              "generic_zero_rays":list(zeros)})
            ids[(subset,zeros)] = ident

    chart_divisors = []
    for ci,(cone,pieces) in enumerate(zip(cones,chart_pieces)):
        divisors = []
        for axis, ray in enumerate(cone):
            subset = (ray,)
            zeros = pieces.get(subset)
            if zeros is None or (subset,zeros) not in ids:
                continue
            ideal = [coords[axis]]
            extra = [i for i in zeros if i!=ray]
            if extra:
                ideal.append(coords[cone.index(extra[0])])
            # 'ideal' is in the chart's strict ring for intrinsic mode. An
            # additional ambient presentation removes any ambiguity for reuse.
            ambient_ideal = ideal + ([chart_equations[ci]] if mode=="intrinsic" and not extra else [])
            divisors.append({"id":ids[(subset,zeros)], "ray_index":ray,
                             "equation":", ".join(ideal), "ideal":ideal,
                             "ambient_ideal":ambient_ideal})
        chart_divisors.append(divisors)
    convention = ("Reduced ambient exceptional divisor" if mode=="embedded" else
                  "Reduced exceptional divisor on the smooth strict transform")
    dual = {"status":"complete","convention":convention+
            " in the algebraic germ over Qbar; cells are geometric strata whose closures meet the origin fiber.",
            "vertices":vertices,"cells":cells}
    return dual,chart_divisors
