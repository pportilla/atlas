"""Geometric dual complex on a smooth terminal fan for a binomial.

In a chart of a unimodular fan, the strict equation has disjoint monomials
determined by the signs of the pairings with delta. Intersections are taken
with the exceptional coordinate hyperplanes, then glued across the toric
chart cover. All claims are over the algebraic closure of Q.
"""

from fractions import Fraction

from .incidence import incidence
from .kernel_json import kernel_json
from .polynomial import LimitError, rational_text
from .binomial_kernel import _encode, pairings, primitive_character, run, signed_powers


def _monomial(coordinates, powers):
    return "*".join(name if power == 1 else f"{name}^{_encode(power)}"
                    for name, power in zip(coordinates,powers) if power) or "1"


def _equation(coordinates, values, coefficient):
    plus, minus = signed_powers(tuple(values))
    left, right = _monomial(coordinates, plus), _monomial(coordinates, minus)
    text = rational_text(coefficient)
    absolute = text.lstrip("-")
    term = right if absolute == "1" else absolute if right == "1" else f"{absolute}*{right}"
    return left + ("+" if text.startswith("-") else "-") + term


def binomial_branch_equations(rays, cone, delta, coefficient, *, values=None):
    """Geometric strict branch equations for a nonprimitive terminal chart.

    The coefficient occurs in the defining equation of the common root family.
    Using the same oriented primitive character in every chart fixes the branch
    labels across overlaps, including when the unit monomial switches sides.
    """
    d, direction = primitive_character(delta)
    if not Fraction(coefficient) or d == 0:
        raise ValueError("Expected a nonzero binomial coefficient and exponent difference.")
    if d == 1:
        return []
    if values is None:
        values = tuple(q for q, in pairings([rays[i] for i in cone],(direction,)))
    if run(f"separated {len(values)} 1", values) == ["0"]:
        raise ValueError("A split terminal binomial chart must have a unit monomial.")
    coordinates = [f"u{i+1}" for i in range(len(delta))]
    plus, minus = signed_powers(tuple(values))
    left, right = _monomial(coordinates, plus), _monomial(coordinates, minus)
    suffix = "" if right == "1" else f"*{right}"
    branches = []
    for j in range(1, d+1):
        root = f"alpha1_{j}"
        branches.append({"root": root, "root_index": j, "equation": f"{left}-{root}{suffix}"})
    return branches


def binomial_complex(rays, cones, delta, coefficient, mode, *, exceptional_rays=None, orders=None):
    """Return the dual complex and the list of exceptional ideals per affine chart."""
    if len(delta)<2 or any(not isinstance(q,int) for q in delta):
        raise ValueError("The exponent difference must be an integer vector.")
    degree, direction = primitive_character(delta)
    if degree<=1:
        return _primitive_complex(rays,cones,delta,coefficient,mode,
                                  exceptional_rays=exceptional_rays,orders=orders)
    coefficient = Fraction(coefficient)
    if not coefficient:
        raise ValueError("A torus binomial must have two nonzero coefficients.")
    exceptional = frozenset(range(len(delta),len(rays)) if exceptional_rays is None else exceptional_rays)
    cones = tuple(tuple(c) for c in cones)
    if not _within_budget(cones, exceptional, len(delta), degree if mode == "intrinsic" else 1):
        raise LimitError("The split binomial exceeds the geometric intersection-stratum budget.")
    signs = tuple(q for q, in pairings(rays,(direction,)))
    weights = [tuple(signs[i] for i in cone) for cone in cones]
    if run(f"separated {len(delta)} {len(cones)}", *weights) == ["0"]:
        raise ValueError("Nonterminal split binomial: both monomials are nonconstant in a chart.")

    # On a sign-separated cone the strict equation is a d-th power of
    # a primitive character equal to c. Every variable in that character is
    # invertible on the strict transform, so its d distinct roots give
    # disjoint geometric branches everywhere, not just on the dense torus.
    dual, chart_divisors = _primitive_complex(rays,cones,direction,1,mode,
                                              exceptional_rays=exceptional,orders=orders)
    value, degree_text = rational_text(coefficient), _encode(degree)
    dual["root_families"]={"alpha1":{
        "degree":degree,"power_value":value,"polynomial":f"T^{degree_text}-({value})",
        "direction":list(direction),
        "definition":f"alpha1_j is root j of T^{degree_text} = {value}, ordered by real part then imaginary part, starting at 1.",
    }}
    if mode=="embedded":
        return dual,chart_divisors

    sources = [*dual["vertices"], *dual["cells"]]
    layout = [{k: item[k] for k in ("id", "vertices", "faces") if k in item} for item in sources]
    sheets = kernel_json("atlas_torus_sections", "binomial_sheets", degree=degree_text, layout=layout)["layout"]
    lifted = {}
    for item, copies in zip(sources, sheets, strict=True):
        for copy in copies:
            copy["root_index"] = int(copy["root_index"])
        lifted[item["id"]] = copies
    for field in ("vertices","cells"):
        dual[field]=[{**item, **copy} for item in dual[field] for copy in lifted[item["id"]]]
    for ci,divisors in enumerate(chart_divisors):
        if not divisors:
            continue
        branches=binomial_branch_equations(rays,cones[ci],delta,coefficient,values=tuple(signs[r] for r in cones[ci]))
        expanded=[]
        for component in divisors:
            for copy, branch in zip(lifted[component["id"]], branches, strict=True):
                ideal=[*component["ideal"],branch["equation"]]
                expanded.append({**component, **copy, "root_equation":branch["equation"],
                                 "ideal":ideal,"ambient_ideal":list(ideal),"equation":", ".join(ideal)})
        chart_divisors[ci]=expanded
    return dual,chart_divisors


def _within_budget(cones, exceptional, n, copies):
    status = run(f"terminal_sections {n} {len(cones)} {len(exceptional)}", (copies,), sorted(exceptional), *cones)
    return status == ["ok"]


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
    if n < 2 or any(not isinstance(q,int) for q in delta) or primitive_character(delta)[0] != 1:
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
    if not _within_budget(cones, exceptional, n, 1):
        raise LimitError("The binomial fan exceeds the intersection-stratum budget.")

    coords = [f"u{i+1}" for i in range(n)]
    from . import binomial_kernel
    rows = binomial_kernel.run(f"incidence {n} {len(rays)} {len(cones)} {len(exceptional)} {int(mode=='intrinsic')}",
                               delta, sorted(exceptional), *rays, *cones)
    pairings, pieces, charts = binomial_kernel.incidence(rows)
    layout = incidence("binomial_layout", width=n, intrinsic=mode=="intrinsic",
                       exceptional=sorted(exceptional), pieces=pieces)
    names = [record["id"] for record in layout]
    vertices, cells = [], []
    for record in layout:
        if "ray_index" in record:
            ray = record["ray_index"]
            record["ray"] = list(rays[ray])
            if orders is not None:
                record["ambient_order"] = orders[ray]
            vertices.append(record)
        else:
            cells.append(record)
    chart_divisors = []
    for cone, visible in zip(cones,charts):
        divisors = []
        equation = _equation(coords,tuple(pairings[r] for r in cone),coefficient)
        for axis,ray in enumerate(cone):
            if ray not in visible:
                continue
            number = visible[ray]
            zeros = pieces[number][1]
            extra = [r for r in zeros if r!=ray]
            ideal = [coords[axis], *(coords[cone.index(r)] for r in extra)]
            ambient = [*ideal, equation] if mode=="intrinsic" and not extra else ideal.copy()
            divisors.append({"id":names[number], "ray_index":ray, "equation":", ".join(ideal),
                             "ideal":ideal, "ambient_ideal":ambient})
        chart_divisors.append(divisors)
    convention = ("Reduced ambient exceptional divisor" if mode=="embedded" else
                  "Reduced exceptional divisor on the smooth strict transform")
    dual = {"status":"complete","convention":convention+
            " in the algebraic germ over Qbar; cells are geometric strata whose closures meet the origin fiber.",
            "vertices":vertices,"cells":cells}
    return dual,chart_divisors
