"""A'Campo contributions for smooth/SNC divisors and A-series surfaces.

A contribution (N,chi) represents (1-T^N)^(-chi) in the Lefschetz zeta
convention. chi is taken on the open divisor over the origin, with every
other component of the total transform removed. All input equations refer
to the original function; a nonvanishing unit preserves its multiplicities.
See N. A'Campo, La fonction zêta d'une monodromie, Comment. Math. Helv. 50
(1975), 233–248, https://doi.org/10.1007/BF02565748 .
"""

import re

from .polynomial import InputError, LimitError, parse


def _term(component, multiplicity, chi, method):
    return {"component":component,"multiplicity":multiplicity,
            "euler_characteristic":chi,"method":method}


def _chain_terms(result):
    # Only the verified native normal form is interpreted here. The expressions
    # in arbitrary saved JSON are never passed to a symbolic expression parser.
    certificate = result.get("certificate")
    chart = certificate.get("normalized_chart") if isinstance(certificate, dict) else None
    if not isinstance(chart,dict) or not isinstance(chart.get("equation"),str):
        return None
    match = re.fullmatch(r"a\*b\+c\^([0-9]{1,3})", chart["equation"])
    if not match or not 2 <= (k:=int(match.group(1))) <= 128:
        return None
    scale = 1
    if "input_divisor" in result:
        # uv+z^k is geometrically irreducible. Thus a nonreduced divisor
        # supported on it has exactly one active factor, of uniform weight m.
        info = result["input_divisor"]
        factors = info.get("factors",[]) if isinstance(info,dict) else []
        if not isinstance(factors,list) or len(factors)!=1 or not isinstance(factors[0],dict):
            return None
        scale = factors[0].get("multiplicity")
        if type(scale) is not int or scale<1:
            return None

    # Every E_i from a point blowup is P² before subsequent centers. On an
    # intermediate E_i the tangent section is the pair of lines uv=0.
    # The preceding divisor, if present, is the line z=0, making a triangle.
    # Either complement has chi=0. Later centers lie in the deleted boundary.
    terms = []
    for i in range(1,k//2+1):
        last_even = 2*i==k
        method = ("The last point exceptional is P² minus a smooth conic; a preceding "
                  "boundary line, if present, meets the conic twice." if last_even else
                  "The complement of the two tangent lines (and the preceding boundary line, if present) has Euler characteristic zero.")
        terms.append(_term(f"E{i}",2*i*scale,int(last_even),method))
    if k%2:
        # Over the final line, the projective-line fibers have two punctures
        # except where the strict and old exceptional sections meet. There
        # the fiber is once-punctured, with chi=1. The preceding divisor may
        # remove a different, twice-punctured fiber, contributing zero.
        label = f"E{(k+1)//2}"
        if result.get("mode")=="intrinsic":
            label += " (auxiliary embedded line blowup)"
        terms.append(_term(label,k*scale,1,
            "The final line exceptional has two deleted sections agreeing over one point. Only that once-punctured fiber contributes."))
    return terms


def _snc_terms(named):
    # In the identity resolution the inverse image of the origin is one point.
    # It belongs to the open part of one component iff exactly one passes through it.
    chi = int(len(named)==1)
    return [_term(name, multiplicity, chi,
                  "Identity resolution: the origin lies on a single smooth component." if chi else
                  "Identity resolution: this open component misses the origin, where multiple transverse components meet.")
            for name,multiplicity in named]


def _simple_coordinates(p, names):
    # If f=x^e*h with h(0) nonzero, the coordinate powers give the entire
    # divisor of the germ. This includes monomials multiplied by a general unit.
    powers = tuple(min(e[j] for e in p) for j in range(len(names)))
    if powers not in p or not any(powers):
        return None
    return [(x,m) for x,m in zip(names,powers) if m]


def _factor_snc(p, names):
    # This optional recognizer has a bounded workload. A QQ factor smooth
    # through the origin is a single geometric local branch. Independent
    # differentials make the active factors part of local analytic coordinates.
    import sympy as sp
    if len(p)>192 or len(names)>8 or max(map(sum,p))>24:
        return None
    xs=tuple(map(sp.Symbol,names))
    function=sp.Poly.from_dict(p,*xs,domain=sp.QQ)
    factors=[]
    gradients=[]
    zero=(0,)*len(xs)
    for factor, power in function.factor_list()[1]:
        if factor.coeff_monomial(zero):
            continue
        gradients.append([factor.diff(x).coeff_monomial(zero) for x in xs])
        factors.append((str(factor.as_expr()).replace("**","^"),power))
    if not factors or len(factors)>len(xs) or sp.Matrix(gradients).rank()!=len(factors):
        return None
    return factors


def _recorded_snc(factors, names):
    """Test transversality using the exact factorization retained by the divisor lift."""
    import sympy as sp
    if not isinstance(factors,list) or not factors or not isinstance(names,list) or len(factors)>len(names):
        return None
    rows, data = [], []
    for item in factors:
        if not isinstance(item,dict) or type(item.get("multiplicity")) is not int or item["multiplicity"]<1:
            return None
        name=item.get("equation")
        try:
            factor=parse(name,names)
        except (InputError,LimitError,TypeError,RecursionError):
            return None
        rows.append([factor.get(tuple(int(a==j) for a in range(len(names))),0) for j in range(len(names))])
        data.append((name,item["multiplicity"]))
    return data if sp.Matrix(rows).rank()==len(data) else None


def crossing_components(result):
    """The components (name, multiplicity) of the divisor through the origin
    when it has normal crossings there, so that the identity is an embedded
    resolution near the origin; None if this is not recognized."""
    if not isinstance(result,dict) or result.get("status")!="resolved":
        return None
    variables=result.get("variables")
    lift=result.get("input_divisor")
    if isinstance(lift,dict):
        recorded=_recorded_snc(lift.get("factors"),variables)
        if recorded is not None:
            return recorded
    try:
        p=parse(result.get("polynomial"),variables)
    except (InputError,LimitError,TypeError,ValueError,RecursionError):
        # A very large smooth input can still carry an explicit native
        # certificate. In a lifted atlas the weights refer to the original f.
        certificate=result.get("certificate")
        if isinstance(certificate,dict) and certificate.get("method")=="nonzero linear part":
            if "input_divisor" not in result:
                return [("smooth strict component",1)]
            info=result["input_divisor"]
            factors=info.get("factors") if isinstance(info,dict) else None
            if isinstance(factors,list) and len(factors)==1 and isinstance(factors[0],dict):
                m=factors[0].get("multiplicity")
                if type(m) is int and m>0:
                    return [("smooth support component",m)]
        return None
    if min(map(sum,p))==1:
        return [("smooth strict component",1)]
    coordinate=_simple_coordinates(p,variables)
    if coordinate is not None:
        return coordinate
    return _factor_snc(p,variables)


def special_contributions(result):
    """Return exact local contributions or None if these formulas do not apply."""
    if not isinstance(result,dict) or result.get("status")!="resolved":
        return None
    if result.get("engine")=="native-a-series":
        return _chain_terms(result)
    named=crossing_components(result)
    return None if named is None else _snc_terms(named)
