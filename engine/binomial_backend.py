"""Resolve a binomial by blowups of smooth coordinate strata."""
from .binomial_complex import binomial_branch_equations, binomial_complex
from .binomial_fan import build_binomial_fan
from .binomial_kernel import pairings, primitive_character
from .chart_batch import transform_batch
from .toric_backend import monomial
from .polynomial import format_poly


def resolve_binomial(poly, variables, mode, compute=None):
    if len(poly) != 2:
        return None
    left, right = sorted(poly, reverse=True)
    if any(a and b for a, b in zip(left, right)):
        return None
    delta = [a-b for a,b in zip(left,right)]
    degree, direction = primitive_character(delta)

    fan = build_binomial_fan([left, right], variables=variables)
    if fan is None:
        return None
    rays, cones = fan["rays"], fan["cones"]
    values = pairings(rays,(left,right))
    signs = tuple(v for v, in pairings(rays,(direction,))) if degree > 1 else ()
    orders = [min(row) for row in values]
    coefficient = -poly[right] / poly[left]
    dual, divisors = binomial_complex(rays,cones,delta,coefficient,mode,orders=orders)
    exponents = [left, right]
    if max(map(sum, exponents))*max(map(max,rays)) <= 2**31-1:
        transforms,batch = transform_batch(exponents,rays,cones,orders,compute=compute)
    else:
        powers = [(a-order,b-order) for (a,b),order in zip(values,orders)]
        transforms = [[tuple(powers[i][j] for i in cone) for j in range(2)] for cone in cones]
        batch = {"backend":"cpu","arithmetic":"arbitrary precision",
                 "dimension":len(variables),"chart_term_pairs":2*len(cones)}
    coords = [f"u{i+1}" for i in range(len(variables))]
    charts = []
    for i,cone in enumerate(cones):
        mapping = {v:monomial(coords,[rays[k][j] for k in cone]) for j,v in enumerate(variables)}
        a,b = transforms[i]
        strict = format_poly({a:poly[left],b:poly[right]},coords)
        charts.append({"id":f"B{i+1}","coordinates":coords,"ray_indices":cone,
                       "map":mapping,"strict_transform":strict,"exceptional":divisors[i],
                       "pullback_factor":monomial(coords,[orders[k] for k in cone])})
        if degree>1:
            charts[-1]["strict_transform_components"] = binomial_branch_equations(
                rays, cone, delta, coefficient, values=tuple(signs[r] for r in cone))
    result = {"status":"resolved","engine":"binomial-coordinate-blowups",
            "summary":f"Resolved the binomial by {len(fan['steps'])} smooth blowups in {len(charts)} affine charts.",
            "charts":charts,"steps":fan["steps"],"fan":fan,"dual_complex":dual,
            "certificate":{"method":"coordinate blowups and exact binomial restrictions to exceptional strata",
                           "primitive_difference":list(direction),"character_degree":degree,"chart_batch":batch,
                           "description":"The centers lie over the singular locus. Each strict equation either has a unit term, giving disjoint smooth torus sections, or is linear in a coordinate outside the exceptional divisor."}}
    if degree>1:
        result["root_families"] = dual["root_families"]
    return result
