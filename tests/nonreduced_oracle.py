"""Lift an embedded resolution of a reduced support to its effective divisor.

Free affine charts use exact integer valuations of the pulled-back factors.
On a graph chart the strict ideal is saturated inside its ambient equations.
Every ambient exceptional prime is locally Cartier on the smooth chart, so
membership in its ordinary powers computes its generic order of vanishing.
"""
from functools import cmp_to_key, lru_cache
from math import prod
import re

from engine.reference import sp, QQ

from engine.blowup_ideal import ideal_basis, ideal_subset, saturate
from engine.polynomial import InputError


def _text(expr):
    return str(expr).replace("**","^")


def _read(source, names):
    from engine.atlas import _expression
    return _expression(source, names)


@lru_cache(maxsize=64)
def _flint_roots(coefficients):
    """FLINT root order: real decreasing, then Re decreasing, |Im| increasing, Im positive first.

    Distinct real parts of algebraic roots are separated by exact real-root
    isolation of Res_T(p(T),p(2X-T)). No displayed decimal is used.
    """
    x,t=sp.symbols("X T")
    p=sp.Poly.from_list(list(reversed(coefficients)), x)
    nr=int(p.count_roots(-sp.oo,sp.oo)); degree=p.degree()
    exact = p.all_roots(radicals=True)
    if nr==degree:
        return tuple(reversed(exact))
    if degree-nr==2:
        return (*reversed(exact[:nr]),exact[-1],exact[-2])
    sums=sp.resultant(p.as_expr().subs(x,t),p.as_expr().subs(x,2*x-t),t)
    bins=sp.Poly(sums,x).sqf_part().intervals(eps=sp.Rational(1,1000))
    positives=[]
    for j in range(nr+1,degree,2):
        # Complex conjugates occupy adjacent SymPy indices, negative first.
        root=sp.CRootOf._new(p,j)
        rect=root._get_interval()
        if rect.ay<0:
            raise ArithmeticError("Unexpected complex root ordering.")
        while True:
            matches=[k for k,((a,b),_) in enumerate(bins) if a<=rect.bx and rect.ax<=b]
            if len(matches)==1:
                break
            rect=rect.refine()
        positives.append([matches[0],rect,j])
    def compare(a,b):
        if a[0]!=b[0]:
            return 1 if a[0]<b[0] else -1
        while True:
            if a[1].by<=b[1].ay: return -1
            if b[1].by<=a[1].ay: return 1
            a[1]=a[1].refine();b[1]=b[1].refine()
    ans=list(reversed(exact[:nr]))
    for _,_,j in sorted(positives,key=cmp_to_key(compare)):
        ans.extend([exact[j],exact[j-1]])
    return tuple(ans)


def _algebraics(records, used):
    values={}
    for name in records:
        if name not in used: continue
        raw=records[name]["exact"]
        match=re.fullmatch(r"PolynomialRootIndexed\(List\(([-\d, ]+)\), (\d+)\)",raw)
        if not match:
            raise ArithmeticError("An exact algebraic chart coefficient was not recognized.")
        coefficients=tuple(int(i) for i in match[1].split(","))
        values[name]=_flint_roots(coefficients)[int(match[2])-1]
    return values


def _power_product(factors):
    if not factors: return "1"
    return "*".join(f"({text})^{m}" if m!=1 else f"({text})" for text,m in factors)


def _valuation(variables, ambient, prime, total, domain):
    # In the regular ambient ring this height-one prime is invertible. Its
    # positive powers have no embedded primes; P^k=P^(k).
    prime=tuple(f for f in prime if not ideal_subset(variables,(f,),ambient,domain=domain))
    if not prime: raise ArithmeticError("A reported divisor has zero ambient ideal.")
    current=ideal_basis(variables,(*ambient,*prime),domain=domain)
    order=0
    while ideal_subset(variables,(total,),current,domain=domain):
        order+=1
        current=ideal_basis(variables,(*ambient,*(p*q for p in prime for q in current)),domain=domain)
    return order


def _uniform_lift(atlas, multiplicity, unit, symbols):
    """When every branch has the same weight, take powers directly in each
    existing chart ring. This preserves FLINT's algebraic numbers verbatim."""
    vertices=atlas["dual_complex"]["vertices"]
    weights={}
    for vertex in vertices:
        value=next((vertex[key] for key in ("total_multiplicity","ambient_order","multiplicity")
                    if key in vertex),None)
        if value is None: return False
        weights[vertex["id"]]=value*multiplicity
    divisors=[]
    for chart in atlas["charts"]:
        strict=chart["strict_transform"]
        if not isinstance(strict,str) or "ambient_ideal" in chart:
            return False
        factor=chart.get("pullback_factor")
        if factor is None:
            total=chart.get("ambient_total_transform")
            suffix=f"*({strict})"
            if total and total.endswith(suffix):
                factor=total[:-len(suffix)]
            elif not chart.get("exceptional"):
                factor="1"
            else:
                return False
        divisors.append(factor)
    for chart, factor in zip(atlas["charts"],divisors):
        strict=chart["strict_transform"]
        chart["reduced_strict_transform"]=strict
        if "strict_transform_on_exceptional" in chart:
            chart["reduced_strict_transform_on_exceptional"]=chart.pop("strict_transform_on_exceptional")
        power=[(strict,multiplicity)]
        if unit != 1:
            # Keep the published algebraic labels. Substitution with these
            # labels as formal coordinates specializes to the same unit.
            from engine.affine_kernel import pullbacks
            names = {a: sp.Symbol(a) for a in chart["coordinates"]}
            needed = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", " ".join(chart["map"].values())))
            coefficients = sorted(needed & atlas.get("algebraic_numbers", {}).keys())
            gen = atlas.get("coefficient_field", {}).get("generator")
            if gen in needed:
                coefficients.append(gen)
            names.update((a, sp.Symbol(a)) for a in coefficients)
            mapping = tuple(_read(chart["map"][str(v)], names) for v in symbols)
            pulled, = pullbacks(tuple(names.values()), mapping, symbols, (unit,))
            power.insert(0, (_text(pulled), 1))
        chart["strict_transform"]=_power_product(power)
        chart["pullback_factor"]=_power_product([(factor,multiplicity)]) if factor!="1" else "1"
        chart["ambient_total_transform"]=f"({chart['pullback_factor']})*({chart['strict_transform']})"
        for c in (*chart.get("exceptional",()),*chart.get("ambient_components",())):
            for key in ("total_multiplicity","multiplicity","ambient_order"):
                if key in c: c[key]=weights[c["id"]]
            c["total_multiplicity"]=weights[c["id"]]
    _set_global_weights(atlas,weights)
    return True


def _strict_parts(variables, ambient, divisors, pulled):
    """Remove each Cartier divisor from each effective principal divisor.

    In a local trivialization for a divisor with equation d, an ideal J is
    contained in (d) exactly when J has positive order there, and J : (d)
    cancels one factor d. These quotients glue. The exceptional divisors in
    a chart are distinct; subtracting one leaves the orders of the others
    unchanged. Rational conjugates have equal orders. Thus repeated quotient
    gives both the orders and the ideal of the residual strict transform.
    """
    from engine import kernel
    native = kernel.strict_parts(variables, ambient, divisors, pulled)
    if native is None:
        # Work in the preimage of the Cartier ideals in the ambient ring.
        # (J:P) mod A agrees with (J/A):(P/A) whenever A is in J,P.
        # The wide native block order handles large rings and exponents.
        from engine.groebner import colon
        native = []
        primes = [ideal_basis(variables, (*ambient, *part)) for part in divisors]
        if any(p == (1,) or ideal_subset(variables, p, ambient) for p in primes):
            raise ArithmeticError("Expected a nonempty divisor of positive codimension.")
        for g in pulled:
            if ideal_subset(variables, (g,), ambient):
                raise ArithmeticError("A factor vanishes identically on a chart component.")
            residual = ideal_basis(variables, (*ambient, g))
            orders = []
            for prime in primes:
                order = 0
                while ideal_subset(variables, residual, prime):
                    quotient = colon(residual, prime, variables)
                    if quotient is None or quotient == residual:
                        raise ArithmeticError("Could not lower the order of the strict ideal.")
                    residual = quotient
                    order += 1
                orders.append(order)
            native.append((tuple(orders), residual))
    if native is not None:
        return native
    ring = QQ.old_poly_ring(*variables, order="grevlex")
    reference = []
    for g in pulled:
        ideal = ring.ideal(*ambient, g)
        orders = []
        for generators in divisors:
            divisor = ring.ideal(*ambient, *generators)
            order = 0
            while divisor.subset(ideal):
                ideal = ideal.quotient(ring.ideal(*generators))
                order += 1
            orders.append(order)
        residual = tuple(ring.to_sympy(h) for h in ideal.gens)
        basis = tuple(sp.groebner(residual, *variables, order="grevlex", domain=QQ).exprs)
        reference.append((tuple(orders), basis))
    if native is not None and reference != native:
        raise AssertionError("The kernel's exceptional orders or strict ideals differ from SymPy's.")
    return reference


def _lift_local(chart, saved, factors, unit, symbols):
    """Weight a serialized blowup chart and its chart in the exported record."""
    from itertools import combinations_with_replacement, product
    from engine.affine_kernel import pullbacks
    from engine.local_blowups import _is_strict
    coords = tuple(map(sp.Symbol, saved["coordinates"]))
    names = dict(zip(saved["coordinates"], coords))
    ambient = tuple(_read(g, names) for g in saved["ambient_ideal"])
    mapping = tuple(_read(saved["map"][str(v)], names) for v in symbols)
    pulled = pullbacks(coords, mapping, symbols, (unit, *(f for f, _ in factors)))
    scalar, *polynomials = pulled
    exceptional = [c for c in saved["components"] if not _is_strict(c["id"])]
    if len(factors) == 1:
        # The existing strict ideal and all exceptional orders already belong
        # to this single factor; the unit has order zero over the origin.
        weight = factors[0][1]
        weights = {c["id"]: c["multiplicity"] * weight for c in exceptional}
        strict = [{**c, "multiplicity": weight} for c in saved["components"] if _is_strict(c["id"])]
    else:
        divisors = [tuple(_read(g, names) for g in c["ideal"]) for c in exceptional]
        parts = _strict_parts(coords, ambient, divisors, polynomials)
        weights = dict.fromkeys((c["id"] for c in exceptional), 0)
        strict = []
        for k, ((orders, ideal), (_, weight)) in enumerate(zip(parts, factors), 1):
            for c, order in zip(exceptional, orders):
                weights[c["id"]] += weight * order
            if ideal != (1,):
                strict.append({"id": f"S{k}", "multiplicity": weight, "nu": 1, "ideal": [_text(g) for g in ideal]})
    saved["components"] = [{**c, "multiplicity": weights[c["id"]]} for c in exceptional] + strict
    total = [(_text(g), weight) for g, (_, weight) in zip(polynomials, factors)]
    if scalar != 1:
        total.insert(0, (_text(scalar), 1))
    chart["ambient_total_transform"] = _power_product(total)
    # Retain the original strict ideal of each factor with its multiplicity.
    # Their product, times the pullback of the unit, gives the divisor ideal.
    chart["reduced_strict_transform"] = chart["strict_transform"]
    powers = [combinations_with_replacement(c["ideal"], c["multiplicity"]) for c in strict]
    generators = []
    for choice in product(*powers):
        row = [g for terms in choice for g in terms]
        if scalar != 1:
            row.insert(0, _text(scalar))
        generators.append(_power_product([(g, row.count(g)) for g in dict.fromkeys(row)]))
    chart["strict_transform"] = generators
    for comp in chart["exceptional"]:
        comp["total_multiplicity"] = weights[comp["id"]]
    return weights


@lru_cache(maxsize=256)
def _weight_embedding(indexed):
    """A SymPy-readable name for the marked FLINT generator in the display record."""
    match = re.fullmatch(r"PolynomialRootIndexed\(List\(([-\d, ]+)\), (\d+)\)", indexed)
    if not match:
        raise ArithmeticError("Unrecognized exact root of a weighted chart.")
    coeffs = tuple(map(int, match[1].split(",")))
    index = int(match[2]) - 1
    if len(coeffs) == 2:
        return str(sp.Rational(-coeffs[0], coeffs[1]))
    if len(coeffs) == 3:
        # FLINT gives the larger real root or the positive imaginary root
        # first. Its irreducible integral polynomial has positive leading coefficient.
        c, b, a = coeffs
        from engine.field_coefficient import rational_sqrt
        return str(rational_sqrt(b*b - 4*a*c, sp.Rational(1 if index == 0 else -1, 2*a),
                                 sp.Rational(-b, 2*a)))
    from engine.field_coefficient import marked_root_display
    description = marked_root_display(coeffs)
    if description is None:
        return str(_flint_roots(coeffs)[index])
    scale, normalized, positions = description
    x = sp.Symbol("x")
    poly = sp.PurePoly.from_list(normalized, x, domain=sp.ZZ)
    root = scale * sp.CRootOf._new(poly, positions[index])
    return str(root)



def _lift_free(atlas, chart, factors, unit, symbols):
    """Pull back factors in the marked coefficient field and remove the coordinate axes."""
    from engine.affine_kernel import chart_data, pullbacks
    from engine.atlas import _field, _record_text
    coordinates = tuple(map(sp.Symbol, chart["coordinates"]))
    names = {v.name: v for v in coordinates}
    # Only numbers occurring in the coordinate map enter the published
    # weight field. Excluded points in the atlas may use a larger extension.
    field = _field(atlas, {"map": chart["map"]}, names, symbols)
    ring, target, modulus = coordinates, symbols, None
    if field:
        parameter, modulus, _, embedding = field
        field_name = "weight_generator"
        while field_name in names or field_name in chart["map"]:
            field_name += "_"
        generator = sp.Symbol(field_name)
        rename = {parameter: generator}
        names = {name: value.xreplace(rename) for name, value in names.items()}
        modulus = modulus.xreplace(rename)
        ring, target = (*coordinates, generator), (*symbols, generator)
        if embedding == "root 1 (FLINT order)":  # the marked A-series quadratic
            data = atlas["coefficient_field"]
            pattern = re.escape(data["generator"]) + r"\^2-\(([-\d/]+)\)"
            match = re.fullmatch(pattern, data["minimal_polynomial"])
            if not match:
                raise ArithmeticError("Unrecognized quadratic chart field.")
            from engine.field_coefficient import rational_sqrt
            exact = str(rational_sqrt(match[1]))
        else:
            exact = _weight_embedding(embedding)
        chart["weight_field"] = {"generator": field_name, "exact": exact}
    mapping = tuple(_read(chart["map"][v.name], names) for v in symbols)
    if field:
        mapping = (*mapping, generator)
    axes = {}
    for component in chart.get("exceptional", ()):
        variable = _read(component["equation"], names)
        if variable not in coordinates:
            raise ArithmeticError("A free embedded chart needs coordinate exceptional components.")
        axes[variable] = component["id"]
    local = dict.fromkeys(axes.values(), 0)
    strict = []
    original = []
    # A unit at the origin has order zero on every germ-relevant divisor.
    scalar, *pulled = pullbacks(ring, mapping, target, (unit, *(f for f, _ in factors)), modulus)
    if scalar != 1:
        strict.append((_record_text(scalar, ring), 1))
        original.append((_record_text(scalar, ring), 1))
    for (factor, weight), total in zip(factors, pulled):
        orders, proper = chart_data(ring, mapping, tuple(axes), factor, target, modulus)
        for label, (order, _) in zip(axes.values(), orders):
            local[label] += weight*order
        strict.append((_record_text(proper, ring), weight))
        original.append((_record_text(total, ring), weight))
    chart["ambient_total_transform"] = _power_product(original)
    chart["strict_transform"] = _power_product(strict)
    chart["pullback_factor"] = _power_product([(str(v), local[label]) for v, label in axes.items() if local[label]])
    return local


def _lift_graph(atlas, chart, factors, unit, symbols, weights):
    """Compute over Q[theta]/(m), with graph ideals lifted to Q[axes,theta].

    Saturation and ideal quotient commute with taking the preimage in
    this presentation. Descent to any embedding preserves the Cartier
    orders and strict equations. The field embedding is only needed to
    label the coefficients for display.
    """
    from engine.affine_kernel import pullbacks
    from engine.atlas import _record_text
    field = atlas.get("coefficient_field", {})
    if atlas.get("algebraic_numbers"):
        from engine.graph_record import lift_graph
        base = tuple(v.name for v in symbols)
        return lift_graph(atlas, chart, [(_record_text(f,symbols), m) for f,m in factors],
                          _record_text(unit,symbols), base)
    if field.get("generator"):
        from tests.graph_monodromy_oracle import _field_coefficients
        coefficients = _field_coefficients(field)
    coordinates = tuple(map(sp.Symbol, chart["coordinates"]))
    names = {v.name: v for v in coordinates}
    target = tuple(symbols)
    relation = None
    if field.get("generator"):
        label = "weight_generator"
        while label in names:
            label += "_"
        theta = sp.Symbol(label)
        names[field["generator"]] = theta
        relation = sp.Add(*(sp.Rational(c)*theta**k for k, c in enumerate(reversed(coefficients))))
        coordinates = (*coordinates, theta)
        chart["weight_field"] = {"generator": label, "exact": field["exact_root"]}
    mapping = tuple(_read(chart["map"][s.name], names) for s in target)
    scalar, *images = pullbacks(coordinates, mapping, target, (unit, *(f for f, _ in factors)), modulus=relation)
    pieces = [(f, multiplicity) for f, (_, multiplicity) in zip(images, factors)]
    if scalar != 1:
        pieces.insert(0, (scalar, 1))
    original = [(_record_text(f, coordinates), m) for f, m in pieces]
    ambient = tuple(_read(f, names) for f in chart["ambient_ideal"])
    if relation is not None:
        ambient = (*ambient, relation)
    total, = pullbacks(coordinates, coordinates, coordinates, (prod(f**m for f, m in pieces),), modulus=relation)
    pivot = _read(chart["pivot_generator"], names)
    strict = saturate(coordinates, (*ambient, total), pivot)
    proper = [g for g in strict if not ideal_subset(coordinates, (g,), ambient)]
    local = {}
    for comp in chart["exceptional"]:
        label = comp["id"]
        if label in weights:
            local[label] = weights[label]
            continue
        prime = tuple(_read(f, names) for f in comp.get("ideal", [comp["equation"]]))
        data = _strict_parts(coordinates, ambient, (prime,), (total,))
        local[label] = data[0][0][0]
    chart["ambient_total_transform"] = _power_product(original)
    chart["strict_transform"] = [_record_text(g, coordinates) for g in proper]
    return local


def _lift(atlas, factors, unit, symbols):
    weights = {}
    saved = atlas.get("atlas", []) if atlas["engine"] == "canonical-blowups" else []
    for index, chart in enumerate(atlas["charts"]):
        local_record = saved[index] if saved else None
        if "fibre" in chart and "components" in chart:
            local_record = chart
        if local_record is not None:
            local = _lift_local(chart, local_record, factors, unit, symbols)
        else:
            chart["reduced_strict_transform"] = chart["strict_transform"]
            if "strict_transform_on_exceptional" in chart:
                chart["reduced_strict_transform_on_exceptional"] = chart.pop("strict_transform_on_exceptional")
            if "ambient_ideal" in chart:
                local = _lift_graph(atlas, chart, factors, unit, symbols, weights)
            else:
                local = _lift_free(atlas, chart, factors, unit, symbols)
        for ident, order in local.items():
            if weights.setdefault(ident, order) != order:
                raise ArithmeticError("Original divisor orders disagree on overlapping charts.")
        for component in (*chart.get("exceptional", ()), *chart.get("ambient_components", ())):
            order = local[component["id"]]
            for key in ("total_multiplicity", "multiplicity", "ambient_order"):
                if key in component:
                    component[key] = order
            component["total_multiplicity"] = order
    _set_global_weights(atlas, weights)


def _set_global_weights(atlas,weights):
    for vertex in atlas["dual_complex"]["vertices"]:
        order=weights[vertex["id"]]
        for key in ("multiplicity","ambient_order","total_multiplicity"):
            if key in vertex: vertex[key]=order
        vertex["total_multiplicity"]=order
    for step in atlas.get("steps",[]):
        if step.get("exceptional_id") in weights:
            step["total_multiplicity"]=weights[step["exceptional_id"]]
        for key in ("strict_multiplicity","multiplicity"):
            if key in step: step["reduced_support_"+key]=step.pop(key)


def nonreduced_resolution(p, variables, mode, resolve_support):
    """Resolve a nonreduced embedded divisor; leave reduced germs to the caller.

    resolve_support receives a sparse polynomial dictionary and must return
    an embedded atlas (or its ordinary inconclusive result).
    """
    symbols=tuple(sp.Symbol(v) for v in variables)
    from engine.affine_kernel import rational_factor_rows, sparse_terms, _sparse_expression, _sparse_terms
    from engine.atlas import _polynomial
    scalar,parts=rational_factor_rows(symbols, _polynomial(p, symbols))
    active=[];unit=sp.sympify(scalar)
    zero = (0,)*len(symbols)
    for row,m,_ in parts:
        factor = _sparse_expression(symbols, row)
        if zero not in dict(_sparse_terms(row)):
            active.append((factor,m))
        else:
            unit*=factor**m
    if not any(m>1 for _,m in active):
        return None
    if mode=="intrinsic":
        raise InputError("The germ has a repeated component through the origin. An intrinsic resolution requires the underlying reduced hypersurface.")
    sparse=dict(sparse_terms(symbols, prod(f for f,_ in active)))
    support=_polynomial(sparse, symbols)
    atlas=resolve_support(sparse)
    metadata={"reduced_equation":_text(support),"unit_at_origin":_text(unit),
              "factors":[{"equation":_text(f),"multiplicity":m} for f,m in active]}
    atlas["input_divisor"]=metadata
    if atlas["status"]!="resolved":
        return atlas
    if atlas["engine"] != "canonical-blowups":
        # Reconstruct from the weighted chart maps after the lift. For the
        # origin engine the owned charts have no separate producer: lift them
        # in place with their display records.
        atlas.pop("atlas", None)
    multiplicities={m for _,m in active}
    if len(multiplicities)!=1 or not _uniform_lift(atlas,next(iter(multiplicities)),unit,symbols):
        _lift(atlas,active,unit,symbols)
    atlas["multiplicity"]=min(map(sum,p))
    atlas["summary"]="Embedded resolution of the divisor with multiplicities. "+atlas["summary"]
    atlas.setdefault("notes",[]).append("The geometric dual complex and smoothness certificate refer to the reduced support. The strict-transform equations and exceptional orders include the original multiplicities.")
    return atlas
