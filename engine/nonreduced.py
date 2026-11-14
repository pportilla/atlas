"""Lift an embedded resolution of a reduced support to its effective divisor.

Free affine charts use exact integer valuations of the pulled-back factors.
On a graph chart the strict ideal is saturated inside its ambient equations.
Every ambient exceptional prime is locally Cartier on the smooth chart, so
membership in its ordinary powers computes its generic order of vanishing.
"""
from fractions import Fraction
from functools import cmp_to_key, lru_cache
from math import prod
import re

import sympy as sp
from sympy import QQ

from .blowup_ideal import ideal_basis, ideal_subset, saturate
from .polynomial import InputError


def _text(expr):
    return str(expr).replace("**","^")


def _read(source, names):
    # Chart formulas are generated internally. Remap identifiers so even
    # Python keywords chosen as coordinate names are ordinary indeterminates.
    aliases = {name:f"q_{i}" for i,name in enumerate(names)}
    source = re.sub(r"[A-Za-z][A-Za-z0-9_]*",lambda m:aliases[m.group()], source)
    return sp.sympify(source.replace("^","**"),locals={a:names[name] for name,a in aliases.items()})


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


def _chart_field(atlas, chart):
    field=atlas.get("coefficient_field",{})
    used = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*"," ".join(chart["map"].values())))
    values=_algebraics(atlas.get("algebraic_numbers",{}),used)
    name=field.get("generator")
    if name and field.get("exact_root"):
        values[name]=sp.sympify(field["exact_root"])
    elif name=="theta" and "theta" in used:
        # The native A-series chart specifies either square root; choosing
        # one embedding of the displayed common field fixes the atlas.
        source=field["minimal_polynomial"]
        match=re.fullmatch(r"theta\^2-\(([-\d/]+)\)",source)
        if not match: raise ArithmeticError("Unrecognized quadratic chart field.")
        values[name]=sp.sqrt(sp.Rational(match[1]))
    return QQ.algebraic_field(*dict.fromkeys(values.values())) if values else QQ,values


def _expression(p, symbols):
    return sp.Poly.from_dict({e:sp.Rational(c) for e,c in p.items()},symbols,domain=QQ)


def _display_poly(poly, domain, field_name):
    if domain==QQ:
        return _text(poly.as_expr())
    # Exact rational-polynomial representation of each coefficient in the
    # primitive element. A chart-local field definition accompanies this text.
    t=sp.Symbol(field_name)
    result=0
    for powers,c in poly.rep.to_dict().items():
        scalar=sp.Poly.from_list([sp.Rational(v.numerator,v.denominator) for v in c.to_list()],t).as_expr()
        result+=scalar*prod(a**k for a,k in zip(poly.gens,powers))
    return _text(result)


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
            # Symbolic substitution is exact even before algebraic numbers
            # are evaluated. No cancellation is used to compute a valuation.
            names={a:sp.Symbol(a) for a in chart["coordinates"]}
            for a in atlas.get("algebraic_numbers",{}): names[a]=sp.Symbol(a)
            gen=atlas.get("coefficient_field",{}).get("generator")
            if gen: names[gen]=sp.Symbol(gen)
            pulled=unit.xreplace({v:_read(chart["map"][str(v)],names) for v in symbols})
            power.insert(0,(_text(pulled),1))
        chart["strict_transform"]=_power_product(power)
        chart["pullback_factor"]=_power_product([(factor,multiplicity)]) if factor!="1" else "1"
        chart["ambient_total_transform"]=f"({chart['pullback_factor']})*({chart['strict_transform']})"
        for c in (*chart.get("exceptional",()),*chart.get("ambient_components",())):
            for key in ("total_multiplicity","multiplicity","ambient_order"):
                if key in c: c[key]=weights[c["id"]]
            c["total_multiplicity"]=weights[c["id"]]
    _set_global_weights(atlas,weights)
    return True


def _lift(atlas, factors, unit, symbols):
    weights={}
    for chart in atlas["charts"]:
        nms=chart["coordinates"]
        coordinates=tuple(sp.Symbol(s) for s in nms)
        domain,algebraics=_chart_field(atlas,chart)
        names=dict(zip(nms,coordinates));names.update(algebraics)
        field_name="weight_generator"
        while field_name in names: field_name+="_"
        if algebraics:
            # Record the exact embedding; retained charts may also refer to
            # the original algebraic labels in their blowdown maps.
            chart["weight_field"]={"generator":field_name,"exact":str(domain.ext.as_expr())}
        def display(f):
            return _display_poly(sp.Poly(f,*coordinates,domain=domain),domain,field_name)
        mapping={v:_read(chart["map"][str(v)],names) for v in symbols}
        scalar=sp.Poly(unit.xreplace(mapping),*coordinates,domain=domain)
        polys=[(sp.Poly(f.xreplace(mapping),*coordinates,domain=domain),m) for f,m in factors]
        # Keep original pullbacks factored to avoid expanding large powers.
        original=[(display(p.as_expr()),m) for p,m in polys]
        if not scalar.is_one: original.insert(0,(display(scalar.as_expr()),1))
        chart["ambient_total_transform"]=_power_product(original)
        chart["reduced_strict_transform"]=chart["strict_transform"]
        if "strict_transform_on_exceptional" in chart:
            chart["reduced_strict_transform_on_exceptional"]=chart.pop("strict_transform_on_exceptional")
        local={}
        if "ambient_ideal" in chart:
            ambient=tuple(_read(f,names) for f in chart["ambient_ideal"])
            total=scalar.as_expr()*prod(p.as_expr()**m for p,m in polys)
            pivot=_read(chart["pivot_generator"],names)
            strict=saturate(coordinates,(*ambient,total),pivot,domain=domain)
            chart["strict_transform"]=[display(f) for f in strict
                                        if not ideal_subset(coordinates,(f,),ambient,domain=domain)]
            for comp in chart["exceptional"]:
                prime=tuple(_read(f,names) for f in comp.get("ideal",[comp["equation"]]))
                label=comp["id"]
                if label not in weights:
                    weights[label]=_valuation(coordinates,ambient,prime,total,domain)
                local[label]=weights[label]
        else:
            axes={}
            for comp in chart.get("exceptional",[]):
                variable=_read(comp["equation"],names)
                if variable not in coordinates:
                    raise ArithmeticError("A free embedded chart needs coordinate exceptional components.")
                axes[coordinates.index(variable)]=comp["id"]
            strict=[]
            # A unit at the origin has order zero on every germ-relevant
            # exceptional component. Keep its actual pullback in the equation.
            if not scalar.is_one: strict.append((display(scalar.as_expr()),1))
            for p,m in polys:
                terms=p.rep.to_dict()
                minima={j:min(e[j] for e in terms) for j in axes}
                for j,label in axes.items():
                    local[label]=local.get(label,0)+m*minima[j]
                shifted={tuple(k-minima.get(j,0) for j,k in enumerate(e)):c for e,c in terms.items()}
                proper=sp.Poly.from_dict(shifted,coordinates,domain=domain)
                strict.append((_display_poly(proper,domain,field_name),m))
            chart["strict_transform"]=_power_product(strict)
            chart["pullback_factor"]=_power_product([(str(coordinates[j]),local[label])
                                                     for j,label in axes.items() if local[label]])
        for ident,order in local.items():
            if ident in weights and order!=weights[ident]:
                raise ArithmeticError("Original divisor orders disagree on overlapping charts.")
            weights[ident]=order
        for component in (*chart.get("exceptional",()), *chart.get("ambient_components",())):
            order=local[component["id"]]
            for key in ("total_multiplicity","multiplicity","ambient_order"):
                if key in component: component[key]=order
            component["total_multiplicity"]=order
    _set_global_weights(atlas,weights)


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
    original=_expression(p,symbols)
    scalar,parts=original.factor_list()
    active=[];unit=sp.sympify(scalar)
    for factor,m in parts:
        if factor.coeff_monomial((0,)*len(symbols))==0:
            active.append((factor.as_expr(),m))
        else:
            unit*=factor.as_expr()**m
    if not any(m>1 for _,m in active):
        return None
    if mode=="intrinsic":
        raise InputError("The germ has a repeated component through the origin. An intrinsic resolution requires the underlying reduced hypersurface.")
    support=sp.Poly(prod(f for f,_ in active),*symbols,domain=QQ)
    sparse={e:Fraction(int(c.p),int(c.q)) for e,c in support.terms()}
    atlas=resolve_support(sparse)
    metadata={"reduced_equation":_text(support.as_expr()),"unit_at_origin":_text(unit),
              "factors":[{"equation":_text(f),"multiplicity":m} for f,m in active]}
    atlas["input_divisor"]=metadata
    if atlas["status"]!="resolved":
        return atlas
    multiplicities={m for _,m in active}
    if len(multiplicities)!=1 or not _uniform_lift(atlas,next(iter(multiplicities)),unit,symbols):
        _lift(atlas,active,unit,symbols)
    atlas["multiplicity"]=min(map(sum,p))
    atlas["summary"]="Embedded resolution of the divisor with multiplicities. "+atlas["summary"]
    atlas.setdefault("notes",[]).append("The geometric dual complex and smoothness certificate refer to the reduced support. The strict-transform equations and exceptional orders include the original multiplicities.")
    return atlas
