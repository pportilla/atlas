"""Identify strata with finite geometric parameters on the resolution atlas.

A native computation identifies finite tuples of regular coordinate ratios
on the supporting charts, proves the local affine-space presentations and
checks generic ownership on every geometric sheet.

For intrinsic strata, a finite coordinate eliminant or a univariate
specialization can supply a candidate splitting field. Lines with collinear
exponents supply a polynomial in their primitive Laurent character.
FLINT finds a common primitive element and embeds the original marked
coefficient fields in it. A coordinate section over this field splits by
imposing each of its exact roots on its original equations. Linear
triangular graph leaves are geometrically integral. More general leaves
use absolute prime certificates and ideal-coverage tests. A specialization
never certifies a component or an incidence. Those require proofs on the
unspecialized ideal and on the actual chart overlaps.
"""
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import re

from .chart_expressions import chart_expressions

from .affine_kernel import rational_factor_rows, sparse_terms
from .groebner import groebner
from .local_blowups import _is_strict, _own_failures


def _finite_graph(variables, rows, dimension, *, require_eliminant):
    from .geometric_components import _component_operation, _encoded_support
    if not rows or dimension < 0:
        return None
    heads = (sparse_terms(variables, f)[0][0] for f in rows)
    return _component_operation("finite_graph", support=_encoded_support(heads),
                                dimension=dimension, require_eliminant=require_eliminant)


def finite_complex(charts, dimension, counts):
    """Identify geometric sheets of exceptional intersections over finite parameter bases."""
    from .atlas import _record_text
    from .incidence import _intersections
    from .kernel_json import kernel_json
    payload = []
    for chart in charts:
        def text(f):
            return _record_text(f, chart.variables)
        payload.append({"coordinates": [v.name for v in chart.variables],
                        "ambient_ideal": [text(f) for f in chart.ambient],
                        "components": {label: [text(f) for f in ideal] for label, _, _, ideal in chart.components
                                       if not _is_strict(label)},
                        "map": {name: text(f) for name, f in chart.coordinates},
                        "region": [text(f) for f in chart.region], "fibre": [text(f) for f in chart.fibre],
                        "removed": [[text(f) for f in hole] for hole in chart.removed],
                        "coefficient_field": chart.coefficient_field[3] if chart.coefficient_field else None})
    result = kernel_json("atlas_finite_sheets", "complex", charts=payload, counts=_intersections(counts),
                         dimension=dimension)
    if result is not None:
        for chart in charts:
            boundary = tuple(row for row in chart.components if not _is_strict(row[0]))
            if _own_failures(replace(chart, components=boundary), dimension, {})[0]:
                return None
    return result


def geometric_chart_records(record):
    """Expose the certified local sheet ideals while retaining rational chart equations for replay."""
    from .polynomial import InputError
    charts = record["charts"]
    dual = record.get("dual_complex", {})
    sections = dual.get("finite_sheet_certificate", {}).get("chart_sections")
    if sections is None:
        return charts
    if len(charts) != len(sections):
        raise InputError("The finite sheet certificate has a different number of charts.")
    result = []
    for chart, entry in zip(charts, sections):
        if chart["coordinates"] != entry["coordinates"]:
            raise InputError("The finite sheet chart coordinates differ from the saved resolution.")
        sheets = entry["divisors"]
        if not sheets:
            result.append(chart)
            continue
        rational = chart.get("rational_exceptional", chart["exceptional"])
        rows = []
        for divisor in rational:
            label = divisor["id"]
            if label not in sheets:
                rows.append(divisor)
                continue
            ideal = divisor.get("ideal")
            if ideal is None:
                ideal = [divisor["equation"]]
            for section in sheets[label]:
                generators = [*ideal, *section["restrictions"]]
                rows.append({**divisor, "id": section["id"], "ambient_divisor": label,
                             "ideal": generators, "equation": ", ".join(generators)})
        result.append({**chart, "exceptional": rows, "rational_exceptional": rational})
    return result


def _integer_polynomial(variable, polynomial):
    """Primitive rational irreducible factors, with integer coefficients in descending order."""
    from .affine_kernel import _sparse_terms
    for row, _, _ in rational_factor_rows((variable,), polynomial)[1]:
        entries = dict(_sparse_terms(row))
        degree = max(m[0] for m in entries)
        if degree > 1:
            yield tuple(int(entries.get((j,), 0)) for j in range(degree, -1, -1))


@lru_cache(maxsize=256)
def _field_mark(field):
    """Read a chart's marked embedding, including the early FLINT-index spelling."""
    from .marked_field import description, flint_literal
    from .polynomial import InputError
    t, relation, degree, exact = field
    matches = re.fullmatch(r"root ([1-9][0-9]*) \(FLINT order\)", exact)
    if matches:
        index = int(matches[1])
        factors = rational_factor_rows((t,), relation)[1]
        if len(factors) != 1 or factors[0][1] != 1 or not 1 <= index <= degree:
            return None
        from .affine_kernel import _sparse_terms
        entries = dict(_sparse_terms(factors[0][0]))
        if max(m[0] for m in entries) != degree:
            return None
        return flint_literal(tuple(int(entries.get((j,),0)) for j in range(degree,-1,-1)),index)
    try:
        coeffs,index = description(exact)
    except InputError:
        return None
    if degree != len(coeffs)-1:
        return None
    # Every scalar substitution must satisfy the relation it replaces.
    factors = rational_factor_rows((t,), relation)[1]
    if len(factors) != 1 or factors[0][1] != 1:
        return None
    from .affine_kernel import _sparse_terms
    row = dict(_sparse_terms(factors[0][0]))
    if any(coeffs[degree-j] != row.get((j,), 0) for j in range(degree+1)):
        return None
    return flint_literal(coeffs,index)


@lru_cache(maxsize=2048)
def _root_coordinates(mark, root):
    from .marked_field import program
    from .native_library import field_request
    row = field_request("atlas_field_coefficient", [program(mark), program(root)])
    return None if row is None or row == "outside" else tuple(map(Fraction, row.split()))


@lru_cache(maxsize=1024)
def _root_on_sheet(t, v, polynomial, mark, root):
    """Evaluate a rational bivariate polynomial at two exact QQbar literals."""
    from .marked_field import program
    from .native_library import field_request
    terms = sparse_terms((v,t), polynomial)
    ops = []
    for powers, coefficient in terms:
        ops.extend(("q", str(coefficient)))
        for degree, literal in zip(powers, (root,mark)):
            if degree:
                ops.extend((program(literal), "pow", str(degree), "*"))
    ops.extend(["+"] * (len(terms)-1))
    row = field_request("atlas_field_coefficient", ["q 0", " ".join(ops)])
    if row is None:
        raise ArithmeticError("The exact algebraic coefficient kernel is unavailable.")
    return row in ("", "0")


def _finite_linear_sections(chart, equations, coords, dimension):
    """Split a triangular finite presentation directly in its marked power basis.

    A coordinate with a rational eliminant has exactly the roots listed
    by FLINT. For the chosen field, extract powers for every root and
    impose its linear equation on the *unspecialized* ideal. Unit or
    triangular graph leaves prove emptiness or geometric primality.
    When some roots lie outside the field, a leaf may still be an affine
    graph, but finite-point coverage cannot be decided by omission.
    """
    from .affine_kernel import _sparse_expression, _sparse_terms
    from .marked_field import flint_literal
    t, relation, _, _ = chart.coefficient_field
    ring = (*coords,t)
    api = chart_expressions(chart.variables)
    mark = _field_mark(chart.coefficient_field)
    if not mark:
        return None

    def split(gens):
        basis = tuple(groebner(gens, ring, order="lex").exprs)
        if basis == (1,):
            return []
        if _finite_graph(ring, basis, dimension, require_eliminant=False) is not None:
            return [tuple(f for f in basis if f.free_symbols != {t})]
        for f in basis:
            geometric = f.free_symbols - {t}
            if len(geometric) != 1:
                continue
            v, = geometric
            if max(e[0] for e,_ in sparse_terms((v,t),f)) < 2:
                continue
            if t in f.free_symbols:
                # The norm eliminant contains the conjugate sections on
                # *all* embeddings. Cutting the original ideal discards any
                # roots that belong exclusively to a different embedding.
                norm_basis = groebner((relation,f),(t,v),order="lex")
                norm = next((g for g in norm_basis.exprs if g.free_symbols == {v}),None)
                if norm is None:
                    continue
            else:
                norm = f
            roots = []
            for row, _, _ in rational_factor_rows((v,), norm)[1]:
                entries = dict(_sparse_terms(row))
                degree = max(m[0] for m in entries)
                if degree == 1:
                    roots.append(_sparse_expression((v,), row))
                    continue
                coeffs = tuple(int(entries.get((j,),0)) for j in range(degree,-1,-1))
                for i in range(1,degree+1):
                    literal = flint_literal(coeffs,i)
                    coefficients = _root_coordinates(mark, literal)
                    if coefficients is None:
                        if t in f.free_symbols and not _root_on_sheet(t,v,f,mark,literal):
                            continue
                        return None
                    value = api.Add(*(api.Rational(a)*t**j for j,a in enumerate(coefficients) if a))
                    roots.append(v-value)
            pieces = [split((*basis, each)) for each in roots]
            if any(part is None for part in pieces):
                return None
            return [p for part in pieces for p in part]
        return None

    return split((relation,*equations))


@lru_cache(maxsize=256)
def _power_domain(t, relation, degree):
    from .candidate_field import NativeField
    from .native_polynomial import native_ring
    coeffs = dict(sparse_terms((t,), relation))
    row = tuple(coeffs.get((j,), 0) for j in range(degree, -1, -1))
    if native_ring((t,)):
        # Arithmetic here uses only the presentation. The chart and the
        # diagonal retain the index of the chosen embedding during gluing.
        return NativeField(row, 0, t), t
    from .field_coefficient import indexed_field
    # Any embedding of this same abstract field gives the same calculation
    # in its power basis. Lift the returned ideals to the chart's t again.
    return indexed_field(row, t, 0)


def marked_primes(chart, equations, dimension, branches=False):
    """Certified geometric primes on one field sheet, lifted to rational rows.

    Strip free affine coordinates and work in K[x] using the chart's
    defining field relation. The finite-graph test or the geometric
    decomposition in K[x] certifies the absolute primes and coverage.
    Intersecting primes of a reduced support are distinguished before
    their local transversality is tested by the gluing routine.
    """
    from .field_polynomial import lift
    from .geometric_components import geometric_components
    from .native_polynomial import native_ring
    t, relation, _, _ = chart.coefficient_field
    # A basis on a lower face may retain a *monic multiple* of the
    # primitive integral field relation. Reduce all rows modulo m before
    # testing whether the section descends to Q or counting coordinates.
    from .affine_kernel import pullbacks
    if any(t in g.free_symbols and g != relation for g in equations):
        equations = pullbacks(chart.variables, chart.variables, chart.variables, equations, modulus=relation)
    equations = tuple(g for g in equations if g != relation and g != 0)
    used = {v for g in equations for v in g.free_symbols}
    coords = tuple(v for v in chart.variables if v != t and v in used)
    dim = dimension - (len(chart.variables)-1-len(coords))
    if not coords:
        basis = groebner((relation,*equations), (t,))
        return [] if tuple(basis.exprs) == (1,) else [()] if dim == 0 else None
    if dim < 0:
        return [] if tuple(groebner((relation,*equations), (*coords,t)).exprs) == (1,) else None
    linear = _finite_linear_sections(chart, equations, coords, dim)
    if linear is not None:
        return linear
    if t not in used:
        rational = geometric_components(coords, equations, dim, reduced_support=branches)
        if rational is not None:
            return rational
    field, generator = _power_domain(t, relation, chart.coefficient_field[2])
    if not native_ring(chart.variables):
        equations = tuple(g.xreplace({t: generator}) for g in equations)

    result = geometric_components(coords, equations, dim, domain=field, reduced_support=branches)
    if result is None or native_ring(chart.variables):
        return result
    return [tuple(lift(g, coords, field, t) for g in piece) for piece in result]


def _splitting_polynomials(chart, equations):
    """Candidate defining polynomials from finite coordinates and rational slices.

    A specialization can factor more than the original variety. Its roots
    propose a field only. The unspecialized ideals must subsequently admit
    geometric prime certificates in that field.
    """
    from .affine_kernel import pullbacks
    from .binomial_kernel import edge_parameters
    from .fan_geometry import lattice_pivots
    field = chart.coefficient_field
    t = field[0] if field else None
    active = tuple(v for v in chart.variables if any(v in g.free_symbols for g in equations))
    if not active:
        return
    # A reduced basis incorporates equations of the coordinate flat, e.g.
    # the pullback on the exceptional divisor, before proposing any roots.
    basis = groebner(equations, active, order="lex")
    candidates = tuple(g for g in basis.exprs if g.free_symbols and g.free_symbols != {t})
    api = chart_expressions(active)
    z = api.Dummy("root_parameter")
    for f in candidates:
        variables = tuple(v for v in active if v in f.free_symbols)
        if t in variables:
            continue
        terms = dict(sparse_terms(variables, f))
        support = tuple(sorted(terms))
        pivots = lattice_pivots(support)
        if len(pivots) == 1:
            _, _, powers = edge_parameters(support, pivots[0])
            # Translating the support removes a monomial unit on the torus.
            univariate = api.Add(*(api.Rational(terms[m])*z**j for m,j in zip(support,powers)))
            for coefficients in _integer_polynomial(z, univariate):
                yield coefficients, True, None
    # Eliminate a pre-existing coefficient-field parameter as well. For
    # example x²-theta over Q(sqrt(2)) gives the rational norm x^4-2.
    for f in candidates:
        variables = tuple(v for v in active if v in f.free_symbols and v != t)
        source = (*variables, t) if t in f.free_symbols else variables
        target = (z,t) if t in f.free_symbols else (z,)
        for v in variables:
            for value in (0, 1, -1):
                mapping = tuple(z if w == v else t if w == t else api.Integer(value) for w in source)
                h, = pullbacks(target, mapping, source, (f,))
                if z not in h.free_symbols:
                    continue
                if t in h.free_symbols:
                    elimination = groebner((field[1], h), (t,z), order="lex")
                    rows = [g for g in elimination.exprs if g.free_symbols == {z}]
                else:
                    rows = [h]
                for row in rows:
                    for coefficients in _integer_polynomial(z, row):
                        yield coefficients, len(variables) == 1, (t,z,h) if t in h.free_symbols else None
                if len(variables) == 1:
                    break


def _extend_chart(chart, field, old_value):
    from .affine_kernel import pullbacks
    theta, relation, _, _ = field
    if not chart.coefficient_field:
        return replace(chart, variables=(*chart.variables,theta), ambient=(*chart.ambient,relation),
                       coefficient_field=field)
    t, old_relation, _, _ = chart.coefficient_field
    coords = tuple(v for v in chart.variables if v != t) + (theta,)
    mapping = tuple(old_value if v == t else v for v in chart.variables)

    def move(polys):
        return pullbacks(coords, mapping, chart.variables, polys, modulus=relation)

    ambient = tuple(f for f in chart.ambient if f != old_relation)
    return replace(chart, variables=coords, ambient=(*move(ambient),relation),
                   repair_projection=(chart.repair_projection[0],move(chart.repair_projection[1]))
                                     if chart.repair_projection else (),
                   coordinates=tuple(zip((name for name,_ in chart.coordinates),
                                         move(tuple(g for _,g in chart.coordinates)))),
                   components=tuple((label,n,nu,move(ideal)) for label,n,nu,ideal in chart.components),
                   fibre=move(chart.fibre), region=move(chart.region), removed=tuple(move(hole) for hole in chart.removed),
                   coefficient_field=field)


def splitting_charts(charts, witness):
    """Try a common splitting field on a failed section, and certify before retrying gluing."""
    from .affine_kernel import decode, pullbacks
    from .marked_field import flint_literal
    from .plane_backend import coefficient_field
    index, equations, dimension, *branch_mode = witness
    chart = charts[index]
    branches = branch_mode[0] if branch_mode else True
    api = chart_expressions(chart.variables)
    theta = api.Dummy("boundary_field")
    fields = tuple(dict.fromkeys(c.coefficient_field for c in charts if c.coefficient_field))
    marks = tuple(_field_mark(field) for field in fields)
    if any(mark is None for mark in marks):
        return None
    old_degree = max((f[2] for f in fields), default=1)
    coordinate_roots, seen = [], set()
    T = api.Symbol("T")

    def read(text):
        return decode((T,), text).xreplace({T:theta})

    def attempt(roots):
        generators = tuple(dict.fromkeys((*marks,*roots)))
        # FLINT proves every inclusion in the primitive field it returns.
        common = coefficient_field(generators)
        if common["degree"] <= old_degree:
            return None
        field = (theta, read(common["minimal_polynomial"]), common["degree"], common["exact"])
        images = {m:read(v) for m,v in zip(generators, common["values"])}
        substitutions = {f:images[m] for f,m in zip(fields,marks)}
        grown = _extend_chart(chart,field,substitutions.get(chart.coefficient_field))
        checks = equations
        if chart.coefficient_field:
            t = chart.coefficient_field[0]
            mapping = tuple(substitutions[chart.coefficient_field] if v == t else v for v in chart.variables)
            checks = pullbacks(grown.variables, mapping, chart.variables, equations, modulus=field[1])
        if marked_primes(grown, checks, dimension, branches=branches) is None:
            return None
        return [_extend_chart(c,field,substitutions.get(c.coefficient_field)) for c in charts]

    for coefficients, finite, specialized in _splitting_polynomials(chart, equations):
        roots = [flint_literal(coefficients, i) for i in range(1,len(coefficients))]
        if specialized is not None:
            old_mark = _field_mark(chart.coefficient_field)
            roots = [root for root in roots if _root_on_sheet(*specialized,old_mark,root)]
        if finite:
            coordinate_roots.extend(root for root in roots if root not in coordinate_roots)
        proposal = tuple(dict.fromkeys((*coordinate_roots,*roots)))
        if proposal and proposal not in seen:
            seen.add(proposal)
            grown = attempt(proposal)
            if grown is not None:
                return grown
    return None
