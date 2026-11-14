"""Strata of a certified embedded projective ideal blowup, for A'Campo's
formula and for the topological zeta function."""
import ast
from functools import lru_cache
from itertools import combinations
import re

from engine.reference import sp

from engine.blowup_ideal import ideal_subset
from engine.general_resolution import expression
from tests.nonreduced_oracle import _valuation
from engine.polynomial import InputError, parse_expression


def _algebraic_literal(source):
    """Read an exact scalar expression printed in a historical weighted chart."""
    functions = {"sqrt": sp.sqrt, "CRootOf": sp.CRootOf, "cos": sp.cos, "sin": sp.sin,
                 "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc, "exp": sp.exp, "Rational": sp.Rational}
    constants = {"I": sp.I, "pi": sp.pi}
    def read(node, polynomial=False):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id in constants:
                return constants[node.id]
            if polynomial and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.id):
                return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = read(node.operand, polynomial)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            from operator import add, mul, pow, sub, truediv
            operation = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: truediv, ast.Pow: pow}.get(type(node.op))
            if operation:
                return operation(read(node.left, polynomial), read(node.right, polynomial))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in functions:
                return functions[name](*(read(child, name == "CRootOf" and i == 0)
                                         for i, child in enumerate(node.args)))
        raise InputError("Unrecognized algebraic number in the archived graph.")
    try:
        return read(ast.parse(source.replace("^", "**"), mode="eval").body)
    except (SyntaxError, TypeError, IndexError) as error:
        raise InputError("Invalid algebraic number in the archived graph.") from error


@lru_cache(maxsize=128)
def _archived_field(exact):
    """Recover the defining polynomial and marked root from a historical graph literal."""
    from engine.field_coefficient import indexed_field, marked_root_display
    from engine.marked_field import description
    from engine.native_library import CHECK, _MODE

    def reference():
        root = _algebraic_literal(exact)
        return sp.QQ.algebraic_field(root), root
    if _MODE == "sympy":
        return reference()
    coefficients, flint_index = description(exact)
    order = marked_root_display(tuple(reversed(coefficients)))[2]
    result = indexed_field(coefficients, sp.Symbol("x"), order[flint_index-1])
    if CHECK:
        field, original = reference()
        comparison = field.ext.minpoly
        if (tuple(comparison.all_coeffs()) != coefficients or
                (original != result[1] if len(coefficients) == 2 else not comparison.same_root(original, result[1]))):
            raise AssertionError("The native marked field differs from its historical expression.")
    return result


def _field_coefficients(field):
    coefficients = field.get("minimal_polynomial_coefficients")
    if coefficients:
        if not isinstance(coefficients, list) or not all(
                re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", str(c)) for c in coefficients):
            raise InputError("Expected rational coefficients for the graph's number field.")
        return tuple(map(sp.Rational, coefficients))
    exact = field.get("exact_root", "")
    if not isinstance(exact, str):
        raise InputError("Expected an exact root for the archived graph.")
    return tuple(_archived_field(exact)[0].ext.minpoly.all_coeffs())


def _field(record):
    field = record.get("coefficient_field", {})
    name = field.get("generator")
    if not name:
        return sp.QQ, {}
    if field.get("minimal_polynomial_coefficients"):
        # The Euler reference needs any one embedding of the certified field.
        x = sp.Symbol("x")
        polynomial = sp.Poly.from_list(_field_coefficients(field), x)
        root = sp.CRootOf(polynomial.as_expr(), 0)
        return sp.QQ.algebraic_field(root), {name: root}
    domain, root = _archived_field(field.get("exact_root", ""))
    return domain, {name: root}


def _jacobian_order(coords, base, ambient, prime, domain):
    """The order along V(prime) of the Jacobian of the projection of the chart
    W = V(ambient) to the base coordinates.

    With t the other coordinates, the module of relative differentials of W
    over the base is presented by the matrix of the derivatives of the
    ambient equations in t, and also by the Jacobian matrix of the
    projection, a map between smooth varieties of the same dimension. Its
    zeroth Fitting ideal does not depend on the presentation: the maximal
    minors of the first matrix generate the ideal of the Jacobian
    determinant. The order of an ideal along a divisor is the least order of
    its generators.
    """
    extra = [v for v in coords if v not in base]
    if not extra:
        return 0
    from engine.affine_kernel import frame_minor
    from engine.groebner import _MODE
    rows = [[sp.diff(h, t) for t in extra] for h in ambient] if _MODE != "native" else None
    best = None
    for chosen in combinations(range(len(ambient)), len(extra)):
        if rows is not None:
            minor = sp.expand(sp.Matrix([rows[i] for i in chosen]).det())
        elif domain.is_AlgebraicField:
            from engine.chart_expressions import chart_expressions
            from engine.field_polynomial import lift, lower
            theta = chart_expressions(coords).Dummy("relative_jacobian")
            lifted = tuple(lift(ambient[i], coords, domain, theta) for i in chosen)
            minor = lower(frame_minor((*coords,theta), lifted, extra), coords, domain, theta)
        else:
            minor = frame_minor(coords, tuple(ambient[i] for i in chosen), extra)
        if minor == 0 or ideal_subset(coords, (minor,), ambient, domain=domain):
            continue
        order = _valuation(coords, ambient, prime, minor, domain)
        best = order if best is None else min(best, order)
        if best == 0:
            break
    if best is None:
        raise InputError("The projection of a graph chart is not generically a local isomorphism.")
    return best


def graph_strata(record, polynomial):
    """Numerical data {label: (N, nu)} and strata {labels: chi} over the origin
    of a certified embedded blowup of an ideal, with the method used; None
    when the record is of another kind."""
    if record.get("engine") != "verified-ideal-blowup" or record.get("mode") != "embedded":
        return None
    from engine.blowup_ideal import empty_locus, saturate
    from engine.local_blowups import Chart, _strata

    domain, constants = _field(record)
    factors = record.get("input_divisor", {}).get("factors")
    numbers = {}
    charts = []
    for chart in record["charts"]:
        coords = tuple(map(sp.Dummy, chart["coordinates"]))
        bindings = dict(zip(chart["coordinates"], coords))
        if not set(record["variables"]).issubset(bindings):
            raise InputError("A graph chart must retain the original coordinates.")
        base = tuple(bindings[name] for name in record["variables"])
        bindings.update(constants)
        def read(source):
            return expression(parse_expression(source, list(bindings)), tuple(bindings.values()))

        ambient = tuple(map(read, chart["ambient_ideal"]))
        primes = {e["id"]: tuple(map(read, e["ideal"])) for e in chart["exceptional"]}
        projective = list(map(read, chart["projective_coordinates"]))
        earlier = tuple(projective[:projective.index(1)])
        if any(v not in coords for v in earlier):
            raise InputError("Expected coordinate ratios in a standard graph chart.")
        pivot = read(chart["pivot_generator"])
        if not primes and pivot.subs(dict.fromkeys(base, 0)) != 0:
            # The center misses the origin: locally an isomorphism, and the
            # certificate proves smoothness of the support.
            return {"strict": (min(map(sum, polynomial)), 1)}, {frozenset({"strict"}): 1}, \
                "The centre misses the origin, where the support is smooth"
        f = expression(polynomial, base)
        components = []
        for comp in chart["exceptional"]:
            label = comp["id"]
            if label not in numbers:
                weight = comp.get("total_multiplicity")
                if weight is None:
                    if ideal_subset(coords, (f,), ambient, domain=domain):
                        raise InputError("The input pullback vanishes identically on the ambient chart.")
                    if ideal_subset(coords, (1,), (*ambient, *primes[label]), domain=domain):
                        raise InputError("The exceptional ideal is empty on its chart.")
                    weight = _valuation(coords, ambient, primes[label], f, domain)
                numbers[label] = (weight, 1 + _jacobian_order(coords, base, ambient, primes[label], domain))
            components.append((label, *numbers[label], primes[label]))
        if factors and len(factors) > 1:
            # The strict transform of each factor: its pullback with the
            # exceptional divisor of the chart removed.
            names = dict(zip(record["variables"], base))
            for k, factor in enumerate(factors, 1):
                g = expression(parse_expression(factor["equation"], list(names)), tuple(names.values()))
                ideal = saturate(coords, (*ambient, g), pivot, domain=domain)
                if not empty_locus(coords, ideal, domain=domain):
                    components.append((f"S{k}", factor["multiplicity"], 1, tuple(ideal)))
        else:
            strict = tuple(map(read, chart.get("reduced_strict_transform", chart["strict_transform"])))
            components.append(("strict", factors[0]["multiplicity"] if factors else 1, 1, strict))
        charts.append(Chart(coords, ambient, tuple(components), base, earlier, (), ()))
    data, strata = _strata(charts, domain)
    return data, strata, "Euler integration on disjoint pieces of the projective origin fiber"
