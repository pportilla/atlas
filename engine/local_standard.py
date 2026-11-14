"""Local standard bases and normal forms over QQ at rational affine points."""

from math import lcm, prod

from . import native_polynomial as np
from .algebra_execution import cpu_stage, stage_label
from .algebra_groebner import reduced_basis
from .local_algebra import _homogeneous_local_basis, _leading
from .local_mora import weak_normal_form
from .polynomial import InputError, format_poly
from .rational_point import coordinate_fraction, normalize_point


def _translate(variables, shifts, polynomials):
    if not any(shifts):
        return polynomials
    mapping = {v: v + a for v, a in zip(variables, shifts) if a}
    return tuple(f.xreplace(mapping) for f in polynomials)


def _project_monomials(variables, point, generators):
    # x_i is a unit at p when p_i != 0. On the remaining axes the original
    # and centered coordinates agree. Translation on the unit axes therefore
    # commutes with deleting every term in this projected monomial ideal.
    heads = set()
    for g in generators:
        head, = np.terms(variables, g)
        heads.add(tuple(0 if p else e for p, e in zip(point, head)))
    minimal = []
    for head in sorted(heads, key=lambda e: (sum(e), e)):
        if not any(all(a <= b for a, b in zip(h, head)) for h in minimal):
            minimal.append(head)
    return tuple(np.from_terms(variables, {h: 1}) for h in minimal)


def _nonzero_at_point(variables, point, f):
    # Drop terms killed by zero coordinates. A single surviving term is
    # nonzero without ever expanding a power of a nonzero rational number.
    terms = np.terms(variables, f)
    active = [(e, c) for e, c in terms.items()
              if all(not k or a for k, a in zip(e, point))]
    if len(active) < 2:
        return bool(active)
    # All surviving monomials share a nonzero factor at p. Removing it keeps
    # exponents small for inputs like x^N*(x-2) evaluated at x=2.
    if len({(c < 0) ^ (sum(k for a, k in zip(point, e) if a < 0) % 2)
            for e, c in active}) == 1:
        return True
    minimum = tuple(map(min, zip(*(e for e, _ in active))))
    active = [(tuple(k - m for k, m in zip(e, minimum)), c) for e, c in active]
    if any(k.bit_length() > 16 and abs(a) > 0 and abs(a) != 1
           for e, _ in active for k, a in zip(e, point)):
        # A nonzero residue certifies a nonzero rational evaluation. Clear
        # coefficient denominators so a chosen prime may divide those too.
        common = lcm(*(c.denominator for _, c in active))
        for prime in (1000000007, 1000000009, 998244353):
            if any(a.denominator % prime == 0 for a in point):
                continue
            residues = [a.numerator * pow(a.denominator, -1, prime) % prime for a in point]
            value = sum((common // c.denominator) * c.numerator *
                        prod(pow(v, k, prime) for v, k in zip(residues, e) if k)
                        for e, c in active)
            if value % prime:
                return True
    return bool(sum(c * prod(a**k for a, k in zip(point, e) if k)
                    for e, c in active))


def _local_head(row, order):
    degree = min(map(sum, row))
    return _leading((e for e in row if sum(e) == degree), order)


def _certified_rows(variables, inputs, computed, order):
    """Preserve a supplied standard basis once its local head ideal is certified."""
    with cpu_stage("certify_local_heads", input_rows=len(inputs), basis_rows=len(computed)):
        heads = [_local_head(row, order) for row in inputs]
        certified = [_local_head(row, order) for row in computed]
        if any(not any(h) for h in certified):
            return (np.S.One,)
        # All input rows belong to I. The reverse inclusion of head ideals
        # proves that these very rows form a standard basis, even if redundant.
        # Keeping them prevents a saved weak remainder changing on raw replay.
        if all(any(all(a <= b for a, b in zip(h, target)) for h in heads)
               for target in certified):
            rows = inputs
        else:
            rows, heads = computed, certified
        monic = set()
        for row, head in zip(rows, heads):
            coefficient = row[head]
            monic.add(np.from_terms(variables, row if coefficient == 1 else
                                    {e: c / coefficient for e, c in row.items()}))
        return tuple(sorted(monic, key=np.default_sort_key))


def _standard_basis(variables, generators, order):
    """A local standard basis in coordinates centered at the chosen point."""
    with cpu_stage("inspect_local_generators", generators=len(generators)):
        rows = [np.terms(variables, f) for f in generators if f]
        if not rows:
            return ()
        if any((0,) * len(variables) in row for row in rows):
            return (np.S.One,)
        if len(rows) == 1:
            # Local heads multiply, so one generator is already a standard basis.
            row, = rows
            head = _local_head(row, order)
            return (np.from_terms(variables, {e: c / row[head] for e, c in row.items()}),)
        homogeneous = all(len({sum(e) for e in row}) == 1 for row in rows)

    if homogeneous:
        with stage_label("local_standard_basis"):
            basis = reduced_basis(generators, variables, order=order)
        return _certified_rows(variables, rows, [np.terms(variables, f) for f in basis.exprs], order)

    basis = _homogeneous_local_basis(variables, rows, order)
    # Clearing the homogenizing power in any representation f=sum(a_i*g_i)
    # shows that these projected heads divide every local leading monomial in I.
    # The rows are homogeneous, so setting h=1 never merges distinct terms.
    with cpu_stage("dehomogenize_local_basis", generators=len(basis.exprs)):
        projected = [{e[1:]: c for e, c in np.terms(basis.gens, f).items()} for f in basis.exprs]
    return _certified_rows(variables, rows, projected, order)


def handle(operation, ring, operands, *, known_basis=False):
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Local standard bases require QQ and lex or grevlex order.")
    names = ring["variables"]
    variables = tuple(map(np.Symbol, names))
    point = normalize_point(operands.get("point", [0] * len(names)), len(names))
    shifts = tuple(map(coordinate_fraction, point))
    normal = operation == "local-normal-form"
    if not normal and operation != "local-standard-basis":
        raise InputError(f"Unknown local standard basis operation {operation!r}.")
    result = {"type": operation, "ring": ring, "scope": "point", "point": point}
    generators = tuple(g for f in operands["ideal"] if (g := np.coerce(f)))
    query = np.coerce(operands["polynomial"]) if normal else None
    projected = None
    with cpu_stage("inspect_point_ideal", generators=len(generators)):
        if normal and (not query or not generators):
            result.update(reduction="standard", unit="1",
                          remainder=format_poly(np.terms(variables, query), names))
            return result
        if any(shifts) and generators and all(len(g._terms) == 1 for g in generators):
            projected = _project_monomials(variables, shifts, generators)
        if any(g.is_number for g in generators) or projected == (np.S.One,) or (
                projected is None and any(shifts) and
                any(_nonzero_at_point(variables, shifts, g) for g in generators)):
            if normal:
                result.update(reduction="standard", unit="1", remainder="0")
            else:
                result["generators"] = ["1"]
            return result
    if projected is not None:
        if normal:
            with cpu_stage("local_normal_form", divisors=len(projected)):
                _, remainder, reduction = weak_normal_form(variables, projected, query, ring["order"])
            result.update(unit="1", reduction=reduction, remainder=format_poly(np.terms(variables, remainder), names))
        else:
            result["generators"] = [format_poly(np.terms(variables, f), names) for f in projected]
        return result
    with cpu_stage("center_at_point", variables=len(names)):
        centered = _translate(variables, shifts, generators)
    if normal and known_basis:
        # Only private worker metadata can identify rows already certified at
        # this same point and order. Wire generator arrays require a fresh basis.
        with cpu_stage("restore_local_basis", generators=len(centered)):
            basis = centered
    else:
        basis = _standard_basis(variables, centered, ring["order"])
    if normal:
        if basis == (np.S.One,):
            result.update(reduction="standard", unit="1", remainder="0")
            return result
        with cpu_stage("center_query_at_point", variables=len(names)):
            query, = _translate(variables, shifts, (query,))
        with cpu_stage("local_normal_form", divisors=len(basis)):
            unit, remainder, reduction = weak_normal_form(variables, basis, query, ring["order"])
        outputs = (unit, remainder)
    else:
        outputs = basis
    with cpu_stage("restore_point_coordinates", polynomials=len(outputs)):
        restored = _translate(variables, tuple(-a for a in shifts), outputs)
        texts = [format_poly(np.terms(variables, f), names) for f in restored]
    if normal:
        unit_text, remainder_text = texts
        result.update(unit=unit_text, remainder=remainder_text, reduction=reduction)
    else:
        result["generators"] = texts
    return result
