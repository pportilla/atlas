"""Irreducible factorizations in ordered rational polynomial rings."""

from time import perf_counter_ns

from . import native_polynomial as np
from .affine_kernel import _request, _ring
from .algebra_execution import cpu_stage, record_stage
from .kernel_json import kernel_json
from .gpu_factor_division import factor_division_hooks
from .polynomial import InputError, format_poly, integer_from_decimal, sparse_row


def factor_polynomial(variables, polynomial, *, order="grevlex", squarefree=False):
    """Return the rational unit and distinct monic irreducibles with their multiplicities."""
    variables = tuple(variables)
    polynomial = np.coerce(polynomial)
    if order not in ("lex", "grevlex"):
        raise InputError("Factorization requires lex or grevlex order.")
    if (len(set(variables)) != len(variables) or
            any(not isinstance(v, np.Expr) or not v.is_Symbol for v in variables) or
            not polynomial.free_symbols.issubset(variables)):
        raise InputError("Supply a polynomial in the declared rational coordinates.")
    if not polynomial:
        raise InputError("The zero polynomial has no irreducible factorization.")
    # Even a constant in the empty coordinate space can be read in a
    # temporary one-variable kernel ring. Its result has no factors.
    kernel_vars = variables or (np.Symbol("v0"),)
    ring = _ring(kernel_vars)
    started = perf_counter_ns()
    with factor_division_hooks():
        algorithm, unit_row, *rows = _request([
            "squarefree_factorization" if squarefree else "ordered_factorization", ring.header, order, ring.text(polynomial),
        ])
    unit = np.from_row(kernel_vars, unit_row).scalar
    factors = [(np.from_row(variables, row), integer_from_decimal(exponent))
               for exponent, row in zip(rows[::2], rows[1::2])]
    factors.sort(key=(lambda part: part[1]) if squarefree else (lambda part: np.default_sort_key(part[0])))
    details = {} if algorithm == "rational_constant" else {"order": order}
    record_stage("polynomial_factorization", "cpu", algorithm=algorithm,
                 elapsed_ns=perf_counter_ns() - started, **details)
    return unit, tuple(factors)


def absolute_factor_record(ring, variables, polynomial):
    """Present native absolute factors in their common marked coefficient field."""
    with factor_division_hooks(), cpu_stage("absolute_factorization", order=ring["order"]):
        data = kernel_json("atlas_absolute_factorization", "factor", width=len(variables),
                           polynomial=np.row(variables, polynomial), order=ring["order"])
    names = ring["variables"]
    field = data["field"]
    if field is not None:
        generator = "a"
        index = 0
        while generator in names:
            index += 1
            generator = f"a{index}"
        coefficients = field["coefficients"]
        minimal = {(j,): integer_from_decimal(c) for j, c in enumerate(coefficients)}
        description = {"generator": generator, "minimal_polynomial": format_poly(minimal, [generator]),
                       "degree": field["degree"], "root_index": field["index"], "embedding": field["exact"]}
        ring = {**ring, "field": description}
        names = [*names, generator]
    return {"type": "absolute-factorization", "ring": ring, "unit": data["unit"],
            "factors": [{"polynomial": format_poly(sparse_row(entry["polynomial"]), names),
                         "multiplicity": integer_from_decimal(entry["multiplicity"])} for entry in data["factors"]]}
