"""Transport chart rows, marked fields, and requests from native prime-stratum incidence."""
import ctypes
import json
from dataclasses import replace
from functools import cache, lru_cache

from . import native_polynomial as np
from .affine_kernel import _sparse_expression, rational_factor_rows, sparse_terms
from .local_blowups import _is_strict, _own_failures, _present
from .blowup_ideal import smooth_locus
from .groebner import groebner
from .native_library import _library
from .polynomial import encode_sparse_row, integer_text

_Callback = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)


@lru_cache(maxsize=8192)
def rows(axes, ideal):
    if np.native_ring(axes):
        return tuple(np.row(axes, f) for f in ideal)
    return tuple(encode_sparse_row(dict(sparse_terms(axes, f))) for f in ideal)


def values(axes, encoded):
    return tuple(_value(axes, row) for row in encoded)


@lru_cache(maxsize=8192)
def _value(axes, row):
    return np.from_row(axes,row) if np.native_ring(axes) else _sparse_expression(axes,row)


@lru_cache(maxsize=256)
def chart_input(chart):
    axes = chart.variables
    return dict(width=len(axes), ambient=rows(axes, chart.ambient), owned=rows(axes, (*chart.fibre,*chart.region)),
                holes=[rows(axes, z) for z in chart.removed], maps=rows(axes, tuple(f for _,f in chart.coordinates)),
                theta=axes.index(chart.coefficient_field[0]) if chart.coefficient_field else -1,
                parent=chart.repair_projection[0] if chart.repair_projection else -1,
                repair=rows(axes, chart.repair_projection[1]) if chart.repair_projection else (),
                components=[(name, rows(axes, ideal), _is_strict(name)) for name, _, _, ideal in chart.components])


def operation(payload, handle, *, symbol="atlas_chart_gluing"):
    # Preserve exceptions (including requested field extensions) across ctypes.
    errors = []
    buffer = None

    @_Callback
    def callback(raw):
        nonlocal buffer
        try:
            reply = handle(json.loads(raw))
            buffer = ctypes.create_string_buffer(json.dumps(reply, separators=(",", ":")).encode("ascii"))
            return ctypes.addressof(buffer)
        except BaseException as exc:
            errors.append(exc)
            return None

    function = getattr(_library(),symbol)
    function.argtypes = [ctypes.c_char_p, _Callback]
    function.restype = ctypes.c_char_p
    reply = json.loads(function(json.dumps(payload,separators=(",", ":")).encode("ascii"), callback))
    if errors:
        raise errors[0]
    if "error" in reply:
        raise ArithmeticError(reply["error"])
    return reply["result"]


def chart_complex(charts, overlap, transition, dimension=0, intrinsic=False, *, character=None, absolute=False,
                  pieces=None):
    from .absolute_sections import absolute_primes
    from .atlas_gluing import _NeedsSplittingField, _geometric_primes
    from .constructible_euler import euler_characteristic

    @cache
    def target_basis(i, equations):
        return groebner(equations,charts[i].variables)

    transition = cache(transition)

    def handle(task):
        op, i = task["op"], task["i"]
        c = charts[i]
        axes = c.variables
        if op == "transition":
            direct = transition(i,task["j"])
            if direct is None:
                return None
            transfer, holes = direct
            record = dict(holes=[rows(axes,tuple(h)) for h in holes])
            fractions = getattr(transfer,"coordinate_fractions",None)
            if fractions is not None:
                record["map"] = [rows(axes,fractions[v]) for v in charts[task["j"]].variables]
            return record
        if op == "transport":
            transfer, _ = transition(i,task["j"])
            basis = target_basis(i,values(axes,task["source"]))
            target = values(charts[task["j"]].variables, task["target"])
            return rows(axes,tuple(transfer(f,basis) for f in target))
        if op == "overlap":
            coordinates, _, _, graph, removed = overlap(i,task["j"])
            return dict(graph=rows(coordinates,tuple(graph)), removed=[rows(coordinates,tuple(h)) for h in removed])
        if op == "snc":
            restricted = replace(c, ambient=values(axes,task["carrier"]),
                                 components=tuple((str(j),1,1,values(axes,ideal)) for j,ideal in enumerate(task["branches"])),
                                 fibre=(*c.fibre,*c.region), region=())
            return not _own_failures(restricted,task["dimension"],{})[0]
        if op == "factors":
            f, = values(axes,[task["equation"]])
            return [(row, integer_text(m)) for row,m,_ in rational_factor_rows(axes,f)[1]]
        equations = values(axes,task["ideal"])
        if op == "present":
            return _present(axes,equations,c.removed)
        if op == "euler":
            return integer_text(c.one_sheet(euler_characteristic(axes,equations,c.removed)))
        if op == "smooth":
            report = smooth_locus(axes,equations,task["dimension"])
            return None if report["smooth"] else rows(axes,tuple(report["bad_ideal"]))
        if op == "split":
            raise _NeedsSplittingField(i,equations,task["dimension"])
        if op == "geometric":
            primes = _geometric_primes(c,equations,task["dimension"],character,reduced_support=task["reduced"])
        elif op == "absolute":
            primes = absolute_primes(c,equations,task["dimension"])
        else:
            raise ValueError("Unknown chart gluing request.")
        return None if primes is None else [rows(axes,tuple(ideal)) for ideal in primes]

    payload = dict(charts=[chart_input(c) for c in charts], dimension=dimension, intrinsic=intrinsic,
                   absolute=absolute, operation="complex")
    if pieces is not None:
        payload.update(operation="groups",pieces=[(i, label or "", rows(charts[i].variables, prime))
                                                  for (i,label),prime in pieces])
    result = operation(payload,handle)
    if result is None or pieces is None:
        return result
    return [dict(pieces[k] for k in group) for group in result]
