"""Transport ordered QQ ideal operations and their scheduled Gröbner tasks."""
import ctypes
import json
from contextlib import ExitStack

from . import native_polynomial as np
from .affine_kernel import _sparse_expression, sparse_terms
from .chart_expressions import chart_expressions
from .kernel_json import kernel_json
from .native_library import _library
from .polynomial import encode_sparse_row, sparse_row


_BasisDispatch = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)


def scalar_ideal(operation, generators, query=0):
    return kernel_json("atlas_scalar_ideal", operation, ideal=[np.row((), f) for f in generators],
                       query=np.row((), query))


def ordered_ideal(operation, variables, left, right, order, basis, stage, *, removed=(), by=None):
    variables = tuple(variables)
    expressions = chart_expressions(variables)
    native = np.native_ring(variables) or not variables
    auxiliary = expressions.Dummy("inverse" if operation == "saturate" else "intersection")
    coordinates = (*variables, auxiliary)

    def rows(axes, generators):
        if native:
            return [np.row(axes, f) for f in generators]
        return [encode_sparse_row(dict(sparse_terms(axes, f))) for f in generators]

    def values(axes, encoded):
        if native:
            return tuple(np.from_row(axes, r) for r in encoded)
        if not axes:
            return tuple(expressions.Rational(next(iter(sparse_row(r).values()), 0)) for r in encoded)
        return tuple(_sparse_expression(axes, r) for r in encoded)

    payload = dict(operation=operation, width=len(variables), order=order, left=rows(variables, left),
                   right=rows(variables, right), native_ring=np.native_ring(variables), removed=[variables.index(v) for v in removed])
    if by is not None:
        payload['by'], = rows(variables, [by])
    buffer = None
    errors = []
    with ExitStack() as cleanup:
        stages = []

        @_BasisDispatch
        def dispatch(raw):
            nonlocal buffer
            try:
                task = json.loads(raw)
                result = None
                if task['event'] == 'basis':
                    axes = tuple(coordinates[i] for i in task['ring'])
                    gb = basis(task['label'], axes, values(axes, task['generators']), task['order'])
                    result = rows(axes, gb.exprs)
                elif task['event'] == 'begin':
                    scope = ExitStack()
                    cleanup.callback(scope.close)
                    scope.enter_context(stage(task['label'], **task['counts']))
                    stages.append(scope)
                elif task['event'] == 'end':
                    stages.pop().close()
                else:
                    raise ValueError("Unknown native ideal scheduling event.")
                buffer = ctypes.create_string_buffer(json.dumps(result).encode('ascii'))
                return ctypes.addressof(buffer)
            except BaseException as exc:
                errors.append(exc)
                return None

        function = _library().atlas_ordered_ideal
        function.argtypes = [ctypes.c_char_p, _BasisDispatch]
        function.restype = ctypes.c_char_p
        reply = json.loads(function(json.dumps(payload).encode('ascii'), dispatch))
        if errors:
            raise errors[0]
    if 'error' in reply:
        raise ArithmeticError(reply['error'])
    output = reply['result']
    axes = tuple(coordinates[i] for i in output['ring'])
    return axes, values(axes, output['generators']), order
