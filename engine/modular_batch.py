"""Validate integer arrays and transport native residue batches to the selected device."""
import ctypes
import json
import re
from time import perf_counter_ns

from .algebra_execution import current_execution, record_fallback, record_stage
from .gpu_devices import torch_device_context, triton_device
from .native_library import _library
from .polynomial import InputError, integer_text

ACTIONS = {
    "reduce": (), "add": ("other",), "subtract": ("other",), "multiply": ("other",),
    "divide": ("other",), "inverse": (), "reconstruct": (), "hensel": ("other",),
    "crt": ("other", "other_modulus"),
}
_WORD = ctypes.c_int64
_PTR = ctypes.POINTER(_WORD)
_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, _WORD, _WORD, _WORD,
                             ctypes.c_int, ctypes.c_int, _PTR, _PTR)


def integer(value):
    if type(value) is int:
        return integer_text(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
        return value.lstrip("+")
    raise InputError("Supply exact integers or signed decimal integer strings.")


def operands(action, values, modulus, other=None, other_modulus=None, field_polynomial=None):
    if not isinstance(action, str) or action not in ACTIONS:
        raise InputError("Choose a modular operation from " + ", ".join(ACTIONS) + ".")
    present = {key for key, value in (("other", other), ("other_modulus", other_modulus)) if value is not None}
    if present != set(ACTIONS[action]):
        raise InputError(f"Unexpected or missing operands for {action}.")
    result = {"action": action, "modulus": integer(modulus)}
    def element(value):
        if field_polynomial is None:
            return integer(value)
        coefficients = value if isinstance(value, (tuple,list)) else (value,)
        return [integer(c) for c in coefficients]
    if field_polynomial is not None:
        if action not in ("reduce", "add", "subtract", "multiply", "divide", "inverse"):
            raise InputError("Choose an arithmetic operation for the finite extension field.")
        if not isinstance(field_polynomial, (tuple,list)):
            raise InputError("Supply the defining polynomial as an ordered coefficient array.")
        result["field_polynomial"] = [integer(c) for c in field_polynomial]
    for key, array in (("values", values), ("other", other)):
        if key == "other" and array is None:
            continue
        if not isinstance(array, (tuple, list)):
            raise InputError("Supply an array of exact integers.")
        result[key] = [element(v) for v in array]
    if other_modulus is not None:
        result["other_modulus"] = integer(other_modulus)
    return result


_FIELD_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   _WORD, _WORD, ctypes.c_int, ctypes.c_int, _PTR, _PTR)


def _field_launch(device, action, degree, count, columns, characteristic, exponent, bits, words, packed, answers):
    import torch
    from .triton_buffers import to_device, to_host
    from .triton_ir import launch_kernel
    with torch_device_context(device):
        host = torch.frombuffer((_WORD*words).from_address(ctypes.addressof(packed.contents)), dtype=torch.int64)
        inputs = to_device(host,device)
        output = torch.empty(count*degree, dtype=torch.int64, device=device)
        block = 32
        launch_kernel("finite_field_batch", (action,degree,block), ((count+block-1)//block,),
                      inputs, inputs[count*columns:], output, count, characteristic, exponent, bits)
        destination = (_WORD*output.numel()).from_address(ctypes.addressof(answers.contents))
        torch.frombuffer(destination,dtype=torch.int64).copy_(to_host(output))


def _launch(device, action, count, modulus, other_modulus, bound, input_columns, output_columns, packed, answers):
    import torch
    from .triton_buffers import to_device, to_host
    from .triton_ir import launch_kernel
    def view(pointer, size):
        return torch.frombuffer((_WORD*size).from_address(ctypes.addressof(pointer.contents)), dtype=torch.int64)
    block = 128
    with torch_device_context(device):
        inputs = to_device(view(packed, input_columns*count), device).split(count)
        output = torch.empty(output_columns*count, dtype=torch.int64, device=device)
        outputs = output.split(count)
        launch_kernel("residue_batch", (action, block), ((count + block-1)//block,),
                      inputs[0], inputs[min(1,input_columns-1)], inputs[-1], outputs[0], outputs[-1],
                      count, modulus, other_modulus, bound)
        view(answers, output.numel()).copy_(to_host(output))


def handle(operation, ring, request):
    execution = current_execution()
    scope = execution.replay
    device = None
    if scope.compute == "gpu" and scope.backend in ("auto", "triton"):
        device = triton_device(scope.index)
    error = []
    extension = "field_polynomial" in request
    callback_type = _FIELD_CALLBACK if extension else _CALLBACK
    launch = _field_launch if extension else _launch
    @callback_type
    def batch(*args):
        try:
            launch(device["torch_device"], *args)
            return 1
        except Exception as exc:
            error.append(str(exc))
            return 0

    fn = _library().atlas_finite_field_batch if extension else _library().atlas_modular_batch
    fn.argtypes, fn.restype = [ctypes.c_char_p, callback_type], ctypes.c_char_p
    started = perf_counter_ns()
    response = json.loads(fn(json.dumps(request).encode("ascii"), batch if device else callback_type()))
    if "error" in response:
        raise InputError(response["error"])
    result = response["result"]
    ran = result.pop("device_batch")
    details = {"entries": len(request["values"]), "elapsed_ns": perf_counter_ns()-started}
    if ran:
        details["device"] = scope.record_device("triton", device)
    elif scope.compute == "gpu" and request["values"]:
        reason = ("The field degree, element size or batch exceeds the Triton kernel bounds." if extension else
                  "Modulus or reduction inputs exceed the Triton word range.")
        record_fallback(error[0] if error else (reason
                        if device else "Modular arithmetic requires a usable Triton device."))
    record_stage(("finite_field_" if extension else "modular_")+request["action"], "triton" if ran else "cpu", **details)
    return dict(type="modular-values", ring=ring, **result)


def modular_arithmetic(action, values, modulus, *, other=None, other_modulus=None, field_polynomial=None, compute="auto"):
    """Return a batch of exact residues, lifted roots or bounded rational reconstructions.

    >>> r = modular_arithmetic("hensel", [3, 4], 7, other=[-2, 0, 1], compute="gpu")
    >>> r.values  # (10, 39)
    >>> r.modulus  # 49

    Docs: examples/polynomial-ideals.md#compute-with-residues-and-lift-roots
    """
    from .algebra import dispatch
    from .session_algebra import _typed
    from .session_polynomial import QQ
    data = operands(action, values, modulus, other, other_modulus, field_polynomial)
    request = {"operation": "modular-arithmetic", "ring": {"variables": []}, "operands": data, "compute": compute}
    return _typed(dispatch(request), QQ)
