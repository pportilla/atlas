"""Load the exact kernel without importing a polynomial expression library."""
import ctypes
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import os
from pathlib import Path
import sys
from threading import Lock

_LIBRARY = Path(__file__).resolve().parents[1] / "native" / "build" / (
    "libgroebner.dylib" if sys.platform == "darwin" else "libgroebner.so")

_MODE = os.getenv("ATLAS_GROEBNER", "native")
if _MODE not in ("native", "sympy", "check"):
    raise ValueError("ATLAS_GROEBNER must be native, sympy, or check.")
CHECK = _MODE == "check"


@lru_cache(maxsize=1)
def _kernel():
    try:
        library = ctypes.CDLL(str(_LIBRARY))
    except OSError:
        return None
    library.atlas_groebner.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner.restype = ctypes.c_void_p
    library.atlas_groebner_error.restype = ctypes.c_char_p
    library.atlas_groebner_text.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_text.restype = ctypes.c_char_p
    library.atlas_groebner_seconds.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_seconds.restype = ctypes.c_double
    library.atlas_groebner_reduce.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner_reduce.restype = ctypes.c_void_p
    library.atlas_groebner_colon.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner_colon.restype = ctypes.c_void_p
    try:
        library.atlas_ideal.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        library.atlas_ideal.restype = ctypes.c_void_p
    except AttributeError:  # built before native/ideal.cpp existed: run make
        return None
    library.atlas_groebner_release.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_release.restype = None
    library.atlas_groebner_free.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_free.restype = None
    if hasattr(library, "atlas_groebner_use_batch"):
        library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
        library.atlas_groebner_use_batch.restype = None
        if hasattr(library, "atlas_groebner_swap_batch"):
            library.atlas_groebner_swap_batch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]
            library.atlas_groebner_swap_batch.restype = None
            library.atlas_groebner_cpu_replays.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
            library.atlas_groebner_cpu_replays.restype = None
            if hasattr(library, "atlas_groebner_cpu_trace_counts"):
                library.atlas_groebner_cpu_trace_counts.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
                library.atlas_groebner_cpu_trace_counts.restype = None
        # Standalone library users can opt in through the environment even
        # after the library has loaded. Old binaries use the same permanent
        # dispatcher so concurrent Atlas requests never swap a global hook.
        library.atlas_groebner_use_batch(ctypes.cast(_dispatch_batch, ctypes.c_void_p))
    return library


_UINTS = ctypes.POINTER(ctypes.c_uint32)
_TRACE = ctypes.CFUNCTYPE(ctypes.c_int, *([ctypes.c_uint32] * 3 + [_UINTS] * 6))
_BATCH = ctypes.CFUNCTYPE(ctypes.c_int, _UINTS, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                          _UINTS, ctypes.c_uint32, _UINTS, _UINTS)
_LEGACY = {"1": "opencl", "triton": "triton", "reference": "reference"}
_replay = ContextVar("atlas_replay", default=None)
_old_hook_lock = Lock()
_old_scope_count = 0


def current_replay_scope():
    return _replay.get()


@lru_cache(maxsize=1)
def _opencl_replay():
    path = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    try:
        function = ctypes.CDLL(str(path)).atlas_replay_opencl_index
    except (OSError, AttributeError):
        return None
    function.argtypes = [*_BATCH._argtypes_, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


@lru_cache(maxsize=1)
def _opencl_trace():
    path = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    try:
        function = ctypes.CDLL(str(path)).atlas_trace_screen_opencl_index
    except (OSError, AttributeError):
        return None
    function.argtypes = [*_TRACE._argtypes_, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def _use_gpu(library):
    """Install the standalone OpenCL replay hook for legacy library callers."""
    path = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    try:
        plugin = ctypes.CDLL(str(path))
        if not plugin.atlas_replay_opencl_available():
            return False
        library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
        library.atlas_groebner_use_batch.restype = None
        library.atlas_groebner_use_batch(ctypes.cast(plugin.atlas_replay_opencl, ctypes.c_void_p))
        library._replay_device = plugin
        return True
    except (OSError, AttributeError):
        return False


def _cpu_counts(library):
    batches, primes = ctypes.c_uint64(), ctypes.c_uint64()
    library.atlas_groebner_cpu_replays(ctypes.byref(batches), ctypes.byref(primes))
    layers, pairs = ctypes.c_uint64(), ctypes.c_uint64()
    if hasattr(library, "atlas_groebner_cpu_trace_counts"):
        library.atlas_groebner_cpu_trace_counts(ctypes.byref(layers), ctypes.byref(pairs))
    return batches.value, primes.value, layers.value, pairs.value


class _ReplayScope:
    def __init__(self, compute, backend, index):
        legacy = _LEGACY.get(os.getenv("ATLAS_GROEBNER_GPU"))
        # Automatic dispatch keeps the measured threaded CPU route unless the
        # caller has opted into replay offload with the legacy setting.
        if compute is None:
            compute = "auto" if legacy or os.getenv("RESOLUTION_GPU") == "auto" else "cpu"
        if compute == "auto" and legacy and backend is None:
            compute = "gpu"
        if compute not in ("cpu", "auto", "gpu"):
            raise ValueError("Unknown replay compute policy.")
        if compute == "cpu":
            backend, index = "auto", 0
        else:
            from .gpu_devices import gpu_preference
            if backend is None:
                backend = gpu_preference()
                if backend == "auto" and legacy and compute == "gpu":
                    backend = legacy
            if backend not in ("auto", "triton", "opencl", "reference"):
                raise ValueError("Unknown replay backend.")
            if index is None:
                index = int(os.getenv("RESOLUTION_GPU_INDEX", 0))
            if not 0 <= index < 2**31:
                raise ValueError("GPU index must be nonnegative and below 2^31.")
        self.compute, self.backend, self.index = compute, backend, index
        self.counts = {}
        self.trace_counts = {}
        self.reasons = []
        self.device = None
        self.library = None
        self.start = None
        self.unavailable = set()
        self.trace_unavailable = set()
        self.interrupted = None

    def settings(self):
        return {"compute": self.compute, "backend": self.backend, "index": self.index}

    def count(self, backend, batches, primes):
        if batches:
            found = self.counts.setdefault(backend, {"batches": 0, "primes": 0})
            found["batches"] += batches
            found["primes"] += primes

    def collect_cpu(self):
        if self.start is not None:
            now = _cpu_counts(self.library)
            self.count("cpu", now[0] - self.start[0], now[1] - self.start[1])
            if now[2] > self.start[2]:
                self.count_trace("cpu", now[2] - self.start[2], now[3] - self.start[3])
            self.start = now

    def report(self):
        if self.interrupted is not None:
            raise self.interrupted
        self.collect_cpu()
        if not self.counts and not self.trace_counts and not self.reasons:
            return None
        backend = next(iter(self.counts), None) if len(self.counts) <= 1 else "mixed"
        data = {"backend": backend,
                "batches": sum(item["batches"] for item in self.counts.values()),
                "primes": sum(item["primes"] for item in self.counts.values()),
                "backends": {key: dict(count) for key, count in self.counts.items()}}
        if self.device is not None:
            data["device"] = self.device
        if self.trace_counts:
            data["trace"] = {"backends": {key: dict(counts) for key, counts in self.trace_counts.items()}}
        if self.reasons:
            data["fallback_reasons"] = list(self.reasons)
        return data

    def merge(self, report):
        if report:
            for backend, count in report["backends"].items():
                self.count(backend, count["batches"], count["primes"])
            for backend, count in report.get("trace", {}).get("backends", {}).items():
                self.count_trace(backend, count["batches"], count["pairs"])
            self.device = report.get("device", self.device)
            for reason in report.get("fallback_reasons", []):
                if reason not in self.reasons:
                    self.reasons.append(reason)

    def count_trace(self, backend, batches, pairs):
        count = self.trace_counts.setdefault(backend, {"batches": 0, "pairs": 0})
        count["batches"] += batches
        count["pairs"] += pairs

    def screen(self, args):
        choices = ("triton", "opencl") if self.backend == "auto" else (self.backend,)
        for backend in choices:
            if backend in self.trace_unavailable:
                continue
            try:
                if backend == "opencl":
                    function = _opencl_trace()
                    if function is None or function(*args, self.index):
                        raise RuntimeError("OpenCL trace kernel or device unavailable")
                    device = {"backend": backend, "index": self.index}
                else:
                    from .gpu_devices import triton_device
                    device = triton_device(self.index)
                    if device is None:
                        raise RuntimeError("Triton device unavailable")
                    from .gpu_trace import reduce_rows
                    reduce_rows(args, device["torch_device"])
            except Exception as exc:
                self.trace_unavailable.add(backend)
                self.reasons.append(f"{backend} trace: {exc}")
                continue
            self.count_trace(backend, 1, args[1])
            self.device = device
            return 0
        return 1

    def run(self, args):
        choices = ("triton", "opencl") if self.backend == "auto" else (self.backend,)
        if self.compute != "gpu":
            choices = ()
        for backend in choices:
            if backend in self.unavailable:
                continue
            try:
                if backend == "opencl":
                    execute = _opencl_replay()
                    if execute is None:
                        raise RuntimeError("indexed OpenCL replay plugin unavailable")
                    if execute(*args, self.index) != 0:
                        raise RuntimeError("OpenCL replay failed")
                    device = {"backend": "opencl", "index": self.index}
                else:
                    from . import gpu_replay
                    device = None
                    if backend == "triton":
                        from .gpu_devices import triton_device
                        device = triton_device(self.index)
                        if device is None:
                            raise RuntimeError("Triton device unavailable")
                    gpu_replay.run_batch(args, device)
            except Exception as exc:
                self.unavailable.add(backend)
                self.reasons.append(f"{backend}: {exc}")
                continue
            self.count(backend, 1, args[5])
            self.device = device
            return 0
        # Native thread counters record the host execution after we return.
        # The old binary doesn't provide those counters.
        if self.start is None:
            self.count("cpu", 1, args[5])
        return 1


@_BATCH
def _dispatch_batch(*args):
    scope = current_replay_scope()
    try:
        if scope is None:
            if os.getenv("ATLAS_GROEBNER_GPU") not in _LEGACY:
                return 1
            scope = _ReplayScope(None, None, None)
        return scope.run(args)
    except BaseException as exc:
        # Exceptions must not cross a ctypes callback. Restore interruptions
        # in Python after the native operation returns.
        if scope is not None:
            scope.interrupted = exc
        return 1


@_TRACE
def _dispatch_trace(*args):
    scope = current_replay_scope()
    if scope is None or scope.interrupted is not None:
        return 1
    try:
        return scope.screen(args)
    except BaseException as exc:
        scope.interrupted = exc
        return 1


@contextmanager
def _legacy_replay_scope(library):
    # Old binaries cannot save or query the installed hook. The first active
    # solve reinstalls the permanent dispatcher after standalone users, and
    # concurrent scopes share that pointer without rewriting it mid-replay.
    global _old_scope_count
    with _old_hook_lock:
        if not _old_scope_count:
            library.atlas_groebner_use_batch(ctypes.cast(_dispatch_batch, ctypes.c_void_p))
        _old_scope_count += 1
    try:
        yield
    finally:
        with _old_hook_lock:
            _old_scope_count -= 1


@contextmanager
def replay_scope(compute=None, backend=None, index=None):
    """Select the modular replay policy for this synchronous solve and its worker payloads."""
    scope = _ReplayScope(compute, backend, index)
    outer = current_replay_scope()
    if outer is not None:
        outer.collect_cpu()
    token = _replay.set(scope)
    try:
        library = _kernel()
        if library is None or not hasattr(library, "atlas_groebner_use_batch"):
            yield scope
        elif not hasattr(library, "atlas_groebner_swap_batch"):
            with _legacy_replay_scope(library):
                yield scope
        else:
            scope.library = library
            scope.start = _cpu_counts(library)
            pointer = ctypes.cast(_dispatch_batch, ctypes.c_void_p) if scope.compute == "gpu" else ctypes.c_void_p()
            active = ctypes.c_int(1)
            # An inner CPU scope clears the outer GPU trace hook on this thread.
            trace_swap = getattr(library, "atlas_groebner_swap_trace_screen", None)
            enable_trace = scope.compute == "gpu" and scope.backend != "reference"
            trace = ctypes.cast(_dispatch_trace, ctypes.c_void_p) if enable_trace else ctypes.c_void_p()
            if trace_swap:
                trace_swap.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                trace_swap.restype = None
                trace_swap(ctypes.byref(trace))
            library.atlas_groebner_swap_batch(ctypes.byref(pointer), ctypes.byref(active))
            try:
                yield scope
            finally:
                library.atlas_groebner_swap_batch(ctypes.byref(pointer), ctypes.byref(active))
                if trace_swap:
                    trace_swap(ctypes.byref(trace))
                scope.collect_cpu()
                scope.start = None
        if scope.interrupted is not None:
            raise scope.interrupted
    finally:
        _replay.reset(token)
        if outer is not None and outer.start is not None:
            # Nested work has its own report. The parent may merge it once.
            outer.start = _cpu_counts(outer.library)


class Unavailable(ArithmeticError):
    """The binary or its packed representation is unavailable for this input."""


@lru_cache(maxsize=1)
def _library():
    lib = _kernel()
    if lib is None or not hasattr(lib, "atlas_affine"):
        raise Unavailable("The native affine-map kernel is unavailable.")
    lib.atlas_affine.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.atlas_affine.restype = ctypes.c_void_p
    lib.atlas_groebner_error.restype = ctypes.c_char_p
    return lib


def affine_request(lines):
    data = ("\n".join(lines) + "\n").encode("utf-8")
    lib = _library()
    if lines[0].startswith("lattice_") and not hasattr(lib, "atlas_lattice_abi"):
        raise Unavailable("The installed kernel predates the torus lattice operations.")
    if lines[0] in ("parse_input", "parse_quick") and not hasattr(lib, "atlas_input_abi"):
        raise Unavailable("The installed kernel predates the public input parser.")
    answer = lib.atlas_affine(data, len(data))
    if not answer:
        message = lib.atlas_groebner_error().decode("utf-8")
        if message in ("Unsupported number of variables.", "Degree out of range.", "Exponent out of range."):
            raise Unavailable(message)
        raise ArithmeticError(message)
    try:
        return ctypes.string_at(answer).decode("ascii").splitlines()
    finally:
        lib.atlas_groebner_release(answer)


def ideal_request(header, rows):
    """Send sparse rational polynomial rows to one whole-ideal operation."""
    library = _kernel()
    if library is None:
        raise Unavailable("The native ideal kernel is unavailable.")
    payload = (header + "\n" + "".join(row + "\n" for row in rows)).encode("ascii")
    answer = library.atlas_ideal(payload, len(payload))
    if not answer:
        raise Unavailable(library.atlas_groebner_error().decode("utf-8"))
    try:
        return ctypes.string_at(answer).decode("ascii").splitlines()
    finally:
        library.atlas_groebner_release(answer)


def field_request(name, lines):
    """Call a QQbar operation, returning None for binaries predating that operation."""
    library = _kernel()
    function = getattr(library, name, None)
    if function is None:
        return None
    function.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    function.restype = ctypes.c_void_p
    text = ("\n".join(lines) + "\n").encode("ascii")
    answer = function(text, len(text))
    if not answer:
        raise ArithmeticError(library.atlas_groebner_error().decode("utf-8"))
    try:
        return ctypes.string_at(answer).decode("ascii").strip()
    finally:
        library.atlas_groebner_release(answer)
