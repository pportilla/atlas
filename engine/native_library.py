"""Load the exact kernel without importing a polynomial expression library."""
import ctypes
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import os
from pathlib import Path
import sys
from threading import Lock, get_ident

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
    if hasattr(library, "atlas_groebner_modular"):
        library.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                                   ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double)]
        library.atlas_groebner_modular.restype = ctypes.c_void_p
    library.atlas_groebner_route.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_route.restype = ctypes.c_int
    library.atlas_groebner_error.restype = ctypes.c_char_p
    library.atlas_groebner_text.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_text.restype = ctypes.c_char_p
    library.atlas_groebner_seconds.argtypes = [ctypes.c_void_p]
    library.atlas_groebner_seconds.restype = ctypes.c_double
    library.atlas_groebner_reduce.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    library.atlas_groebner_reduce.restype = ctypes.c_void_p
    if hasattr(library, "atlas_groebner_contains"):
        library.atlas_groebner_contains.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        library.atlas_groebner_contains.restype = ctypes.c_int
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
            if hasattr(library, "atlas_groebner_certification_counts"):
                library.atlas_groebner_certification_counts.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
                library.atlas_groebner_certification_counts.restype = None
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
# Runtime initialization is shared by solves in a process. Triton specializes
# the trace kernel on the prime, so a different prime still has a cold cost.
_warm_trace = set()


def current_replay_scope():
    # asyncio.to_thread and callers using copy_context can inherit Python
    # state. Native hooks and monotone counters belong to their OS thread.
    scope = _replay.get()
    return scope if scope is not None and scope.thread == get_ident() else None


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


@lru_cache(maxsize=1)
def _opencl_device_probe():
    path = _LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
    try:
        probe = ctypes.CDLL(str(path)).atlas_groebner_opencl_device_status
    except (OSError, AttributeError):
        return None
    probe.argtypes = [ctypes.c_size_t]
    probe.restype = ctypes.c_int
    return probe


def algebra_device_unavailable(backend, index):
    """Return True only if every algebra accelerator allowed by this policy is absent."""
    if backend in ("triton", "auto"):
        from .gpu_devices import triton_device
        if triton_device(index) is not None:
            return False
        if backend == "triton":
            return True
    if backend in ("opencl", "auto"):
        if _opencl_replay() is None and _opencl_trace() is None:
            return True
        probe = _opencl_device_probe()
        # Older algebra plugins and inconclusive probes may still execute.
        return probe is not None and probe(index) == 0
    return False  # Preserve an explicitly requested reference replay.


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


_HostCounts = namedtuple("_HostCounts", "batches primes layers pairs bases attempts lift_ns verify_ns")


def _cpu_counts(library):
    batches, primes = ctypes.c_uint64(), ctypes.c_uint64()
    library.atlas_groebner_cpu_replays(ctypes.byref(batches), ctypes.byref(primes))
    layers, pairs = ctypes.c_uint64(), ctypes.c_uint64()
    if hasattr(library, "atlas_groebner_cpu_trace_counts"):
        library.atlas_groebner_cpu_trace_counts(ctypes.byref(layers), ctypes.byref(pairs))
    proof = (ctypes.c_uint64 * 4)()
    if hasattr(library, "atlas_groebner_certification_counts"):
        library.atlas_groebner_certification_counts(proof)
    return _HostCounts(batches.value, primes.value, layers.value, pairs.value, *proof)


class _ReplayScope:
    def __init__(self, compute, backend, index):
        legacy = _LEGACY.get(os.getenv("ATLAS_GROEBNER_GPU"))
        if compute is None:
            if legacy:
                compute = "gpu"
            else:
                compute = "auto" if os.getenv("RESOLUTION_GPU") == "auto" else "cpu"
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
        self.thread = get_ident()
        self.counts = {}
        self.trace_counts = {}
        self.reasons = []
        self.device = None
        self.devices = {}
        self.certification = dict(bases=0, attempts=0, lift_ms=0.0, verify_ms=0.0)
        self.library = None
        self.start = None
        self.unavailable = set()
        self.trace_unavailable = set()
        self.trace_profiles = {}
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
            start = self.start
            now = _cpu_counts(self.library)
            self.count("cpu", now.batches - start.batches, now.primes - start.primes)
            if now.layers > start.layers:
                self.count_trace("cpu", now.layers - start.layers, now.pairs - start.pairs)
            self.certification["bases"] += now.bases - start.bases
            self.certification["attempts"] += now.attempts - start.attempts
            self.certification["lift_ms"] += (now.lift_ns - start.lift_ns) / 1e6
            self.certification["verify_ms"] += (now.verify_ns - start.verify_ns) / 1e6
            self.start = now

    def record_device(self, backend, device=None):
        if backend not in self.devices:
            if backend == "opencl" and device is None:
                # The rank plugin supplies names for the same OpenCL GPU indices.
                # It is optional even when the algebra plugin is installed.
                from .native_backend import gpu_devices
                try:
                    device = next((d for d in gpu_devices() if d["index"] == self.index), None)
                except (OSError, AttributeError, ValueError):
                    pass
            self.devices[backend] = {"index": self.index, **(device or {}), "backend": backend}
        self.device = self.devices[backend]
        return self.device

    def report(self):
        if self.interrupted is not None:
            raise self.interrupted
        self.collect_cpu()
        if not self.counts and not self.trace_counts and not self.reasons and not any(self.certification.values()):
            return None
        backend = next(iter(self.counts), None) if len(self.counts) <= 1 else "mixed"
        data = {"backend": backend,
                "batches": sum(item["batches"] for item in self.counts.values()),
                "primes": sum(item["primes"] for item in self.counts.values()),
                "backends": {key: dict(count) for key, count in self.counts.items()}}
        if self.device is not None:
            data["device"] = dict(self.device)
            data["devices"] = {key: dict(device) for key, device in self.devices.items()}
        if any(self.certification.values()):
            data["certification"] = dict(self.certification)
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
            self.devices.update(report.get("devices", {}))
            for key, value in report.get("certification", {}).items():
                self.certification[key] += value
            for reason in report.get("fallback_reasons", []):
                if reason not in self.reasons:
                    self.reasons.append(reason)

    def count_trace(self, backend, batches, pairs):
        count = self.trace_counts.setdefault(backend, {"batches": 0, "pairs": 0})
        count["batches"] += batches
        count["pairs"] += pairs

    def screen(self, args):
        if self.compute == "auto":
            from .trace_policy import fastest_backend
            prime, jobs, width = args[:3]
            shape = (prime, jobs, width, args[4][width])
            backend = fastest_backend(self.trace_profiles, shape, _warm_trace, self.trace_unavailable)
            if backend is None:
                return 1
            choices = (backend,)
        else:
            choices = ("triton", "opencl") if self.backend == "auto" else (self.backend,)
        for backend in choices:
            if backend in self.trace_unavailable:
                continue
            try:
                if backend == "opencl":
                    function = _opencl_trace()
                    if function is None or function(*args, self.index):
                        raise RuntimeError("OpenCL trace kernel or device unavailable")
                    device = None
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
            self.record_device(backend, device)
            _warm_trace.add((backend, self.index, args[0] if backend == "triton" else None))
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
                    device = None
                else:
                    from . import gpu_replay
                    device = None
                    if backend == "triton":
                        from .gpu_devices import triton_device
                        device = triton_device(self.index)
                        if device is None:
                            raise RuntimeError("Triton device unavailable")
                    if gpu_replay.run_batch(args, device):
                        raise RuntimeError("Modular replay failed")
            except Exception as exc:
                self.unavailable.add(backend)
                self.reasons.append(f"{backend}: {exc}")
                continue
            self.count(backend, 1, args[5])
            if backend != "reference":
                self.record_device(backend, device)
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
        if scope is not None and scope.interrupted is not None:
            return 1
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
            if trace_swap and scope.compute == "auto" and scope.backend != "reference":
                from .trace_policy import load_profiles
                scope.trace_profiles = load_profiles(scope.backend, scope.index)
            enable_trace = (scope.compute == "gpu" or scope.trace_profiles) and scope.backend != "reference"
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
