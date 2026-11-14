"""Machine-local evidence for automatic Gröbner trace dispatch."""
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile


def default_path(backend, index):
    cache = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "resolution-atlas" / f"groebner-trace-{backend}-{index}.json"


def _affinity():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


@lru_cache(maxsize=2)
def _host(backend):
    # Measure the actual installed reducers. A checkout may have newer source
    # than its loaded library, or use different compiler optimization flags.
    root = Path(__file__).resolve().parents[1]
    from .native_library import _LIBRARY
    digest = sha256()
    device_file = (_LIBRARY.with_name(_LIBRARY.name.replace("libgroebner", "libgroebner_opencl"))
                   if backend == "opencl" else root / "engine/gpu_trace.py")
    for path in (_LIBRARY, device_file):
        digest.update(path.read_bytes())
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    return dict(cpu=cpu, architecture=platform.machine(), system=platform.system(), kernels=digest.hexdigest())


def _device_identity(device):
    keys = ("backend", "index", "name", "vendor", "version", "driver", "runtime", "architecture")
    result = {key: device[key] for key in keys if key in device}
    if device["backend"] == "triton":
        import triton
        import torch
        runtime_version = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
        if device.get("runtime") == "xpu":
            runtime_version = getattr(torch.version, "xpu", None)
        result.update(triton=triton.__version__, torch=torch.__version__,
                      runtime_version=runtime_version)
    return result


def _selected_device(backend, index):
    if backend == "opencl":
        from .native_backend import gpu_devices
        return next((dict(device, backend=backend) for device in gpu_devices() if device["index"] == index), None)
    from .gpu_devices import triton_device
    return triton_device(index)


def _duration(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _shape(row):
    fields = ("prime", "jobs", "width", "edges")
    key = tuple(row[field] for field in fields)
    if any(type(n) is not int for n in key) or min(key[:3]) < 1 or key[3] < 0:
        raise ValueError("Invalid trace shape")
    return key


def _savings(row):
    if type(row["samples"]) is not int or row["samples"] < 3:
        return 0
    if not all(_duration(row[k]) for k in ("cpu_ms", "gpu_ms", "cpu_low_ms", "gpu_high_ms")):
        raise ValueError("Invalid trace timings")
    # Require both a material median improvement and separation of the middle
    # halves of the measurements. Charge startup against the conservative gap.
    if row["gpu_ms"] >= 0.90 * row["cpu_ms"]:
        return 0
    return max(0, row["cpu_low_ms"] - row["gpu_high_ms"])


def write_profile(path, device, records, cold_start_ms):
    """Save measured row shapes from the profiler. Readers see a complete file."""
    if cold_start_ms is not None and not _duration(cold_start_ms):
        raise ValueError("Invalid trace startup time")
    records = list(records)
    for row in records:
        _shape(row)
        _savings(row)
    data = dict(version=2, host=_host(device["backend"]), threads=_affinity(), device=_device_identity(device),
                cold_start_ms=cold_start_ms, rows=records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as tmp:
        try:
            json.dump(data, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


@lru_cache(maxsize=16)
def _read(path, file_signature, backend, index, threads):
    # The stat signature in the cache key detects a replaced calibration.
    data = json.loads(path.read_bytes())
    if data["version"] != 2 or data["threads"] != threads:
        return None
    device = data["device"]
    if (device["backend"], device["index"]) != (backend, index) or data["host"] != _host(backend):
        return None
    cold = data["cold_start_ms"]
    if cold is not None and not _duration(cold):
        return None
    savings = {}
    for row in data["rows"]:
        key, gain = _shape(row), _savings(row)
        # Repeated observations for one shape must all support its use.
        savings[key] = min(savings.get(key, gain), gain)
    savings = {key: gain for key, gain in savings.items() if gain > 0}
    return (device, cold, savings) if savings else None


def load_profiles(preference, index):
    profiles = {}
    override = os.getenv("ATLAS_TRACE_PROFILE")
    choices = ("triton", "opencl") if preference == "auto" else (preference,)
    for backend in choices:
        path = Path(override) if override else default_path(backend, index)
        try:
            stat = path.stat()
            if stat.st_size > (4 << 20):
                continue
            profile = _read(path, (stat.st_ino, stat.st_mtime_ns, stat.st_size), backend, index, _affinity())
            if profile:
                profiles[backend] = profile
        except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
            continue
    return profiles


def fastest_backend(profiles, shape, warmed, unavailable):
    candidates = []
    for backend, (device, cold, savings) in profiles.items():
        gain = savings.get(shape, 0)
        if backend in unavailable or gain <= 0:
            continue
        warm = (backend, device["index"], shape[0] if backend == "triton" else None) in warmed
        if not warm and (cold is None or gain <= cold):
            continue
        candidates.append((gain if warm else gain - cold, backend, device))
    for _, backend, device in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            found = _selected_device(backend, device["index"])
            if found and _device_identity(found) == device:
                return backend
        except Exception:
            pass
        unavailable.add(backend)
    return None
