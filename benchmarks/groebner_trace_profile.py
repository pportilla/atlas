"""Profile real modular Gröbner trace packets and calibrate dispatch on this host."""
import argparse
from contextlib import contextmanager
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median, quantiles
import subprocess
import sys
from time import perf_counter_ns

from groebner_parallel import ROOT, fingerprint, hardware


U32 = C.c_uint32
PTR = C.POINTER(U32)
SCREEN = C.CFUNCTYPE(C.c_int, U32, U32, U32, *([PTR] * 6))
PROFILE = C.CFUNCTYPE(None, *([U32] * 7), *([C.c_double] * 3))
OPENCL_PARTS = ("upload_alloc_ms", "kernel_wall_ms", "download_wall_ms", "kernel_event_ms", "download_event_ms")


@contextmanager
def callback(swap, function):
    swap.argtypes, swap.restype = [C.POINTER(C.c_void_p)], None
    old = C.cast(function, C.c_void_p)
    swap(C.byref(old))
    try:
        yield
    finally:
        swap(C.byref(old))


def solve(lib, source):
    mod = lib.atlas_groebner_modular
    mod.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(C.c_long), C.POINTER(C.c_double)]
    mod.restype = C.c_void_p
    lib.atlas_groebner_text.argtypes, lib.atlas_groebner_text.restype = [C.c_void_p], C.c_char_p
    lib.atlas_groebner_error.restype = C.c_char_p
    lib.atlas_groebner_free.argtypes, lib.atlas_groebner_free.restype = [C.c_void_p], None
    counts = (C.c_long * 7)()
    handle = mod(source, len(source), counts, None)
    if not handle:
        raise RuntimeError(lib.atlas_groebner_error().decode())
    try:
        if counts[3] < 1:
            raise RuntimeError("Missing rational verification")
        return lib.atlas_groebner_text(handle)
    finally:
        lib.atlas_groebner_free(handle)


def capture(lib, source):
    cpu = lib.atlas_groebner_trace_rows_cpu
    cpu.argtypes, cpu.restype = SCREEN._argtypes_, C.c_int
    packets, errors = [], []

    @SCREEN
    def capture_rows(p, jobs, width, initial, offsets, pivots, columns, coefficients, out):
        try:
            edges = offsets[width]
            sizes = (jobs * width, width + 1, width, edges, edges)
            arrays = []
            digest = hashlib.sha256()
            for ptr, length in zip((initial, offsets, pivots, columns, coefficients), sizes):
                data = C.string_at(ptr, 4 * length)
                digest.update(data)
                arrays.append((U32 * length).from_buffer_copy(data))
            if cpu(p, jobs, width, *arrays, out):
                raise RuntimeError("CPU row reduction failed")
            shape = {"prime": p, "jobs": jobs, "width": width, "edges": edges, "pivots": sum(arrays[2]),
                     "initial_nonzeros": sum(bool(value) for value in arrays[0]), "inputs_sha256": digest.hexdigest()}
            packets.append((shape, arrays, C.string_at(out, 4 * jobs * width)))
        except BaseException as exc:
            errors.append(exc)
            return 1
        return 0

    with callback(lib.atlas_groebner_swap_trace_screen, capture_rows):
        basis = solve(lib, source)
    if errors:
        raise errors[0]
    if not packets:
        raise RuntimeError("The selected problem produced no trace packets")
    return basis, packets


def gpu_dispatch(policy, args):
    # Use the normal requested-GPU policy to include Python dispatch and all transfers.
    if policy.screen(args):
        raise RuntimeError(f"GPU trace failed: {policy.report()}")
    return 0


def timed_rows(run, shape, arrays, expected):
    out = (U32 * (shape["width"] * shape["jobs"]))()
    args = (shape["prime"], shape["jobs"], shape["width"], *arrays, out)
    start = perf_counter_ns()
    status = run(args)
    elapsed_ms = (perf_counter_ns() - start) / 1e6
    if status or bytes(out) != expected:
        raise RuntimeError(f"Row reduction failed or yielded different coefficients: {shape}")
    return elapsed_ms


def cold_dispatch(root, backend, index, size):
    from engine.native_library import _ReplayScope, _kernel
    from groebner_hard import katsura, payload
    names, polys = katsura(size)
    _, packets = capture(_kernel(), payload(polys, len(names), "grevlex"))

    # Older comparison trees predate trace_policy. Their requested-GPU path
    # discovers the device inside screen(), without an upfront probe.
    def run(call):
        if (root / "engine/trace_policy.py").exists():
            from engine.trace_policy import _selected_device
            if _selected_device(backend, index) is None:
                raise RuntimeError("GPU discovery failed")
        return gpu_dispatch(_ReplayScope("gpu", backend, index), call)

    return timed_rows(run, *packets[0])


def host_stages(lib, source, expected, packets):
    stages = []

    @PROFILE
    def record(degree, prime, jobs, width, edges, rank, device, setup, rows, echelon):
        stages.append({"degree": degree, "prime": prime, "jobs": jobs, "width": width, "edges": edges,
                       "rank": rank, "on_device": bool(device), "setup_ms": 1000 * setup,
                       "cpu_reduction_ms": 1000 * rows, "echelon_ms": 1000 * echelon})

    with callback(lib.atlas_groebner_swap_trace_profile, record):
        if solve(lib, source) != expected:
            raise RuntimeError("The instrumented solver returned a different certified basis")
    if len(stages) != len(packets):
        raise RuntimeError("Instrumented solver produced different trace batches")
    for stage, (shape, _, _) in zip(stages, packets):
        if stage["on_device"] or any(stage[key] != shape[key] for key in ("prime", "jobs", "width", "edges")):
            raise RuntimeError("Instrumented CPU solve changed the reducer shape or backend")
    return stages


def opencl_split(path, index):
    lib = C.CDLL(str(path))
    execute = lib.atlas_trace_screen_opencl_index
    execute.argtypes, execute.restype = [*SCREEN._argtypes_, C.c_size_t], C.c_int
    last = lib.atlas_trace_profile_last
    last.argtypes, last.restype = [C.POINTER(C.c_double)], None

    def sample(shape, arrays, expected):
        elapsed = timed_rows(lambda args: execute(*args, index), shape, arrays, expected)
        parts = (C.c_double * len(OPENCL_PARTS))()
        last(parts)
        return dict(total_ms=elapsed, **dict(zip(OPENCL_PARTS, parts)))
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Built production checkout")
    parser.add_argument("--native-profile", type=Path, help="Matched solver with the optional trace profile hook")
    parser.add_argument("--opencl-profile", type=Path, help="Matched OpenCL plugin built with ATLAS_OPENCL_PROFILE")
    parser.add_argument("--backend", choices=("triton", "opencl"), default="opencl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sizes", nargs="+", type=int, choices=(6, 7, 8), default=[6, 7, 8])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--write-auto-profile", type=Path, help="Write calibration using this build's dispatch format")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 3 or args.index < 0 or len(args.sizes) != len(set(args.sizes)):
        parser.error("Use distinct sizes, at least three repeats and a nonnegative device index.")
    if args.opencl_profile and args.backend != "opencl":
        parser.error("--opencl-profile requires --backend opencl")
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from engine.native_library import _ReplayScope, _kernel
    from groebner_hard import katsura, payload

    lib = _kernel()
    if not lib:
        raise RuntimeError(f"Missing native kernel in {root}")
    diagnostics = C.CDLL(str(args.native_profile.resolve())) if args.native_profile else lib
    split = opencl_split(args.opencl_profile.resolve(), args.index) if args.opencl_profile else None

    if args.cold_probe:
        print(cold_dispatch(root, args.backend, args.index, args.sizes[0]))
        return

    if not hasattr(diagnostics, "atlas_groebner_swap_trace_profile"):
        parser.error("The optional native timing hook is missing. Specify a matched --native-profile library.")
    devices = hardware(root)
    device = next((d for d in devices["devices"] if d["backend"] == args.backend and d["index"] == args.index), None)
    if device is None:
        raise RuntimeError("The selected physical GPU could not be enumerated")
    cold_command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--backend", args.backend,
                    "--index", str(args.index), "--sizes", str(args.sizes[0]), "--cold-probe"]
    cold = [float(subprocess.check_output(cold_command, text=True)) for _ in range(3)]
    policy = _ReplayScope("gpu", args.backend, args.index)
    cpu = lib.atlas_groebner_trace_rows_cpu
    report = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "hardware": devices, "build": fingerprint(root), "device": device,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "profiling_libraries_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in (("core", args.native_profile), ("opencl", args.opencl_profile)) if path},
              "repeats": args.repeats, "cold_first_packet_ms": cold,
              "method": "Each problem uses its own prepared input arrays, remainder and certified basis from the "
                        "production solver. An untimed CPU/GPU warmup precedes alternating timed calls. CPU/GPU "
                        "wall times include normal row reduction to a host output buffer. Native stage costs come "
                        "from separate CPU solves. OpenCL event and synchronizing wall times are diagnostic.",
              "problems": {}}
    calibration = []
    first_warm = None
    for size in args.sizes:
        names, polys = katsura(size)
        source = payload(polys, len(names), "grevlex")
        basis, packets = capture(lib, source)
        batches = [dict(shape, samples=[]) for shape, _, _ in packets]
        cpu_run = lambda args: cpu(*args)
        gpu_run = lambda args: gpu_dispatch(policy, args)
        for packet in packets:
            timed_rows(cpu_run, *packet)
            timed_rows(gpu_run, *packet)
            if split:
                split(*packet)
        for i in range(args.repeats):
            stages = host_stages(diagnostics, source, basis, packets)
            for packet, batch, stage in zip(packets, batches, stages):
                measurements = {}
                order = (("cpu_ms", cpu_run), ("gpu_ms", gpu_run))
                for name, run in (reversed(order) if i % 2 else order):
                    measurements[name] = timed_rows(run, *packet)
                measurements["host"] = {key: stage[key] for key in ("setup_ms", "cpu_reduction_ms", "echelon_ms")}
                batch.update(degree=stage["degree"], rank=stage["rank"])
                if split:
                    measurements["opencl_profile"] = split(*packet)
                batch["samples"].append(measurements)
        for batch in batches:
            samples = batch["samples"]
            cpu_ms = [sample["cpu_ms"] for sample in samples]
            # Auto competes with a native CPU call that avoids Python entirely.
            native_ms = [min(sample["cpu_ms"], sample["host"]["cpu_reduction_ms"]) for sample in samples]
            gpu_ms = [sample["gpu_ms"] for sample in samples]
            record = {key: batch[key] for key in ("prime", "jobs", "width", "edges")}
            record.update(samples=len(samples), cpu_ms=median(native_ms), gpu_ms=median(gpu_ms),
                          cpu_low_ms=quantiles(native_ms, n=4, method="inclusive")[0],
                          gpu_high_ms=quantiles(gpu_ms, n=4, method="inclusive")[2])
            calibration.append(record)
            batch["median_ms"] = {"cpu": median(cpu_ms), "gpu": record["gpu_ms"],
                                  **{key: median(s["host"][key] for s in samples) for key in samples[0]["host"]}}
            if split:
                batch["opencl_median_ms"] = {key: median(s["opencl_profile"][key] for s in samples)
                                             for key in samples[0]["opencl_profile"]}
        if first_warm is None:
            first_warm = batches[0]["median_ms"]["gpu"]
        report["problems"][f"Katsura-{size}"] = {"basis_sha256": hashlib.sha256(basis).hexdigest(), "batches": batches}
        print(f"Katsura-{size}: {len(batches)} exact batch comparisons per round", flush=True)
    cold_start = max(0.0, median(cold) - first_warm)
    report["cold_start_ms"] = cold_start
    if args.write_auto_profile:
        from engine.trace_policy import write_profile
        write_profile(args.write_auto_profile, device, calibration, cold_start)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
