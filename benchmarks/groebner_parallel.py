"""Measure full, certified Katsura solves with CPU, GPU and automatic dispatch."""
import argparse
import ctypes
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("trace", "replay", "lift", "verify")


def worker(root, policy, backend, device_index, size):
    # Each worker imports the Python bindings and loads the libraries from its
    # assigned build. The workload generator is shared across all builds and modes.
    sys.path.insert(0, str(root))
    from engine.native_library import _kernel, replay_scope
    from groebner_hard import katsura, payload

    lib = _kernel()
    if lib is None:
        raise RuntimeError(f"No built Gröbner kernel in {root}")
    lib.atlas_groebner_route.argtypes = [ctypes.c_void_p]
    lib.atlas_groebner_route.restype = ctypes.c_int
    lib.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double)]
    lib.atlas_groebner_modular.restype = ctypes.c_void_p
    print("ready", flush=True)
    for line in sys.stdin:
        operation = line.strip()
        if operation not in ("solve", "profile"):
            raise ValueError(f"Unknown benchmark operation {operation!r}")
        counts, seconds = (ctypes.c_long * 7)(), (ctypes.c_double * 4)()
        start = perf_counter_ns()
        with replay_scope(policy, backend=backend, index=device_index) as scope:
            names, polynomials = katsura(size)
            data = payload(polynomials, len(names), "grevlex")
            if operation == "solve":
                handle = lib.atlas_groebner(data, len(data))
            else:
                handle = lib.atlas_groebner_modular(data, len(data), counts, seconds)
            if not handle:
                raise RuntimeError(lib.atlas_groebner_error().decode())
            try:
                if operation == "solve" and lib.atlas_groebner_route(handle) != 1:
                    raise RuntimeError(f"Katsura-{size} did not take the certified modular route")
                basis = lib.atlas_groebner_text(handle)
            finally:
                lib.atlas_groebner_free(handle)
            receipt = scope.report()
        elapsed = (perf_counter_ns() - start) / 1e6
        sample = {"solve_ms": elapsed, "execution": receipt, "basis": basis.decode("ascii")}
        if operation == "profile":
            sample["phase_ms"] = {key: value * 1e3 for key, value in zip(PHASES, seconds)}
            sample["counts"] = dict(zip(("primes", "rejected", "traces", "verifications",
                                        "trace_words", "basis_size", "replay_callback_used"), counts))
        print(json.dumps(sample, separators=(",", ":")), flush=True)


def fingerprint(root):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    binaries = [root / "native/build" / ("lib" + name + suffix)
                for name in ("groebner", "groebner_opencl", "resolution_opencl")]
    sources = sorted((root / "engine").glob("*.py")) + sorted((root / "native").glob("*.cpp"))
    for pattern in ("*.hpp", "*.inc", "*.cl"):
        sources += sorted((root / "native").glob(pattern))
    sources.append(root / "native/groebner.mk")
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"source_tree_sha256": digest.hexdigest(),
            "libraries_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries if p.is_file()}}


def hardware(root):
    sys.path.insert(0, str(root))
    from engine.gpu_devices import available_devices
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"cpu": cpu, "platform": platform.platform(), "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "devices": available_devices()}


def measure(args, roots, cases, size):
    labels = [f"{version}_{policy}" for version, policy in cases]
    commands = [[sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(roots[version]),
                 policy, args.backend, str(args.index), str(size)] for version, policy in cases]
    env = dict(os.environ, ATLAS_GROEBNER="native", RESOLUTION_GPU_BACKEND=args.backend,
               RESOLUTION_GPU_INDEX=str(args.index), PYTHONDONTWRITEBYTECODE="1")
    if args.calibration:
        env["ATLAS_TRACE_PROFILE"] = str(args.calibration.resolve())
    report = {"generators": size + 1, "variables": size + 1,
              "samples": {label: [] for label in labels}, "summary": {}}
    reference = None

    def validate(raw, label):
        nonlocal reference
        sample = json.loads(raw)
        basis = sample.pop("basis").encode("ascii")
        if not basis:
            raise RuntimeError(f"{label}: empty reduced basis")
        if reference is None:
            reference = basis
            report["basis"] = {"sha256": hashlib.sha256(basis).hexdigest(), "bytes": len(basis),
                               "polynomials": len(basis.splitlines())}
        if basis != reference:
            raise RuntimeError(f"{label}: reduced basis differs from the baseline CPU result")
        receipt = sample["execution"] or {}
        policy = label.split("_")[-1]
        backend = args.backend if policy == "gpu" else "cpu"
        replay_backends = receipt.get("backends", {})
        trace_backends = receipt.get("trace", {}).get("backends", {})
        if policy == "cpu" and ((set(replay_backends) | set(trace_backends)) - {"cpu"}):
            raise RuntimeError(f"{label}: a CPU-only solve used another backend: {receipt}")
        replay = replay_backends.get(backend, {})
        trace = trace_backends.get(backend, {})
        if policy == "auto":
            if not any(b.get("batches", 0) for group in (trace_backends, replay_backends) for b in group.values()):
                raise RuntimeError(f"{label}: automatic dispatch did not report any execution: {receipt}")
        elif not (replay.get("batches", 0) or trace.get("batches", 0)):
            raise RuntimeError(f"{label}: requested {backend} execution was not recorded: {receipt}")
        if label == "candidate_gpu" and args.require_gpu_trace and not trace.get("pairs", 0):
            raise RuntimeError(f"{label}: no trace pairs executed on {backend}: {receipt}")
        if "counts" in sample and not sample["counts"]["verifications"]:
            raise RuntimeError(f"{label}: no exact verification recorded")
        return sample

    with ExitStack() as stack:
        sessions = []
        for cmd in commands:
            session = stack.enter_context(subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, text=True, env=env))
            stack.callback(session.kill)
            if session.stdout.readline().strip() != "ready":
                raise RuntimeError(f"Worker startup failed: {cmd}")
            sessions.append(session)

        def request(i, operation):
            session = sessions[i]
            session.stdin.write(operation + "\n")
            session.stdin.flush()
            return validate(session.stdout.readline(), labels[i])

        for i in range(len(labels)):
            request(i, "solve")
        for repeat in range(args.repeats):
            order = list(range(len(labels)))
            if (repeat // len(order)) % 2:
                order.reverse()
            offset = repeat % len(order)
            for i in order[offset:] + order[:offset]:
                warm = request(i, "solve")
                start = perf_counter_ns()
                p = subprocess.run(commands[i], input="solve\n", capture_output=True, text=True,
                                   env=env, timeout=300)
                fresh_ms = (perf_counter_ns() - start) / 1e6
                if p.returncode:
                    raise RuntimeError(f"{labels[i]} fresh solve failed:\n{p.stderr}")
                ready, result = p.stdout.splitlines()
                if ready != "ready":
                    raise RuntimeError("Fresh worker did not initialize")
                fresh = validate(result, labels[i])
                profile = request(i, "profile")
                report["samples"][labels[i]].append({"round": repeat + 1, "warm_ms": warm["solve_ms"],
                    "fresh_ms": fresh_ms, "warm_execution": warm["execution"], "fresh_execution": fresh["execution"],
                    "phase_ms": profile["phase_ms"], "counts": profile["counts"]})
    for label, samples in report["samples"].items():
        summary = {key: median(s[key] for s in samples) for key in ("warm_ms", "fresh_ms")}
        summary["phase_ms"] = {name: median(s["phase_ms"][name] for s in samples) for name in PHASES}
        report["summary"][label] = summary
        print(f"Katsura-{size} {label}", json.dumps(summary), flush=True)
    for scope in ("warm_ms", "fresh_ms"):
        report.setdefault("speedup", {})[scope] = {
            "candidate_cpu_over_candidate_auto": report["summary"]["candidate_cpu"][scope] / report["summary"]["candidate_auto"][scope],
            "baseline_gpu_over_candidate_gpu": report["summary"]["baseline_gpu"][scope] / report["summary"]["candidate_gpu"][scope],
            "baseline_cpu_over_candidate_gpu": report["summary"]["baseline_cpu"][scope] / report["summary"]["candidate_gpu"][scope],
            "candidate_cpu_over_candidate_gpu": report["summary"]["candidate_cpu"][scope] / report["summary"]["candidate_gpu"][scope],
            "baseline_cpu_over_candidate_cpu": report["summary"]["baseline_cpu"][scope] / report["summary"]["candidate_cpu"][scope]
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Built checkout before the optimization")
    parser.add_argument("--candidate", type=Path, default=ROOT, help="Built checkout after the optimization")
    parser.add_argument("--backend", choices=("opencl", "triton"), default="opencl")
    parser.add_argument("--index", type=int, default=int(os.getenv("RESOLUTION_GPU_INDEX", "0")))
    parser.add_argument("--sizes", type=int, nargs="+", choices=(6, 7, 8), default=[7],
                        help="Katsura parameters, each gives n+1 generators in n+1 variables")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--require-gpu-trace", action="store_true",
                        help="Reject a candidate GPU solve unless a trace batch completed on the device")
    parser.add_argument("--calibration", type=Path, help="Machine-specific profile for automatic trace dispatch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.index < 0 or len(set(args.sizes)) != len(args.sizes):
        parser.error("Use distinct sizes, positive repetitions and a nonnegative GPU index")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    cases = [(version, policy) for version in roots for policy in ("cpu", "gpu")]
    cases.append(("candidate", "auto"))
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": f"Katsura {args.sizes} over Q, grevlex, known basis prefix 0",
        "backend": args.backend, "device_index": args.index, "hardware": hardware(roots["candidate"]),
        "builds": {version: fingerprint(root) for version, root in roots.items()},
        "repetitions": args.repeats, "require_gpu_trace": args.require_gpu_trace,
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).with_name("groebner_hard.py").read_bytes()).hexdigest(),
        "timing": {
            "warm": "One untimed full solve per persistent worker; wall time inside the worker from "
                    "compute selection and input construction through native route selection, certification, "
                    "complete reduced basis text, freeing its handle and collecting the execution report.",
            "fresh": "Fresh interpreter from parent process launch to receipt of the full basis via stdout "
                     "and process exit. Filesystem and device compiler caches may be warm.",
            "phase_ms": "Separate diagnostic runs call atlas_groebner_modular directly; the native phase "
                        "times do not include work performed before the default route switches to modular.",
            "order": "Each round rotates the starting case; each complete rotation reverses direction. "
                     "Solves run one at a time. OS activity is uncontrolled."
        }
    }
    report["problems"] = {}
    for size in args.sizes:
        report["problems"][f"Katsura-{size}"] = measure(args, roots, cases, size)
    if len(args.sizes) == 1:
        report.update(report["problems"].popitem()[1])
        del report["problems"]
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
