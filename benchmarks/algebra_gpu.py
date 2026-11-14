"""Compare complete Phase 1 CLI results on the CPU, GPU, and an unavailable device.

Run python3 benchmarks/algebra_gpu.py --backend opencl after make gpu.
Add --workload normal-form for the degree-120 reduction modulo (x^8, y^9).
Use --fallback-only on a CPU-only system. Each sample starts a fresh Atlas process.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from math import comb
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.groebner_hard import katsura
from engine.polynomial import format_poly, parse_expression

NF_TERMS = comb(122, 2)
NF_REMAINDER = {(i, j): Fraction(comb(120, i) * comb(120 - i, j), 7)
                for i in range(8) for j in range(9)}


def run_cli(arguments, backend, index):
    env = dict(os.environ, ATLAS_GROEBNER="native", RESOLUTION_GPU_BACKEND=backend,
               RESOLUTION_GPU_INDEX=str(index))
    start = perf_counter()
    proc = subprocess.run([sys.executable, str(ROOT / "atlas"), *arguments], env=env,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"Atlas exited {proc.returncode}")
    data = json.loads(proc.stdout)
    return data, (perf_counter() - start) * 1000


def device_stages_for(report, mode, backend, index):
    expected_policy = "cpu" if mode == "cpu" else "gpu"
    if report["requested"] != expected_policy:
        raise RuntimeError(f"{mode} request recorded the wrong policy: {report}")
    device_stages = [s for s in report["stages"] if s["backend"] in {"opencl", "triton"}]
    if mode == "gpu":
        if not device_stages or any(s["backend"] != backend or s["device"]["index"] != index for s in device_stages):
            raise RuntimeError(f"No confirmed work on the selected backend {backend}: {report}")
    elif device_stages or report["backend"] != "cpu":
        raise RuntimeError(f"{mode} should have used only the CPU: {report}")
    if mode == "fallback" and not report["fallback_reasons"]:
        raise RuntimeError(f"The unavailable device was not reported: {report}")
    return device_stages


def check_groebner(report, mode, device_stages, scaled, result):
    if result["type"] != "basis" or not result["generators"]:
        raise RuntimeError(f"{mode} did not return a nonempty basis")
    if device_stages:
        for stage in device_stages:
            if not any(stage.get(k, 0) > 0 for k in ("pairs", "primes")):
                raise RuntimeError(f"Expected nonzero modular work on the device: {stage}")
        operations = {s["operation"] for s in device_stages}
        required = {"groebner_trace", "groebner_replay"} if scaled else {"groebner_trace"}
        if not required.issubset(operations):
            raise RuntimeError(f"The selected GPU did not execute the expected trace/replay: {report}")
    else:
        for operation, counter in (("groebner_trace", "pairs"), ("groebner_replay", "primes")):
            if not any(s["operation"] == operation and s["backend"] == "cpu" and s[counter] > 0
                       for s in report["stages"]):
                raise RuntimeError(f"{mode} did not report nonzero {operation} work on the CPU: {report}")
    certification = [s for s in report["stages"] if s["operation"] == "groebner_certification"]
    if not any(s["backend"] == "cpu" and s["bases"] > 0 and s["attempts"] > 0 for s in certification):
        raise RuntimeError(f"{mode} did not certify a lifted basis on the host: {report}")


def check_normal_form(report, mode, device_stages, result):
    ring = {"field": "QQ", "variables": ["x", "y"], "order": "lex"}
    if result["type"] != "polynomial" or result["ring"] != ring:
        raise RuntimeError(f"{mode} returned the wrong type or ring")
    if parse_expression(result["polynomial"], ring["variables"], _reference=True) != NF_REMAINDER:
        raise RuntimeError(f"{mode} remainder differs from the multinomial coefficients")
    if device_stages:
        expected = {"operation": "monomial_normal_form", "host_verified": True, "terms": NF_TERMS,
                    "removed": NF_TERMS - len(NF_REMAINDER), "divisors": 2}
        if len(device_stages) != 1 or any(device_stages[0].get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"The GPU did not report the expected certified monomial reductions: {report}")
    elif not any(s["operation"] == "normal_form" and s["backend"] == "cpu" for s in report["stages"]):
        raise RuntimeError(f"{mode} did not report the CPU normal form: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("groebner", "normal-form"), default="groebner")
    parser.add_argument("--backend", choices=["opencl", "triton"], default="opencl")
    parser.add_argument("--index", type=int, default=0, help="Backend-local ordinal for the hardware run")
    parser.add_argument("--size", type=int, choices=[6, 7, 8], default=6, help="Katsura parameter")
    parser.add_argument("--scale-bits", type=int, choices=(0, 192), default=192,
                        help="Scale the last variable by 2^192 for GPU replay; use 0 for the original ideal")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fallback-only", action="store_true", help="Run just the CPU and unavailable-index cases")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 <= args.index < 2**31:
        parser.error("Use a positive repeat count and an index between 0 and 2^31-1")
    devices, _ = run_cli(["devices"], args.backend, args.index)
    selected = next((d for d in devices if d["backend"] == args.backend and d["index"] == args.index), None)
    if selected is None and not args.fallback_only:
        parser.exit(2, f"No physical {args.backend} device at index {args.index}. Use --fallback-only to test CPU fallback.\n")
    missing_index = max((d["index"] for d in devices if d["backend"] == args.backend), default=-1) + 1
    if args.workload == "groebner":
        names, polys = katsura(args.size)
        command = ["algebra", "groebner", "--vars", ",".join(names), "--order", "grevlex", "--json"]
        scale = 2**args.scale_bits
        for poly in polys:
            scaled = {powers: coefficient * scale**powers[-1] for powers, coefficient in poly.items()}
            command.extend(["--gen", format_poly(scaled, names)])
        workload = f"Katsura-{args.size} over QQ, grevlex, last variable scaled by 2^{args.scale_bits}"
    else:
        command = ["algebra", "normal-form", "--vars", "x,y", "--order", "lex", "--gen", "x^8",
                   "--gen", "y^9", "--poly", "(1+x+y)^120/7", "--json"]
        workload = "(1+x+y)^120/7 modulo (x^8,y^9) over QQ, lex"

    modes = ["cpu", "fallback"] if args.fallback_only else ["cpu", "gpu", "fallback"]
    samples = {mode: [] for mode in modes}
    reference = None
    for repeat in range(args.repeats):
        offset = repeat % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            index = missing_index if mode == "fallback" else args.index
            compute = "cpu" if mode == "cpu" else "gpu"
            result, elapsed = run_cli([*command, "--compute", compute], args.backend, index)
            report = result["execution"]
            device_stages = device_stages_for(report, mode, args.backend, index)
            algebra = result["result"]
            if result["status"] != "ok":
                raise RuntimeError(f"{mode} returned status {result['status']}")
            if args.workload == "groebner":
                check_groebner(report, mode, device_stages, args.scale_bits, algebra)
            else:
                check_normal_form(report, mode, device_stages, algebra)
            if reference is None:
                reference = algebra
            elif algebra != reference:
                raise RuntimeError(f"{mode} returned a different ring or exact result")
            samples[mode].append({"end_to_end_ms": round(elapsed, 3), "execution": report})
    print(json.dumps({
        "workload": workload,
        "device": selected,
        "unavailable_index": missing_index, "exactly_equal": True,
        **({"basis_size": len(reference["generators"])} if args.workload == "groebner" else
           {"input_terms": NF_TERMS, "kept": len(NF_REMAINDER), "equal_to_oracle": True}),
        "result_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "timing": "Fresh CLI process through exact host verification and complete JSON result decoding (milliseconds)",
        "median_ms": {m: median(s["end_to_end_ms"] for s in values) for m, values in samples.items()},
        "samples": samples,
    }, indent=2))


if __name__ == "__main__":
    main()
