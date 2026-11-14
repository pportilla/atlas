"""Time the Groebner kernel on classical hard ideals, where coefficients swell.

For each ideal: the kernel's default entry point (exact strategies and, when
they swell, the Hilbert-driven or the modular route), the modular route alone, and with
--singular Singular's std and modStd. With --gpu the replay of the modular
route is also timed on the OpenCL device against the host's threads.
Only the computation is timed.
"""
import argparse
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.groebner import _kernel  # noqa: E402


def katsura(n):
    names = [f"u{i}" for i in range(n + 1)]
    polys = []
    for m in range(n):
        terms = {}
        for i in range(-n, n + 1):
            if abs(m - i) <= n:
                e = [0] * (n + 1)
                e[abs(i)] += 1
                e[abs(m - i)] += 1
                terms[tuple(e)] = terms.get(tuple(e), 0) + 1
        e = [0] * (n + 1)
        e[m] = 1
        terms[tuple(e)] = terms.get(tuple(e), 0) - 1
        polys.append(terms)
    terms = {}
    for i in range(-n, n + 1):
        e = [0] * (n + 1)
        e[abs(i)] = 1
        terms[tuple(e)] = terms.get(tuple(e), 0) + 1
    terms[tuple([0] * (n + 1))] = -1
    polys.append(terms)
    return names, polys


def cyclic(n):
    names = [f"x{i}" for i in range(n)]
    polys = []
    for k in range(1, n):
        terms = {}
        for i in range(n):
            e = [0] * n
            for j in range(k):
                e[(i + j) % n] += 1
            terms[tuple(e)] = terms.get(tuple(e), 0) + 1
        polys.append(terms)
    polys.append({tuple([1] * n): 1, tuple([0] * n): -1})
    return names, polys


PROBLEMS = [("Katsura-5", katsura(5), "grevlex"), ("Katsura-6", katsura(6), "grevlex"), ("Katsura-7", katsura(7), "grevlex"),
            ("Cyclic-6", cyclic(6), "grevlex"), ("Cyclic-5", cyclic(5), "lex"), ("Katsura-4", katsura(4), "lex"),
            ("Katsura-5", katsura(5), "lex")]


def payload(polys, variables, order):
    lines = [";".join(f"{c} {' '.join(map(str, e))}" for e, c in p.items() if c) for p in polys]
    return (f"{variables} {1 if order == 'lex' else 0} 0\n" + "".join(line + "\n" for line in lines)).encode()


def singular(names, polys, order, limit):
    def text(p):
        return "+".join("*".join([f"({c})"] + [f"{v}^{k}" for v, k in zip(names, e) if k]) for e, c in p.items() if c)
    script = ('system("--ticks-per-sec",1000000);\nLIB "modstd.lib";\n'
              f'ring r=0,({",".join(names)}),{"lp" if order == "lex" else "dp"};\nideal i={",".join(map(text, polys))};\n'
              'int a=rtimer; ideal g=std(i); print("std "+string(rtimer-a));\n'
              'a=rtimer; ideal h=modStd(i); print("modStd "+string(rtimer-a));\nquit;\n')
    times = {}
    try:
        output = subprocess.run(["Singular", "-q"], input=script, capture_output=True, text=True, timeout=limit).stdout
    except subprocess.TimeoutExpired as expired:
        output = (expired.stdout or b"").decode() if isinstance(expired.stdout, bytes) else (expired.stdout or "")
    for row in output.splitlines():
        name, _, micros = row.partition(" ")
        if name in ("std", "modStd") and micros.isdigit():
            times[name] = int(micros) / 1e6
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--singular", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--limit", type=float, default=120, help="seconds allowed to Singular per ideal")
    arguments = parser.parse_args()
    library = _kernel()
    if library is None:
        raise SystemExit("The kernel is not built. Run make.")
    library.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double)]
    library.atlas_groebner_modular.restype = ctypes.c_void_p
    library.atlas_groebner_route.argtypes = [ctypes.c_void_p]
    use_singular = arguments.singular and shutil.which("Singular")
    device = None
    if arguments.gpu:
        plugin = ROOT / "native" / "build" / ("libgroebner_opencl.dylib" if sys.platform == "darwin" else "libgroebner_opencl.so")
        if plugin.is_file():
            device = ctypes.CDLL(str(plugin))
            if not device.atlas_replay_opencl_available():
                device = None
        if device is None:
            raise SystemExit("No OpenCL device for the replay. Run make gpu.")
        library.atlas_groebner_use_batch.argtypes = [ctypes.c_void_p]
        library.atlas_groebner_replay_compare.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
                                                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_long)]
        library.atlas_groebner_replay_compare.restype = ctypes.c_long
    print("times in ms; 'route' is what the default entry point ended up using")
    for name, (names, polys), order in PROBLEMS:
        data = payload(polys, len(names), order)
        best = {"default": None, "modular": None}
        texts = {}
        counts = (ctypes.c_long * 7)()
        for _ in range(arguments.repeats):
            for kind in best:
                if kind == "default":
                    handle = library.atlas_groebner(data, len(data))
                    route = ("exact", "modular", "Hilbert")[library.atlas_groebner_route(handle)]
                else:
                    handle = library.atlas_groebner_modular(data, len(data), counts, None)
                if not handle:
                    raise SystemExit(f"The kernel failed on {name}.")
                elapsed = library.atlas_groebner_seconds(handle)
                best[kind] = elapsed if best[kind] is None else min(best[kind], elapsed)
                texts[kind] = library.atlas_groebner_text(handle)
                library.atlas_groebner_free(handle)
        if texts["default"] != texts["modular"]:
            raise SystemExit(f"The two routes disagree on {name}.")
        row = (f"{name:10s} {order:8s} default {best['default']*1e3:9.1f} ({route:7s})  modular {best['modular']*1e3:9.1f}"
               f" ({counts[0]:3d} primes)")
        if use_singular:
            times = singular(names, polys, order, arguments.limit)
            for kind in ("std", "modStd"):
                row += f"  Singular {kind} " + (f"{times[kind]*1e3:9.1f}" if kind in times else f"> {arguments.limit:.0f} s")
        print(row, flush=True)
        if device is not None:
            library.atlas_groebner_use_batch(ctypes.cast(device.atlas_replay_opencl, ctypes.c_void_p))
            for count in (64, 512, 2048):
                seconds, sizes = (ctypes.c_double * 3)(), (ctypes.c_long * 3)()
                differ = library.atlas_groebner_replay_compare(data, len(data), count, seconds, sizes)
                if differ:
                    raise SystemExit(f"The device and the host disagree on {name}.")
                print(f"{'':19s} replay for {count:4d} primes: host {seconds[1]*1e3:8.1f}  device {seconds[2]*1e3:8.1f}")
            library.atlas_groebner_use_batch(None)


if __name__ == "__main__":
    main()
