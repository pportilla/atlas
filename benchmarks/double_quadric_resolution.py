"""Time certified local embedded resolutions on the same double-quadric germs."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
Q = "(x*y-z*t+u^2)^2"
CASES = {
    "ordinary_double_point": "x^2+y^2+z^2+t^2+u^2",
    "smooth_quintic_section": Q + "+x^5+2*y^5+3*z^5+4*t^5+5*u^5",
    "nodal_quintic_section": Q + "+x^5+y^5+z^5+t^5+u^5+x^6+2*y^6+3*z^6+5*t^6+7*u^6",
}


def measure(solver, equation, timeout):
    if solver == "Atlas":
        cmd = [str(ROOT / "atlas"), "resolve", equation, "--vars", "x,y,z,t,u", "--mode", "embedded",
               "--exhaustive", "--compute", "cpu", "--json"]
        script = ""
    else:
        cmd = ["Singular", "-q", "--no-rc"]
        script = ('LIB "resolve.lib"; ring r=0,(x,y,z,t,u),dp; ideal I=' + equation + ';\n'
                  'list atlasBenchCharts=resolve(I,1,"L","A");\n'
                  'print("CHARTS "+string(size(atlasBenchCharts[1]))); quit;\n')
    start = monotonic()
    with subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, start_new_session=True) as child:
        try:
            out, _ = child.communicate(script, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            if isinstance(exc, KeyboardInterrupt):
                raise
            return {"status": "timeout", "seconds": round(monotonic() - start, 3)}
    seconds = round(monotonic() - start, 3)
    if child.returncode:
        return {"status": "error", "seconds": seconds, "diagnostic": out[-1000:]}
    if solver == "Atlas":
        r = json.loads(out)
        if r["status"] != "resolved" or r["certificate"].get("atlas_verification") != "Embedded normal crossings on the owned origin fibre":
            raise ValueError("Atlas did not certify embedded normal crossings")
        charts = len(r["atlas"])
    else:
        lines = out.splitlines()
        counts = [line for line in lines if line.startswith("CHARTS ")]
        if "result is o.k." not in out or not counts:
            error = next((line.strip() for line in lines if line.lstrip().startswith("?")), out[-400:])
            return {"status": "error", "seconds": seconds, "diagnostic": error}
        charts = int(counts[-1].split()[1])
    if equation == CASES["ordinary_double_point"] and charts != 5:
        raise ValueError("The ordinary double point should have five charts.")
    return {"status": "certified", "seconds": seconds, "charts": charts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.trials < 1 or args.timeout <= 0 or not shutil.which("Singular"):
        p.error("Supply positive limits and install Singular with resolve.lib.")
    checkout = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    report = {
        "cpu": platform.machine(), "platform": platform.platform(), "timeout_seconds": args.timeout,
        "solvers": {"Atlas": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "Singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0]},
        "atlas_working_tree_modified": bool(checkout),
        "atlas_kernel_sha256": {file.name: sha256(file.read_bytes()).hexdigest()
                                for file in sorted((ROOT / "native" / "build").glob("lib*"))
                                if file.stem in ("libgroebner", "libresolution")
                                and file.suffix in (".so", ".dylib")},
        "scope": "Fresh process, QQ, CPU, embedded resolution at the origin, includes chart verification and output; no zeta computation.",
        "cases": CASES, "samples": [],
    }
    info = Path("/proc/cpuinfo")
    if info.is_file():
        report["cpu"] = next((s.split(":", 1)[1].strip() for s in info.read_text().splitlines()
                              if s.startswith("model name")), report["cpu"])
    for trial in range(1, args.trials + 1):
        for case, equation in CASES.items():
            for solver in (("Atlas", "Singular") if trial % 2 else ("Singular", "Atlas")):
                row = dict(trial=trial, case=case, solver=solver, **measure(solver, equation, args.timeout))
                report["samples"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(row, flush=True)


if __name__ == "__main__":
    main()
