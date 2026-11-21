"""Time complete exact rational matrix results and paired CRT through public Atlas APIs."""
import argparse
from fractions import Fraction
from pathlib import Path
import os
import random
from statistics import median
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.algebra import dispatch
from engine.gpu_devices import triton_device
from engine.matrix_algebra import matrix_analysis


class Singular:
    def __init__(self):
        self.process = subprocess.Popen(["Singular", "-q", "--no-rc"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.run('ring R=0,(x,y),lp; option(redSB); LIB "matrix.lib";')
        self.run("""proc solveM(matrix A,matrix b) {
            module I=std(module(A)); module K=std(syz(module(A)));
            module W=std(module(transpose(A)));
            int consistent=size(reduce(module(b),I))==0;
            matrix X[ncols(A)][1]; if(consistent){X=lift(module(A),module(b));}
            string d="None"; if(nrows(A)==ncols(A)){d=string(det(A));}
            return(list(I,K,W,X,consistent,d)); }""")

    def run(self, script):
        p = self.process
        p.stdin.write(script + '\nprint("ATLASEND");\n')
        p.stdin.flush()
        lines = []
        for line in p.stdout:
            if line.strip() == "ATLASEND":
                break
            if line.strip() and not line.lstrip().startswith("//"):
                lines.append(line.strip())
        else:
            raise RuntimeError("Singular terminated")
        if any(line.startswith("?") for line in lines):
            raise RuntimeError("\n".join(lines))
        return lines

    def close(self):
        self.process.stdin.write("quit;\n")
        self.process.stdin.close()
        self.process.wait()


def declaration(name, rows, width):
    values = ",".join(str(n) for row in rows for n in row)
    return f"matrix {name}[{len(rows)}][{width}]" + ("="+values if values else "") + ";"


def verify_spaces(solver, result, rows, rhs):
    """Compare subspaces in both directions; the two solvers may choose different bases."""
    def columns(basis, height):
        return list(zip(*basis)) if basis else [[0] for _ in range(height)]
    m,n = len(rows),len(rows[0])
    declarations = [declaration("AI",columns(result.image,m),max(1,len(result.image))),
                    declaration("AK",columns(result.kernel,n),max(1,len(result.kernel))),
                    declaration("AE",result.rref,n)]
    if result.solution.particular is not None:
        declarations.append(declaration("AX",[[v] for v in result.solution.particular],1))
        declarations.append('print((A*AX==B) && (A*output[4]==B));')
    for obj, part in (("module(AI)",1),("module(AK)",2),("module(transpose(AE))",3)):
        declarations.append(f'print(size(reduce({obj},output[{part}]))+size(reduce(output[{part}],std({obj})))==0);')
    declarations.append('print(output[5]); print(output[6]); print(size(output[1]));')
    expected = (["1"] if result.solution.particular is not None else []) + ["1"]*3
    expected += [str(int(result.solution.status != "inconsistent")),str(result.determinant),str(result.rank)]
    actual = solver.run("\n".join(declarations))
    if actual != expected:
        raise AssertionError((actual,expected))


def timings(solvers, trials, check=None):
    out = {name: [] for name in solvers}
    for i in range(trials+1):
        names = list(solvers)
        start_at = i % len(names)
        for name in names[start_at:]+names[:start_at]:
            start = perf_counter()
            result = solvers[name]()
            if i:
                out[name].append((perf_counter()-start)*1000)
            if check:
                check(name,result)
    return out


def cases():
    random_source = random.Random(19463)
    yield "rational-square", [[Fraction(-1,3),2],[3,4]], [2,-4]
    yield "dependent-inconsistent", [[1,2,4],[2,4,8]], [3,5]
    a = [[random_source.randint(-4,4) for _ in range(6)] for _ in range(6)]
    yield "dense-6x6", a, [sum(row) for row in a]
    a = [[random_source.randint(-15,15) for _ in range(6)] for _ in range(80)]
    yield "tall-80x6", a, [sum(row) for row in a]
    a = [[random_source.randint(-5,5) for _ in range(14)] for _ in range(12)]
    yield "rectangular-12x14", a, [sum(row) for row in a]


def compare_rank_kernels(device, trials):
    import ctypes
    import numpy as np
    from engine import gpu_rank

    rng = np.random.default_rng(139)
    for rows, cols in ((2,3),(6,7),(16,7),(32,7),(64,7),(80,7),(128,7),(256,7)):
        data = rng.integers(99999,size=(rows,cols),dtype=np.int32)
        pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        elapsed = {"fused":[],"tiled":[]}
        for i in range(trials+1):
            for mode in (("tiled","fused") if i%2 else ("fused","tiled")):
                with patch.object(gpu_rank,"_single_program",lambda r,c: mode=="fused"):
                    start = perf_counter()
                    ids = gpu_rank.independent_rows(pointer,rows,cols,1000000007,device)
                    if i:
                        elapsed[mode].append((perf_counter()-start)*1000)
                    if len(ids) != min(rows,cols):
                        raise AssertionError("unexpected rank for deterministic full-rank witness")
        print(f"rank-{rows}x{cols}",{k:round(median(v),3) for k,v in elapsed.items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=5)
    parser.add_argument("--cpu-only",action="store_true")
    parser.add_argument("--rank-trials",type=int,default=30,help="extra trials of each isolated rank kernel")
    args = parser.parse_args()
    if args.trials < 1 or args.rank_trials < 0:
        parser.error("Use positive comparison trials and nonnegative rank trials.")
    gpu = not args.cpu_only
    os.environ["RESOLUTION_GPU_BACKEND"] = "triton"
    if gpu:
        device = triton_device(0)
        if not device:
            parser.error("Requires physical Triton device zero.")
        print(device,flush=True)
    for label,rows,rhs in cases():
        singular = Singular()
        try:
            singular.run(declaration("A",rows,len(rows[0]))+declaration("B",[[v] for v in rhs],1)+"list output;")
            cpu = matrix_analysis(rows,rhs=rhs,compute="cpu")
            def solve(compute):
                return matrix_analysis(rows,rhs=rhs,compute=compute)
            def check(name,result):
                if name != "singular" and (result.result != cpu.result or
                                           (name.startswith("gpu") and result.execution["backend"]!="triton")):
                    raise AssertionError(result.record)
            solvers = {"cpu":lambda: solve("cpu"), "singular":lambda: singular.run("output=solveM(A,B);")}
            if gpu:
                solvers["gpu"] = lambda: solve("gpu")
                if len(rows[0])+1 <= 8:
                    def tiled():
                        with patch("engine.gpu_rank._single_program",lambda r,c:False):
                            return solve("gpu")
                    solvers["gpu_tiled"] = tiled
            runs = timings(solvers,args.trials,check)
            verify_spaces(singular,cpu,rows,rhs)
            print(label, {k:round(median(v),3) for k,v in runs.items()},
                  {k:[round(x,3) for x in v] for k,v in runs.items()},flush=True)
        finally:
            singular.close()
    # Paired CRT is a stage of modular basis computation. Force coefficients
    # beyond one 62-bit pair and compare complete bases across modes and Singular.
    scale = 2**301+123
    request = {"operation":"groebner","ring":{"field":"QQ","variables":["x","y"],"order":"lex"},
               "operands":{"ideal":[f"x^2+{scale}*y-1","x*y-2"]}}
    expected = dispatch(dict(request,compute="cpu"))["result"]
    def check_basis(compute,record):
        if compute == "singular":
            return
        if record["result"] != expected:
            raise AssertionError("basis mismatch")
        if compute=="gpu" and not any(s["operation"]=="crt_pair_lift" and s["backend"]=="triton"
                                      for s in record["execution"]["stages"]):
            raise AssertionError("paired CRT did not run on the GPU")
    singular = Singular()
    try:
        singular.run(f"ideal I=x^2+{scale}*y-1,x*y-2; ideal G;")
        solvers = {"cpu":lambda: dispatch(dict(request,compute="cpu")), "singular":lambda: singular.run("G=std(I);")}
        if gpu:
            solvers["gpu"] = lambda: dispatch(dict(request,compute="gpu"))
        runs = timings(solvers,args.trials,check_basis)
        out = singular.run("ideal E="+",".join(expected["generators"])+"; print(size(reduce(G,std(E)))+size(reduce(E,G)));")
        if out != ["0"]:
            raise AssertionError(out)
        print("301-bit-CRT-ideal", {k:round(median(v),3) for k,v in runs.items()},
              {k:[round(x,3) for x in v] for k,v in runs.items()},flush=True)
    finally:
        singular.close()
    if gpu and args.rank_trials:
        compare_rank_kernels(device,args.rank_trials)


if __name__=="__main__":
    main()
