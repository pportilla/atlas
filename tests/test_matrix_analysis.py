"""Exact matrix API through the session, CLI, worker, and replay."""
from fractions import Fraction
import json
import os
import shlex
import subprocess
import unittest
from unittest.mock import patch

from engine.algebra import calculate_algebra
from engine.gpu_devices import triton_device
from engine.matrix_algebra import matrix_analysis
from engine.session import new_session
from tests.cli_support import ROOT, run_prompt


class MatrixAnalysisTests(unittest.TestCase):
    def test_solution_status_and_subspaces(self):
        a = [[0, -2, 1], [0, 4, -2]]
        m = matrix_analysis(a, rhs=["1/3", "-2/3"], compute="cpu")
        self.assertEqual((m.rank,m.determinant,m.pivot_columns), (1,None,(1,)))
        self.assertEqual(m.image, ((-2,4),))
        self.assertEqual(m.kernel, ((1,0,0),(0,Fraction(1,2),1)))
        self.assertEqual(m.solution, ("underdetermined",(0,Fraction(-1,6),0)))
        self.assertEqual(m.rref, ((0,1,Fraction(-1,2)),(0,0,0)))
        self.assertEqual(matrix_analysis(a, rhs=[1,1]).solution, ("inconsistent",None))
        unique = matrix_analysis([["-1/4",0], [1,3]], rhs=[2,1])
        self.assertEqual((unique.determinant,unique.solution), (Fraction(-3,4),("unique",(-8,3))))
        self.assertEqual(matrix_analysis([[1,2],[2,4]]).determinant, 0)
        self.assertEqual(matrix_analysis([]).determinant, 1)
        self.assertEqual(matrix_analysis([],columns=3,rhs=[]).kernel, ((1,0,0),(0,1,0),(0,0,1)))
        self.assertEqual(matrix_analysis([[],[]],rhs=[0,1]).solution, ("inconsistent",None))

    def test_large_rationals_bad_witness_and_failure_preserves_saved_value(self):
        wide = 2**15100+1
        matrix = [[wide,0],[wide,1000000007]]
        exact = matrix_analysis(matrix,rhs=[wide,wide+2*1000000007],compute="cpu")
        with patch("engine.tangent_linear._device_row_basis", return_value=[0]):
            alternative = matrix_analysis(matrix,rhs=[wide,wide+2*1000000007],compute="gpu")
        self.assertEqual(alternative.result,exact.result)
        self.assertEqual((exact.solution,exact.determinant), (("unique",(1,2)),wide*1000000007))
        self.assertTrue(alternative.execution["fallback_reasons"])
        with patch("engine.matrix_algebra._determinant", return_value=("123", {}, 0)):
            wrong = matrix_analysis([[1,2],[3,4]],compute="gpu")
        self.assertEqual(wrong.determinant,-2)
        self.assertTrue(any("determinant failed" in msg for msg in wrong.execution["fallback_reasons"]))
        session = new_session(None)
        list(session.run('M = matrix_analysis([[1]])'))
        for bad in ('[[1],[2,3]]','[[1.2]]','[["2/0"]]','[[True]]'):
            with self.assertRaises(ValueError):
                list(session.run(f'M = matrix_analysis({bad})'))
            self.assertEqual(session.variables["M"].rank,1)

    def test_replay_help_cli_and_worker(self):
        request = dict(operation="matrix-analysis",ring={"variables":[]},
                       operands={"rows":[["-1/3",2],[3,2]], "rhs":[2,3]},compute="cpu")
        status,record = calculate_algebra(request)
        self.assertEqual(status,200)
        from tests.http_support import endpoint, post
        with endpoint() as url:
            status,response = post(url,request["operation"],request["operands"],ring=request["ring"])
            self.assertEqual((status,response["result"]),(200,record["result"]))
        for args,source in ((shlex.split(record["replay"]["command"]),None),
                            (["./atlas","run","-"],record["replay"]["script"])):
            proc = subprocess.run(args,input=source,cwd=ROOT,text=True,capture_output=True,check=True)
            self.assertEqual(json.loads(proc.stdout)["result"], record["result"])
        session = new_session(None)
        list(session.run('M = matrix_analysis([[1,2]])'))
        from engine.session_help import algebra_value_help
        with patch("engine.matrix_algebra.handle",side_effect=AssertionError("help computed")):
            state = dict(session.variables["M"].__dict__)
            card = algebra_value_help("M",session.variables["M"],session.variables)
            self.assertIn("1 by 2 matrix over QQ",card)
            self.assertIn("examples/algebra-api.md#exact-rational-linear-algebra",card)
            self.assertEqual(state,session.variables["M"].__dict__)
            for expr in ("M.rank","M.kernel","M.solution"):
                self.assertIn(expr,card)
                list(session.run(expr))
        status,output,error = run_prompt('M=matrix_analysis([[1,2]],rhs=[1])\nhelp M\nsol=M.solution\n'
                                        'help sol\nprint(M.rank,sol.status)\nprint(sol.particular[0])\nquit\n')
        self.assertEqual((status,error),(0,""))
        self.assertIn("1 by 2 matrix over QQ",output)
        self.assertIn("sol.particular",output)
        self.assertIn("1 underdetermined",output)

    def test_physical_device_matches_cpu(self):
        if triton_device() is None:
            self.skipTest("physical Triton accelerator required")
        cases = [([[1,2],[2,4]],[3,6]), ([[1,2],[2,4]],[3,0]),
                 ([[1,0],[1,1000000007]],[0,1000000007]), ([[1,2],[0,3]],[1,2])]
        with patch.dict(os.environ,{"RESOLUTION_GPU_BACKEND":"triton"}):
            for rows,rhs in cases:
                cpu = matrix_analysis(rows,rhs=rhs,compute="cpu")
                gpu = matrix_analysis(rows,rhs=rhs,compute="gpu")
                self.assertEqual(gpu.result,cpu.result)
                self.assertTrue(any(stage["operation"]=="matrix_row_selection" and stage["backend"]=="triton"
                                    for stage in gpu.execution["stages"]))
                self.assertTrue(any(stage["operation"]=="matrix_determinant" and stage["host_verified"]
                                    for stage in gpu.execution["stages"]))
            # All supported direct determinant dimensions, including negative
            # determinants, rational row contents, and more than a warp of permutations.
            from random import Random
            random = Random(1492)
            for n in range(1,7):
                rows = [[Fraction((2**220+1) * random.randint(-3,3), 7) for _ in range(n)] for _ in range(n)]
                cpu = matrix_analysis(rows,compute="cpu")
                gpu = matrix_analysis(rows,compute="gpu")
                self.assertEqual(gpu.result,cpu.result)
                self.assertTrue(any(s["operation"]=="matrix_determinant" for s in gpu.execution["stages"]))
