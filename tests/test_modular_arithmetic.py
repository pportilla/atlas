"""Cross-check residue kernels, reconstruction and Hensel lifting."""
from fractions import Fraction
from math import gcd, isqrt
import json
import shlex
import subprocess
import unittest
from unittest.mock import patch

from engine.algebra import calculate_algebra
from engine.gpu_devices import triton_device
from engine.modular_batch import modular_arithmetic
from engine.session import new_session
from engine.session_help import algebra_value_help
from tests.cli_support import ROOT, run_prompt, run_script


class ModularArithmetic(unittest.TestCase):
    def test_lifts_and_reconstruction(self):
        for m in [3, 7, 11, 16, 23, 37, 53]:
            bound = isqrt((m-1)//2)
            fractions = {a*pow(b,-1,m)%m: Fraction(a,b) for b in range(1,bound+1) if gcd(b,m)==1
                         for a in range(-bound,bound+1) if gcd(a,b)==1}
            for mode in ("cpu", "gpu"):
                value = modular_arithmetic("reconstruct",list(range(m)),m,compute=mode)
                self.assertEqual(value.values, tuple(fractions.get(a) for a in range(m)))
        for mode in ("gpu","cpu"):
            roots = modular_arithmetic("hensel",[-3,3],7,other=[-2,0,1],compute=mode)
            self.assertEqual((roots.modulus, roots.values),(49,(39,10)))
            more = modular_arithmetic("hensel",roots.values,49,other=[-2,0,1],compute=mode)
            for root, reduced in zip(more.values,roots.values):
                self.assertEqual((root%49,root*root%2401),(reduced,2))
            lift = modular_arithmetic("crt",[1,5,0],6,other=[7,3,-1],other_modulus=35,compute=mode)
            self.assertEqual(lift.modulus,210)
            for v,a,b in zip(lift.values,[1,5,0],[7,3,-1]):
                self.assertEqual((v%6,v%35),(a,b%35))

    def test_physical_word_boundaries(self):
        if not triton_device():
            self.skipTest("Physical Triton device required")
        p = 2**31-1
        vals = [-(1<<63),-5,0,p,p+1,(1<<63)-1]*31
        other = [7,11,1,9,2,3]*31
        for action, options, reference in [
            ("reduce",{},[a%p for a in vals]),
            ("add",{"other":other},[(a+b)%p for a,b in zip(vals,other)]),
            ("subtract",{"other":other},[(a-b)%p for a,b in zip(vals,other)]),
            ("multiply",{"other":other},[(a*b)%p for a,b in zip(vals,other)]),
            ("divide",{"other":other},[(a*pow(b,-1,p))%p for a,b in zip(vals,other)])]:
            cpu = modular_arithmetic(action,vals,p,compute="cpu",**options)
            gpu = modular_arithmetic(action,vals,p,compute="gpu",**options)
            self.assertEqual(gpu.execution["backend"],"triton",gpu.execution)
            self.assertEqual(gpu.values,cpu.values)
            self.assertEqual(list(cpu.values),reference)
        for action, vals, options in [("inverse",other,{}),("reconstruct",vals,{}),
                                      ("crt",vals,{"other":other,"other_modulus":p-1}),
                                      ("hensel",[1,-1],{"other":[-1,0,1]})]:
            g = modular_arithmetic(action,vals,p,compute="gpu",**options)
            self.assertEqual(g.execution["backend"],"triton",g.execution)
            self.assertEqual(g.values,modular_arithmetic(action,vals,p,compute="cpu",**options).values)

    def test_wide_empty_and_invalid(self):
        m=2**16384+1
        result=modular_arithmetic("multiply",[m-2],m,other=[m-3],compute="gpu")
        self.assertEqual((result.values,result.modulus),((6,),m))
        self.assertEqual(result.execution["backend"],"cpu")
        self.assertEqual(modular_arithmetic("inverse",[],3).values,())
        for action, vals, m, opts in [("inverse",[0],7,{}),("inverse",[4],10,{}),
                                      ("crt",[2],6,{"other":[3],"other_modulus":9}),
                                      ("add",[1],3,{"other":[]}),("hensel",[0],7,{"other":[0,0,1]}),
                                      ("hensel",[1],7,{"other":[-2,0,1]}),("reduce",[1],1,{})]:
            for mode in ("cpu","gpu"):
                with self.assertRaises(ValueError):
                    modular_arithmetic(action,vals,m,compute=mode,**opts)

    def test_interfaces_and_saved_help(self):
        request={"operation":"modular-arithmetic","ring":{"variables":[]},"compute":"cpu",
                 "operands":{"action":"hensel","values":[3,4],"modulus":7,"other":[-2,0,1]}}
        status, record=calculate_algebra(request)
        self.assertEqual(status,200,record)
        script=run_script(record["replay"]["script"])
        shell=subprocess.run(shlex.split(record["replay"]["command"]),cwd=ROOT,capture_output=True,text=True,check=True)
        for run in (script,shell):
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout)["result"],record["result"])
        session=new_session(None)
        source='v=modular_arithmetic("crt",[2],3,other=[3],other_modulus=5)'
        list(session.run(source))
        before=dict(session.variables)
        with patch("engine.modular_batch.handle",side_effect=AssertionError("help performed a calculation")):
            card=algebra_value_help("v",before["v"],session.variables)
            self.assertEqual(before,session.variables)
            self.assertIn("examples/polynomial-ideals.md#compute-with-residues-and-lift-roots",card)
            for line in card.splitlines():
                if line.strip().startswith("v."):
                    list(session.run(line.strip()))
        code,text,errors=run_prompt(source+"\nhelp v\nquit\n")
        self.assertEqual((code,errors),(0,""))
        self.assertIn("v.modulus",text)
