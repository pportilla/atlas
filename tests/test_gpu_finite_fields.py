"""Check extension-field arithmetic against independent polynomial arithmetic and public replay."""
import json
import random
import shlex
import subprocess
import threading
import unittest
from urllib.request import Request, urlopen

from engine.algebra import dispatch
from engine.app import AtlasServer
from engine.gpu_devices import triton_device
from engine.modular_batch import modular_arithmetic
from engine.polynomial import InputError
from tests.cli_support import ROOT, run_prompt, run_script
from tests.http_support import QuietHandler


CUBIC = [1, 1, 0, 1]
AES = [1, 1, 0, 1, 1, 0, 0, 0, 1]
ACTIONS = ("reduce", "add", "subtract", "multiply", "divide", "inverse")


def reduced(a, p, relation):
    a = list(a) if isinstance(a, (list, tuple)) else [a]
    coefficients = [v % p for v in a]
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()
    while len(coefficients) >= len(relation):
        lead = coefficients.pop()
        shift = len(coefficients) - len(relation) + 1
        for j, value in enumerate(relation[:-1]):
            coefficients[shift+j] = (coefficients[shift+j]-lead*value) % p
        while coefficients and coefficients[-1] == 0:
            coefficients.pop()
    return coefficients


def product(a, b, p, relation):
    work = [0] * (len(a) + len(b))
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            work[i+j] += first*second
    return reduced(work, p, relation)


def oracle(action, a, b, p, relation):
    a, b = reduced(a, p, relation), reduced(b, p, relation)
    if action == "reduce":
        return a
    if action in ("add", "subtract"):
        size = max(len(a), len(b))
        a += [0] * (size - len(a))
        b += [0] * (size - len(b))
        sign = 1 if action == "add" else -1
        return reduced([x+sign*y for x, y in zip(a, b)], p, relation)
    if action == "multiply":
        return product(a, b, p, relation)
    base = a if action == "inverse" else b
    assert base
    power = p**(len(relation)-1) - 2
    ans = [1]
    while power:
        if power & 1:
            ans = product(ans, base, p, relation)
        base = product(base, base, p, relation)
        power //= 2
    return ans if action == "inverse" else product(a, ans, p, relation)


def request(action, values, p, poly, *, other=None, compute="cpu"):
    args = dict(action=action, modulus=p, field_polynomial=poly, values=values)
    if other is not None:
        args["other"] = other
    return dict(operation="modular-arithmetic", ring={"variables": []}, operands=args, compute=compute)


class FiniteFieldArithmeticTests(unittest.TestCase):
    def check_matrix(self, mode):
        rng = random.Random(783)
        for p, relation in ((2, [1, 1]), (5, CUBIC), (2, AES), (2147483647, [1, 0, 1])):
            degree = len(relation)-1
            a = [[], 2**700+3, [-2**601-1, p-1][:2*degree-1]]
            b = [1, -1, [1]]
            for _ in range(62):
                a.append([rng.randrange(-p, 2*p) for _ in range(2*degree-1)])
                row = [rng.randrange(-p, 2*p) for _ in range(degree)]
                b.append(row if reduced(row, p, relation) else 1)
            for action in ACTIONS:
                with self.subTest(mode=mode, p=p, degree=degree, action=action):
                    values = b if action == "inverse" else a
                    rhs = b if action not in ("inverse", "reduce") else None
                    record = dispatch(request(action, values, p, relation, other=rhs, compute=mode))
                    want = [oracle(action, left, right, p, relation) for left, right in zip(values, b)]
                    self.assertEqual([[int(c) for c in row] for row in record["result"]["values"]], want)
                    self.assertEqual(record["execution"]["backend"], "cpu" if mode == "cpu" else "triton",
                                     record["execution"])

    def test_cpu(self):
        self.check_matrix("cpu")

    def test_physical_triton(self):
        if triton_device() is None:
            self.skipTest("Requires a physical Triton device")
        self.check_matrix("gpu")

    def test_characteristic_two_device_degrees(self):
        if triton_device() is None:
            self.skipTest("Requires a physical Triton device")
        for degree, mask in enumerate((0x3, 0x7, 0xb, 0x13, 0x25, 0x43, 0x83, 0x11b), 1):
            with self.subTest(degree=degree):
                poly = [(mask >> i) & 1 for i in range(degree+1)]
                samples = [[(value >> i) & 1 for i in range(degree)] for value in range(1, 1 << degree)]
                for action in ("multiply", "inverse"):
                    rhs = list(reversed(samples)) if action == "multiply" else None
                    record = dispatch(request(action, samples, 2, poly, other=rhs, compute="gpu"))
                    self.assertEqual(record["execution"]["backend"], "triton")
                    want = [oracle(action, a, b, 2, poly) for a, b in zip(samples, rhs or samples)]
                    self.assertEqual(record["result"]["values"], [list(map(str, row)) for row in want])

    def test_typed_values_and_field(self):
        batch = modular_arithmetic("reduce", [[], -2, [0, 0, 0, 1]], 5, field_polynomial=[-4, 1, 5, 1])
        self.assertEqual(batch.values, ((), (3,), (4, 4)))
        self.assertEqual(batch.modulus, 5)
        self.assertEqual(batch.field_polynomial, tuple(CUBIC))
        empty = modular_arithmetic("inverse", [], 5, field_polynomial=CUBIC)
        self.assertEqual(empty.values, ())

    def test_zero_and_malformed_fields(self):
        for mode in ("cpu", "gpu"):
            bad = [request("inverse", [[5], [1]], 5, CUBIC),
                   request("divide", [[]], 5, CUBIC, other=[CUBIC]),
                   request("add", [[1]], 5, CUBIC, other=[]),
                   request("reduce", [[1, "1/2"]], 5, CUBIC),
                   request("reduce", [1], 5, "x^3+x+1")]
            bad.extend(request("reduce", [], p, CUBIC) for p in (-1, 0, 1, 4, 561))
            bad.extend(request("reduce", [], 5, poly) for poly in ([], [1], [1, 2], [-1, 0, 1], [1, 2, 1]))
            for req in bad:
                with self.subTest(req=req, mode=mode):
                    with self.assertRaises(InputError):
                        dispatch(dict(req, compute=mode))

    def test_unsupported_device_shapes_preserve_exact_values(self):
        p = 2**127 - 1
        for action, prime, poly, row in (("inverse", p, [1, 0, 1], [p-1, 2]),
                                         ("multiply", 2, [1, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1]*10),
                                         ("inverse", 2147483647, [-5, 0, 0, 1], [2, 3]),
                                         ("reduce", 5, CUBIC, [1, 0, 0, 0, 0, 1])):
            with self.subTest(prime=prime, action=action):
                record = dispatch(request(action, [row], prime, poly, compute="gpu",
                                          other=[row] if action == "multiply" else None))
                self.assertEqual(record["execution"]["backend"], "cpu")
                self.assertTrue(record["execution"]["fallback_reasons"])
                self.assertEqual(record["result"]["values"], [list(map(str, oracle(action, row, row, prime, poly)))])

    def test_prompt_http_and_saved_replay(self):
        req = request("divide", [[], [1, 2, -5], -3], 5, CUBIC, other=[[1, 1], 1, [-7, 2, 1]])
        server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        try:
            post = Request(f"http://127.0.0.1:{server.server_port}/api/algebra", data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
            with urlopen(post, timeout=30) as received:
                record = json.load(received)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(record["result"], dispatch(req)["result"])
        replay = subprocess.run(shlex.split(record["replay"]["command"]), cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        for proc in (replay, run_script(record["replay"]["script"])):
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["result"], record["result"])
        status, output, err = run_prompt('v=modular_arithmetic("reduce", [[0], [0,0,0,1]],5,'
                                         'field_polynomial=[1,1,0,1],compute="cpu")\nprint(v.values)\nquit\n')
        self.assertEqual(status, 0, err)
        self.assertIn("((), (4, 4))", output)
