"""Provenance, isolation and portability of reusable bases over the public API."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import shlex
import subprocess
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine.algebra import calculate_algebra
from engine.algebra_basis_store import AlgebraBasisStore
from engine.app import AtlasServer
from tests.test_cli_session import ROOT, run_script
from tests.test_http import QuietHandler


RING = {"field": "QQ", "variables": ["x", "y"], "order": "lex"}


@contextmanager
def endpoint(**limits):
    with AtlasServer(("127.0.0.1", 0), QuietHandler) as server:
        server.algebra_bases = AlgebraBasisStore(**limits)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/api/algebra"
        finally:
            server.shutdown()
            thread.join()


def post(url, operation, operands, ring=RING, **extra):
    request = {"operation": operation, "operands": operands, "ring": ring, "compute": "cpu", **extra}
    req = Request(url, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
    try:
        response = urlopen(req, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


class HttpBasisTests(unittest.TestCase):
    def test_reuse_in_two_workers_with_portable_replays(self):
        with endpoint() as url:
            status, record = post(url, "groebner", {"ideal": ["x^2-1", "y-x"]})
            self.assertEqual(status, 200, record)
            ref = record["basis_ref"]
            self.assertEqual(record["result"]["generators"], ["x-y", "y^2-1"])
            with patch("engine.algebra_basis_store.calculate_algebra", wraps=calculate_algebra) as worker:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    reductions = [pool.submit(post, url, op, {"basis_ref": ref, "polynomial": p})
                                  for op, p in [("normal-form", "x^2+y/2"), ("membership", "y^2-1")]]
                    (a, remainder), (b, member) = [future.result() for future in reductions]
                self.assertEqual(worker.call_count, 2)
                for call in worker.call_args_list:
                    self.assertIs(call.kwargs["known_basis"], True)
                    self.assertEqual(call.args[0]["operands"]["ideal"], record["result"]["generators"])
            self.assertEqual((a, b), (200, 200))
            self.assertEqual(remainder["result"]["polynomial"], "1/2*y+1")
            self.assertIs(member["result"]["value"], True)
        replay = remainder["replay"]
        self.assertNotIn(ref, str(replay))
        shell = subprocess.run(shlex.split(replay["command"]), cwd=ROOT, capture_output=True,
                               text=True, timeout=30)
        script = run_script(replay["script"])
        for process in (shell, script):
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["result"], remainder["result"])

    def test_reject_cross_ring_forgery_and_expiration(self):
        with endpoint(max_entries=1) as url:
            _, record = post(url, "groebner", {"ideal": ["x"]})
            ref = record["basis_ref"]
            args = {"basis_ref": ref, "polynomial": "x"}
            for changed in ({"field": "RR"}, {"order": "grevlex"}, {"variables": ["y", "x"]}):
                status, error = post(url, "membership", args, RING | changed)
                self.assertEqual((status, error["error"]["code"]), (400, "basis_ring_mismatch"))
            for op, operands in [("normal-form", {**args, "ideal": ["1"]}),
                                 ("quotient", {"basis_ref": ref, "other_ideal": ["x"]}),
                                 ("membership", {**args, "basis_ref": 1})]:
                self.assertEqual(post(url, op, operands)[0], 400)
            self.assertEqual(post(url, "normal-form", args, known_basis=True)[0], 400)
            with endpoint() as other:
                status, error = post(other, "membership", args)
                self.assertEqual((status, error["error"]["code"]), (410, "stale_basis_ref"))
            post(url, "groebner", {"ideal": ["y"]})
            status, error = post(url, "membership", args)
            self.assertEqual((status, error["error"]["code"]), (410, "stale_basis_ref"))

    def test_oversized_and_zero_bases(self):
        with endpoint(max_bytes=1) as url:
            status, basis = post(url, "groebner", {"ideal": ["x"]})
            self.assertEqual(status, 200)
            self.assertNotIn("basis_ref", basis)
            self.assertEqual(basis["result"]["generators"], ["x"])
        with endpoint() as url:
            ring = {"variables": ["x"]}
            _, basis = post(url, "groebner", {"ideal": []}, ring)
            status, member = post(url, "membership", {"basis_ref": basis["basis_ref"], "polynomial": "1"}, ring)
            self.assertEqual(status, 200, member)
            self.assertIs(member["result"]["value"], False)
