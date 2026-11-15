"""Helper programs run in a forked solver exactly as they run as programs."""
import json
import os
import subprocess
import time
import unittest
from unittest.mock import patch

from engine import native_process
from engine.native_process import BUILD


def program(name, request, **options):
    return subprocess.run([str(BUILD / name)], input=request, capture_output=True, text=True, **options)


@unittest.skipUnless(all(native_process.available(name) for name in native_process.HELPERS),
                     "Build the helper libraries with make.")
class ForkedHelperPrograms(unittest.TestCase):
    def assertSameRun(self, name, request, **options):
        forked = native_process.run(name, input=request, **options)
        expected = program(name, request, **{k: v for k, v in options.items() if k != "timeout"})
        self.assertEqual((forked.returncode, forked.stdout), (expected.returncode, expected.stdout))
        return forked

    def test_answers_and_exit_statuses_match_the_programs(self):
        self.assertSameRun("newton_fan", json.dumps({"exponents": [[2, 0, 0], [0, 3, 0], [0, 0, 5]], "exhaustive": False}))
        self.assertSameRun("newton_fan", json.dumps({"exponents": "invalid"}))
        self.assertSameRun("face_gb", "1 2\n1 2\n-1 0\n")
        self.assertSameRun("face_gb", "not a face\n")
        self.assertSameRun("graph_verify", "smooth_reduced_graph 3 1\n1 2 0 0;-1 0 2 1;-1 0 2 0;-2 1 1 0\n")
        self.assertSameRun("graph_verify", "unknown\n")
        roots = {"field_roots": [{"coefficients": ["-2", "0", "1"], "index": 1},
                                 {"coefficients": ["-3", "0", "1"], "index": 1}]}
        # The quick plane request bounds memory beyond the address space in use.
        for exhaustive in (False, True):
            answer = self.assertSameRun("plane", json.dumps(dict(roots, exhaustive=exhaustive)))
            self.assertEqual(json.loads(answer.stdout)["degree"], 4)
        curve = {"terms": [[0, 2, "1"], [3, 0, "-1"]], "variables": ["x", "y"], "mode": "embedded", "exhaustive": False}
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        self.assertSameRun("plane", json.dumps(curve), env=env)

    def test_timeouts_stop_and_reap_the_forked_helper(self):
        def sleeper(name, args, stdin, stdout, stderr, env):
            pid = os.fork()
            if pid == 0:
                try:
                    time.sleep(30)
                finally:
                    os._exit(0)
            return pid

        with patch.object(native_process, "_start", side_effect=sleeper), \
                patch.object(native_process, "_wait", wraps=native_process._wait) as wait:
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                native_process.run("face_gb", input="1 2 1 0 1 1\n", timeout=.1)
            self.assertLess(time.monotonic() - start, 5)
        pid = wait.call_args.args[0]
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_graph_proof_in_a_solver_with_replay_hooks(self):
        from engine.native_library import replay_scope
        from engine.polynomial import encode_sparse_row, parse
        equation = encode_sparse_row(parse("(x+y+z)^2-z*(x-y)^2", list("xyz")))
        request = f"smooth_reduced_graph 3 1\n{equation}\n"
        expected = program("graph_verify", request)
        self.assertEqual(expected.returncode, 0)
        self.assertTrue(expected.stdout)
        with replay_scope("auto"):
            forked = native_process.run("graph_verify", input=request)
        self.assertEqual((forked.returncode, forked.stdout), (expected.returncode, expected.stdout))

    def test_long_running_helper_uses_the_given_descriptors(self):
        import socket
        ours, theirs = socket.socketpair()
        with open(os.devnull, "wb") as devnull, theirs:
            child = native_process.start("face_gb", ["--serve"], stdin=theirs.fileno(), stdout=theirs.fileno(),
                                         stderr=devnull.fileno())
        try:
            ours.settimeout(10)
            ours.sendall(b"1 2 1 0 1 1\n")
            self.assertEqual(ours.recv(128), b"nondegenerate 0\n")
            self.assertIsNone(child.poll())
        finally:
            ours.close()
            child.kill()
            child.wait()
        self.assertIsNotNone(child.returncode)


if __name__ == "__main__":
    unittest.main()
