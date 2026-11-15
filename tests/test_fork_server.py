"""Workers forked from a preloaded engine behave like `python -m` workers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from engine import fork_server

ROOT = Path(__file__).resolve().parents[1]
WORKER = [sys.executable, "-m", "engine.worker"]


def _ready(server, seconds=30):
    deadline = time.monotonic() + seconds
    while not server.is_ready():
        if server.broken or time.monotonic() > deadline:
            return False
        time.sleep(.02)
    return True


def _gone(pid, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(.02)
    return False


@unittest.skipUnless(fork_server.supported(), "The fork server needs Linux pidfds and descriptor passing.")
class ForkServerTests(unittest.TestCase):
    def setUp(self):
        self.server = fork_server._Connection.start()
        self.assertIsNotNone(self.server)
        self.addCleanup(self._stop_server)
        self.assertTrue(_ready(self.server))
        patches = (patch.object(fork_server, "_enabled", True), patch.object(fork_server, "_root", self.server),
                   patch.object(fork_server, "_inherited", None))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def _stop_server(self):
        self.server.close()
        self.server.process.wait(timeout=10)

    def popen(self, args=WORKER, **options):
        options.setdefault("cwd", ROOT)
        return fork_server.popen(args, **options)

    def test_forked_resolution_worker_matches_the_interpreter(self):
        request = json.dumps({"polynomial": "(x+y)^2+z^3+x^3", "variables": list("xyz"), "mode": "embedded",
                              "monodromy": True})

        def record(process):
            with process:
                out, _ = process.communicate(request, timeout=120)
            self.assertEqual(process.returncode, 0)
            status, data = json.loads(out)
            for key in ("elapsed_ms", "execution"):
                data.pop(key, None)
            return status, data

        forked = self.popen(stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
        self.assertIsInstance(forked, fork_server.ForkedProcess)
        interpreted = subprocess.Popen(WORKER, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                       start_new_session=True)
        self.assertEqual(record(forked), record(interpreted))

    def test_exit_status_streams_and_missing_modules(self):
        done = fork_server.run([sys.executable, "-m", "engine.no_such_module"], capture_output=True, text=True,
                               cwd=ROOT, timeout=30)
        self.assertEqual(done.returncode, 1)
        self.assertIn("No module named engine.no_such_module", done.stderr)
        done = fork_server.run(WORKER, input="{", capture_output=True, text=True, cwd=ROOT, timeout=30)
        self.assertEqual(done.returncode, 0)
        self.assertEqual(json.loads(done.stdout)[0], 500)

    def test_startup_settings_and_other_programs_use_the_interpreter(self):
        env = dict(os.environ, PYTHONHASHSEED="17")
        process = self.popen(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        self.assertIsInstance(process, subprocess.Popen)
        process.wait()
        process = fork_server.popen([sys.executable, "-c", "pass"])
        self.assertIsInstance(process, subprocess.Popen)
        process.wait()

    def test_requests_do_not_wait_for_a_preloading_server(self):
        warming = fork_server._Connection.start()
        try:
            with patch.object(fork_server, "_root", warming):
                start = time.monotonic()
                process = self.popen([sys.executable, "-m", "engine.no_such_module"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                self.assertLess(time.monotonic() - start, .1)
                self.assertIsInstance(process, subprocess.Popen)
                self.assertEqual(process.wait(), 1)
        finally:
            warming.close()
            warming.process.kill()
            warming.process.wait()

    def test_a_server_that_never_started_is_not_restarted(self):
        import socket
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        theirs.close()
        dead = fork_server._Connection(ours, fork_server._startup_env(os.environ))
        starts = []
        missing = [sys.executable, "-m", "engine.no_such_module"]
        with patch.object(fork_server, "_root", dead), \
                patch.object(fork_server._Connection, "start", side_effect=lambda: starts.append(1)):
            for _ in range(2):
                process = self.popen(missing, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.assertIsInstance(process, subprocess.Popen)
                self.assertEqual(process.wait(), 1)
        self.assertEqual(starts, [])
        ours.close()

    def test_session_kill_and_concurrent_waits(self):
        # The worker waits for its request on the open pipe.
        process = self.popen(stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
        self.assertEqual(os.getpgid(process.pid), process.pid)
        codes = []
        waiter = threading.Thread(target=lambda: codes.append(process.wait()))
        waiter.start()
        with self.assertRaises(subprocess.TimeoutExpired):
            process.wait(timeout=.2)
        fork_server.kill_session(process)
        self.assertEqual(process.wait(timeout=10), -signal.SIGKILL)
        waiter.join(10)
        self.assertEqual(codes, [-signal.SIGKILL])
        process.stdin.close()
        process.stdout.close()

    def test_children_stop_with_their_requester(self):
        process = self.popen(stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
        self.server.close()
        self.assertTrue(_gone(process.pid))
        self.assertEqual(self.server.process.wait(timeout=10), 0)
        process.stdin.close()
        process.stdout.close()

    def test_race_lane_watches_its_coordinator_by_descriptor_number(self):
        # A lane reads the coordinator pipe named on its command line and stops
        # its session when the coordinator's end closes.
        read_fd, write_fd = os.pipe()
        try:
            process = self.popen([sys.executable, "-m", "engine.exhaustive_resolution", "newton", str(read_fd)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
                                 pass_fds=(read_fd,))
        finally:
            os.close(read_fd)
        self.assertIsInstance(process, fork_server.ForkedProcess)
        with self.assertRaises(subprocess.TimeoutExpired):
            process.wait(timeout=.3)
        os.close(write_fd)
        self.assertEqual(process.wait(timeout=10), -signal.SIGKILL)
        process.stdin.close()
        process.stdout.close()


if __name__ == "__main__":
    unittest.main()
