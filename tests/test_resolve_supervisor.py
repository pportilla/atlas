import json
import os
from pathlib import Path
import signal
import shutil
import sys
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "native" / "build" / "resolve_supervisor"


@unittest.skipUnless(SUPERVISOR.is_file(), "Requires the native Linux resolve supervisor")
class ResolveSupervisor(unittest.TestCase):
    def test_deadline_interrupt_and_worker_exit_reap_detached_children(self):
        self.exercise_worker(False)

    def test_borrowed_worker_matches_deadlines_signals_and_process_cleanup(self):
        from engine.preloaded_cli import _context
        try:
            _context(Path("/proc/self"))
        except (OSError, ValueError) as exc:
            self.skipTest(f"Cannot share a preloaded interpreter in this execution context: {exc}")
        self.exercise_worker(True)

    def exercise_worker(self, borrowed):
        with tempfile.TemporaryDirectory() as dirname:
            folder = Path(dirname)
            (folder / "engine").mkdir()
            entry = folder / "atlas"
            entry.touch()
            # The fixture issues the same start/done notices as the real CLI.
            # Fork a child and grandchild in separate sessions so killing just
            # the original process group would leave both tasks behind.
            (folder / "engine" / "supervised_cli.py").write_text("""
import json, os, pathlib, signal, sys, time
control, _, action, filename = sys.argv[1:]
if control in ("timeout", "stopped"):
    print(control)
    sys.exit(3 if control == "timeout" else 2)
assert ("engine.fork_server" in sys.modules) == (os.environ["EXPECT_BORROWED_CLI"] == "yes")
fd = int(control)
os.set_inheritable(fd, False)
os.write(fd, b'Q' + (250).to_bytes(8, sys.byteorder) if action == "timeout" else b'E')
if os.fork() == 0:
    os.setsid()
    grand = os.fork()
    if grand == 0:
        os.setsid()
        time.sleep(5)
        os._exit(0)
    pathlib.Path(filename).write_text(json.dumps([os.getpid(), grand]))
    time.sleep(5)
    os._exit(0)
while not pathlib.Path(filename).exists():
    time.sleep(.002)
if action == "crash":
    os._exit(1)
if action == "done":
    os.write(fd,b'D')
    print('done')
else:
    time.sleep(5)
""")
            env = dict(os.environ, EXPECT_BORROWED_CLI="yes" if borrowed else "no")
            if borrowed:
                (folder / "native/build").mkdir(parents=True)
                (folder / "native/build/resolve_supervisor").symlink_to(SUPERVISOR)
                for source in ("__init__.py", "fork_server.py", "preloaded_cli.py"):
                    shutil.copy2(ROOT / "engine" / source, folder / "engine" / source)
                # This isolated checkout owns a separate socket. Keep the
                # server alive until all the standalone invocations complete.
                owner = subprocess.Popen([sys.executable, "-c", """
from engine import fork_server
import sys, time
with fork_server.serving(prepare_now=True):
    while not fork_server._root.is_ready():
        time.sleep(.005)
    print("ready", flush=True)
    sys.stdin.read()
"""], cwd=folder, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                self.addCleanup(lambda: None if owner.poll() is not None else owner.communicate(timeout=10))
                self.assertEqual(owner.stdout.readline(), "ready\n")
            for action, expected, sent in (("timeout", 3, None), ("crash", 2, None),
                                           ("done", 0, None), ("interrupt", 130, signal.SIGINT),
                                           ("terminate", 143, signal.SIGTERM)):
                with self.subTest(action=action):
                    pidfile = folder / action
                    with subprocess.Popen([SUPERVISOR, entry, "resolve", action, pidfile],
                                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                        if sent:
                            deadline = time.monotonic() + 4
                            while (not pidfile.exists() or not pidfile.stat().st_size) and time.monotonic() < deadline:
                                time.sleep(.005)
                            self.assertTrue(pidfile.exists())
                            proc.send_signal(sent)
                        stdout, stderr = proc.communicate(timeout=5)
                    self.assertEqual(proc.returncode, expected)
                    self.assertEqual(stdout.strip(), {"timeout":"timeout", "crash":"stopped", "done":"done"}.get(action, ""))
                    self.assertEqual(stderr, "Resolution interrupted.\n" if sent == signal.SIGINT else "")
                    self.assertFalse(any(Path(f"/proc/{pid}").exists() for pid in json.loads(pidfile.read_text())))
            if borrowed:
                # A different interpreter setting forces an isolated exec.
                proc = subprocess.run([SUPERVISOR, entry, "resolve", "done", folder / "cold"],
                                      env=dict(env, PYTHONHASHSEED="1843", EXPECT_BORROWED_CLI="no"),
                                      capture_output=True, text=True, timeout=5)
                self.assertEqual((proc.returncode, proc.stdout, proc.stderr), (0, "done\n", ""))
                owner.communicate(timeout=10)
