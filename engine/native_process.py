"""Run a native helper program's entry in a forked copy of this process.

The helper programs (plane, jung, newton_fan, graph_verify, face_gb, weighted) read a request
on standard input and answer on standard output, so a calculation can be
stopped at a deadline or bounded in memory without stopping the solver.
Starting one used to cost 15-20 ms before any arithmetic, mostly the dynamic
linker binding FLINT and Normaliz. Each program's entry is also built into a
shared library. When this process has loaded it, a fork of the process runs the
entry with the program's standard streams, environment and exit status, and the
same timeouts apply.

A fork copies only the calling thread. The child runs system calls and the
native entry, never Python objects of other threads, and it resolves no
symbols: the parent loads the entry, the C stream reset and the thread-count
setters first. Atlas's other threads wait on pipes (a lane's coordinator
watch, a verifier's reader). Without the library, off Linux, or in a process
that has native threads of its own, such as an OpenMP team left by an
earlier computation, the program is started as before.
"""
from functools import lru_cache
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import warnings

BUILD = Path(__file__).resolve().parents[1] / "native" / "build"
_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
# Program name -> (library, entry). graph_verify's entry is in the solver kernel.
HELPERS = {
    "plane": ("libplane", "atlas_plane_main"),
    "jung": ("libjung", "atlas_jung_main"),
    "newton_fan": ("libnewton_fan", "atlas_newton_fan_main"),
    "face_gb": ("libface_gb", "atlas_face_gb_main"),
    "graph_verify": ("libgroebner", "atlas_graph_verify_main"),
    "weighted": ("libgroebner", "atlas_weighted_main"),
}
# Thread counts that the helper programs read from their environment at start.
_THREAD_SETTINGS = {"OMP_NUM_THREADS": "omp_set_num_threads", "OPENBLAS_NUM_THREADS": "openblas_set_num_threads"}


@lru_cache(maxsize=1)
def _process_symbols():
    """The C stream reset calls and the BLAS/OpenMP thread-count setters, resolved before any fork."""
    libc = ctypes.CDLL(None)
    # Discard buffered data, then clear the end-of-file and error indicators:
    # a solver started as `python3 -` has read its C stdin to the end.
    calls = []
    purge, clear = getattr(libc, "__fpurge", None), getattr(libc, "clearerr", None)
    if purge is not None and clear is not None:
        streams = [ctypes.c_void_p.in_dll(libc, name) for name in ("stdin", "stdout")]
        for function in (purge, clear):
            function.argtypes, function.restype = [ctypes.c_void_p], None
            calls += [(function, stream) for stream in streams]
    setters = {}
    for library in ("libgomp.so.1", "libopenblas.so.0"):
        try:
            loaded = ctypes.CDLL(library, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            continue
        for variable, setter in _THREAD_SETTINGS.items():
            function = getattr(loaded, setter, None)
            if function is None:
                continue
            function.argtypes = [ctypes.c_int]
            setters.setdefault(variable, []).append(function)
    return calls, setters


def c_stream_reset():
    """A call that gives a fork of this process fresh C standard streams: no data, no end of file.

    Resolve it before forking: the child then resolves no symbols.
    """
    calls, _ = _process_symbols()

    def child():
        for call, stream in calls:
            call(stream)
    return child


@lru_cache(maxsize=None)
def _entry(name):
    if not sys.platform.startswith("linux"):
        return None
    library, symbol = HELPERS[name]
    try:
        if library == "libgroebner":
            from .native_library import _kernel
            loaded = _kernel()
        else:
            loaded = ctypes.CDLL(str(BUILD / (library + _SUFFIX)))
        function = getattr(loaded, symbol)
    except (OSError, AttributeError):
        return None
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    function.restype = ctypes.c_int
    reset = getattr(loaded, "atlas_groebner_reset_fork_child", None) if library == "libgroebner" else None
    return function, reset


def preload():
    """Warm cheap helper entries. Load Normaliz only for a full Newton fan."""
    for name in HELPERS:
        if name != "newton_fan":
            _entry(name)


def _native_threads():
    """Whether this process has threads that Python did not start, such as an OpenMP team.

    A fork copies only the calling thread. A helper that enters a parallel
    region in the child then waits for the team of its parent, which is not
    there. The fan program does so when it computes a Hilbert basis.
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("Threads:"):
                    return int(line.split()[1]) > threading.active_count()
    except (OSError, ValueError, IndexError):
        pass
    return False


def available(name):
    """Whether the helper can run in a fork of this process. Otherwise its program is started."""
    return (_entry(name) is not None and all(
        thread.daemon for thread in threading.enumerate() if thread is not threading.current_thread())
        and not _native_threads())


def _start(name, args, stdin, stdout, stderr, env):
    """Fork a child that runs the helper's entry on these descriptors. Return its pid."""
    function, reset = _entry(name)
    calls, setters = _process_symbols()
    argv = [str(BUILD / name).encode(), *(str(a).encode() for a in args)]
    array = (ctypes.c_char_p * (len(argv) + 1))(*argv, None)
    threads = [] if env is None else [(function, int(env[variable])) for variable, functions in setters.items()
                                      if env.get(variable, "").isdigit() for function in functions]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid:
        return pid
    code = 127
    try:
        # The program's process state: its streams, default signal handling,
        # no descriptors of the solver, and its own environment.
        import gc
        gc.disable()
        for sig in (signal.SIGPIPE, signal.SIGINT, signal.SIGXFSZ):
            signal.signal(sig, signal.SIG_DFL)
        for fd, target in ((stdin, 0), (stdout, 1), (stderr, 2)):
            if fd != target:
                os.dup2(fd, target)
        for entry in os.listdir("/proc/self/fd"):
            if int(entry) > 2:
                try:
                    os.close(int(entry))
                except OSError:
                    pass
        # Buffered C stream data belongs to the solver, not to the program.
        for call, stream in calls:
            call(stream)
        if env is not None:
            os.environ.clear()
            os.environ.update(env)
        for setter, count in threads:
            setter(count)
        if reset is not None:
            reset()
        code = function(len(argv), array)
    except BaseException:
        code = 127
    finally:
        os._exit(code & 0xff)


def _wait(pid):
    while True:
        try:
            _, status = os.waitpid(pid, 0)
            return os.waitstatus_to_exitcode(status)
        except InterruptedError:
            continue


def run(name, args=(), *, input="", timeout=None, env=None):
    """subprocess.run([program, *args], input=input, capture_output=True, text=True, timeout=timeout, env=env)."""
    if not available(name):
        return subprocess.run([str(BUILD / name), *map(str, args)], input=input, capture_output=True, text=True,
                              timeout=timeout, env=env)
    pipes = [os.pipe() for _ in range(3)]
    try:
        pid = _start(name, args, pipes[0][0], pipes[1][1], pipes[2][1], env)
    except BaseException:
        for pair in pipes:
            for fd in pair:
                os.close(fd)
        raise
    for fd in (pipes[0][0], pipes[1][1], pipes[2][1]):
        os.close(fd)
    process = _Child(name, pid, stdin=pipes[0][1], stdout=pipes[1][0], stderr=pipes[2][0])
    try:
        out, err = process.communicate(input, timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate("", None)
        raise subprocess.TimeoutExpired([str(BUILD / name), *map(str, args)], timeout) from None
    except BaseException:
        process.kill()
        process.wait()
        raise
    return subprocess.CompletedProcess([str(BUILD / name), *map(str, args)], process.returncode, out, err)


def start(name, args, *, stdin, stdout, stderr):
    """A long-running helper on the given descriptors (Popen's kill/wait/poll), or None."""
    if not available(name):
        return None
    return _Child(name, _start(name, args, stdin, stdout, stderr, None))


class _Child:
    def __init__(self, name, pid, *, stdin=None, stdout=None, stderr=None):
        self.args, self.pid, self.returncode = [str(BUILD / name)], pid, None
        self._fds = {"stdin": stdin, "stdout": stdout, "stderr": stderr}

    def poll(self):
        if self.returncode is None:
            done, status = os.waitpid(self.pid, os.WNOHANG)
            if done:
                self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self):
        if self.returncode is None:
            self.returncode = _wait(self.pid)
        return self.returncode

    def kill(self):
        if self.returncode is None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def communicate(self, input, timeout):
        import select
        import time
        end = None if timeout is None else time.monotonic() + timeout
        data = memoryview((input or "").encode())
        writer = self._fds["stdin"]
        if writer is not None and not data:
            os.close(writer)
            self._fds["stdin"] = writer = None
        readers = {self._fds[name]: name for name in ("stdout", "stderr") if self._fds[name] is not None}
        chunks = {name: [] for name in readers.values()}
        try:
            while readers or writer is not None:
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired(self.args, timeout)
                poller = select.poll()
                for fd in readers:
                    poller.register(fd, select.POLLIN)
                if writer is not None:
                    poller.register(writer, select.POLLOUT)
                for fd, _ in poller.poll(None if remaining is None else remaining * 1000):
                    if fd == writer:
                        try:
                            written = os.write(writer, data[:select.PIPE_BUF])
                        except BrokenPipeError:
                            written = len(data)
                        data = data[written:]
                        if not data:
                            os.close(writer)
                            self._fds["stdin"] = writer = None
                    elif fd in readers:
                        chunk = os.read(fd, 65536)
                        if chunk:
                            chunks[readers[fd]].append(chunk)
                        else:
                            os.close(fd)
                            self._fds[readers.pop(fd)] = None
        except BaseException:
            for name in ("stdin", "stdout", "stderr"):
                if self._fds[name] is not None:
                    os.close(self._fds[name])
                    self._fds[name] = None
            raise
        self.wait()
        text = {name: b"".join(parts).decode() for name, parts in chunks.items()}
        return text.get("stdout", ""), text.get("stderr", "")
