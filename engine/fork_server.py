"""Start Atlas Python workers by forking a preloaded interpreter.

Every isolated resolution used to start `python -m engine...`: an interpreter,
the engine imports and the dynamic linking of the native libraries came before
any calculation, once per worker. A fork server pays that cost once. It imports
the engine, loads the native libraries and never computes, so each child it
forks starts from a clean, warm image.

A child calls setsid(), receives its standard streams and declared descriptors
from the requester, runs the module as __main__ and exits like the interpreter
would. Its exit status reaches the requester through a pipe. Children forked for
a requester are killed when that requester's connection closes, which is how a
coordinator's death still reaches its descendants.

The server declines a request, and the caller starts the interpreter as before,
when the request needs interpreter settings fixed at startup, or when the engine
sources or native libraries changed after the server loaded them.
"""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# Startup-only settings: the interpreter and the engine read these at import.
_STARTUP_PREFIXES = ("PYTHON", "LD_", "ATLAS_GROEBNER")
_MAX_FDS = 16
_READY = b'{"ready":true}'
_MESSAGE = 1 << 16
_PRELOAD = (
    "engine.worker", "engine.resolve", "engine.exhaustive_resolution", "engine.resolution_search",
    "engine.newton_backend", "engine.newton_trials", "engine.newton_faces", "engine.newton_repair",
    "engine.newton_fan", "engine.origin_backend", "engine.local_blowups", "engine.general_resolution",
    "engine.reduced_centre", "engine.atlas", "engine.strata", "engine.monomial_atlas", "engine.plane_atlas",
    "engine.plane_backend", "engine.point_atlas", "engine.graph_atlas", "engine.graph_record",
    "engine.identity_atlas", "engine.atlas_finite", "engine.atlas_gluing", "engine.atlas_torus",
    "engine.weighted_record", "engine.resolution_zetas", "engine.monodromy", "engine.topological",
    "engine.monodromy_special", "engine.chain_backend", "engine.toric_backend", "engine.native_backend",
    "engine.binomial_backend", "engine.binomial_complex", "engine.monomial_construction",
    "engine.geometric_components", "engine.resolution_complex", "engine.native_process",
)


def supported():
    import socket
    return (sys.platform.startswith("linux") and hasattr(os, "pidfd_open") and hasattr(socket, "send_fds")
            and hasattr(socket, "SOCK_SEQPACKET"))


def _startup_env(env):
    return {key: value for key, value in env.items() if key.startswith(_STARTUP_PREFIXES)}


# ---------------------------------------------------------------- requester

_lock = threading.Lock()
_enabled = False
_root = None        # this process's own server
_inherited = None   # the server that forked this process
_retired = []       # stale servers kept until their children finish


def enable():
    """Let this process start its workers from a fork server."""
    global _enabled
    _enabled = supported()


@contextmanager
def serving(*, prepare_now=False):
    """Start an entry point's workers from a fork server while it runs.

    On exit the previous setting returns and this entry point's servers stop,
    so a process that calls an entry point in place keeps its own behaviour.
    """
    global _enabled, _root
    previous = _enabled
    enable()
    try:
        if prepare_now:
            prepare()
        yield
    finally:
        _enabled = previous
        if not previous:
            with _lock:
                servers = [*_retired, *([_root] if _root is not None else [])]
                _retired.clear()
                _root = None
            for server in servers:
                server.close()
                if server.process is None:
                    continue
                if not server.served:
                    # Nothing was forked from it: do not wait for its preload.
                    server.process.kill()
                try:
                    server.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.process.kill()
                    server.process.wait()


def prepare():
    """Start this process's server ahead of its first worker."""
    if _inherited is None and _enabled:
        with _lock:
            _own_server()


def _own_server():
    global _root
    if _root is not None and _root.broken:
        if not _root.served:
            return None  # It never started: this process starts interpreters.
        _root.close()
        _root = None
    if _root is None:
        _root = _Connection.start()
    return _root


class _Connection:
    def __init__(self, sock, env, process=None, *, ready=False):
        self.sock, self.env, self.process = sock, env, process
        self.lock = threading.Lock()
        self.broken = False
        self.served = False
        self.ready = ready

    def is_ready(self):
        """Whether the server has finished preloading. A request never waits for that."""
        if self.ready or self.broken:
            return self.ready
        with self.lock:
            if not self.ready and _readable(self.sock.fileno(), 0):
                try:
                    data = self.sock.recv(_MESSAGE)
                except OSError:
                    data = b""
                if data == _READY:
                    self.ready = True
                else:
                    self.broken = True
        return self.ready

    @classmethod
    def start(cls):
        import socket
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        env = dict(os.environ)
        try:
            process = subprocess.Popen([sys.executable, "-m", "engine.fork_server", str(theirs.fileno())],
                                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       pass_fds=(theirs.fileno(),), start_new_session=True)
        except OSError:
            ours.close()
            return None
        finally:
            theirs.close()
        return cls(ours, _startup_env(env), process)

    def close(self):
        self.broken = True
        try:
            self.sock.close()
        except OSError:
            pass

    def spawn(self, request, fds):
        """Return (pid, status_fd), or a reason why this server declines."""
        import socket
        with self.lock:
            if self.broken:
                return "unavailable"
            try:
                socket.send_fds(self.sock, [json.dumps(request).encode()], fds)
                data, received, _, _ = socket.recv_fds(self.sock, _MESSAGE, 1)
            except OSError:
                self.broken = True
                return "unavailable"
        if not data:
            self.broken = True
            for fd in received:
                os.close(fd)
            return "unavailable"
        reply = json.loads(data)
        if "pid" not in reply or len(received) != 1:
            for fd in received:
                os.close(fd)
            return reply.get("declined", "unavailable")
        self.served = True
        return reply["pid"], received[0]

    def retire(self):
        import socket
        with self.lock:
            try:
                socket.send_fds(self.sock, [json.dumps({"op": "retire"}).encode()], [])
            except OSError:
                self.broken = True


def _server_for(env):
    """The connection to use with this child environment, starting one if this process may."""
    global _root
    if _inherited is not None:
        return None if _inherited.broken or _startup_env(env) != _inherited.env else _inherited
    if not _enabled:
        return None
    with _lock:
        server = _own_server()
        if server is None or _startup_env(env) != server.env:
            return None
    # Until the server has preloaded the engine, start the interpreter as
    # before rather than wait for it.
    return server if server.is_ready() else None


def attached():
    """Whether this process starts workers from a fork server: its own, or the one that forked it."""
    return _enabled or _inherited is not None


def available(env=None):
    """Whether popen() would fork an engine module with this environment from a server."""
    return _server_for(dict(os.environ if env is None else env)) is not None


def _module_command(args):
    if (isinstance(args, (list, tuple)) and len(args) >= 3 and args[0] == sys.executable and args[1] == "-m"
            and isinstance(args[2], str) and args[2].startswith("engine.") and all(isinstance(a, str) for a in args)):
        return args[2], list(args[3:])
    return None


_SUPPORTED = {"stdin", "stdout", "stderr", "text", "cwd", "env", "start_new_session", "pass_fds"}


def popen(args, **options):
    """subprocess.Popen, forked from a preloaded engine when this process has a server."""
    command = _module_command(args)
    if command is not None and not set(options) - _SUPPORTED:
        env = dict(os.environ if options.get("env") is None else options["env"])
        server = _server_for(env)
        if server is not None:
            process = _forked(server, args, command, env, options)
            if process is not None:
                return process
    return subprocess.Popen(args, **options)


def run(args, *, input=None, capture_output=False, timeout=None, check=False, **options):
    """subprocess.run on top of `popen`. A timeout stops the child's whole session."""
    if capture_output:
        options["stdout"] = options["stderr"] = subprocess.PIPE
    if input is not None:
        options["stdin"] = subprocess.PIPE
    with popen(args, **options) as process:
        try:
            out, err = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            kill_session(process)
            process.communicate()
            raise subprocess.TimeoutExpired(args, timeout, output=error.output, stderr=error.stderr) from None
        except BaseException:
            kill_session(process)
            raise
    result = subprocess.CompletedProcess(args, process.returncode, out, err)
    if check:
        result.check_returncode()
    return result


def kill_session(process):
    """Stop a child and the native helpers in its session or process group."""
    if process.poll() is not None:
        return
    try:
        if isinstance(process, ForkedProcess):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _forked(server, args, command, env, options):
    module, arguments = command
    targets, fds, close_after, parent = [], [], [], {}
    try:
        for target, name in ((0, "stdin"), (1, "stdout"), (2, "stderr")):
            value = options.get(name)
            if value is None:
                fd = target
                try:
                    os.fstat(fd)
                except OSError:
                    # A closed inherited stream: the child gets /dev/null instead.
                    fd = os.open(os.devnull, os.O_RDWR)
                    close_after.append(fd)
            elif value == subprocess.PIPE:
                r, w = os.pipe()
                fd, parent[name] = (r, w) if target == 0 else (w, r)
                close_after.append(fd)
            elif value == subprocess.DEVNULL:
                fd = os.open(os.devnull, os.O_RDWR)
                close_after.append(fd)
            elif value == subprocess.STDOUT and target == 2:
                fd = fds[1]
            elif isinstance(value, int):
                fd = value
            else:
                fd = value.fileno()
            targets.append(target)
            fds.append(fd)
        for fd in options.get("pass_fds", ()):
            targets.append(fd)
            fds.append(fd)
        request = {"op": "spawn", "module": module, "args": arguments, "env": env, "targets": targets,
                   "cwd": str(options.get("cwd") or os.getcwd())}
        reply = server.spawn(request, fds)
        if reply == "stale":
            reply = _replace_stale(server, request, fds)
        if not isinstance(reply, tuple):
            for fd in parent.values():
                os.close(fd)
            parent.clear()
            return None
    finally:
        for fd in close_after:
            os.close(fd)
    pid, status = reply
    return ForkedProcess(args, pid, status, parent, bool(options.get("text")))


def _replace_stale(server, request, fds):
    # Engine sources or native libraries changed. Children already started keep
    # the old server until they finish. New work loads the current code.
    global _root
    with _lock:
        if server is not _root:
            return "unavailable"
        server.retire()
        _retired.append(server)
        _root = None
        fresh = _own_server()
    if fresh is None or not fresh.is_ready():
        return "unavailable"
    return fresh.spawn(request, fds)


def _readable(fd, timeout):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(None if timeout is None else max(0, timeout) * 1000))


def _text_encoding():
    choose = getattr(subprocess, "_text_encoding", None)
    if choose is not None:
        return choose()
    import locale
    return locale.getpreferredencoding(False)


class ForkedProcess:
    """The subset of subprocess.Popen used by Atlas, for a child of the fork server."""

    def __init__(self, args, pid, status_fd, pipes, text):
        self.args, self.pid, self.returncode = args, pid, None
        # Several threads may wait for the same child, as with Popen. The
        # status descriptor stays open until this object is released.
        self._status = status_fd
        self._status_lock = threading.Lock()
        self.text_mode = text
        self.encoding = _text_encoding() if text else None
        self._input = None
        self._chunks = {}
        self.stdin = self.stdout = self.stderr = None
        if "stdin" in pipes:
            raw = io.open(pipes["stdin"], "wb", -1)
            self.stdin = io.TextIOWrapper(raw, encoding=self.encoding, write_through=True) if text else raw
        for name in ("stdout", "stderr"):
            if name in pipes:
                raw = io.open(pipes[name], "rb", -1)
                setattr(self, name, io.TextIOWrapper(raw, encoding=self.encoding) if text else raw)

    def _read_status(self, timeout):
        if self.returncode is not None:
            return self.returncode
        if not _readable(self._status, timeout):
            return None
        with self._status_lock:
            if self.returncode is not None:
                return self.returncode
            data = b""
            while len(data) < 4:
                chunk = os.read(self._status, 4 - len(data))
                if not chunk:
                    break
                data += chunk
            if len(data) == 4:
                self.returncode = struct.unpack("i", data)[0]
            else:
                # The server stopped without reporting: stop the orphan as well.
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except OSError:
                    pass
                self.returncode = -signal.SIGKILL
        return self.returncode

    def poll(self):
        return self._read_status(0)

    def wait(self, timeout=None):
        end = None if timeout is None else time.monotonic() + timeout
        while self.returncode is None:
            remaining = None if end is None else end - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(self.args, timeout)
            self._read_status(remaining)
        return self.returncode

    def send_signal(self, sig):
        if self.poll() is None:
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                pass

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def communicate(self, input=None, timeout=None):
        end = None if timeout is None else time.monotonic() + timeout
        if self._input is None:
            data = input or b""
            if self.text_mode and isinstance(data, str):
                data = data.encode(self.encoding)
            self._input = memoryview(data)
            if self.stdin is not None:
                self.stdin.flush()
                if not data:
                    self.stdin.close()
        readers = {}
        for name in ("stdout", "stderr"):
            stream = getattr(self, name)
            if stream is not None and not stream.closed:
                self._chunks.setdefault(name, [])
                readers[stream.fileno()] = name
        writer = self.stdin.fileno() if self.stdin is not None and not self.stdin.closed else None
        while readers or writer is not None:
            remaining = None if end is None else end - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(self.args, timeout, *self._collected())
            poller = select.poll()
            for fd in readers:
                poller.register(fd, select.POLLIN)
            if writer is not None:
                poller.register(writer, select.POLLOUT)
            events = dict(poller.poll(None if remaining is None else max(0, remaining) * 1000))
            rlist = [fd for fd in readers if fd in events]
            wlist = [writer] if writer is not None and writer in events else []
            if writer in wlist:
                try:
                    written = os.write(writer, self._input[:select.PIPE_BUF])
                except BrokenPipeError:
                    written = len(self._input)
                self._input = self._input[written:]
                if not self._input:
                    self.stdin.close()
                    writer = None
            for fd in rlist:
                chunk = os.read(fd, 32768)
                if chunk:
                    self._chunks[readers[fd]].append(chunk)
                else:
                    getattr(self, readers.pop(fd)).close()
        self.wait(None if end is None else max(0, end - time.monotonic()))
        return self._collected()

    def _collected(self):
        def output(name):
            if name not in self._chunks:
                return None
            data = b"".join(self._chunks[name])
            if self.text_mode:
                data = data.decode(self.encoding).replace("\r\n", "\n").replace("\r", "\n")
            return data
        return output("stdout"), output("stderr")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for name in ("stdout", "stderr", "stdin"):
            stream = getattr(self, name)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self.wait()

    def __del__(self):
        if self._status is not None:
            try:
                os.close(self._status)
            except OSError:
                pass


# ------------------------------------------------------------------- server

def _code_signature():
    """Engine sources and native build products at server start."""
    paths = [*ROOT.joinpath("engine").glob("*.py"), *ROOT.joinpath("native", "build").glob("*")]
    stamps = []
    for path in paths:
        try:
            stamps.append((str(path), os.stat(path).st_mtime_ns))
        except OSError:
            pass
    return sorted(stamps)


def _preload():
    import importlib
    for name in _PRELOAD:
        try:
            importlib.import_module(name)
        except Exception:
            pass  # A child imports what it needs and reports its own errors.
    try:
        from .native_library import _kernel
        _kernel()
        from .native_process import preload
        preload()
    except Exception:
        pass


def _serve(root_fd):
    import socket
    root = socket.socket(fileno=root_fd)
    from . import preloaded_cli
    signature = _code_signature()
    _preload()
    startup = _startup_env(os.environ)
    try:
        root.send(_READY)
    except OSError:
        return
    poller = select.poll()
    connections = {root.fileno(): root}
    listener = preloaded_cli.listener(ROOT)
    visitors = {}
    guardians = {}
    if listener is not None:
        poller.register(listener, select.POLLIN)
    owners = {}        # connection fd -> set of child pids
    children = {}      # pidfd -> (pid, status write fd, connection fd)
    retired = False
    poller.register(root.fileno(), select.POLLIN)

    def stop(connection_fd):
        connection = connections.pop(connection_fd)
        poller.unregister(connection_fd)
        for pid in owners.pop(connection_fd, ()):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        connection.close()

    while (connections or guardians) and not (retired and not children and not guardians):
        for fd, _ in poller.poll():
            if listener is not None and fd == listener.fileno():
                try:
                    connection, _ = listener.accept()
                except BlockingIOError:
                    continue
                # Never wait on another user's socket or admit it to the pool.
                _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if retired or uid != os.geteuid() or len(visitors) >= 16:
                    connection.close()
                    continue
                visitors[connection.fileno()] = connection
                poller.register(connection, select.POLLIN)
                continue
            if fd in visitors:
                connection = visitors.pop(fd)
                poller.unregister(fd)
                received = []
                try:
                    data, received, flags, _ = socket.recv_fds(connection, _MESSAGE, 5)
                    pid = preloaded_cli.accept(connection, ROOT, data, received, flags,
                                               lambda: not retired and _code_signature() == signature)
                    if pid is not None:
                        pidfd = os.pidfd_open(pid)
                        guardians[pidfd] = pid
                        poller.register(pidfd, select.POLLIN)
                except OSError:
                    pass
                finally:
                    connection.close()
                    for extra in received:
                        os.close(extra)
                continue
            if fd in guardians:
                poller.unregister(fd)
                os.waitpid(guardians.pop(fd), 0)
                os.close(fd)
                continue
            if fd in children:
                pid, status, owner = children.pop(fd)
                poller.unregister(fd)
                _, code = os.waitpid(pid, 0)
                os.close(fd)
                try:
                    os.write(status, struct.pack("i", os.waitstatus_to_exitcode(code)))
                except OSError:
                    pass
                os.close(status)
                owners.get(owner, set()).discard(pid)
                continue
            if fd not in connections:
                continue
            connection = connections[fd]
            try:
                data, received, flags, _ = socket.recv_fds(connection, _MESSAGE, _MAX_FDS)
            except OSError:
                data, received, flags = b"", [], 0
            if not data:
                for extra in received:
                    os.close(extra)
                if fd == root.fileno():
                    retired = True
                stop(fd)
                continue
            try:
                request = json.loads(data)
                if request.get("op") == "retire" and fd == root.fileno():
                    retired = True
                    continue
                if retired and fd == root.fileno():
                    reply = {"declined": "retired"}
                elif flags & socket.MSG_CTRUNC or request.get("op") != "spawn":
                    reply = {"declined": "invalid"}
                elif _startup_env(request["env"]) != startup:
                    reply = {"declined": "environment"}
                elif fd == root.fileno() and _code_signature() != signature:
                    reply = {"declined": "stale"}
                elif (not isinstance(request.get("module"), str) or not request["module"].startswith("engine.")
                      or len(request["targets"]) != len(received)):
                    reply = {"declined": "invalid"}
                else:
                    reply = None
            except (ValueError, KeyError, TypeError, AttributeError):
                reply = {"declined": "invalid"}
            if reply is not None:
                for extra in received:
                    os.close(extra)
                try:
                    socket.send_fds(connection, [json.dumps(reply).encode()], [])
                except OSError:
                    stop(fd)
                continue
            status_r, status_w = os.pipe()
            ready_r, ready_w = os.pipe()
            ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            pid = os.fork()
            if pid == 0:
                _child(request, received, theirs.fileno(), ready_w)
            theirs.close()
            # Like Popen(start_new_session=True), answer once the child leads
            # its own session, so the requester can signal its process group.
            os.close(ready_w)
            os.read(ready_r, 1)
            os.close(ready_r)
            for extra in received:
                os.close(extra)
            pidfd = os.pidfd_open(pid)
            children[pidfd] = (pid, status_w, fd)
            owners.setdefault(fd, set()).add(pid)
            poller.register(pidfd, select.POLLIN)
            connections[ours.fileno()] = ours
            poller.register(ours.fileno(), select.POLLIN)
            try:
                socket.send_fds(connection, [json.dumps({"pid": pid}).encode()], [status_r])
            except OSError:
                stop(fd)
            os.close(status_r)


def _child(request, received, connection_fd, ready):
    """Become the requested module's __main__. Never returns."""
    import socket
    import fcntl
    code = 1
    try:
        os.setsid()
        os.write(ready, b"1")
        # Park the kept descriptors above every target, close the server's
        # descriptors, then place the streams and declared descriptors.
        base = max(256, *(target + 1 for target in request["targets"]))
        high = [fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, base) for fd in [*received, connection_fd]]
        for name in os.listdir("/proc/self/fd"):
            fd = int(name)
            if fd not in high:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for fd, target in zip(high, request["targets"]):
            os.dup2(fd, target, inheritable=True)
        for fd in high[:-1]:
            os.close(fd)
        global _inherited, _enabled
        _enabled = True
        _inherited = _Connection(socket.socket(fileno=high[-1]), _startup_env(request["env"]), ready=True)
        os.chdir(request["cwd"])
        os.environ.clear()
        os.environ.update(request["env"])
        code = _run_main(request["module"], request["args"])
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(code)


def _run_main(module, args):
    """Execute the module's code as `python -m module args` would."""
    import atexit
    import builtins
    import importlib.util
    import types
    code = 0
    try:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.loader is None:
            print(f"{sys.executable}: No module named {module}", file=sys.stderr)
            return 1
        main = types.ModuleType("__main__")
        main.__dict__.update(__file__=spec.origin, __cached__=spec.cached, __loader__=spec.loader,
                             __package__=spec.parent, __spec__=spec, __builtins__=builtins)
        sys.modules["__main__"] = main
        sys.argv = [spec.origin, *args]
        exec(spec.loader.get_code(module), main.__dict__)
    except SystemExit as stop:
        if stop.code is None:
            code = 0
        elif isinstance(stop.code, int):
            code = stop.code
        else:
            print(stop.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        code = 128 + signal.SIGINT
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    shutdown = getattr(threading, "_shutdown", None)
    if shutdown is not None:
        shutdown()
    atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    return code


if __name__ == "__main__":
    # Serve from the importable module so children share its state.
    from engine.fork_server import _serve as serve
    serve(int(sys.argv[1]))
